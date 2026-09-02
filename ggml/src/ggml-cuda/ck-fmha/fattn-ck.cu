// Composable Kernel (ck_tile) FMHA forward dispatch for the ggml HIP backend.
//
// Compiled only when GGML_HIP_CK_FMHA is defined; links against the vendored
// libck_tile_fmha.so (C ABI in ck_tile_fmha_c_api.h). Converts the ggml
// GGML_OP_FLASH_ATTN_EXT problem into the CK batch-mode FMHA args and, on
// success, writes the F32 result into dst. Any unsupported case returns false
// so the caller falls back to the built-in ggml FlashAttention kernels.

#include "fattn-ck.cuh"

#ifdef GGML_HIP_CK_FMHA

#include "../convert.cuh"
#include "ck_tile_fmha_c_api.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct ck_fa_env {
    bool enabled;   // GGML_CK_FA=1 (or GGML_CUDA_FATTN_FORCE=ck)
    bool causal;    // GGML_CK_FA_CAUSAL=1: use CK top-left causal mask instead of the explicit ggml mask
    bool decode;    // GGML_CK_FA_DECODE=1: allow the CK path for seqlen_q==1 (decode)
};

const ck_fa_env & ck_fa_get_env() {
    static const ck_fa_env env = [] {
        auto flag = [](const char * name) {
            const char * v = getenv(name);
            return v && v[0] && v[0] != '0';
        };
        const char * force = getenv("GGML_CUDA_FATTN_FORCE");
        const bool force_ck = force && std::string(force) == "ck";
        return ck_fa_env{ flag("GGML_CK_FA") || force_ck, flag("GGML_CK_FA_CAUSAL"), flag("GGML_CK_FA_DECODE") };
    }();
    return env;
}

} // namespace

bool ggml_cuda_flash_attn_ext_ck(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ck_fa_env & env = ck_fa_get_env();
    if (!env.enabled) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // CK instances are built for fp16/bf16 KV, dense head dims 128/256, no ALiBi, no soft-cap.
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_BF16)) {
        return false;
    }
    if (mask && mask->type != GGML_TYPE_F16) {
        return false;
    }

    const int64_t hdim_q   = Q->ne[0];
    const int64_t hdim_v   = V->ne[0];
    const int64_t seqlen_q = Q->ne[1];
    const int64_t nhead_q  = Q->ne[2];
    const int64_t batch    = Q->ne[3];
    const int64_t seqlen_k = K->ne[1];
    const int64_t nhead_k  = K->ne[2];

    if (hdim_q != hdim_v || (hdim_q != 128 && hdim_q != 256)) {
        return false;
    }
    if (nhead_k == 0 || nhead_q % nhead_k != 0) {
        return false;
    }
    if (K->ne[0] != hdim_q || K->ne[3] != batch || V->ne[1] != seqlen_k || V->ne[2] != nhead_k || V->ne[3] != batch) {
        return false;
    }
    if (seqlen_q == 1 && !env.decode) {
        return false; // decode is opt-in; the built-in vec kernel is usually better here
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false; // ALiBi / logit soft-cap not covered by the built CK instances
    }

    // Contiguous unit stride on the head dim is required to map ggml strides -> CK element strides.
    if (Q->nb[0] != ggml_element_size(Q) || K->nb[0] != ggml_element_size(K) || V->nb[0] != ggml_element_size(V)) {
        return false;
    }
    if (!ggml_is_contiguously_allocated(Q)) {
        return false; // Q is converted with a flat F32->fp16/bf16 pass
    }
    // The explicit mask is broadcast over heads (ne[2]==1); ne[3] may broadcast over batch.
    if (mask && (mask->ne[2] != 1 || (mask->ne[3] != 1 && mask->ne[3] != batch))) {
        return false;
    }
    if (mask && mask->nb[0] != ggml_element_size(mask)) {
        return false;
    }

    const bool is_bf16 = (K->type == GGML_TYPE_BF16);

    ggml_cuda_pool & pool   = ctx.pool();
    cudaStream_t     stream = ctx.stream();

    const int64_t n_q = ggml_nelements(Q);
    const int64_t n_o = ggml_nelements(dst);

    // fp16 and bf16 are both 2 bytes; allocate raw scratch and reinterpret.
    ggml_cuda_pool_alloc<char> q_buf(pool, n_q * sizeof(half));
    ggml_cuda_pool_alloc<char> o_buf(pool, n_o * sizeof(half));

    // Q: F32 -> fp16/bf16, preserving ggml's contiguous [hdim_q, seqlen_q, nhead_q, batch] order.
    if (is_bf16) {
        ggml_get_to_bf16_cuda(GGML_TYPE_F32)(Q->data, (nv_bfloat16 *) q_buf.get(), n_q, stream);
    } else {
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(Q->data, (half *) q_buf.get(), n_q, stream);
    }

    const int64_t esk = ggml_element_size(K);
    const int64_t esv = ggml_element_size(V);

    ck_fa_params p;
    memset(&p, 0, sizeof(p));

    p.hdim_q       = (int32_t) hdim_q;
    p.hdim_v       = (int32_t) hdim_v;
    p.batch        = (int32_t) batch;
    p.nhead_q      = (int32_t) nhead_q;
    p.nhead_k      = (int32_t) nhead_k;
    p.seqlen_q     = (int32_t) seqlen_q;
    p.seqlen_k     = (int32_t) seqlen_k;
    p.max_seqlen_q = (int32_t) seqlen_q;

    p.q_ptr = q_buf.get();
    p.k_ptr = K->data;
    p.v_ptr = V->data;
    p.o_ptr = o_buf.get();

    // Q scratch is contiguous [hdim_q, seqlen_q, nhead_q, batch].
    p.stride_q       = hdim_q;
    p.nhead_stride_q = hdim_q * seqlen_q;
    p.batch_stride_q = hdim_q * seqlen_q * nhead_q;

    // K/V map directly from ggml byte strides (row-major V: stride is the token stride).
    p.stride_k       = K->nb[1] / esk;
    p.nhead_stride_k = K->nb[2] / esk;
    p.batch_stride_k = K->nb[3] / esk;

    p.stride_v       = V->nb[1] / esv;
    p.nhead_stride_v = V->nb[2] / esv;
    p.batch_stride_v = V->nb[3] / esv;

    // O scratch matches dst layout [hdim_v, nhead_q, seqlen_q, batch] (contiguous).
    p.stride_o       = hdim_v * nhead_q;
    p.nhead_stride_o = hdim_v;
    p.batch_stride_o = hdim_v * nhead_q * seqlen_q;

    p.scale_s = scale;
    p.dtype   = is_bf16 ? CK_FA_BF16 : CK_FA_FP16;
    p.has_lse = 0;
    p.window_left  = -1;
    p.window_right = -1;

    if (env.causal) {
        // Use CK's built-in causal mask; ignore the explicit ggml mask.
        p.bias_mode = CK_FA_BIAS_NONE;
        p.mask_mode = CK_FA_MASK_TOP_LEFT;
    } else if (mask) {
        // Feed the ggml additive mask as an elementwise bias [seqlen_q, seqlen_k].
        const int64_t esm = ggml_element_size(mask);
        p.bias_ptr          = mask->data;
        p.bias_mode         = CK_FA_BIAS_ELEMENTWISE;
        p.mask_mode         = CK_FA_MASK_NONE;
        p.stride_bias       = mask->nb[1] / esm;                          // seqlen_q stride
        p.nhead_stride_bias = 0;                                          // broadcast over heads
        p.batch_stride_bias = (mask->ne[3] == batch) ? mask->nb[3] / esm : 0;
    } else {
        p.bias_mode = CK_FA_BIAS_NONE;
        p.mask_mode = CK_FA_MASK_NONE;
    }

    if (!ck_tile_fmha_fwd_supported(&p)) {
        return false;
    }
    if (ck_tile_fmha_fwd(&p, (void *) stream) != 0) {
        return false;
    }

    // CK wrote o in the input dtype; convert into the F32 dst (same element order).
    if (is_bf16) {
        ggml_get_to_fp32_cuda(GGML_TYPE_BF16)(o_buf.get(), (float *) dst->data, n_o, stream);
    } else {
        ggml_get_to_fp32_cuda(GGML_TYPE_F16)(o_buf.get(), (float *) dst->data, n_o, stream);
    }

    return true;
}

#endif // GGML_HIP_CK_FMHA
