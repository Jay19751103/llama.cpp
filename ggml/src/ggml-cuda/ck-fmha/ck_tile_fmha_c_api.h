// C ABI shim for ck_tile FMHA forward, for out-of-tree integration (e.g. llama.cpp
// ggml-cuda/HIP backend on gfx1100). Keep this header in sync with the copy in
// composable_kernel (example/ck_tile/01_fmha/ck_tile_fmha_c_api.h). POD only, stable C ABI.
//
// SPDX-License-Identifier: MIT
#ifndef CK_TILE_FMHA_C_API_H
#define CK_TILE_FMHA_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// dtype
enum ck_fa_dtype { CK_FA_FP16 = 0, CK_FA_BF16 = 1 };
// mask mode (matches ck_tile mask_enum)
enum ck_fa_mask { CK_FA_MASK_NONE = 0, CK_FA_MASK_TOP_LEFT = 1, CK_FA_MASK_BOTTOM_RIGHT = 2 };
// bias mode (matches ck_tile bias_enum: no_bias / elementwise)
enum ck_fa_bias { CK_FA_BIAS_NONE = 0, CK_FA_BIAS_ELEMENTWISE = 1 };

// All strides are in ELEMENTS (not bytes). Batch mode only. Row-major V.
typedef struct ck_fa_params {
    // problem size
    int32_t hdim_q;
    int32_t hdim_v;
    int32_t batch;
    int32_t nhead_q;
    int32_t nhead_k;
    int32_t seqlen_q;
    int32_t seqlen_k;
    int32_t max_seqlen_q;

    // device pointers
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // additive bias/mask [seqlen_q, seqlen_k], or NULL
    void*       o_ptr;
    void*       lse_ptr;  // scratch [batch, nhead_q, seqlen_q] fp32, or NULL if has_lse==0

    // strides (elements)
    int64_t stride_q, stride_k, stride_v, stride_o, stride_bias;
    int64_t nhead_stride_q, nhead_stride_k, nhead_stride_v, nhead_stride_o, nhead_stride_bias, nhead_stride_lse;
    int64_t batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_o, batch_stride_bias, batch_stride_lse;

    float   scale_s;      // softmax scale (1/sqrt(d) applied by caller if 0? no: pass explicit)
    int32_t dtype;        // ck_fa_dtype
    int32_t mask_mode;    // ck_fa_mask
    int32_t bias_mode;    // ck_fa_bias
    int32_t has_lse;      // 1 if lse_ptr valid and instances have lse
    int32_t window_left;  // for masks; -1 if unused
    int32_t window_right; // for masks; -1 if unused
} ck_fa_params;

// Returns 0 on success (kernel launched), negative if no matching instance / unsupported.
// 'stream' is a hipStream_t.
int ck_tile_fmha_fwd(const ck_fa_params* p, void* stream);

// Returns 1 if a matching instance is available for these params, else 0. Does not launch.
int ck_tile_fmha_fwd_supported(const ck_fa_params* p);

#ifdef __cplusplus
}
#endif

#endif // CK_TILE_FMHA_C_API_H
