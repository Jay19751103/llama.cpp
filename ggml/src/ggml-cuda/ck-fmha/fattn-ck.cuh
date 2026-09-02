// Composable Kernel (ck_tile) FMHA forward dispatch for the ggml HIP backend.
//
// The CK path is opt-in at build time (GGML_HIP_CK_FMHA) and at runtime (GGML_CK_FA=1).
// When either is off, ggml_cuda_flash_attn_ext_ck() is a no-op returning false so the
// caller falls back to the built-in ggml FlashAttention kernels.
#pragma once

#include "../common.cuh"

#ifdef GGML_HIP_CK_FMHA
// Try to run GGML_OP_FLASH_ATTN_EXT via the vendored ck_tile FMHA library.
// Returns true if CK handled dst (result written), false to fall back.
bool ggml_cuda_flash_attn_ext_ck(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#else
static inline bool ggml_cuda_flash_attn_ext_ck(ggml_backend_cuda_context &, ggml_tensor *) {
    return false;
}
#endif // GGML_HIP_CK_FMHA
