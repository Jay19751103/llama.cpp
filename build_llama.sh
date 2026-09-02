#!/usr/bin/env bash
#
# build_llama.sh — build the HIP llama.cpp binaries for one or more AMD GPU archs,
# with an optional ck_tile FMHA (composable_kernel) integration.
#
# The ck path is OPT-IN via --ck. When NOT given, nothing CK-related is built,
# linked or staged (the llama.cpp CMake flag GGML_HIP_CK_FMHA stays OFF and the
# fattn-ck path compiles out to stubs).
#
# Building the CK library itself is controlled separately by --build_ck:
#   * --build_ck (implies --ck): (re)compile the `ck_tile_fmha` SHARED target in
#     the sibling composable_kernel checkout, then copy the resulting .so into the
#     vendored ggml tree at ggml/src/ggml-cuda/ck-fmha/lib/libck_tile_fmha.so.
#   * --ck without --build_ck: skip compiling CK and reuse the .so already
#     vendored in the ggml tree.
#
# When --ck IS given the script then:
#   1. configures llama.cpp with -DGGML_HIP_CK_FMHA=ON and compiles the ck
#      dispatch path, then builds, and
#   2. copies the vendored ggml-tree libck_tile_fmha.so into ./build/bin so it
#      sits next to the binaries (auto-loaded when GGML_CK_FA=1).
#
# The arch list is exactly what llama.cpp / CMake accept for GPU_TARGETS, e.g.
#   --archs gfx1100
#   --archs "gfx1151;gfx1100"
# Supported archs for the ck FMHA path:
#   gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1152 gfx1201 gfx1202
#
# Usage:
#   ./build_llama.sh [--archs "gfx1151;gfx1100"] [--ck] [--build_ck] [--native]
#                    [--jobs N] [--rocm /opt/rocm] [--clean] [--ck-clean]
#
# --native bakes -march=native into the CPU backend (faster, but the binaries
# only run on CPUs like the build host). Omit it (default) for portable builds
# that you move to another machine / a different CPU (e.g. EPYC).
#
# Env overrides: ARCHS, JOBS, ROCM, CK_DIR.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR"

ARCHS="${ARCHS:-gfx1151}"
ROCM="${ROCM:-/opt/rocm}"
JOBS="${JOBS:-$(nproc)}"
CK_DIR="${CK_DIR:-$(cd "$LLAMA_DIR/.." && pwd)/composable_kernel}"
WITH_CK=0
BUILD_CK=0
CLEAN=0
CK_CLEAN=0
# GGML_NATIVE bakes -march=native into the CPU backend; OFF keeps binaries
# portable across CPUs (needed when moving build/bin to a different machine).
NATIVE=OFF

while [ $# -gt 0 ]; do case "$1" in
  --archs)    ARCHS="$2"; shift 2;;
  --ck)       WITH_CK=1; shift;;
  --build_ck) BUILD_CK=1; WITH_CK=1; shift;;
  --native)   NATIVE=ON; shift;;
  --jobs)     JOBS="$2"; shift 2;;
  --rocm)     ROCM="$2"; shift 2;;
  --ck-dir)   CK_DIR="$2"; shift 2;;
  --clean)    CLEAN=1; shift;;
  --ck-clean) CK_CLEAN=1; shift;;
  -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 1;;
esac; done

# Normalize separators: accept both ';' and ',' -> CMake wants ';'.
ARCHS_CMAKE="${ARCHS//,/;}"

HIP_CLANGXX="$ROCM/llvm/bin/clang++"
BUILD="$LLAMA_DIR/build"
BIN="$BUILD/bin"
CK_LIB_DIR="$LLAMA_DIR/ggml/src/ggml-cuda/ck-fmha/lib"
VENDORED="$CK_LIB_DIR/libck_tile_fmha.so"

echo "==> archs=$ARCHS_CMAKE  ck=$WITH_CK  build_ck=$BUILD_CK  native=$NATIVE  jobs=$JOBS  rocm=$ROCM"
[ -x "$HIP_CLANGXX" ] || { echo "missing HIP compiler: $HIP_CLANGXX" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Step 1+2: build the ck_tile FMHA shared library and vendor it into llama.cpp.
# Only when --build_ck is given; otherwise the vendored ggml-tree .so is reused.
# ---------------------------------------------------------------------------
if [ "$BUILD_CK" = 1 ]; then
  [ -d "$CK_DIR" ] || { echo "composable_kernel dir not found: $CK_DIR (use --ck-dir)" >&2; exit 1; }
  [ -f "$CK_DIR/example/ck_tile/01_fmha/ck_tile_fmha_c_api.cpp" ] || {
    echo "ck C-ABI shim missing in $CK_DIR; apply 0001-CK-patch.patch first" >&2; exit 1; }

  CK_BUILD="$CK_DIR/build_fmha"
  [ "$CK_CLEAN" = 1 ] && { echo "==> clean $CK_BUILD"; rm -rf "$CK_BUILD"; }

  echo "==> configuring composable_kernel ck_tile_fmha ($ARCHS_CMAKE)"
  cmake -S "$CK_DIR" -B "$CK_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$HIP_CLANGXX" \
    -DCMAKE_HIP_COMPILER="$HIP_CLANGXX" \
    -DCMAKE_CXX_FLAGS="-Wno-gnu-line-marker -fbracket-depth=1024" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DENABLE_CLANG_CPP_CHECKS=OFF \
    -DGPU_TARGETS="$ARCHS_CMAKE" \
    -DBUILD_DEV=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_CK_EXAMPLES=ON \
    -DBUILD_CK_TUTORIALS=OFF \
    -DBUILD_CK_TILE_ENGINE=OFF \
    -DBUILD_CK_PROFILER=OFF \
    -DBUILD_CK_DEVICE_INSTANCES=OFF \
    -DFMHA_FWD_ENABLE_APIS=fwd \
    -DFMHA_FWD_RECEIPT=2 \
    -DFMHA_FWD_OPTDIM=128,256

  echo "==> building ck_tile_fmha (this generates + compiles the FMHA instances)"
  cmake --build "$CK_BUILD" --target ck_tile_fmha -j"$JOBS"

  CK_SO="$(find "$CK_BUILD" -name libck_tile_fmha.so -print -quit)"
  [ -n "$CK_SO" ] || { echo "build finished but libck_tile_fmha.so not found under $CK_BUILD" >&2; exit 1; }

  echo "==> vendoring $(basename "$CK_SO") -> $VENDORED"
  mkdir -p "$CK_LIB_DIR"
  cp -f "$CK_SO" "$VENDORED"
  echo "    $(du -h "$VENDORED" | cut -f1)"
fi

# ---------------------------------------------------------------------------
# Step 3: configure + build llama.cpp (CK on/off decided by --ck).
# ---------------------------------------------------------------------------
[ "$CLEAN" = 1 ] && { echo "==> clean $BUILD"; rm -rf "$BUILD"; }

CK_CMAKE_FLAG=OFF
[ "$WITH_CK" = 1 ] && CK_CMAKE_FLAG=ON

echo "==> configuring llama.cpp (HIP $ARCHS_CMAKE, GGML_HIP_CK_FMHA=$CK_CMAKE_FLAG)"
cmake -S "$LLAMA_DIR" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_HIP_CK_FMHA="$CK_CMAKE_FLAG" \
  -DAMDGPU_TARGETS="$ARCHS_CMAKE" -DGPU_TARGETS="$ARCHS_CMAKE" \
  -DCMAKE_HIP_COMPILER="$HIP_CLANGXX" \
  -DGGML_NATIVE="$NATIVE"

echo "==> building llama binaries"
cmake --build "$BUILD" \
  --target llama-bench llama-server llama-cli test-backend-ops -j"$JOBS"

echo
echo "==> DONE. binaries in $BIN"
if [ "$WITH_CK" = 1 ]; then
  # Stage the vendored ggml-tree .so next to the binaries.
  # With --build_ck it was just refreshed above; without it we reuse the committed copy.
  if [ -f "$VENDORED" ]; then
    mkdir -p "$BIN"
    cp -f "$VENDORED" "$BIN/libck_tile_fmha.so"
    echo "    staged $VENDORED -> $BIN/libck_tile_fmha.so (auto-loaded when GGML_CK_FA=1)."
  else
    echo "    WARN: $VENDORED missing; run with --build_ck to build it" >&2
  fi
  cat <<EOF

Run with ck FMHA (dense d128, e.g. Llama):
  GGML_CK_FA=1 GGML_CK_FA_CAUSAL=1 GGML_CK_FA_DECODE=1 GGML_CUDA_FATTN_FORCE=ck \\
    $BIN/llama-bench -m model.gguf -fa 1 -ngl 99

Correctness gate (must be N/N; do NOT set GGML_CK_FA_CAUSAL):
  GGML_CK_FA=1 GGML_CUDA_FATTN_FORCE=ck $BIN/test-backend-ops -o FLASH_ATTN_EXT
EOF
else
  echo "    (built without ck_tile FMHA; pass --ck to enable it)"
fi
