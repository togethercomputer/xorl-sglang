# Kimi-K3 serving image (aarch64 / sm_90 + sm_100a + sm_103a).
#
# Base ships stock SGLang (editable at /sgl-workspace/sglang), DeepEP source
# (deepseek-ai@d28bd67 at /sgl-workspace/DeepEP), the deep_gemm pip package,
# and the CUDA 13 toolchain (nvcc + /usr/local/cuda/include/cccl).
#
# This image adds the four Kimi-K3-specific pieces that stock lacks:
#   1. the Kimi-K3 SGLang code (this repo), editable-installed
#   2. DeepEP patch + rebuild:
#        topk 11->16, SWITCH_HIDDEN += 3584, EP>8 SourceMeta alignment,
#        cross-node timeout headroom, CUDA-13 cccl include; rebuilt for
#        sm_90, sm_100a, and sm_103a
#   3. DeepGEMM upgrade to 0.1.5.post1:
#        official MegaMoE runtime-JIT header with Kimi-K3 SiTU support
#   4. FlashInfer CuTeDSL MLA DCP patch:
#        apply the seven runtime-file diffs; exclude tests absent from the wheel
#
# Build (on/for aarch64; nvcc cross-compiles the DeepEP cubin, no GPU needed):
#   docker build -f docker/kimi_k3/kimi_k3_cu13.Dockerfile \
#     --build-arg 'TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a' -t kimi-k3 .
#
# The FlashInfer MXFP4 MoE runner cubins are installed in the image below.
# The runner is auto-selected on SM100/103; the remaining kernel sources
# JIT-compile from the installed FlashInfer wheel on first launch and are
# cached.

FROM lmsysorg/sglang:v0.5.16 AS base

ARG SGL_DEEP_GEMM_VERSION="0.1.5.post1"

# Current Kimi-K3 source auto-discovers and builds its PyO3 extensions.
ARG RUST_VERSION="1.90.0"
ENV RUSTUP_HOME="/usr/local/rustup" \
    CARGO_HOME="/usr/local/cargo" \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      unzip \
      wget && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain "${RUST_VERSION}" && \
    cargo --version && \
    rustc --version && \
    rm -rf /var/lib/apt/lists/*

# Build one DeepEP wheel with native cubins for Hopper, B200, and GB300.
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0a;10.3a"

# --- 1. Kimi-K3 SGLang code (replaces the base's stock sglang, editable) ---
# Keep the installed extension modules, but discard Rust and pip build
# artifacts that are not used at runtime.
RUN rm -rf /sgl-workspace/sglang && \
    git clone --branch kimi-k3 \
      https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    rm -rf .git && \
    test ! -e .git && \
    pip install -e python --no-deps && \
    rm -rf \
      rust/target \
      rust/sglang-grpc/target \
      rust/sglang-mm/target \
      rust/sglang-server/target \
      /usr/local/cargo/registry \
      /root/.cache/pip

# --- 2. DeepEP: patch (topk16 / hidden3584 / SourceMeta / cccl) + multi-arch rebuild ---
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    bash /sgl-workspace/sglang/docker/kimi_k3/apply_deepep_k3_patch.sh && \
    rm -rf /sgl-workspace/DeepEP/build /sgl-workspace/DeepEP/dist

# --- 3. DeepGEMM: upgrade to the first release with Kimi-K3 SiTU ---
# The v0.5.16 base contains DeepGEMM 0.1.4.post1.
RUN python3 -m pip install --no-deps --force-reinstall \
    "sgl-deep-gemm==${SGL_DEEP_GEMM_VERSION}"

# Install the pinned FlashInfer MXFP4 MoE runner cubin pool.
ARG TRTLLM_GEN_MOE_CUBIN_URL="https://github.com/sgl-project/whl/releases/download/trtllm_gen_moe_cubin_20260617/trtllm_gen_moe_cubin_pool_20260617_v0613rc1.zip"
ARG TRTLLM_GEN_MOE_CUBIN_SHA256="4900501cbe782a76b08a5858f9f07152287b97cb68114466dac286366b66c192"
ARG TRTLLM_GEN_MOE_CUBIN_ARCHIVE_ROOT="trtllm_gen_moe_cubin_pool_20260617_v0613rc1"
ENV SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL="/opt/trtllm_gen_moe_cubin_pool"

RUN cubin_archive="/tmp/trtllm_gen_moe_cubin_pool.zip" && \
    cubin_extract_dir="/tmp/trtllm_gen_moe_cubin_extract" && \
    wget --no-verbose --output-document="${cubin_archive}" \
      "${TRTLLM_GEN_MOE_CUBIN_URL}" && \
    echo "${TRTLLM_GEN_MOE_CUBIN_SHA256}  ${cubin_archive}" | \
      sha256sum --check --strict - && \
    mkdir -p "${cubin_extract_dir}" && \
    unzip -q "${cubin_archive}" -d "${cubin_extract_dir}" && \
    test ! -e "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}" && \
    mv "${cubin_extract_dir}/${TRTLLM_GEN_MOE_CUBIN_ARCHIVE_ROOT}" \
      "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}" && \
    test "$(find "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}" \
      -type f -name '*.cubin' | wc -l)" -eq 1696 && \
    rm -f "${cubin_archive}" && \
    rm -rf "${cubin_extract_dir}"

# Reinstall the matching FlashInfer package trio before patching its Python
# sources. A mixed Python/cubin/JIT-cache installation fails at import time.
RUN python3 -m pip uninstall -y \
      flashinfer-python flashinfer-cubin flashinfer-jit-cache && \
    rm -rf /root/.cache/flashinfer /root/.cache/pip && \
    python3 -m pip install --no-deps \
      "flashinfer-python==0.6.17" && \
    python3 -m pip install --no-deps \
      "flashinfer-cubin==0.6.17" \
      --index-url https://flashinfer.ai/whl && \
    python3 -m pip install --no-deps \
      "flashinfer-jit-cache==0.6.17" \
      --index-url https://flashinfer.ai/whl/cu130 && \
    python3 -c 'from importlib.metadata import version; expected = "0.6.17"; packages = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"); actual = {package: version(package).split("+", 1)[0] for package in packages}; assert all(value == expected for value in actual.values()), actual' && \
    rm -rf /root/.cache/pip

ENV FLASHINFER_VERSION="0.6.17"

# --- 4. FlashInfer: CuTeDSL MLA decode-context-parallel runtime patch ---
# The FlashInfer CuTeDSL MLA decode-context-parallel patch is NOT applied on
# 0.6.17: DCP is upstream there. flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
# gained enable_dcp / cp_world / cp_rank / causal_seqlens_kv_global in 0.6.17, and
# none of those exist in 0.6.15 -- which is what the patch was adding. Measured
# against an installed 0.6.17 tree: 112 of the patch's 147 hunks are already
# present, and of the 32 that still fail, 96% of the added lines in mla/_core.py
# and 85% in mla_dispatch.py are already there, the remainder being docstrings and
# helper names upstream renamed. Keeping the patch would fail the build for no
# gain (its --dry-run gate is fatal).
#
# docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt is retained for the 0.6.15
# lineage; delete it once no branch pins 0.6.15. DCP still needs a functional
# test on 0.6.17 before this is called done.
WORKDIR /sgl-workspace/sglang
