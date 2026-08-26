"""XoRL exact-serving extensions to the upstream batch-invariant op set.

Ported from togethercomputer/xorl-sglang `main`
(`sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, commit c08786bd3),
where these functions live inline in the upstream module. On this branch the
additive capability is kept out of upstream files: the overrides twin for
`sglang.srt.batch_invariant_ops.batch_invariant_ops` re-exports everything
here onto the public module so main-lineage callers keep working.

Bit-relevant contracts (see xorl docs/k3): the LM-head chunk GEMM pins its
reduction geometry through `lookup_mm_config` / `lookup_tiera_mm_config`;
the residual-tree RMSNorm fixes an adjacent-pair reduction order; the router
GEMM/topk pin FP32 arithmetic order.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    _rms_norm_kernel,
    mean_dim,
    rms_norm_batch_invariant,
)
from sglang.xorl.bi.bi_gemm_configs import baseline_mm_config, lookup_mm_config
from sglang.xorl.bi.bi_gemm_tiera import (
    lookup_tiera_mm_config,
    lookup_tiera_router_config,
)
from triton.runtime.errors import OutOfResources

_MM_CONFIG_OOM_SHAPES: set[tuple] = set()


def _launch_with_config_fallback(launch, dtype, M, N, K, out_itemsize=None):
    """Launch a shape-pinned exact GEMM, falling back only on resource failure."""
    tiera_cfg = lookup_tiera_mm_config(dtype, M, N, K, out_itemsize=out_itemsize)
    if tiera_cfg is not None:
        try:
            launch(tiera_cfg)
            return
        except OutOfResources:
            pass

    key = (str(dtype), M, N, K, out_itemsize)
    if key in _MM_CONFIG_OOM_SHAPES:
        launch(baseline_mm_config(dtype))
        return
    try:
        launch(lookup_mm_config(dtype, M, N, K, out_itemsize=out_itemsize))
    except OutOfResources:
        _MM_CONFIG_OOM_SHAPES.add(key)
        launch(baseline_mm_config(dtype))


@triton.jit
def _add_residual_square_kernel(
    input_ptr,
    residual_ptr,
    residual_out_ptr,
    sq_ptr,
    input_row_stride: tl.constexpr,
    residual_row_stride: tl.constexpr,
    residual_out_row_stride: tl.constexpr,
    sq_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 1 of the fused residual-add RMSNorm: residual add in orig dtype and
    the per-element square in float32.

        residual_out = (x + residual).to(orig_dtype)
        sq           = residual_out.float() ** 2     # float32

    ``sq`` is then reduced by the existing batch-invariant ``mean_dim`` kernel,
    so the variance reduction order is bit-identical to the eager
    ``x.pow(2).mean(-1)`` path this replaces.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    in_row = input_ptr + row_idx * input_row_stride
    res_row = residual_ptr + row_idx * residual_row_stride
    res_out_row = residual_out_ptr + row_idx * residual_out_row_stride
    sq_row = sq_ptr + row_idx * sq_row_stride
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(in_row + col_idx, mask=mask, other=0.0)
        r = tl.load(res_row + col_idx, mask=mask, other=0.0)
        # Match torch's elementwise add for low-precision dtypes: upcast to
        # float32, add, round the result back to the original dtype. The
        # normalization then operates on this rounded value.
        s = (x.to(tl.float32) + r.to(tl.float32)).to(x.dtype)
        tl.store(res_out_row + col_idx, s, mask=mask)
        s_f32 = s.to(tl.float32)
        tl.store(sq_row + col_idx, s_f32 * s_f32, mask=mask)


@triton.jit
def _rms_normalize_with_var_kernel(
    input_ptr,
    var_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 2 of the fused residual-add RMSNorm: normalize by a precomputed
    per-row variance and multiply weight in float32, casting last.

        out = (x.float() * rsqrt(var + eps) * weight.float()).to(orig_dtype)

    ``tl.rsqrt`` bit-matches ``torch.rsqrt(var + eps)`` used by forward_native
    (``1.0 / tl.sqrt`` does not).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    in_row = input_ptr + row_idx * input_row_stride
    out_row = output_ptr + row_idx * output_row_stride
    var = tl.load(var_ptr + row_idx)
    inv_rms = tl.rsqrt(var + eps)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(in_row + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        output_f32 = x.to(tl.float32) * inv_rms * weight.to(tl.float32)
        tl.store(out_row + col_idx, output_f32.to(x.dtype), mask=mask)


def fused_add_rms_norm_batch_invariant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch-invariant fused residual-add + RMS normalization.

    Returns ``(output, residual_out)`` using the residual-tree family:
    BF16 residual addition, fixed-order fp32 variance reduction, fp32 affine
    multiply, and a final cast to the input dtype.

    The eager path is ~12 small launches per call; here it is three: a fused
    residual-add+square, the existing batch-invariant ``mean_dim`` reduction
    (reused verbatim so the variance is bit-identical to ``x.pow(2).mean(-1)``),
    and a fused normalize. The caller must only dispatch here when that exact
    configuration holds.
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape == residual.shape, "Input and residual must share a shape"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    residual_2d = residual.reshape(-1, residual.shape[-1]).contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    residual_out = torch.empty_like(input_2d)
    sq = torch.empty((n_rows, n_cols), dtype=torch.float32, device=input.device)

    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _add_residual_square_kernel[grid](
        input_2d,
        residual_2d,
        residual_out,
        sq,
        input_2d.stride(0),
        residual_2d.stride(0),
        residual_out.stride(0),
        sq.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reuse the batch-invariant mean reduction verbatim: variance is then
    # bit-identical to the residual-tree family's x.pow(2).mean(-1).
    var = mean_dim(sq, -1, keepdim=True).reshape(-1).contiguous()

    output = torch.empty_like(input_2d)
    _rms_normalize_with_var_kernel[grid](
        residual_out,
        var,
        weight,
        output,
        residual_out.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape), residual_out.reshape(original_shape)


@triton.jit
def _square_kernel(
    input_ptr,
    sq_ptr,
    input_row_stride: tl.constexpr,
    sq_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """No-residual analog of the fused-add stage 1: per-element square in float32.

        sq = x.float() ** 2

    Reduced by ``mean_dim`` for a variance bit-identical to forward_native's
    ``x.float().pow(2).mean(-1)`` (fp32 ``s * s`` == ``pow(x, 2)``).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    in_row = input_ptr + row_idx * input_row_stride
    sq_row = sq_ptr + row_idx * sq_row_stride
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(in_row + col_idx, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        tl.store(sq_row + col_idx, x_f32 * x_f32, mask=mask)


def rms_norm_residual_tree_batch_invariant(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Single-tensor RMSNorm through the serving residual tree (family-2).

    The no-residual analog of :func:`fused_add_rms_norm_batch_invariant`,
    bit-matching its normalization of an already-summed residual stream
    (``mean_dim`` variance + ``tl.rsqrt`` + fp32 weight multiply, cast last).
    Vendored identically in xorl (``sglang_rms_norm_batch_invariant``), where
    the trainer applies it to pre-summed single-tensor residual-tree sites
    (input layernorm at layer>0, final norm). Serving's own dispatch always has
    the residual stream in hand and uses the fused-add form; this function
    exists so the cross-engine family gates can pin both engines to one tree.
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    sq = torch.empty((n_rows, n_cols), dtype=torch.float32, device=input.device)

    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _square_kernel[grid](
        input_2d,
        sq,
        input_2d.stride(0),
        sq.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    var = mean_dim(sq, -1, keepdim=True).reshape(-1).contiguous()

    output = torch.empty_like(input_2d)
    _rms_normalize_with_var_kernel[grid](
        input_2d,
        var,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


# --------------------------------------------------------------------------- #
# RMSNorm kernel-family contract
#
# Two batch-invariant RMSNorm kernel families coexist, and they disagree at
# 1 ulp on rare bf16 boundary values (~2/524288 at [4096, 128]), so silently
# swapping one for the other seeds K3 divergence that amplifies downstream
# (five such seed elements are enough to open a 2.99e-5 K3).
# Each family is pinned to the serving site-class that executes it:
#   - "serving_no_residual" (family-1): the looped ``tl.sum`` + ``1.0/tl.sqrt``
#     kernel (``rms_norm_batch_invariant``), dispatched when ``residual is
#     None`` under batch-invariant mode. Site-classes: qk-norm, layer-0 input
#     layernorm.
#   - "serving_residual_tree" (family-2): the ``mean_dim`` + ``tl.rsqrt``
#     residual-tree kernels (``fused_add_rms_norm_batch_invariant`` /
#     ``rms_norm_residual_tree_batch_invariant``), dispatched for residual
#     calls under the rl-on-policy lane. Site-classes: input layernorm at
#     layer>0, post-attention layernorm, final norm.
# Every call site must name its family through ``bi_rms_norm`` /
# ``bi_fused_add_rms_norm``; never call the family kernels directly. Mirrored
# in xorl's ``xorl.ops.batch_invariant_ops``.
# --------------------------------------------------------------------------- #
RMS_NORM_FAMILY_NO_RESIDUAL = "serving_no_residual"
RMS_NORM_FAMILY_RESIDUAL_TREE = "serving_residual_tree"
RMS_NORM_FAMILIES = (RMS_NORM_FAMILY_NO_RESIDUAL, RMS_NORM_FAMILY_RESIDUAL_TREE)
RMSNormFamily = Literal["serving_no_residual", "serving_residual_tree"]


def bi_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    family: RMSNormFamily,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Single-tensor batch-invariant RMSNorm with an explicit kernel family.

    The single sanctioned entry point for both no-residual family kernels; the
    ``family`` keyword must name the serving site-class of the call site.

    ``zero_centered`` is the Gemma-style Qwen3.5 form: an fp32 upcast with the
    ``1 + weight`` scale folded in fp32, cast back last (mirrors xorl's
    ``fast_zero_centered_batch_invariant_rms_norm`` forward). It is an affine
    fold around the SAME family-1 reduction tree — not a third family — and only
    exists in no-residual form (Qwen3.5 residual-tree norms run the native
    path, never a batch-invariant kernel).
    """
    if zero_centered:
        if family != RMS_NORM_FAMILY_NO_RESIDUAL:
            raise ValueError(
                "zero-centered RMSNorm only exists in the 'serving_no_residual' family; "
                "Qwen3.5 residual-tree norms run the native path, not a batch-invariant kernel"
            )
        return rms_norm_batch_invariant(
            input.float(), 1.0 + weight.float(), eps=eps
        ).type_as(input)
    if family == RMS_NORM_FAMILY_NO_RESIDUAL:
        return rms_norm_batch_invariant(input, weight, eps=eps)
    if family == RMS_NORM_FAMILY_RESIDUAL_TREE:
        return rms_norm_residual_tree_batch_invariant(input, weight, eps=eps)
    raise ValueError(
        f"Unknown RMSNorm family {family!r}; expected one of {RMS_NORM_FAMILIES}"
    )


def bi_fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    family: RMSNormFamily,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add batch-invariant RMSNorm with an explicit kernel family.

    Only the serving residual tree has a fused-add kernel; requesting the
    no-residual family with a residual stream is a contract violation (serving
    never runs family-1 on a residual site) and raises.
    """
    if family == RMS_NORM_FAMILY_RESIDUAL_TREE:
        return fused_add_rms_norm_batch_invariant(input, residual, weight, eps=eps)
    if family == RMS_NORM_FAMILY_NO_RESIDUAL:
        raise ValueError(
            "RMSNorm family 'serving_no_residual' has no fused-add kernel: residual "
            "site-classes are 'serving_residual_tree' by the cross-engine contract"
        )
    raise ValueError(
        f"Unknown RMSNorm family {family!r}; expected one of {RMS_NORM_FAMILIES}"
    )


# --------------------------------------------------------------------------- #
# Batch-invariant fused LM-head selected-token log-probability
#
# The K3 lm-head contract, vendored identically in xorl and SGLang
# (python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py). Both engines
# compute per-token logprobs of given token ids from bit-exact bf16 hidden
# states and the bf16 lm-head weight through the SAME reduction trees, so the
# results are bitwise identical cross-engine:
#   1. chunk GEMM — ``matmul_kernel_persistent`` with the family's fixed bf16
#      tile config and an fp32 output buffer. bf16xbf16 products are exact in
#      fp32, so reading the weight in bf16 with tensor-core fp32 accumulation
#      equals a GEMM over materialized fp32 upcasts with the same tree (and
#      deletes the fp32 weight copy the eager paths materialize).
#   2. chunk stats — per-row max and sum(exp(x - chunk_max)) in a fixed
#      sequential BLOCK loop (same discipline as ``_rms_norm_kernel``), plus
#      the selected-token logit gather.
#   3. merge — global max over chunk maxima (exact), then the rescaled sumexp
#      accumulated in pinned chunk order; lse = gmax + log(acc).
# All transcendentals stay inside these kernels: tl.exp/tl.log measured
# bit-identical across triton 3.5.1 (serving venv) and 3.7.1 (trainer venv),
# as is the fixed-tile tl.dot fp32 accumulator. VOCAB_CHUNK and STATS_BLOCK are
# contract constants — changing either changes the bits (the LSE reduction
# tree). The chunk GEMM's tile config is shape-keyed via bi_gemm_configs: only
# its BLOCK_SIZE_K (pinned there) is bit-relevant.
# Forward-only; the trainer wraps it in an autograd.Function (ops/loss).
# --------------------------------------------------------------------------- #

BI_LM_HEAD_VOCAB_CHUNK = 8192
_BI_LM_HEAD_STATS_BLOCK = 1024


@triton.jit
def _lm_head_chunk_stats_kernel(
    logits_ptr,
    token_ids_ptr,
    sel_ptr,
    m_ptr,
    s_ptr,
    temp_ptr,
    logits_row_stride,
    n_cols,
    col_offset,
    chunk_idx,
    n_chunks,
    BLOCK_SIZE: tl.constexpr,
    HAS_TEMP: tl.constexpr,
):
    """Per-row chunk statistics over an fp32 logits tile [N, n_cols]:
    chunk max, sum(exp(x - chunk_max)) in a fixed sequential block loop, and
    the selected-token logit when ``token_ids[row]`` falls in this chunk.
    With HAS_TEMP, logits are scaled by 1/temp[row] before the statistics and
    the selected logit; the fp32 divide runs in-kernel so every engine
    computes the identical scale (elementwise, so batch-invariance holds)."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * logits_row_stride
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temp_ptr + row)
    else:
        inv_t = 1.0

    row_max = float("-inf")
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_ptr + col_idx, mask=mask, other=float("-inf"))
        if HAS_TEMP:
            vals = vals * inv_t
        row_max = tl.maximum(row_max, tl.max(vals))

    sum_exp = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_ptr + col_idx, mask=mask, other=float("-inf"))
        if HAS_TEMP:
            vals = vals * inv_t
        e = tl.exp(vals - row_max)
        sum_exp += tl.sum(tl.where(mask, e, 0.0))

    tl.store(m_ptr + row * n_chunks + chunk_idx, row_max)
    tl.store(s_ptr + row * n_chunks + chunk_idx, sum_exp)

    tok = tl.load(token_ids_ptr + row)
    local = tok - col_offset
    in_chunk = (local >= 0) & (local < n_cols)
    sel = tl.load(row_ptr + local, mask=in_chunk, other=0.0)
    if HAS_TEMP:
        sel = sel * inv_t
    tl.store(sel_ptr + row, sel, mask=in_chunk)


@triton.jit
def _lm_head_lse_merge_kernel(
    m_ptr,
    s_ptr,
    lse_ptr,
    n_chunks,
):
    """lse[row] = gmax + log(sum_c s_c * exp(m_c - gmax)), chunks in pinned order."""
    row = tl.program_id(0).to(tl.int64)
    base = row * n_chunks
    gmax = float("-inf")
    for c in range(n_chunks):
        gmax = tl.maximum(gmax, tl.load(m_ptr + base + c))
    acc = 0.0
    for c in range(n_chunks):
        acc += tl.load(s_ptr + base + c) * tl.exp(tl.load(m_ptr + base + c) - gmax)
    tl.store(lse_ptr + row, gmax + tl.log(acc))


def _bi_lm_head_chunk_gemm_fp32(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    """Launch the family's persistent matmul with the shape-keyed bf16 config and
    an fp32 output buffer (the fp32 store path keeps the raw accumulator bits)."""
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    M, K = a.shape
    _, N = b.shape

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    def _launch(config):
        matmul_kernel_persistent[grid](
            a,
            b,
            out,
            None,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            NUM_SMS=NUM_SMS,
            A_LARGE=a.numel() > 2**31,
            B_LARGE=b.numel() > 2**31,
            # Extent-based, not numel: out may be a column slice of a wider buffer
            # (bi_lm_head_full_logits), where offsets span stride(0), not numel.
            # Constexpr index width only; stored values are unchanged.
            C_LARGE=(out.stride(0) * (M - 1) + out.stride(1) * (N - 1) + 1) > 2**31,
            HAS_BIAS=False,
            **config,
        )

    _launch_with_config_fallback(
        _launch, a.dtype, M, N, K, out_itemsize=out.element_size()
    )


def bi_lm_head_selected_logprob(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token ``log p(token_ids)`` for the LM head, batch-invariant and
    cross-engine bit-exact (the K3 lm-head contract).

    Args:
        hidden: ``[N, H]`` bf16 hidden states (pre lm-head).
        weight: ``[V, H]`` bf16 lm-head weight (kept resident in bf16; no fp32
            copy is materialized).
        token_ids: ``[N]`` integer token ids to score (callers must pre-clamp
            ignored positions to a valid id and mask outputs downstream).
        temperature: optional ``[N]`` fp32 per-row temperatures (> 0). Logits
            are scaled by ``1/temperature[row]`` inside the stats kernel (the
            divide runs in-kernel, so engines sharing the contract compute the
            identical scale). ``None`` is the exact temperature-1.0 path.

    Returns:
        ``(logprob, lse, selected)`` — all ``[N]`` fp32; ``logprob = selected - lse``
        (temperature-scaled when ``temperature`` is given).
    """
    assert hidden.ndim == 2 and weight.ndim == 2, "hidden and weight must be 2D"
    assert hidden.shape[1] == weight.shape[1], "hidden dim mismatch"
    assert (
        hidden.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    ), "the lm-head contract takes bf16 hidden/weight (fp32 upcast is exact inside the GEMM)"
    assert hidden.is_cuda, "CUDA only"

    hidden = hidden.contiguous()
    token_ids = token_ids.contiguous().to(device=hidden.device, dtype=torch.int64)
    n_tokens = hidden.shape[0]
    vocab = weight.shape[0]
    n_chunks = (vocab + vocab_chunk - 1) // vocab_chunk
    if temperature is not None:
        temperature = (
            temperature.reshape(-1)
            .to(device=hidden.device, dtype=torch.float32)
            .contiguous()
        )
        assert temperature.shape[0] == n_tokens, "temperature must be per-row [N]"
        assert bool((temperature > 0).all()), "temperature must be > 0"

    chunk_max = torch.empty(
        (n_tokens, n_chunks), dtype=torch.float32, device=hidden.device
    )
    chunk_sumexp = torch.empty_like(chunk_max)
    selected = torch.zeros(n_tokens, dtype=torch.float32, device=hidden.device)
    lse = torch.empty(n_tokens, dtype=torch.float32, device=hidden.device)
    logits_buf = torch.empty(
        (n_tokens, vocab_chunk), dtype=torch.float32, device=hidden.device
    )

    for chunk_idx, col_start in enumerate(range(0, vocab, vocab_chunk)):
        col_end = min(col_start + vocab_chunk, vocab)
        n_cols = col_end - col_start
        logits_c = logits_buf[:, :n_cols]
        # [H, C] transposed view of the resident bf16 weight — the persistent
        # kernel takes explicit strides, so no copy is made.
        _bi_lm_head_chunk_gemm_fp32(hidden, weight[col_start:col_end].t(), logits_c)
        _lm_head_chunk_stats_kernel[(n_tokens,)](
            logits_c,
            token_ids,
            selected,
            chunk_max,
            chunk_sumexp,
            temperature,
            logits_c.stride(0),
            n_cols,
            col_start,
            chunk_idx,
            n_chunks,
            BLOCK_SIZE=_BI_LM_HEAD_STATS_BLOCK,
            HAS_TEMP=temperature is not None,
        )

    _lm_head_lse_merge_kernel[(n_tokens,)](chunk_max, chunk_sumexp, lse, n_chunks)
    # In exact math the selected logit never exceeds the LSE; clamp the one-ulp
    # fp boundary case (p~1 tokens) so contract logprobs are provably <= 0.
    return torch.clamp_max(selected - lse, 0.0), lse, selected


def bi_lm_head_full_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> torch.Tensor:
    """Full ``[N, V]`` fp32 logits through the contract's GEMM (step 1).

    Decode twin of ``bi_lm_head_selected_logprob``, as ONE full-vocab launch:
    N-tiling is bit-free in the persistent kernel (each output element's
    K-chain is set by the pinned BLOCK_SIZE_K alone), so a single ``[N, V]``
    GEMM is bitwise identical to per-chunk column-slice launches while
    skipping ``V / vocab_chunk`` launch overheads and per-launch tail waves
    (~-26% lm-head GEMM time at decode bs1). ``vocab_chunk`` remains the LSE
    contract constant for the stats/merge steps, which still read this buffer
    per-chunk. The launch shape is static for a given ``[N, V]``, so the path
    stays CUDA-graph-safe. Bitwise equality of the two forms is gated (not
    assumed) in the BI lm-head unit suite.
    """
    assert hidden.ndim == 2 and weight.ndim == 2, "hidden and weight must be 2D"
    assert hidden.shape[1] == weight.shape[1], "hidden dim mismatch"
    assert (
        hidden.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    ), "the lm-head contract takes bf16 hidden/weight (fp32 upcast is exact inside the GEMM)"
    assert hidden.is_cuda, "CUDA only"

    hidden = hidden.contiguous()
    n_tokens = hidden.shape[0]
    vocab = weight.shape[0]
    logits = torch.empty((n_tokens, vocab), dtype=torch.float32, device=hidden.device)
    if n_tokens == 0:
        return logits
    _bi_lm_head_chunk_gemm_fp32(hidden, weight.t(), logits)
    return logits


def bi_lm_head_selected_logprob_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contract steps 2-3 (chunk stats + pinned-order LSE merge) over an
    existing fp32 logits tensor.

    Decode twin of ``bi_lm_head_selected_logprob`` for callers that must
    sample from the same logits they score: with ``logits`` from
    ``bi_lm_head_full_logits`` the result is bitwise identical to the fused
    path on the same hidden/weight/temperature — the stats kernel reads the
    same fp32 values through the same block loop, per chunk, in pinned order.
    The temperature > 0 check is device-side (``torch._assert_async``) so the
    decode hot loop stays free of host syncs.
    """
    assert logits.ndim == 2, "logits must be [N, V]"
    assert logits.dtype == torch.float32, "the contract scores fp32 logits"
    assert logits.is_cuda, "CUDA only"
    assert logits.stride(1) == 1, "logits rows must be unit-stride"

    n_tokens, vocab = logits.shape
    token_ids = token_ids.contiguous().to(device=logits.device, dtype=torch.int64)
    assert token_ids.shape[0] == n_tokens, "token_ids must be per-row [N]"
    n_chunks = (vocab + vocab_chunk - 1) // vocab_chunk
    if temperature is not None:
        temperature = (
            temperature.reshape(-1)
            .to(device=logits.device, dtype=torch.float32)
            .contiguous()
        )
        assert temperature.shape[0] == n_tokens, "temperature must be per-row [N]"
        torch._assert_async((temperature > 0).all(), "temperature must be > 0")

    chunk_max = torch.empty(
        (n_tokens, n_chunks), dtype=torch.float32, device=logits.device
    )
    chunk_sumexp = torch.empty_like(chunk_max)
    selected = torch.zeros(n_tokens, dtype=torch.float32, device=logits.device)
    lse = torch.empty(n_tokens, dtype=torch.float32, device=logits.device)

    for chunk_idx, col_start in enumerate(range(0, vocab, vocab_chunk)):
        col_end = min(col_start + vocab_chunk, vocab)
        logits_c = logits[:, col_start:col_end]
        _lm_head_chunk_stats_kernel[(n_tokens,)](
            logits_c,
            token_ids,
            selected,
            chunk_max,
            chunk_sumexp,
            temperature,
            logits_c.stride(0),
            col_end - col_start,
            col_start,
            chunk_idx,
            n_chunks,
            BLOCK_SIZE=_BI_LM_HEAD_STATS_BLOCK,
            HAS_TEMP=temperature is not None,
        )

    _lm_head_lse_merge_kernel[(n_tokens,)](chunk_max, chunk_sumexp, lse, n_chunks)
    # In exact math the selected logit never exceeds the LSE; clamp the one-ulp
    # fp boundary case (p~1 tokens) so contract logprobs are provably <= 0.
    return torch.clamp_max(selected - lse, 0.0), lse, selected


# --------------------------------------------------------------------------- #
# Batch-invariant MoE router GEMM (the K3 router contract)
#
# Vendored identically in xorl and SGLang so the MoE gate/router logits are
# computed through ONE reduction tree cross-engine. Unlike the capture/replay
# lane, live training routes independently of serving (no routing replay), so a
# ~1e-10..1e-4 router-logit reduction-order diff between the two engines' GEMMs
# can flip the top-k expert selection on razor-edge tokens and cause large,
# rare-token logprob divergence. This kernel removes that last term:
#   - bf16 hidden [N, H] @ bf16 gate weight [E, H]^T -> fp32 logits [N, E]
#   - ``matmul_kernel_persistent`` with a pinned tile config and an fp32 output
#     buffer (same discipline as the lm-head contract). bf16xbf16 products are
#     exact in fp32, so reading both operands in bf16 with tensor-core fp32
#     accumulation equals an fp32 GEMM over their (exact) fp32 upcasts, but with
#     the reduction order pinned identically in both engines — and without the
#     fp32 weight/activation copies the eager fp32-router paths materialize.
# num_experts is small (a single BLOCK_SIZE_N tile for the common E <= 128), so
# the whole GEMM is one persistent launch. The config below is part of the
# contract; changing any constant changes the bits.
# The architecture resolver selects this kernel for exact Qwen MoE serving.
# Forward-only; the trainer wraps it in an autograd.Function with a closed-form
# (order-insensitive) backward — gradients do not enter the forward K3.
# --------------------------------------------------------------------------- #

_BI_ROUTER_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_stages": 3,
    "num_warps": 8,
}


def bi_router_gemm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 MoE router logits ``[N, E]`` from bf16 hidden ``[N, H]`` and bf16 gate
    weight ``[E, H]`` (the K3 router contract).

    One persistent bf16-in / fp32-accumulate / fp32-out GEMM with a pinned launch
    config, vendored identically in xorl and SGLang so the router logits (and
    therefore the top-k expert selection) are bitwise identical cross-engine.
    bf16xbf16 products are exact in fp32, so this equals an fp32 GEMM over the
    upcast operands the eager fp32-router paths materialize — minus the fp32
    weight/activation copies and with a reduction order that no longer depends on
    the backend GEMM.

    Args:
        hidden: ``[N, H]`` bf16 hidden states (pre-gate).
        weight: ``[E, H]`` bf16 gate weight (kept resident in bf16; no fp32 copy).

    Returns:
        ``[N, E]`` fp32 router logits.
    """
    assert hidden.ndim == 2 and weight.ndim == 2, "hidden and weight must be 2D"
    assert hidden.shape[1] == weight.shape[1], "hidden dim mismatch"
    assert (
        hidden.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    ), "the router contract takes bf16 hidden/weight (fp32 upcast is exact inside the GEMM)"
    assert hidden.is_cuda, "CUDA only"

    hidden = hidden.contiguous()
    weight = weight.contiguous()
    n_tokens = hidden.shape[0]
    num_experts = weight.shape[0]
    logits = torch.empty(
        (n_tokens, num_experts), dtype=torch.float32, device=hidden.device
    )
    if n_tokens == 0:
        return logits

    # [H, E] transposed view of the resident bf16 gate weight — the persistent
    # kernel takes explicit strides, so no copy is made.
    b = weight.t()
    NUM_SMS = torch.cuda.get_device_properties(hidden.device).multi_processor_count
    M, K = hidden.shape
    N = num_experts

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    def _launch(config):
        matmul_kernel_persistent[grid](
            hidden,
            b,
            logits,
            None,
            M,
            N,
            K,
            hidden.stride(0),
            hidden.stride(1),
            b.stride(0),
            b.stride(1),
            logits.stride(0),
            logits.stride(1),
            NUM_SMS=NUM_SMS,
            A_LARGE=hidden.numel() > 2**31,
            B_LARGE=weight.numel() > 2**31,
            C_LARGE=logits.numel() > 2**31,
            HAS_BIAS=False,
            **config,
        )

    # W3.1 Tier-A: the pinned config launches ~1-4 CTAs on a 132-SM H100 at
    # the decode shapes (M<=256, N=num_experts). Shape-keyed grid/schedule
    # configs selected by the architecture resolver (otherwise lookup returns
    # None and the pinned contract config below launches exactly as before).
    # BK and the per-element reduction chain are unchanged; certified
    # torch.equal vs the pinned config, which also remains the launch-failure
    # fallback.
    tiera_cfg = lookup_tiera_router_config(hidden.dtype, M, N, K)
    if tiera_cfg is not None:
        try:
            _launch(tiera_cfg)
            return logits
        except OutOfResources:
            pass
    _launch(_BI_ROUTER_GEMM_CONFIG)
    return logits


def bi_router_topk_weights(
    topk_vals: torch.Tensor,
    norm_topk_prob: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Renormalize the gathered top-k router scores in a fixed reduction order,
    then cast — the second half of the K3 router contract.

    ``bi_router_gemm`` makes the router logits (and therefore ``torch.topk``
    selection and the gathered top-k softmax scores) bitwise identical
    cross-engine, but the stock renorm ``vals / vals.sum(dim=-1, keepdim=True)``
    is not: the small last-dim reduction ``sum(dim=-1)`` uses a build-dependent
    tree order, so the divisor can differ by ~1 fp32 ulp between the trainer's
    and the server's torch/triton, occasionally flipping the final bf16 weight
    on rare tokens. Elementwise ops (add, divide, round-to-bf16) are IEEE
    correctly-rounded and build-invariant, so accumulating the divisor with a
    pinned left-to-right order makes the top-k weights bit-identical too.

    Args:
        topk_vals: ``[N, top_k]`` fp32 gathered top-k router scores (softmax
            probabilities on the softmax path).
        norm_topk_prob: renormalize the top-k slice to sum to 1 (Qwen3 MoE
            default). When False the scores are only cast (already bit-identical
            cross-engine, since the softmax/top-k that produced them are).
        out_dtype: routing-weight dtype (the model activation dtype, bf16).

    Returns:
        ``[N, top_k]`` ``out_dtype`` routing weights.
    """
    assert (
        topk_vals.dtype == torch.float32
    ), "the router contract renorms fp32 top-k scores"
    if norm_topk_prob:
        if _ROUTER_RENORM_FUSED_ENABLED:
            # W3.4: one fused launch replacing the 7-add chain + divide +
            # cast below, bit-identical (certified by
            # test_bi_router_renorm_fused_a0.py). Armed by the contract
            # resolver selection; ineligible inputs fall back to the unfused chain
            # unchanged.
            from sglang.xorl.bi.router_renorm_fused import (
                fused_router_renorm,
            )

            fused = fused_router_renorm(topk_vals, out_dtype)
            if fused is not None:
                return fused
        denom = topk_vals[..., 0]
        for k in range(1, topk_vals.shape[-1]):
            denom = denom + topk_vals[..., k]
        topk_vals = topk_vals / denom.unsqueeze(-1)
    return topk_vals.to(out_dtype)


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_batch_invariant_OPS: set[str] = set()
_ROUTER_RENORM_FUSED_ENABLED = False
_BI_HEAD_FASTPATH_ENABLED = False
_original_torch_bmm = None


def set_router_renorm_fused_enabled(enabled: bool) -> None:
    global _ROUTER_RENORM_FUSED_ENABLED
    _ROUTER_RENORM_FUSED_ENABLED = bool(enabled)


def set_bi_head_fastpath_enabled(enabled: bool) -> None:
    global _BI_HEAD_FASTPATH_ENABLED
    _BI_HEAD_FASTPATH_ENABLED = bool(enabled)


def is_bi_head_fastpath_enabled() -> bool:
    return _BI_HEAD_FASTPATH_ENABLED

