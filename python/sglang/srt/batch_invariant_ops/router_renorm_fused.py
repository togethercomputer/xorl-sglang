"""Fused fixed-order router renorm for Qwen3.5-family exact serving.

``bi_router_topk_weights`` (batch_invariant_ops.py) renormalizes the gathered
top-k router scores with a pinned LEFT-TO-RIGHT fp32 addition chain, one
fp32 divide, and one cast:

    denom = topk_vals[..., 0]
    for k in range(1, top_k):
        denom = denom + topk_vals[..., k]      # fp32, no downcast
    topk_vals = topk_vals / denom.unsqueeze(-1)
    return topk_vals.to(out_dtype)

At top_k 8 that is 9 kernel launches (7 fp32 adds + 1 divide + 1 cast) per
MoE layer per decode step. This module fuses the whole thing into ONE
deterministic Triton kernel, bit-identical per row:

    acc = v0
    for k in 1..7: acc = f32_add_rn(acc, v_k)   # fp32 THROUGHOUT
    out[k] = bf16_rne(f32_div_rn(v_k, acc))

Numerical contract notes (why this file is exactly this shape):

- The chain is ALL-fp32: IEEE round-to-nearest fp32 adds in left-to-right
  order with NO intermediate downcast (different from the bf16 combine
  chain, which rounds to bf16 at every step). Triton's fp32 ``+`` lowers to
  ``add.f32`` (IEEE RN), same as ATen's CUDAFunctor_add<float>.
- The divide MUST be ``tl.math.div_rn`` (IEEE round-to-nearest fp32
  division, ``div.rn.f32``). Triton's ``/`` operator AND
  ``tl.fdiv(..., ieee_rounding=True)`` both lower to approximate division
  on this build (measured: ~22-30% of random values differ by 1 ulp vs
  torch). Never "simplify" the division back
  to ``/``.
- The accumulation must stay a serial ``tl.static_range`` chain. ``tl.sum``
  is a tree reduction and moves the denominator by 1 ulp on order-sensitive
  rows — the exact bug class the pinned order exists to exclude (this
  function is vendored bit-for-bit into the xorl trainer; the entire safety
  argument is bitwise identity).
- Each row's chain runs inside a single thread, so launch geometry is
  provably outside the accumulation order; the config is pinned anyway.

Capture-safety: one launch, no host sync, one caching-allocator output
allocation per call (the unfused path allocates on every add/div/cast);
Triton compilation happens at the first warmup call, before capture.

Selection: the private architecture resolver, read at dispatch
site in ``bi_router_topk_weights``. Ineligible inputs return ``None`` and
the caller falls back to the unfused code unchanged. The
``norm_topk_prob=False`` path (single cast, nothing to fuse) stays unfused
by design.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# Production router top-k (Qwen3.6-35B-A3B). Other widths fall back.
_FUSED_TOP_K = 8
# Row offsets are computed in int32 inside the kernel.
_MAX_TOTAL_NUMEL = 2**31 - 1

_BLOCK_ROWS = 256
_NUM_WARPS = 4


@triton.jit
def _router_renorm_fused_kernel(
    vals_ptr,
    out_ptr,
    n_rows,
    TOPK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mask = rows < n_rows
    base = rows * TOPK
    # Left-to-right fp32 chain, exactly like the unfused loop. Vectorization
    # is across ROWS; the per-row order is the serial unrolled chain below.
    acc = tl.load(vals_ptr + base, mask=mask, other=0.0)
    for k in tl.static_range(1, TOPK):
        acc = acc + tl.load(vals_ptr + base + k, mask=mask, other=0.0)
    for k in tl.static_range(TOPK):
        v = tl.load(vals_ptr + base + k, mask=mask, other=0.0)
        # IEEE RN division (div.rn.f32). tl.fdiv / the ``/`` operator are
        # approximate on this build and DO NOT match torch bitwise.
        q = tl.math.div_rn(v, acc)
        tl.store(out_ptr + base + k, q.to(tl.bfloat16), mask=mask)


def fused_router_renorm(
    topk_vals: torch.Tensor, out_dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """One-launch fused renorm for the ``norm_topk_prob=True`` path.

    Returns a fresh ``[N, top_k]`` ``out_dtype`` tensor (same output
    semantics as the unfused path's final ``.to``), or ``None`` when the
    input is outside the certified envelope — the caller must then run the
    unfused chain.
    """
    if out_dtype is not torch.bfloat16:
        return None
    if topk_vals.dtype is not torch.float32:
        return None
    if not topk_vals.is_cuda:
        return None
    if topk_vals.dim() != 2:
        return None
    n_rows, top_k = topk_vals.shape
    if top_k != _FUSED_TOP_K:
        return None
    if n_rows == 0:
        return None
    if not topk_vals.is_contiguous():
        return None
    if topk_vals.numel() > _MAX_TOTAL_NUMEL:
        return None

    out = torch.empty((n_rows, top_k), dtype=out_dtype, device=topk_vals.device)
    grid = (triton.cdiv(n_rows, _BLOCK_ROWS),)
    _router_renorm_fused_kernel[grid](
        topk_vals,
        out,
        n_rows,
        TOPK=top_k,
        BLOCK_ROWS=_BLOCK_ROWS,
        num_warps=_NUM_WARPS,
    )
    return out
