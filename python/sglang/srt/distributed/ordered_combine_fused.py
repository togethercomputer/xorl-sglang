"""Fused ordered MoE combine chain for Qwen3.5-family exact serving.

``tensor_model_parallel_ordered_all_reduce`` (communication_op.py) implements
the XORL exact MoE reduction as an all-gather of raw per-rank partials
followed by a LOCAL fixed-order addition chain:

    result = partials[world - 1]
    for rank in range(world - 2, -1, -1):
        result = result + partials[rank]

At world 8 that is 7 separate ``CUDAFunctor_add<BFloat16>`` launches per MoE
layer per step. This module fuses the chain into ONE deterministic Triton
kernel that reproduces the chain's arithmetic bit-exactly. The all-gather
(transport) is untouched: fusion applies strictly after ``partials`` exists.

Numerical contract (the reason this file looks paranoid):

PyTorch's CUDA BF16 add computes every ``a + b`` as
(widen bf16 -> fp32, fp32 add, round-to-nearest-even back to bf16). The
chain therefore ROUNDS TO BF16 AFTER EVERY STEP. The fused kernel must do
exactly that, per element:

    acc = p[W-1]                                   # bf16
    for rank in (W-2 .. 0):
        acc = bf16_rne(f32(acc) + f32(p[rank]))    # round EVERY step

An fp32 accumulator with a single final rounding produces different bytes
(ties at bf16 rounding boundaries resolve differently) and must never ship.
Triton's ``.to(tl.bfloat16)`` lowers to the same RNE conversion the ATen
functor uses (``cvt.rn.bf16.f32`` on SM90); component tests compare it against
the unfused chain directly.

Each element's whole chain runs inside a single thread, so launch geometry
(BLOCK_SIZE, num_warps, grid) is provably outside the accumulation order;
the config below is nevertheless pinned (no autotune) so the shipped path
is one fixed program.

Capture-safety: one kernel launch, no host-side synchronization; the single
output allocation goes through the caching allocator exactly like the
unfused path's final ``result + partials[0]`` (which also allocates a fresh
output every call, including under CUDA-graph capture); Triton compilation
happens on the first (warmup) call, before production capture.

Selection: the private architecture resolver, read at the
dispatch site in communication_op.py. Ineligible inputs make
``fused_ordered_combine`` return ``None`` and the caller falls back to the
unfused chain unchanged.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# The chain is unrolled at compile time (WORLD_SIZE is constexpr); each world
# size compiles one specialization. Production is world 8; anything larger
# than this bound falls back to the unfused chain rather than compiling an
# unboundedly long unrolled program.
_MAX_FUSED_WORLD_SIZE = 16

# Offsets are computed in int32 inside the kernel; keep every reachable
# address strictly inside int32 range.
_MAX_TOTAL_NUMEL = 2**31 - 1

_BLOCK_SIZE = 1024
_NUM_WARPS = 4


@triton.jit
def _ordered_combine_chain_kernel(
    partials_ptr,
    out_ptr,
    numel,
    slab_stride,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    # acc starts as rank (WORLD_SIZE - 1)'s partial, exactly like the loop's
    # ``result = partials[-1]``.
    acc = tl.load(
        partials_ptr + (WORLD_SIZE - 1) * slab_stride + offs, mask=mask, other=0.0
    )
    for step in tl.static_range(WORLD_SIZE - 1):
        rank = WORLD_SIZE - 2 - step
        v = tl.load(partials_ptr + rank * slab_stride + offs, mask=mask, other=0.0)
        # One chain step == one PyTorch bf16 CUDA add: widen to fp32, add,
        # round-to-nearest-even back to bf16. The per-step rounding is the
        # contract; do NOT hoist to an fp32 accumulator.
        acc = (acc.to(tl.float32) + v.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + offs, acc, mask=mask)


def fused_ordered_combine(partials: torch.Tensor) -> Optional[torch.Tensor]:
    """Run the ordered combine chain over ``partials`` in one kernel.

    ``partials`` is the ``(world_size, *input_shape)`` contiguous view of the
    all-gathered buffer. Returns a fresh bf16 tensor of shape
    ``input_shape`` (same output semantics as the unfused chain's final add),
    or ``None`` when the input is not eligible for the fused path — the
    caller must then run the unfused chain.
    """
    if partials.dtype is not torch.bfloat16:
        return None
    if not partials.is_cuda:
        return None
    if partials.dim() < 2:
        return None
    world_size = partials.shape[0]
    if world_size < 2 or world_size > _MAX_FUSED_WORLD_SIZE:
        return None
    # The caller allocates ``gathered`` fresh and contiguous and views it as
    # (world, *input_shape); anything else did not come from that call site.
    if not partials.is_contiguous():
        return None
    total_numel = partials.numel()
    if total_numel == 0 or total_numel > _MAX_TOTAL_NUMEL:
        return None

    slab_numel = total_numel // world_size
    out = torch.empty_like(partials[0])
    grid = (triton.cdiv(slab_numel, _BLOCK_SIZE),)
    _ordered_combine_chain_kernel[grid](
        partials,
        out,
        slab_numel,
        slab_numel,
        WORLD_SIZE=world_size,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return out
