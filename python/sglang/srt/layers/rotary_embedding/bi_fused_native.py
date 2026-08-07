"""Bit-exact fused eager RoPE for the RL numerical-contract lane.

The eager bf16 expression has eight observable rounding points per rotated
pair: cos and sin casts, four products, and two combines. A normal fused
floating-point program lets Triton remove the intermediate f32-to-bf16-to-f32
casts. This kernel spells round-to-nearest-even as integer bit operations so
those boundaries survive optimization while the chain runs in one launch.

Only bf16 Neox-style RoPE is dispatched here. The caller falls back to the
eager implementation for every unsupported dtype, layout, or RoPE style.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rtne_bf16(x):
    """Round f32 to a bf16-exact f32 value without an elidable cast pair."""
    bits = x.to(tl.int32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    # Match c10's fp32-to-bf16 NaN handling: truncate and set the quiet bit.
    nan_bits = (bits & -65536) | 0x00400000
    out = tl.where(x != x, nan_bits, rounded)
    return out.to(tl.float32, bitcast=True)


@triton.jit
def _bi_fused_native_rope_kernel(
    x_ptr,
    out_ptr,
    pos_ptr,
    cache_ptr,
    num_heads,
    stride_xt,
    stride_xh,
    stride_ot,
    stride_oh,
    rotary_half: tl.constexpr,
    head_size: tl.constexpr,
    block: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    token = program_id // num_heads
    head = program_id % num_heads

    offsets = tl.arange(0, block)
    mask = offsets < rotary_half
    position = tl.load(pos_ptr + token).to(tl.int64)

    cache_row = cache_ptr + position * (2 * rotary_half)
    cos = _rtne_bf16(tl.load(cache_row + offsets, mask=mask, other=0.0))
    sin = _rtne_bf16(tl.load(cache_row + rotary_half + offsets, mask=mask, other=0.0))

    input_row = x_ptr + token * stride_xt + head * stride_xh
    x1 = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(input_row + rotary_half + offsets, mask=mask, other=0.0).to(tl.float32)

    x1_cos = _rtne_bf16(x1 * cos)
    x2_sin = _rtne_bf16(x2 * sin)
    out1 = _rtne_bf16(x1_cos - x2_sin)
    x2_cos = _rtne_bf16(x2 * cos)
    x1_sin = _rtne_bf16(x1 * sin)
    out2 = _rtne_bf16(x2_cos + x1_sin)

    output_row = out_ptr + token * stride_ot + head * stride_oh
    tl.store(output_row + offsets, out1.to(tl.bfloat16), mask=mask)
    tl.store(
        output_row + rotary_half + offsets,
        out2.to(tl.bfloat16),
        mask=mask,
    )

    if head_size > 2 * rotary_half:
        for start in range(2 * rotary_half, head_size, block):
            tail = start + tl.arange(0, block)
            tail_mask = tail < head_size
            value = tl.load(input_row + tail, mask=tail_mask, other=0.0)
            tl.store(output_row + tail, value, mask=tail_mask)


def bi_fused_native_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Replay eager Neox RoPE for one ``[tokens, heads, size]`` tensor."""
    tokens, num_heads, head_size = x.shape
    assert x.dtype == torch.bfloat16
    assert x.stride(-1) == 1
    assert cos_sin_cache.dtype == torch.float32

    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if tokens == 0:
        return out

    rotary_half = rotary_dim // 2
    block = max(triton.next_power_of_2(rotary_half), 16)
    _bi_fused_native_rope_kernel[(tokens * num_heads,)](
        x,
        out,
        positions,
        cos_sin_cache,
        num_heads,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        rotary_half=rotary_half,
        head_size=head_size,
        block=block,
        num_warps=1 if rotary_half <= 64 else 2,
    )
    return out
