"""GPU kernels for the exact GLM-5.2 canonical MoE reduction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rtne_bf16_as_f32(value):
    """Round FP32 to BF16 while retaining an FP32 register representation.

    The integer round trip prevents Triton from folding away an intermediate
    FP32->BF16->FP32 cast. Every call is a bit-relevant canonical-tree round
    point. NaNs are quieted, although legal GLM-5.2 partials are finite.
    """
    bits = value.to(tl.int32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    nan_bits = (bits & -65536) | 0x00400000
    result = tl.where(value != value, nan_bits, rounded)
    return result.to(tl.float32, bitcast=True)


@triton.jit
def _load_logical_contributor(
    partials_ptr,
    logical_to_group_ptr,
    offsets,
    n_elements,
    LOGICAL: tl.constexpr,
    IDENTITY_ORDER: tl.constexpr,
):
    physical = LOGICAL
    if not IDENTITY_ORDER:
        physical = tl.load(logical_to_group_ptr + LOGICAL)
    value = tl.load(
        partials_ptr + physical * n_elements + offsets,
        mask=offsets < n_elements,
        other=0.0,
    )
    return value.to(tl.float32)


@triton.jit
def _rounded_add(left, right):
    return _rtne_bf16_as_f32(left + right)


@triton.jit
def _fused_balanced_adjacent_bf16_tree_kernel(
    partials_ptr,
    logical_to_group_ptr,
    output_ptr,
    n_elements,
    CONTRIBUTORS: tl.constexpr,
    IDENTITY_ORDER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    p0 = _load_logical_contributor(
        partials_ptr,
        logical_to_group_ptr,
        offsets,
        n_elements,
        LOGICAL=0,
        IDENTITY_ORDER=IDENTITY_ORDER,
    )
    p1 = _load_logical_contributor(
        partials_ptr,
        logical_to_group_ptr,
        offsets,
        n_elements,
        LOGICAL=1,
        IDENTITY_ORDER=IDENTITY_ORDER,
    )
    level_01 = _rounded_add(p0, p1)

    if CONTRIBUTORS == 2:
        result = level_01
    else:
        p2 = _load_logical_contributor(
            partials_ptr,
            logical_to_group_ptr,
            offsets,
            n_elements,
            LOGICAL=2,
            IDENTITY_ORDER=IDENTITY_ORDER,
        )
        p3 = _load_logical_contributor(
            partials_ptr,
            logical_to_group_ptr,
            offsets,
            n_elements,
            LOGICAL=3,
            IDENTITY_ORDER=IDENTITY_ORDER,
        )
        level_23 = _rounded_add(p2, p3)
        level_03 = _rounded_add(level_01, level_23)

        if CONTRIBUTORS == 4:
            result = level_03
        else:
            p4 = _load_logical_contributor(
                partials_ptr,
                logical_to_group_ptr,
                offsets,
                n_elements,
                LOGICAL=4,
                IDENTITY_ORDER=IDENTITY_ORDER,
            )
            p5 = _load_logical_contributor(
                partials_ptr,
                logical_to_group_ptr,
                offsets,
                n_elements,
                LOGICAL=5,
                IDENTITY_ORDER=IDENTITY_ORDER,
            )
            p6 = _load_logical_contributor(
                partials_ptr,
                logical_to_group_ptr,
                offsets,
                n_elements,
                LOGICAL=6,
                IDENTITY_ORDER=IDENTITY_ORDER,
            )
            p7 = _load_logical_contributor(
                partials_ptr,
                logical_to_group_ptr,
                offsets,
                n_elements,
                LOGICAL=7,
                IDENTITY_ORDER=IDENTITY_ORDER,
            )
            level_47 = _rounded_add(
                _rounded_add(p4, p5),
                _rounded_add(p6, p7),
            )
            level_07 = _rounded_add(level_03, level_47)

            if CONTRIBUTORS == 8:
                result = level_07
            else:
                p8 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=8,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                p9 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=9,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                p10 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=10,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                p11 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=11,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                p12 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=12,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                p13 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=13,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                p14 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=14,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                p15 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=15,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                level_811 = _rounded_add(
                    _rounded_add(p8, p9),
                    _rounded_add(p10, p11),
                )
                level_1215 = _rounded_add(
                    _rounded_add(p12, p13),
                    _rounded_add(p14, p15),
                )
                level_815 = _rounded_add(level_811, level_1215)
                result = _rounded_add(level_07, level_815)

    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)


def fused_balanced_adjacent_bf16_tree(
    partials: torch.Tensor,
    logical_to_group: torch.Tensor,
    *,
    identity_order: bool,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the exact 2/4/8/16-contributor BF16 tree in one kernel."""
    if not partials.is_cuda:
        raise ValueError("The fused canonical BF16 tree requires a CUDA tensor")
    if partials.dtype is not torch.bfloat16:
        raise TypeError("The fused canonical BF16 tree requires BF16 partials")
    if not partials.is_contiguous():
        raise ValueError("The fused canonical BF16 tree requires contiguous partials")
    contributors = partials.shape[0]
    if contributors not in (2, 4, 8, 16):
        raise ValueError(
            "The fused canonical BF16 tree requires 2, 4, 8, or 16 contributors"
        )
    if logical_to_group.shape != (contributors,):
        raise ValueError("logical_to_group must contain one entry per contributor")
    if logical_to_group.device != partials.device:
        raise ValueError("logical_to_group must be on the partials device")
    if logical_to_group.dtype is not torch.int64:
        raise TypeError("logical_to_group must be int64")
    if not logical_to_group.is_contiguous():
        raise ValueError("logical_to_group must be contiguous")

    output_shape = partials.shape[1:]
    if output is None:
        output = torch.empty(output_shape, dtype=partials.dtype, device=partials.device)
    if output.shape != output_shape or output.dtype is not partials.dtype:
        raise ValueError(
            "Fused canonical BF16 tree output metadata does not match partials"
        )
    if output.device != partials.device or not output.is_contiguous():
        raise ValueError(
            "Fused canonical BF16 tree output must be contiguous on the input device"
        )

    n_elements = output.numel()
    block_size = 128
    _fused_balanced_adjacent_bf16_tree_kernel[(triton.cdiv(n_elements, block_size),)](
        partials,
        logical_to_group,
        output,
        n_elements,
        CONTRIBUTORS=contributors,
        IDENTITY_ORDER=identity_order,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


__all__ = ["fused_balanced_adjacent_bf16_tree"]
