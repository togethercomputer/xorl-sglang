"""GPU kernels for exact canonical MoE leaf construction and reduction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp64_add_rn(left, right):
    """Pin one non-reassociable IEEE FP64 add at a declared tree node."""
    return tl.inline_asm_elementwise(
        asm="add.rn.f64 $0, $1, $2;",
        constraints="=d,d,d",
        args=(left, right),
        dtype=tl.float64,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32_fma_rn(multiplicand, multiplier, addend):
    """Pin one-round FP32 FMA for exact contributor-leaf construction."""
    return tl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=(multiplicand, multiplier, addend),
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
    return value.to(tl.float64)


@triton.jit
def _fused_balanced_adjacent_fp64_tree_kernel(
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
    if CONTRIBUTORS == 1:
        result = p0
    else:
        p1 = _load_logical_contributor(
            partials_ptr,
            logical_to_group_ptr,
            offsets,
            n_elements,
            LOGICAL=1,
            IDENTITY_ORDER=IDENTITY_ORDER,
        )
        level_01 = _fp64_add_rn(p0, p1)
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
            if CONTRIBUTORS == 3:
                result = _fp64_add_rn(level_01, p2)
            else:
                p3 = _load_logical_contributor(
                    partials_ptr,
                    logical_to_group_ptr,
                    offsets,
                    n_elements,
                    LOGICAL=3,
                    IDENTITY_ORDER=IDENTITY_ORDER,
                )
                level_23 = _fp64_add_rn(p2, p3)
                level_03 = _fp64_add_rn(level_01, level_23)
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
                    if CONTRIBUTORS == 5:
                        result = _fp64_add_rn(level_03, p4)
                    else:
                        p5 = _load_logical_contributor(
                            partials_ptr,
                            logical_to_group_ptr,
                            offsets,
                            n_elements,
                            LOGICAL=5,
                            IDENTITY_ORDER=IDENTITY_ORDER,
                        )
                        level_45 = _fp64_add_rn(p4, p5)
                        if CONTRIBUTORS == 6:
                            result = _fp64_add_rn(level_03, level_45)
                        else:
                            p6 = _load_logical_contributor(
                                partials_ptr,
                                logical_to_group_ptr,
                                offsets,
                                n_elements,
                                LOGICAL=6,
                                IDENTITY_ORDER=IDENTITY_ORDER,
                            )
                            if CONTRIBUTORS == 7:
                                result = _fp64_add_rn(
                                    level_03, _fp64_add_rn(level_45, p6)
                                )
                            else:
                                p7 = _load_logical_contributor(
                                    partials_ptr,
                                    logical_to_group_ptr,
                                    offsets,
                                    n_elements,
                                    LOGICAL=7,
                                    IDENTITY_ORDER=IDENTITY_ORDER,
                                )
                                level_67 = _fp64_add_rn(p6, p7)
                                level_47 = _fp64_add_rn(level_45, level_67)
                                level_07 = _fp64_add_rn(level_03, level_47)
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
                                    if CONTRIBUTORS == 9:
                                        result = _fp64_add_rn(level_07, p8)
                                    else:
                                        p9 = _load_logical_contributor(
                                            partials_ptr,
                                            logical_to_group_ptr,
                                            offsets,
                                            n_elements,
                                            LOGICAL=9,
                                            IDENTITY_ORDER=IDENTITY_ORDER,
                                        )
                                        level_89 = _fp64_add_rn(p8, p9)
                                        if CONTRIBUTORS == 10:
                                            result = _fp64_add_rn(level_07, level_89)
                                        else:
                                            p10 = _load_logical_contributor(
                                                partials_ptr,
                                                logical_to_group_ptr,
                                                offsets,
                                                n_elements,
                                                LOGICAL=10,
                                                IDENTITY_ORDER=IDENTITY_ORDER,
                                            )
                                            if CONTRIBUTORS == 11:
                                                result = _fp64_add_rn(
                                                    level_07,
                                                    _fp64_add_rn(level_89, p10),
                                                )
                                            else:
                                                p11 = _load_logical_contributor(
                                                    partials_ptr,
                                                    logical_to_group_ptr,
                                                    offsets,
                                                    n_elements,
                                                    LOGICAL=11,
                                                    IDENTITY_ORDER=IDENTITY_ORDER,
                                                )
                                                level_1011 = _fp64_add_rn(p10, p11)
                                                level_811 = _fp64_add_rn(
                                                    level_89, level_1011
                                                )
                                                if CONTRIBUTORS == 12:
                                                    result = _fp64_add_rn(
                                                        level_07, level_811
                                                    )
                                                else:
                                                    p12 = _load_logical_contributor(
                                                        partials_ptr,
                                                        logical_to_group_ptr,
                                                        offsets,
                                                        n_elements,
                                                        LOGICAL=12,
                                                        IDENTITY_ORDER=IDENTITY_ORDER,
                                                    )
                                                    if CONTRIBUTORS == 13:
                                                        result = _fp64_add_rn(
                                                            level_07,
                                                            _fp64_add_rn(
                                                                level_811, p12
                                                            ),
                                                        )
                                                    else:
                                                        p13 = _load_logical_contributor(
                                                            partials_ptr,
                                                            logical_to_group_ptr,
                                                            offsets,
                                                            n_elements,
                                                            LOGICAL=13,
                                                            IDENTITY_ORDER=IDENTITY_ORDER,
                                                        )
                                                        level_1213 = _fp64_add_rn(
                                                            p12, p13
                                                        )
                                                        if CONTRIBUTORS == 14:
                                                            result = _fp64_add_rn(
                                                                level_07,
                                                                _fp64_add_rn(
                                                                    level_811,
                                                                    level_1213,
                                                                ),
                                                            )
                                                        else:
                                                            p14 = _load_logical_contributor(
                                                                partials_ptr,
                                                                logical_to_group_ptr,
                                                                offsets,
                                                                n_elements,
                                                                LOGICAL=14,
                                                                IDENTITY_ORDER=IDENTITY_ORDER,
                                                            )
                                                            if CONTRIBUTORS == 15:
                                                                result = _fp64_add_rn(
                                                                    level_07,
                                                                    _fp64_add_rn(
                                                                        level_811,
                                                                        _fp64_add_rn(
                                                                            level_1213,
                                                                            p14,
                                                                        ),
                                                                    ),
                                                                )
                                                            else:
                                                                p15 = _load_logical_contributor(
                                                                    partials_ptr,
                                                                    logical_to_group_ptr,
                                                                    offsets,
                                                                    n_elements,
                                                                    LOGICAL=15,
                                                                    IDENTITY_ORDER=IDENTITY_ORDER,
                                                                )
                                                                level_1415 = (
                                                                    _fp64_add_rn(
                                                                        p14, p15
                                                                    )
                                                                )
                                                                level_1215 = (
                                                                    _fp64_add_rn(
                                                                        level_1213,
                                                                        level_1415,
                                                                    )
                                                                )
                                                                level_815 = (
                                                                    _fp64_add_rn(
                                                                        level_811,
                                                                        level_1215,
                                                                    )
                                                                )
                                                                result = _fp64_add_rn(
                                                                    level_07,
                                                                    level_815,
                                                                )

    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)


def fused_balanced_adjacent_fp64_tree(
    partials: torch.Tensor,
    logical_to_group: torch.Tensor,
    *,
    identity_order: bool,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a 1..16-way FP64 adjacent tree and cast once to output."""
    if not partials.is_cuda:
        raise ValueError("The fused canonical FP64 tree requires a CUDA tensor")
    if partials.dtype not in {torch.bfloat16, torch.float16}:
        raise TypeError("The fused canonical FP64 tree requires BF16 or FP16 partials")
    if not partials.is_contiguous():
        raise ValueError("The fused canonical FP64 tree requires contiguous partials")
    contributors = partials.shape[0]
    if not 1 <= contributors <= 16:
        raise ValueError("The fused canonical FP64 tree requires 1..16 contributors")
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
            "Fused canonical FP64 tree output metadata does not match partials"
        )
    if output.device != partials.device or not output.is_contiguous():
        raise ValueError(
            "Fused canonical FP64 tree output must be contiguous on the input device"
        )

    n_elements = output.numel()
    block_size = 128
    _fused_balanced_adjacent_fp64_tree_kernel[(triton.cdiv(n_elements, block_size),)](
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


@triton.jit
def _canonical_moe_leaf_fp32_kernel(
    shared_ptr,
    routed_ptr,
    output_ptr,
    n_elements,
    routed_scale,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    routed = tl.load(routed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # This is deliberately one FP32 FMA, not a separately rounded multiply
    # followed by an add. The completed leaf is cast only by the transport-dtype store.
    leaf = _fp32_fma_rn(routed, routed_scale.to(tl.float32), shared)
    tl.store(output_ptr + offsets, leaf, mask=mask)


def fused_canonical_moe_leaf_fp32(
    shared: torch.Tensor,
    routed: torch.Tensor,
    *,
    routed_scale: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build one low-precision transport leaf with one FP32 FMA and final cast."""
    if not shared.is_cuda or not routed.is_cuda:
        raise ValueError("The fused canonical MoE leaf requires CUDA tensors")
    if shared.dtype != routed.dtype or shared.dtype not in {
        torch.bfloat16,
        torch.float16,
    }:
        raise TypeError("The fused canonical MoE leaf requires BF16 or FP16 operands")
    if shared.shape != routed.shape or shared.device != routed.device:
        raise ValueError("Canonical MoE leaf operands must share shape and device")
    if not shared.is_contiguous() or not routed.is_contiguous():
        raise ValueError("Canonical MoE leaf operands must be contiguous")
    if output is None:
        output = torch.empty_like(shared)
    if (
        output.shape != shared.shape
        or output.dtype is not shared.dtype
        or output.device != shared.device
        or not output.is_contiguous()
    ):
        raise ValueError("Canonical MoE leaf output metadata does not match operands")

    n_elements = output.numel()
    if n_elements == 0:
        return output
    block_size = 256
    _canonical_moe_leaf_fp32_kernel[(triton.cdiv(n_elements, block_size),)](
        shared,
        routed,
        output,
        n_elements,
        float(routed_scale),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


__all__ = [
    "fused_balanced_adjacent_fp64_tree",
    "fused_canonical_moe_leaf_fp32",
]
