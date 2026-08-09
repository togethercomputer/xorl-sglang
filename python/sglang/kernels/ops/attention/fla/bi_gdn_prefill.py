# The K3 GDN prefill contract (zero-K3 contract), vendored from xorl-internal.
#
# Under the exact GDN prefill=1 the chunked GDN prefill scan runs the trainer's
# exact stage composition so chunked-prefill outputs and fp32 final states are
# bitwise identical cross-engine:
#
#   l2norm -> chunk-local cumsum -> kkt -> solve_tril -> w_u -> fwd_h -> fwd_o
#
# Stage sourcing (measured on Qwen3.5 shapes, torch.equal):
#   - l2norm / cumsum / kkt / w_u: SGLang's own unfused kernels are already
#     bitwise equal to the trainer's and invariant to the numerics-relevant
#     autotune knobs (kkt BK in {32,64,128} hashes identically), so they are
#     reused, not vendored.
#   - solve_tril: SGLang's two-kernel 16x16+merge variant has a rare fp32 tail
#     vs the trainer (~1e-6 of elements at T>=1024); the trainer's fused merge
#     kernel is vendored below (ieee tril precision).
#   - fwd_h: SGLang's kernel keeps the running state as [V, K] with a
#     transposed MMA order and diverges on a rare tail (<=2.4e-6 of elements);
#     the trainer's [K, V]-order kernel is vendored below, with the pool-layout
#     transpose adapter at the boundary (exact — fp32 transpose).
#   - fwd_o: bitwise equal given a matched h, but SGLang's kernel wants the
#     transposed h buffer; the trainer's kernel is vendored to consume the
#     vendored fwd_h output directly and skip a full h transpose copy.
#
# Kernel bodies are byte-identical to xorl @ the PR base
# (src/xorl/ops/linear_attention/ops/{utils/solve_tril,common/chunk_delta_h,
# common/chunk_o}.py, ops/utils/op.py); only imports/drivers are adapted.
# xorl pins IS_TMA_SUPPORTED=False, USE_CUDA_GRAPH=False and
# check_shared_mem()->False (see xorl ops/linear_attention/utils.py), which is
# mirrored here so the autotune spaces match the trainer exactly.
#
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Portions adapted from flash-linear-attention, MIT License.

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.fla.solve_tril_decode import solve_tril_decode
from sglang.kernels.ops.attention.fla.utils import autotune_cache_kwargs
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd

# Internal choices for the exact Qwen3.5-family GDN serving contract.
# Installed as a unit by the architecture resolver (see
# sglang.kernels.ops.attention.fla.qwen35_gdn_exact) -- there are no
# per-feature environment variables on this surface. The False defaults
# keep every non-contract server on the stock path, bit-for-bit
# unaffected; tests may set these module attributes directly on a fresh
# runner (never after a capture).
BI_GDN_PREFILL_ENABLED = False

# Route the triangular-solve stage to the decode-scheduled kernels
# (solve_tril_decode.py, vendored from the paired xorl trainer) --
# bitwise-identical to the pinned solve_tril by construction and gated in
# test_bi_gdn_solve_tril_decode.py.
# The architecture resolver selects this routing together with the FAST transport
# and never without it: the decode-scheduled solve over the ORACLE decode
# path exceeded the ~2 GiB graph-32 capture headroom at mem-fraction 0.50
# (byte-proven only at 0.48), so that combination is structurally
# unreachable on the public surface.
BI_GDN_SOLVE_TRIL_DECODE = False

CHUNK_SIZE = 64

# xorl ops/linear_attention/utils.py pins these platform switches; mirrored so
# the vendored kernels see the trainer's exact autotune spaces.
IS_TMA_SUPPORTED = False
USE_CUDA_GRAPH = False


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    del arch, tensor_idx
    return False


# --- vendored from xorl ops/linear_attention/ops/utils/op.py -----------------
# The exact path pins the guarded/default exponential program. Ambient FLA
# environment variables must not select a different numerical implementation.


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:

    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        del base, shape, strides, block_shape, _builder
        return None


# --- vendored from xorl ops/linear_attention/ops/utils/solve_tril.py ---------

FLA_TRIL_PRECISION = "ieee"
DOT_PRECISION_AUTOTUNE_LIST = [FLA_TRIL_PRECISION]

# The forward-substitution tl.sum reassociates with num_warps (measured: three
# distinct bitwise outputs for 2/4/8 warps on Qwen3.5 shapes); xorl pins the
# 2-warp variant (SOLVE_TRIL_NUM_WARPS in ops/utils/solve_tril.py) and the
# vendored copy must match. num_stages stays tuned (timing-only).
SOLVE_TRIL_NUM_WARPS = [2]


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"DOT_PRECISION": "ieee"}, num_warps=num_warps, num_stages=num_stages
        )
        for num_warps in SOLVE_TRIL_NUM_WARPS
        for num_stages in [2, 3, 4, 5]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos * H + i_h) * BT
    Ai = Ai + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    if not USE_TMA:
        p_A = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)
        )
        # [16, 16]
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        b_A = tl.where(m_A, b_A, 0)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, 16], [H * 16, 1], [16, 16])
        b_A = desc.load([i_t * 16, offset]).to(tl.float32)
        b_A = tl.where(m_A, b_A, 0)
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        # [16]
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
    if not USE_TMA:
        p_Ai = tl.make_block_ptr(
            Ai, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0)
        )
        tl.store(
            p_Ai,
            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
    else:
        desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"DOT_PRECISION": DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages
        )
        for num_warps in SOLVE_TRIL_NUM_WARPS
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=["H", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
        )
        p_A_22 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
        )
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
        )
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )

    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
        )
        p_Ai_21 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
        )
        p_Ai_22 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
        )
        tl.store(
            p_Ai_11,
            b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_22,
            b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_21,
            b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
    else:
        desc_o.store(
            [i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config(
            {"DOT_PRECISION": DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages
        )
        for num_warps in SOLVE_TRIL_NUM_WARPS
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=["H", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
        )
        p_A_22 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
        )
        p_A_33 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
        )
        p_A_44 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
        )
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.0)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.0)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.0)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.0)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
        )
        p_A_31 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
        )
        p_A_32 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
        )
        p_A_41 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
        )
        p_A_42 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
        )
        p_A_43 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
        )
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
        b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
        b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
        b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
        b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
        b_Ai_22,
        input_precision=DOT_PRECISION,
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
        b_Ai_33,
        input_precision=DOT_PRECISION,
    )

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
        )
        p_Ai_22 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
        )
        p_Ai_33 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
        )
        p_Ai_44 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
        )
        p_Ai_21 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
        )
        p_Ai_31 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
        )
        p_Ai_32 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
        )
        p_Ai_41 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
        )
        p_Ai_42 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
        )
        p_Ai_43 = tl.make_block_ptr(
            Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
        )
        tl.store(
            p_Ai_11,
            b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_22,
            b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_33,
            b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_44,
            b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_21,
            b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_31,
            b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_32,
            b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_41,
            b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_42,
            b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
        tl.store(
            p_Ai_43,
            b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
    else:
        desc_o.store(
            [i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )
        desc_o.store(
            [i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne")
        )


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Compute (I + A)^-1; A strictly lower triangular. Trainer-vendored driver."""
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel

    merge_fn[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=IS_TMA_SUPPORTED,
    )
    return Ai


# --- vendored from xorl ops/linear_attention/ops/common/chunk_delta_h.py -----


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2] if check_shared_mem("ampere") else [2, 1])
        for BV in ([32, 64] if check_shared_mem("ada") else [32])
    ],
    key=["H", "K", "V", "BT", "USE_EXP2"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        p_h1 = tl.make_block_ptr(
            h + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (128, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (192, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(
                tl.float32
            )
            p_g = tl.make_block_ptr(
                g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            ).to(tl.float32)
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[:, None]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[:, None]
                else:
                    b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[:, None]
                else:
                    b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[:, None]
                else:
                    b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT)
            if chunk_offsets is None
            else chunk_offsets,
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return h, v_new, final_state


# --- vendored from xorl ops/linear_attention/ops/common/chunk_o.py -----------

# BK/BV/num_warps are bit-relevant axes of chunk_fwd_kernel_o and their bits flip across triton 3.5->3.7;
# BK128/BV128/w4 reproduces the triton-3.5.1 anchor bits on both (K<=128), so it is pinned by default.
_FWD_O_CONFIGS = [triton.Config({"BK": 128, "BV": 128}, num_warps=4, num_stages=3)]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_G_GAMMA": lambda args: args["g_gamma"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=_FWD_O_CONFIGS,
    key=["H", "K", "V", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )

    b_v = tl.load(p_v, boundary_check=(0, 1))
    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o


# --- contract driver ----------------------------------------------------------


def _pool_to_trainer_state(pool_rows: torch.Tensor) -> torch.Tensor:
    """SGLang pool rows [N, H, V, K] fp32 -> trainer state [N, H, K, V] (exact)."""
    return pool_rows.transpose(-1, -2).contiguous()


def _trainer_state_to_pool(state: torch.Tensor) -> torch.Tensor:
    """Trainer final state [N, H, K, V] fp32 -> pool rows [N, H, V, K] (exact)."""
    return state.transpose(-1, -2).contiguous()


@torch.compiler.disable
def bi_chunk_gated_delta_rule_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    ckpt_batch_rows: torch.Tensor | None = None,
    ckpt_token_starts: list[int] | None = None,
    ckpt_lens: list[int] | None = None,
    ckpt_dst_indices: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    cache_indices_long: torch.LongTensor | None = None,
    return_intermediate_state: bool = False,
):
    """Chunked GDN prefill through the trainer's exact stage composition.

    Reads per-request initial states from ``ssm_states[cache_indices]`` (pool
    layout ``[N, H, V, K]`` fp32) and scatters the fp32 final states back to
    the same slots, bitwise identical to the trainer's
    ``chunk_gated_delta_rule`` outputs and final states.

    When ``ckpt_*`` is given (fp32 chunk-boundary checkpoint sourcing for
    prefix caching), the state at each request's last chunk-aligned boundary
    is recomputed by rescanning its aligned prefix from the same (pre-scan)
    initial state — fp32 chaining is exact, so this equals the full scan's
    internal state at that boundary — and written to
    ``ssm_states[ckpt_dst_indices]``. Entries with ``ckpt_lens == 0`` copy the
    pre-scan state directly.

    Returns the attention output ``o`` of shape ``[1, T, HV, V]``.
    """
    if ssm_states.dtype != torch.float32:
        raise RuntimeError(
            f"the exact GDN prefill requires an fp32 SSM state pool; got {ssm_states.dtype}."
        )
    if q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"the exact GDN prefill is a bf16 contract; got q dtype {q.dtype}."
        )
    # serving hands over non-contiguous split views; the FLA kernels assume
    # dense [B, T, H, D] strides (the default path gets this from input_guard)
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    HV = v.shape[-2]
    HK = k.shape[-2]
    if HV != HK:
        rep = HV // HK
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if cu_seqlens is not None:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)

    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    g = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k,
        beta,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices,
    )
    if BI_GDN_SOLVE_TRIL_DECODE:
        A = solve_tril_decode(
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=k.dtype,
        )
    else:
        A = solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=k.dtype,
        )
    w, u = recompute_w_u_fwd(
        k,
        v,
        beta,
        g_cumsum=g,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    idx = cache_indices.long() if cache_indices_long is None else cache_indices_long
    h0 = _pool_to_trainer_state(ssm_states.index_select(0, idx))

    # fp32 chunk-boundary checkpoints must be sourced BEFORE the in-place
    # final-state scatter below (both read the pre-scan pool slots).
    if ckpt_dst_indices is not None and ckpt_dst_indices.numel() > 0:
        _write_aligned_prefix_checkpoints(
            k=k,
            w=w,
            u=u,
            g=g,
            h0=h0,
            ssm_states=ssm_states,
            cache_indices=idx,
            ckpt_batch_rows=ckpt_batch_rows.long(),
            ckpt_token_starts=ckpt_token_starts,
            ckpt_lens=ckpt_lens,
            ckpt_dst_indices=ckpt_dst_indices,
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    ssm_states.index_copy_(
        0, idx, _trainer_state_to_pool(final_state).to(ssm_states.dtype)
    )

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return (o, _trainer_state_to_pool(h)) if return_intermediate_state else o


def _write_aligned_prefix_checkpoints(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    ckpt_batch_rows: torch.Tensor,
    ckpt_token_starts: list[int],
    ckpt_lens: list[int],
    ckpt_dst_indices: torch.Tensor,
):
    """Rescan chunk-aligned prefixes to produce fp32 chunk-boundary checkpoints.

    The intra-chunk stages (cumsum/kkt/solve_tril/w_u) are strictly chunk-local
    and the boundary is chunk-aligned, so the full-batch w/u/g tensors can be
    sliced directly; fwd_h over the sliced prefix from the same initial state
    reproduces the full scan's state at the boundary bitwise (the kernel's
    chunk loop has no lookahead).
    """
    dst = ckpt_dst_indices.long()
    seg_slices = []
    seg_pos = []
    zero_pos = []
    for pos, (start, length) in enumerate(zip(ckpt_token_starts, ckpt_lens)):
        if length == 0:
            zero_pos.append(pos)
        else:
            seg_slices.append((start, length))
            seg_pos.append(pos)

    if zero_pos:
        pos = torch.tensor(zero_pos, device=ssm_states.device, dtype=torch.long)
        # checkpoint == pre-scan state of the request's working slot
        src_slots = cache_indices.index_select(0, ckpt_batch_rows.index_select(0, pos))
        ssm_states.index_copy_(
            0, dst.index_select(0, pos), ssm_states.index_select(0, src_slots)
        )

    if not seg_slices:
        return

    k_c = torch.cat([k[:, s : s + n] for s, n in seg_slices], dim=1)
    w_c = torch.cat([w[:, s : s + n] for s, n in seg_slices], dim=1)
    u_c = torch.cat([u[:, s : s + n] for s, n in seg_slices], dim=1)
    g_c = torch.cat([g[:, s : s + n] for s, n in seg_slices], dim=1)
    lens = torch.tensor([0] + [n for _, n in seg_slices], dtype=torch.int32)
    cu_c = lens.cumsum(0).to(device=k.device, dtype=torch.int32)
    pos = torch.tensor(seg_pos, device=ssm_states.device, dtype=torch.long)
    h0_c = h0.index_select(0, ckpt_batch_rows.index_select(0, pos)).contiguous()

    _, _, ckpt_state = chunk_gated_delta_rule_fwd_h(
        k=k_c,
        w=w_c,
        u=u_c,
        g=g_c,
        initial_state=h0_c,
        output_final_state=True,
        cu_seqlens=cu_c,
    )
    ssm_states.index_copy_(
        0,
        dst.index_select(0, pos),
        _trainer_state_to_pool(ckpt_state).to(ssm_states.dtype),
    )
