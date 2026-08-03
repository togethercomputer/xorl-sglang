# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py

import contextlib
import logging
import os
from collections import namedtuple
from collections.abc import Callable, Iterable
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from sglang.srt.batch_invariant_ops.bi_gemm_configs import (
    baseline_mm_config,
    lookup_mm_config,
)
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.utils.common import calc_diff, get_bool_env_var

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm

logger = logging.getLogger(__name__)

_ENABLE_MM_DEEPGEMM = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "1"
)
# If true, allows to fallback to batch variant gemm when the shape cannot be run in DeepGEMM
_ENABLE_MM_FALLBACK_VARIANT = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT", "0"
)
# Shapes already reported by the loud mm-fallback error (one report per shape).
_MM_FALLBACK_SHAPES_REPORTED: set[tuple[int, int, int]] = set()
_ENABLE_MM_COMPARISON_TEST = get_bool_env_var(
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_COMPARISON_TEST"
)

if not _ENABLE_MM_DEEPGEMM:
    print("Disable DeepGEMM in batch invariant ops. Performance may be suboptimal.")

# Shapes whose table config exceeded shared memory at launch (Triton's epilogue
# staging for wide-output tiles is version-dependent): remembered so the hot
# path re-launches straight on the pinned baseline without re-raising.
_MM_CONFIG_OOM_SHAPES: set[tuple] = set()


def _launch_with_config_fallback(launch, dtype, M, N, K, out_itemsize=None):
    key = (str(dtype), M, N, K, out_itemsize)
    if key in _MM_CONFIG_OOM_SHAPES:
        launch(baseline_mm_config(dtype))
        return
    try:
        launch(lookup_mm_config(dtype, M, N, K, out_itemsize=out_itemsize))
    except OutOfResources:
        _MM_CONFIG_OOM_SHAPES.add(key)
        launch(baseline_mm_config(dtype))


__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "is_batch_invariant_op_enabled",
    "get_batch_invariant_ops",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
]


_BATCH_INVARIANT_ALL_OPS = {
    "mm",
    "addmm",
    "log_softmax",
    "mean",
    "rms_norm",
    "bmm",
}
_BATCH_INVARIANT_ALIASES = {
    "matmul": "mm",
    "logsoftmax": "log_softmax",
    "log-softmax": "log_softmax",
    "rmsnorm": "rms_norm",
    "rms-norm": "rms_norm",
}


def _normalize_batch_invariant_ops(ops: Iterable[str]) -> set[str]:
    normalized = set()
    for raw_op in ops:
        op = raw_op.strip().lower().replace("-", "_")
        op = _BATCH_INVARIANT_ALIASES.get(op, op)
        if op not in _BATCH_INVARIANT_ALL_OPS:
            raise ValueError(
                f"Unsupported batch-invariant op {raw_op!r}; "
                f"supported values are: {sorted(_BATCH_INVARIANT_ALL_OPS)}"
            )
        normalized.add(op)
    return normalized


def _parse_batch_invariant_ops() -> set[str]:
    raw = os.environ.get("SGLANG_BATCH_INVARIANT_OPS", "all").strip().lower()
    if raw in ("", "1", "true", "yes", "all"):
        return set(_BATCH_INVARIANT_ALL_OPS)
    if raw in ("0", "false", "no", "none"):
        return set()

    return _normalize_batch_invariant_ops(
        part for part in raw.replace(";", ",").split(",") if part.strip()
    )


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        elif c_ptr.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float32:
            c = accumulator.to(tl.float32)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def _matmul_persistent_triton(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert (
        bias is None or bias.dim() == 1
    ), "Currently assuming bias is 1D, let Horace know if you run into this"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    # Shape-keyed on the bit-neutral axes only; BLOCK_SIZE_K stays pinned per
    # dtype (bi_gemm_configs — the shape-keyed table, identical in both engines).
    def _launch(config):
        matmul_kernel_persistent[grid](
            a,
            b,
            c,  #
            bias,
            M,
            N,
            K,  #
            a.stride(0),
            a.stride(1),  #
            b.stride(0),
            b.stride(1),  #
            c.stride(0),
            c.stride(1),  #
            NUM_SMS=NUM_SMS,  #
            A_LARGE=a.numel() > 2**31,
            B_LARGE=b.numel() > 2**31,
            C_LARGE=c.numel() > 2**31,
            HAS_BIAS=bias is not None,
            **config,
        )

    _launch_with_config_fallback(_launch, dtype, M, N, K)
    return c


def _matmul_persistent_deepgemm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    out = torch.empty((M, N), device=a.device, dtype=dtype)

    try:
        deep_gemm.bf16_gemm_nn(a, b, out)
    except RuntimeError:
        return None

    # TODO can this be put in DeepGEMM's `c`?
    if bias is not None:
        out += bias

    return out


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    K, N = b.shape

    # DeepGEMM has minimum dimension requirements for TMA descriptors
    MIN_DEEPGEMM_DIM = 16

    if (
        _ENABLE_MM_DEEPGEMM
        and ENABLE_JIT_DEEPGEMM
        and (a.dtype == torch.bfloat16)
        and (b.dtype == torch.bfloat16)
        and a.is_contiguous()
        and b.transpose(0, 1).is_contiguous()
        and N >= MIN_DEEPGEMM_DIM
    ):
        if _ENABLE_MM_COMPARISON_TEST:
            out_triton = _matmul_persistent_triton(a=a, b=b, bias=bias)
            out_deepgemm = _matmul_persistent_deepgemm(a=a, b=b, bias=bias)
            if out_deepgemm is not None:
                diff = calc_diff(out_triton, out_deepgemm)
                assert diff < 0.0001, f"{diff=} {out_triton=} {out_deepgemm=}"
                return out_deepgemm
            # DeepGEMM failed, use Triton result
            return out_triton

        result = _matmul_persistent_deepgemm(a=a, b=b, bias=bias)
        if result is not None:
            return result
        # DeepGEMM failed (e.g. dimensions too small for TMA descriptors),
        # fall through to batch-invariant Triton persistent kernel

    if _ENABLE_MM_FALLBACK_VARIANT:
        shape = (a.shape[0], K, N)
        if shape not in _MM_FALLBACK_SHAPES_REPORTED:
            _MM_FALLBACK_SHAPES_REPORTED.add(shape)
            logger.error(
                "Batch-invariance is BROKEN for mm shape (M=%d, K=%d, N=%d): "
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT routed this "
                "DeepGEMM-rejected shape to torch.einsum, which is not "
                "batch-invariant. Unset the env var to use the batch-invariant "
                "Triton fallback instead. (Reported once per shape.)",
                shape[0],
                shape[1],
                shape[2],
            )
        out = torch.einsum("ik,kj->ij", a, b)
        if bias is not None:
            out += bias
        return out

    return _matmul_persistent_triton(a=a, b=b, bias=bias)


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    # Load first block to infer dtype and initialize max_val with correct type
    col_idx_init = tl.arange(0, BLOCK_SIZE)
    mask_init = col_idx_init < n_cols
    vals_init = tl.load(
        row_start_ptr + col_idx_init, mask=mask_init, other=-float("inf")
    )
    max_val = tl.max(vals_init)

    # Continue with remaining blocks
    for col_offset in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    # Initialize sum_exp with correct dtype by using tl.sum on a zero vector
    sum_exp = tl.sum(tl.zeros([1], dtype=max_val.dtype))

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)

    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert (
        -input.ndim <= dim < input.ndim
    ), f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        assert input.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "only float types supported for now"
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems


@triton.jit
def bmm_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    B,
    M,
    N,
    K,  #
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """
    Batched matrix multiplication kernel that processes batches in parallel.
    Each tile processes a (BLOCK_SIZE_M, BLOCK_SIZE_N) output block for a specific batch.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_batch = num_pid_m * num_pid_n
    num_tiles_total = B * num_tiles_per_batch

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Process tiles in a deterministic order: batch-major ordering
    for tile_id in tl.range(start_pid, num_tiles_total, NUM_SMS, flatten=True):
        # Decompose tile_id into batch and within-batch tile
        batch_idx = tile_id // num_tiles_per_batch
        tile_in_batch = tile_id % num_tiles_per_batch

        pid_m, pid_n = _compute_pid(
            tile_in_batch, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Add batch offset
        if A_LARGE or B_LARGE:
            batch_idx_typed = batch_idx.to(tl.int64)
        else:
            batch_idx_typed = batch_idx

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + (
                batch_idx_typed * stride_ab
                + offs_am[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                batch_idx_typed * stride_bb
                + offs_k[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = (
            c_ptr
            + batch_idx_typed * stride_cb
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        elif c_ptr.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float32:
            c = accumulator.to(tl.float32)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    # Process batches in parallel with our persistent kernel
    if a.ndim == 3 and b.ndim == 3:
        # Check constraints
        assert a.shape[0] == b.shape[0], "Batch sizes must match"
        assert a.shape[2] == b.shape[1], "Incompatible dimensions"
        assert a.dtype == b.dtype, "Incompatible dtypes"

        B = a.shape[0]
        M = a.shape[1]
        K = a.shape[2]
        N = b.shape[2]
        dtype = a.dtype

        # Allocate output
        if out is None:
            c = torch.empty((B, M, N), device=a.device, dtype=dtype)
        else:
            c = out

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        # Use fixed kernel configuration for determinism
        configs = {
            torch.bfloat16: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
            torch.float16: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
            torch.float32: {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
        }

        config = configs.get(dtype)
        if config is None:
            raise ValueError(
                f"Unsupported dtype {dtype} for bmm_batch_invariant. "
                f"Supported dtypes are: {list(configs.keys())}"
            )

        # Grid: limit by NUM_SMS for persistent kernel approach
        num_tiles_per_batch = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(
            N, config["BLOCK_SIZE_N"]
        )
        num_tiles_total = B * num_tiles_per_batch
        grid = (min(NUM_SMS, num_tiles_total),)

        bmm_kernel_persistent[grid](
            a,
            b,
            c,  #
            B,
            M,
            N,
            K,  #
            a.stride(0),
            a.stride(1),
            a.stride(2),  #
            b.stride(0),
            b.stride(1),
            b.stride(2),  #
            c.stride(0),
            c.stride(1),
            c.stride(2),  #
            NUM_SMS=NUM_SMS,  #
            A_LARGE=a.numel() > 2**31,
            B_LARGE=b.numel() > 2**31,
            C_LARGE=c.numel() > 2**31,
            **config,
        )

        return c
    else:
        raise ValueError(
            f"bmm_batch_invariant expects 3D tensors, "
            f"got shapes {a.shape} and {b.shape}"
        )


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each block handles one row of the input tensor.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel.

    RMS Norm normalizes the input by the root mean square and scales by weight:
    output = input / sqrt(mean(input^2) + eps) * weight

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tensor with RMS normalization applied along the last dimension
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch-invariant wrapper for RMS normalization.

    This function provides a deterministic, batch-invariant implementation
    of RMS normalization for use with the batch_invariant mode.

    Adapted from @https://github.com/vllm-project/vllm/blob/66a168a197ba214a5b70a74fa2e713c9eeb3251a/vllm/model_executor/layers/batch_invariant.py#L649

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor
    """
    return rms_norm(input, weight, eps=eps)


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

    Returns ``(output, residual_out)`` bit-matching ``RMSNorm.forward_native``'s
    RL residual path for the closed dense Qwen3 recipe
    (``cast_x_before_out_mul=True``, ``fp32_residual=False``,
    ``SGLANG_RMSNORM_FP32_WEIGHT_MUL=1``, ``post_residual_addition is None``).

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
    # bit-identical to the eager forward_native path's x.pow(2).mean(-1).
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
    """Full ``[N, V]`` fp32 logits through the contract's chunk GEMM (step 1).

    Decode twin of ``bi_lm_head_selected_logprob``: each vocab chunk is the
    same fixed-tile bf16 GEMM launch, stored into a column slice of one full
    fp32 buffer instead of a per-chunk scratch. ``tl.dot`` accumulation is
    independent of the output strides, so every element is bit-identical to
    the fused path's chunk logits. All launch shapes are static for a given
    ``[N, V]``, so the chunk loop is safe to capture in a CUDA graph.
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
    for col_start in range(0, vocab, vocab_chunk):
        col_end = min(col_start + vocab_chunk, vocab)
        _bi_lm_head_chunk_gemm_fp32(
            hidden, weight[col_start:col_end].t(), logits[:, col_start:col_end]
        )
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
# Enable via XORL_MOE_BI_ROUTER=1 (xorl) / SGLANG_BI_ROUTER=1 (sglang).
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
        **_BI_ROUTER_GEMM_CONFIG,
    )
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
        denom = topk_vals[..., 0]
        for k in range(1, topk_vals.shape[-1]):
            denom = denom + topk_vals[..., k]
        topk_vals = topk_vals / denom.unsqueeze(-1)
    return topk_vals.to(out_dtype)


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_batch_invariant_OPS: set[str] = set()
_original_torch_bmm = None


def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE


def get_batch_invariant_ops() -> tuple[str, ...]:
    return tuple(sorted(_batch_invariant_OPS)) if _batch_invariant_MODE else ()


def is_batch_invariant_op_enabled(op: str) -> bool:
    op = _BATCH_INVARIANT_ALIASES.get(op, op)
    return _batch_invariant_MODE and op in _batch_invariant_OPS


def enable_batch_invariant_mode(
    enable_bmm: bool = True,
    *,
    ops: Optional[Iterable[str]] = None,
):
    global _batch_invariant_MODE, _batch_invariant_LIB, _batch_invariant_OPS, _original_torch_bmm
    if _batch_invariant_MODE:
        return

    _batch_invariant_OPS = (
        _parse_batch_invariant_ops()
        if ops is None
        else _normalize_batch_invariant_ops(ops)
    )
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    if "mm" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
    if "addmm" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
    if "log_softmax" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl(
            "aten::_log_softmax", _log_softmax_batch_invariant, "CUDA"
        )
    if "mean" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")

    if enable_bmm and "bmm" in _batch_invariant_OPS:
        _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "CUDA")

        # Also monkeypatch torch.bmm directly as a fallback
        _original_torch_bmm = torch.bmm
        torch.bmm = bmm_batch_invariant


def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB, _batch_invariant_OPS, _original_torch_bmm
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    if _original_torch_bmm is not None:
        torch.bmm = _original_torch_bmm
        _original_torch_bmm = None
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None
    _batch_invariant_OPS = set()


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    # NOTE: the exit path must re-register ops from scratch instead of
    # restoring the saved (destroyed) torch.library.Library handle. The old
    # implementation destroyed the live Library on exit and then restored the
    # stale handle with _batch_invariant_MODE=True, so batch-invariant mode
    # reported enabled with zero ops actually registered.
    was_enabled = _batch_invariant_MODE
    if enabled == was_enabled:
        yield
        return

    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    try:
        yield
    finally:
        # Re-enabling re-reads SGLANG_BATCH_INVARIANT_OPS and uses the default
        # enable_bmm rather than restoring the exact prior registration. All
        # in-tree callers use the defaults, so this is equivalent in practice.
        if was_enabled:
            enable_batch_invariant_mode()
        else:
            disable_batch_invariant_mode()


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    return AttentionBlockSize(block_m=16, block_n=16)
