"""Exact two-round SwiGLU shared with the XoRL trainer."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_HOPPER_WIDE_MIN_ROWS = 192
_HOPPER_OTHER_MIN_ROWS = 512
_WIDE_INTERMEDIATE_SIZE = 8192


def exact_fused_swiglu_min_rows(intermediate_size: int) -> int:
    """Return the measured conservative Hopper crossover."""
    if intermediate_size >= _WIDE_INTERMEDIATE_SIZE:
        return _HOPPER_WIDE_MIN_ROWS
    return _HOPPER_OTHER_MIN_ROWS


def _use_exact_fused_swiglu(input_tensor: torch.Tensor) -> bool:
    if not input_tensor.is_cuda or not input_tensor.is_contiguous():
        return False
    if input_tensor.dtype not in (torch.bfloat16, torch.float16):
        return False
    major, minor = torch.cuda.get_device_capability(input_tensor.device)
    if (major, minor) != (9, 0):
        return False
    rows = input_tensor.numel() // input_tensor.shape[-1]
    intermediate_size = input_tensor.shape[-1] // 2
    return rows >= exact_fused_swiglu_min_rows(intermediate_size)


def _native_exact_silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    intermediate_size = input_tensor.shape[-1] // 2
    return (
        F.silu(input_tensor[..., :intermediate_size])
        * input_tensor[..., intermediate_size:]
    )


@triton.jit
def _exact_silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    intermediate_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    for block_start in range(0, intermediate_size, block_size):
        columns = block_start + tl.arange(0, block_size)
        mask = columns < intermediate_size
        gate = tl.load(
            input_ptr + row * 2 * intermediate_size + columns,
            mask=mask,
            other=0.0,
        )
        up = tl.load(
            input_ptr + row * 2 * intermediate_size + intermediate_size + columns,
            mask=mask,
            other=0.0,
        )
        gate_f32 = gate.to(tl.float32)
        activated = gate_f32 * tl.sigmoid(gate_f32)
        result = activated.to(gate.dtype) * up
        tl.store(output_ptr + row * intermediate_size + columns, result, mask=mask)


def exact_silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    """Execute one exact SwiGLU program through its fastest admitted realization."""
    if input_tensor.shape[-1] % 2:
        raise ValueError("SwiGLU input width must be even")
    if not _use_exact_fused_swiglu(input_tensor):
        return _native_exact_silu_and_mul(input_tensor)

    original_shape = input_tensor.shape
    input_2d = input_tensor.view(-1, original_shape[-1])
    rows = input_2d.shape[0]
    intermediate_size = input_2d.shape[1] // 2
    output = torch.empty(
        (rows, intermediate_size),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    _exact_silu_and_mul_kernel[(rows,)](
        input_2d,
        output,
        intermediate_size,
        1024,
    )
    return output.view(*original_shape[:-1], intermediate_size)
