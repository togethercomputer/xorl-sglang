"""Exact FP32-intermediate SwiGLU shared with the XoRL trainer."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _use_fp32_fused_swiglu(input_tensor: torch.Tensor) -> bool:
    if not input_tensor.is_cuda or not input_tensor.is_contiguous():
        return False
    if input_tensor.dtype not in (torch.bfloat16, torch.float16):
        return False
    major, minor = torch.cuda.get_device_capability(input_tensor.device)
    if (major, minor) != (9, 0):
        return False
    return True


def _fp32_silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    intermediate_size = input_tensor.shape[-1] // 2
    gate = input_tensor[..., :intermediate_size].float()
    up = input_tensor[..., intermediate_size:].float()
    return (F.silu(gate) * up).to(input_tensor.dtype)


def two_round_silu_and_mul_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    """Execute the former program with a BF16 boundary after SiLU."""
    intermediate_size = input_tensor.shape[-1] // 2
    activated = F.silu(input_tensor[..., :intermediate_size].float()).to(
        input_tensor.dtype
    )
    return (activated * input_tensor[..., intermediate_size:]).to(input_tensor.dtype)


@triton.jit
def _fp32_silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    intermediate_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block_start = tl.program_id(1) * block_size
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
    result = activated * up.to(tl.float32)
    tl.store(output_ptr + row * intermediate_size + columns, result, mask=mask)


def fp32_silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    """Execute the one-round SwiGLU program through its fastest admitted realization."""
    if input_tensor.shape[-1] % 2:
        raise ValueError("SwiGLU input width must be even")
    if not _use_fp32_fused_swiglu(input_tensor):
        return _fp32_silu_and_mul(input_tensor)

    original_shape = input_tensor.shape
    input_2d = input_tensor.view(-1, original_shape[-1])
    rows = input_2d.shape[0]
    intermediate_size = input_2d.shape[1] // 2
    output = torch.empty(
        (rows, intermediate_size),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    block_size = min(1024, triton.next_power_of_2(intermediate_size))
    _fp32_silu_and_mul_kernel[(rows, triton.cdiv(intermediate_size, block_size))](
        input_2d,
        output,
        intermediate_size,
        block_size,
    )
    return output.view(*original_shape[:-1], intermediate_size)
