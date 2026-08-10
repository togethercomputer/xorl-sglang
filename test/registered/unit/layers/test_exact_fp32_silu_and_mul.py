import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/batch_invariant_ops/bi_silu_and_mul.py"
)
_SPEC = importlib.util.spec_from_file_location("sglang_exact_fp32_silu_and_mul", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _one_round_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    gate, up = input_tensor.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(input_tensor.dtype)


def test_cpu_fallback_uses_one_round_program():
    values = torch.tensor(
        [[0.5, -1.25, 3.0, -0.75], [-2.0, 0.125, 1.5, 8.0]],
        dtype=torch.bfloat16,
    )
    actual = _MODULE.fp32_silu_and_mul(values)
    assert torch.equal(actual, _one_round_reference(values))
    assert not torch.equal(actual, _MODULE.two_round_silu_and_mul_reference(values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for exact fused SwiGLU")
@pytest.mark.parametrize("shape", [(192, 24576), (512, 4096)])
def test_forward_is_byte_exact_to_one_round_reference(shape):
    torch.manual_seed(4)
    input_tensor = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()

    actual = _MODULE.fp32_silu_and_mul(input_tensor)
    expected = _one_round_reference(input_tensor)

    assert torch.equal(actual, expected)
