from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.batch_invariant_ops.bi_silu_and_mul import (
    exact_fused_swiglu_min_rows,
    exact_silu_and_mul,
)
from sglang.srt.layers.activation import SiluAndMul


def _two_round_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    gate, up = input_tensor.chunk(2, dim=-1)
    activated = F.silu(gate.float()).to(input_tensor.dtype)
    return (activated * up).to(input_tensor.dtype)


def test_hopper_shape_resolver_thresholds():
    assert exact_fused_swiglu_min_rows(12288) == 192
    assert exact_fused_swiglu_min_rows(8192) == 192
    assert exact_fused_swiglu_min_rows(2048) == 512


def test_on_policy_swiglu_selects_exact_auto_resolver():
    execution = SimpleNamespace(
        deterministic=SimpleNamespace(rl_on_policy_target="xorl")
    )
    with patch("sglang.srt.layers.activation.get_exec", return_value=execution):
        operation = SiluAndMul()

    assert operation._forward_method.__name__ == "forward_exact"


def test_nonexact_swiglu_preserves_standard_dispatch():
    execution = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=None))
    with patch("sglang.srt.layers.activation.get_exec", return_value=execution):
        operation = SiluAndMul()

    assert operation._forward_method is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for exact fused SwiGLU"
)
@pytest.mark.parametrize("shape", [(192, 24576), (512, 4096)])
def test_exact_swiglu_is_byte_exact_to_two_round_reference(shape):
    torch.manual_seed(4)
    input_tensor = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()

    actual = exact_silu_and_mul(input_tensor)
    expected = _two_round_reference(input_tensor)

    assert torch.equal(actual, expected)
