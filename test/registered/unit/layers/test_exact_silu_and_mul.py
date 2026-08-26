from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.activation import SiluAndMul
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.xorl.bi.bi_silu_and_mul import (
    fp32_silu_and_mul,
)

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-large")


def _one_round_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    gate, up = input_tensor.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(input_tensor.dtype)


EXACT_MODE_FIELDS = (
    "glm52_exact_mode",
    "dsv4_flash_exact_mode",
    "qwen35_gdn_exact_mode",
    "qwen3_dense_exact_mode",
)


def _execution(*, target=None, exact_mode=None):
    values = {field: field == exact_mode for field in EXACT_MODE_FIELDS}
    values["rl_on_policy_target"] = target
    return SimpleNamespace(**values)


@pytest.mark.parametrize("exact_mode", EXACT_MODE_FIELDS)
def test_resolved_exact_contract_selects_exact_swiglu(exact_mode):
    execution = _execution(target="xorl", exact_mode=exact_mode)
    with patch("sglang.srt.layers.activation.get_server_args", return_value=execution):
        operation = SiluAndMul()

    assert operation._forward_method.__name__ == "forward_exact"


@pytest.mark.parametrize("target", [None, "xorl", "fsdp", "xorl-batch-invariant"])
def test_target_without_exact_contract_preserves_standard_dispatch(target):
    execution = _execution(target=target)
    with patch("sglang.srt.layers.activation.get_server_args", return_value=execution):
        operation = SiluAndMul()

    assert operation._forward_method is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for exact fused SwiGLU"
)
@pytest.mark.parametrize("shape", [(192, 24576), (512, 4096)])
def test_exact_swiglu_is_byte_exact_to_one_round_reference(shape):
    torch.manual_seed(4)
    input_tensor = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()

    actual = fp32_silu_and_mul(input_tensor)
    expected = _one_round_reference(input_tensor)

    assert torch.equal(actual, expected)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
