from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_no_combine_down_hook_updates_initialized_route_output(monkeypatch):
    monkeypatch.setattr(fused_moe, "_is_cuda", False)
    monkeypatch.setattr(fused_moe, "_is_hip", False)
    monkeypatch.setattr(fused_moe, "_is_xpu", False)
    monkeypatch.setattr(fused_moe, "_is_musa", False)
    monkeypatch.setattr(fused_moe, "_has_vllm_ops", False)
    monkeypatch.setattr(
        fused_moe,
        "get_exec",
        lambda: SimpleNamespace(
            moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
        ),
    )

    route_base = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.bfloat16
    )
    kernel_outputs = []

    def fake_invoke(*args, **_kwargs):
        output = args[3]
        kernel_outputs.append(tuple(output.shape))
        if output.shape[-1] == 4:
            output.fill_(1)
        else:
            assert tuple(output.shape) == tuple(route_base.shape)
            output.copy_(route_base)

    monkeypatch.setattr(fused_moe, "invoke_fused_moe_kernel", fake_invoke)

    hook_observations = []

    def after_down(_activated, routed, _weights, _ids):
        hook_observations.append(routed.clone())
        routed.add_(10)

    hooks = SimpleNamespace(after_gate_up=None, after_down=after_down)
    hidden = torch.ones((1, 2), dtype=torch.bfloat16)
    w1 = torch.ones((2, 4, 2), dtype=torch.bfloat16)
    w2 = torch.ones((2, 3, 2), dtype=torch.bfloat16)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)

    actual = fused_moe._fused_moe_kernel_sequence(
        hidden,
        w1,
        w2,
        weights,
        ids,
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.tensor(0, dtype=torch.int32),
        {"BLOCK_SIZE_M": 1},
        None,
        False,
        b1=None,
        b2=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        activation="silu",
        is_gated=True,
        no_combine=True,
        inplace=False,
        apply_router_weight_on_input=False,
        routed_scaling_factor=1.0,
        gemm1_alpha=None,
        gemm1_limit=None,
        filter_expert=False,
        hooks=hooks,
        swiglu_limit=None,
        gate_up_interleaved=False,
        a1_q=None,
    )

    assert kernel_outputs == [(2, 4), (1, 2, 3)]
    assert len(hook_observations) == 1
    assert torch.equal(hook_observations[0], route_base)
    assert torch.equal(actual, route_base + 10)


def test_no_combine_can_force_active_lora_destination_without_hooks(monkeypatch):
    monkeypatch.setattr(fused_moe, "_is_cuda", False)
    monkeypatch.setattr(fused_moe, "_is_hip", False)
    monkeypatch.setattr(fused_moe, "_is_xpu", False)
    monkeypatch.setattr(fused_moe, "_is_musa", False)
    monkeypatch.setattr(fused_moe, "_has_vllm_ops", False)
    monkeypatch.setattr(
        fused_moe,
        "get_exec",
        lambda: SimpleNamespace(
            moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
        ),
    )

    route_base = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.bfloat16
    )
    kernel_outputs = []

    def fake_invoke(*args, **_kwargs):
        output = args[3]
        kernel_outputs.append(tuple(output.shape))
        if output.shape[-1] == 4:
            output.fill_(1)
        else:
            output.copy_(route_base)

    monkeypatch.setattr(fused_moe, "invoke_fused_moe_kernel", fake_invoke)
    hidden = torch.ones((1, 2), dtype=torch.bfloat16)
    w1 = torch.ones((2, 4, 2), dtype=torch.bfloat16)
    w2 = torch.ones((2, 3, 2), dtype=torch.bfloat16)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)

    actual = fused_moe._fused_moe_kernel_sequence(
        hidden,
        w1,
        w2,
        weights,
        ids,
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.tensor(0, dtype=torch.int32),
        {"BLOCK_SIZE_M": 1},
        None,
        False,
        b1=None,
        b2=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        activation="silu",
        is_gated=True,
        no_combine=True,
        inplace=False,
        apply_router_weight_on_input=False,
        routed_scaling_factor=1.0,
        gemm1_alpha=None,
        gemm1_limit=None,
        filter_expert=False,
        hooks=None,
        force_intermediate_output=True,
        swiglu_limit=None,
        gate_up_interleaved=False,
        a1_q=None,
    )

    assert kernel_outputs == [(2, 4), (1, 2, 3)]
    assert torch.equal(actual, route_base)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
