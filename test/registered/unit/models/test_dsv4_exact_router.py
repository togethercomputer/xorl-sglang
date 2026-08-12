from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def test_deterministic_dsv4_router_keeps_fp32_serving_gemm(monkeypatch) -> None:
    from sglang.kernels.ops.attention import dsv4 as dsv4_ops
    from sglang.srt.models import deepseek_v2

    gate = deepseek_v2.MoEGate.__new__(deepseek_v2.MoEGate)
    torch.nn.Module.__init__(gate)
    gate.is_deepseek_v4 = True
    gate._glm52_exact_router = False
    gate.weight = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.bfloat16))

    monkeypatch.setattr(deepseek_v2, "use_intel_amx_backend", lambda _: False)
    monkeypatch.setattr(
        deepseek_v2,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=True)
        ),
    )
    calls = []

    def fake_linear_bf16_fp32(x, weight):
        calls.append((x, weight))
        return torch.full((x.shape[0], weight.shape[0]), 7.0, dtype=torch.float32)

    monkeypatch.setattr(dsv4_ops, "linear_bf16_fp32", fake_linear_bf16_fp32)
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    actual = gate(hidden)

    assert calls == [(hidden, gate.weight)]
    assert actual.dtype is torch.float32
    assert torch.equal(actual, torch.full((2, 3), 7.0))


def test_dsv4_exact_mxfp4_ep_pins_repeatable_marlin_block() -> None:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        select_marlin_moe_block_size_m,
    )

    common = dict(
        dsv4_exact_mode=True,
        num_tokens=48,
        topk=6,
        local_experts=32,
        hidden_size=4096,
        intermediate_size=2048,
        is_mxfp4_marlin=True,
        clamp_limit=10.0,
    )
    assert select_marlin_moe_block_size_m(global_experts=256, **common) == 64
    assert select_marlin_moe_block_size_m(global_experts=32, **common) == 16
    assert (
        select_marlin_moe_block_size_m(
            global_experts=256, **{**common, "dsv4_exact_mode": False}
        )
        == 16
    )


def test_hash_topk_can_disable_unarmed_pdl(monkeypatch) -> None:
    from sglang.kernels.ops.attention.dsv4 import moe

    calls = []

    class FakeModule:
        def hash_topk(self, _logits, _input_ids, _table, weights, ids, _scale):
            weights.fill_(0.5)
            ids.fill_(1)

    monkeypatch.setattr(
        moe,
        "_jit_hash_topk_module",
        lambda use_pdl=None: calls.append(use_pdl) or FakeModule(),
    )
    weights, ids = moe.hash_topk(
        router_logits=torch.zeros(3, 4),
        input_ids=torch.zeros(3, dtype=torch.int64),
        tid2eid=torch.zeros(8, 2, dtype=torch.int32),
        use_pdl=False,
    )

    assert calls == [False]
    assert torch.equal(weights, torch.full((3, 2), 0.5))
    assert torch.equal(ids, torch.ones(3, 2, dtype=torch.int32))
