from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _is_contiguous_or_gkn_transpose_view,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_serving_contiguous_and_gkn_transpose_views_are_supported():
    gkn = torch.empty(3, 8, 16)

    assert _is_contiguous_or_gkn_transpose_view(gkn)
    assert _is_contiguous_or_gkn_transpose_view(gkn.transpose(1, 2))
    assert not _is_contiguous_or_gkn_transpose_view(gkn[:, ::2, :])


def test_strided_down_weight_disables_tma_and_consumes_config_flag(monkeypatch):
    monkeypatch.setattr(fused_moe, "_down_moe_use_tma", lambda: True)
    contiguous = torch.empty(3, 8, 16)
    strided = contiguous.transpose(1, 2)

    contiguous_config = {"USE_TMA": True}
    assert fused_moe._resolve_down_moe_tma(contiguous, contiguous_config)
    assert "USE_TMA" not in contiguous_config

    strided_config = {"USE_TMA": True}
    assert not fused_moe._resolve_down_moe_tma(strided, strided_config)
    assert "USE_TMA" not in strided_config

    monkeypatch.setattr(fused_moe, "_down_moe_use_tma", lambda: False)
    unsupported_config = {"USE_TMA": True}
    assert not fused_moe._resolve_down_moe_tma(contiguous, unsupported_config)
    assert "USE_TMA" not in unsupported_config


def test_small_moe_sum_compile_respects_both_batch_invariance_contracts(monkeypatch):
    deterministic = SimpleNamespace(enable_deterministic_inference=False)
    monkeypatch.setattr(
        fused_moe,
        "get_exec",
        lambda: SimpleNamespace(deterministic=deterministic),
    )
    monkeypatch.setattr(fused_moe, "is_batch_invariant_mode_enabled", lambda: False)

    assert fused_moe._use_moe_sum_reduce_torch_compile(32)
    assert not fused_moe._use_moe_sum_reduce_torch_compile(33)

    deterministic.enable_deterministic_inference = True
    assert not fused_moe._use_moe_sum_reduce_torch_compile(32)

    deterministic.enable_deterministic_inference = False
    monkeypatch.setattr(fused_moe, "is_batch_invariant_mode_enabled", lambda: True)
    assert not fused_moe._use_moe_sum_reduce_torch_compile(32)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
