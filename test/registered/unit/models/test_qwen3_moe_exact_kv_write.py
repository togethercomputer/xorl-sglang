from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.batch_invariant_ops import (
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
)
from sglang.srt.models.qwen3_moe import (
    _can_fuse_qwen3_set_kv_buffer,
    _qwen3_moe_norm_family,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _exec(*, rl_target):
    return SimpleNamespace(
        deterministic=SimpleNamespace(
            rl_on_policy_target=rl_target,
            qwen3_dense_exact_mode=rl_target == "xorl",
        )
    )


def test_native_deepep_qwen3_moe_uses_architecture_owned_norm_families():
    with patch(
        "sglang.srt.models.qwen3_moe.get_exec",
        return_value=_exec(rl_target="xorl"),
    ):
        assert _qwen3_moe_norm_family(layer_id=0) == RMS_NORM_FAMILY_NO_RESIDUAL
        assert _qwen3_moe_norm_family(layer_id=1) == RMS_NORM_FAMILY_RESIDUAL_TREE
        assert _qwen3_moe_norm_family(residual=True) == RMS_NORM_FAMILY_RESIDUAL_TREE


def test_exact_native_rope_retains_ordinary_kv_cache_write():
    with (
        patch(
            "sglang.srt.models.qwen3_moe.enable_fused_set_kv_buffer",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.qwen3_moe.get_exec",
            return_value=_exec(rl_target="xorl"),
        ),
    ):
        assert not _can_fuse_qwen3_set_kv_buffer(
            object(), compatible_with_fused_kv_buffer=True
        )


def test_ordinary_cuda_rope_may_own_kv_cache_write():
    with (
        patch(
            "sglang.srt.models.qwen3_moe.enable_fused_set_kv_buffer",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.qwen3_moe.get_exec",
            return_value=_exec(rl_target=None),
        ),
    ):
        assert _can_fuse_qwen3_set_kv_buffer(
            object(), compatible_with_fused_kv_buffer=True
        )


def test_incompatible_rope_never_owns_kv_cache_write():
    with (
        patch(
            "sglang.srt.models.qwen3_moe.enable_fused_set_kv_buffer",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.qwen3_moe.get_exec",
            return_value=_exec(rl_target=None),
        ),
    ):
        assert not _can_fuse_qwen3_set_kv_buffer(
            object(), compatible_with_fused_kv_buffer=False
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
