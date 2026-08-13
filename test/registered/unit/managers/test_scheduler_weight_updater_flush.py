"""Mandatory prefix-cache flush on weight sync for radix-enabled exact RL lanes.

Guarded bug class: with the radix cache enabled under ``rl_on_policy_target``,
a weight update whose client passes ``flush_cache=False`` (the xorl API's
``cache_invalidation_mode="auto"`` does exactly that for BF16-KV endpoints)
used to leave pre-update KV in the tree.  The next prefix hit then spliced
stale-policy bytes into new-policy decision logprobs — nonzero K3 with no
error anywhere.  The engine owns the numerical contract, so the flush must
not depend on the client remembering the flag.
"""

from unittest.mock import Mock

import pytest

from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager(flush_cache: Mock) -> SchedulerWeightUpdaterManager:
    return SchedulerWeightUpdaterManager(
        tp_worker=Mock(),
        draft_worker=None,
        tp_cpu_group=Mock(),
        memory_saver_adapter=Mock(),
        flush_cache=flush_cache,
        is_fully_idle=Mock(return_value=True),
    )


def _recv_req(flush_cache: bool) -> UpdateWeightFromDiskReqInput:
    return UpdateWeightFromDiskReqInput(model_path="dummy", flush_cache=flush_cache)


def test_weight_update_forces_radix_flush_for_exact_rl():
    flush = Mock(return_value=True)
    with get_context().override_server_args(
        rl_on_policy_target="xorl", disable_radix_cache=False
    ):
        _manager(flush).flush_cache_after_weight_update(_recv_req(flush_cache=False))
    flush.assert_called_once()


@pytest.mark.parametrize(
    "overrides",
    [
        # Outside the exact-RL contract the client's flush_cache=False stands.
        {"rl_on_policy_target": None, "disable_radix_cache": False},
        # With radix disabled there is no tree to flush.
        {"rl_on_policy_target": "xorl", "disable_radix_cache": True},
    ],
)
def test_weight_update_respects_client_flag_outside_the_contract(overrides):
    flush = Mock(return_value=True)
    with get_context().override_server_args(**overrides):
        _manager(flush).flush_cache_after_weight_update(_recv_req(flush_cache=False))
    flush.assert_not_called()


def test_mandatory_flush_failure_raises_instead_of_serving_stale_kv():
    flush = Mock(return_value=False)
    with get_context().override_server_args(
        rl_on_policy_target="xorl", disable_radix_cache=False
    ):
        with pytest.raises(RuntimeError, match="stale-policy"):
            _manager(flush).flush_cache_after_weight_update(
                _recv_req(flush_cache=False)
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
