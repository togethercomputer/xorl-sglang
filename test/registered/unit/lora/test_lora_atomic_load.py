# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CPU regressions for transactional dynamic LoRA loading."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.lora import lora_manager as lora_manager_module
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _result(success, adapters, message=""):
    return LoRAUpdateOutput(
        success=success,
        error_message=message,
        loaded_adapters=adapters,
    )


def _manager():
    manager = LoRAManager.__new__(LoRAManager)
    existing_ref = LoRARef(
        lora_id="existing-id",
        lora_name="existing",
        lora_path="/adapters/existing",
        pinned=False,
    )
    manager.base_hf_config = SimpleNamespace(vocab_size=32)
    manager.configs = {existing_ref.lora_id: object()}
    manager.loras = {existing_ref.lora_id: object()}
    manager.lora_refs = {existing_ref.lora_id: existing_ref}
    manager.num_pinned_loras = 0
    manager.pending_lora_load_events = {}
    manager.validate_new_adapter = MagicMock()
    return manager


def test_disk_load_failure_removes_partial_local_state():
    manager = _manager()
    ref = LoRARef(
        lora_id="candidate-id",
        lora_name="candidate",
        lora_path="/adapters/incomplete",
        pinned=False,
    )

    def fail_after_partial_insert(_):
        manager.loras[ref.lora_id] = object()
        raise RuntimeError("missing adapter weights")

    manager.load_lora_weights = fail_after_partial_insert
    with patch("sglang.srt.lora.lora_manager.LoRAConfig", return_value=object()):
        result = manager._load_lora_adapter(ref)

    assert not result.success
    assert result.error_message == "missing adapter weights"
    assert set(manager.configs) == {"existing-id"}
    assert set(manager.loras) == {"existing-id"}
    assert set(manager.lora_refs) == {"existing-id"}
    assert result.loaded_adapters == {"existing": "/adapters/existing"}


def test_tensor_load_failure_removes_partial_local_state():
    manager = _manager()
    ref = LoRARef(
        lora_id="candidate-id",
        lora_name="candidate",
        lora_path="__tensor__",
        pinned=False,
    )

    def fail_after_partial_insert(*_):
        manager.loras[ref.lora_id] = object()
        raise RuntimeError("malformed tensor payload")

    manager.load_lora_weights_from_tensors = fail_after_partial_insert
    with patch.object(
        lora_manager_module.LoRAConfig, "from_dict", return_value=object()
    ):
        result = manager._load_lora_adapter_from_tensors(ref, {}, {})

    assert not result.success
    assert result.error_message == "malformed tensor payload"
    assert set(manager.configs) == {"existing-id"}
    assert set(manager.loras) == {"existing-id"}
    assert set(manager.lora_refs) == {"existing-id"}


def test_local_rollback_is_idempotent_and_clears_resident_slot():
    manager = _manager()
    ref = LoRARef(
        lora_id="candidate-id",
        lora_name="candidate",
        lora_path="/adapters/candidate",
        pinned=True,
    )
    manager.configs[ref.lora_id] = object()
    manager.loras[ref.lora_id] = object()
    manager.lora_refs[ref.lora_id] = ref
    manager.num_pinned_loras = 1
    manager.memory_pool = MagicMock()
    manager.memory_pool.remove_lora.return_value = 1
    manager._notify_lora_slots_updated = MagicMock()

    first = manager.rollback_lora_adapter(ref.lora_id)
    manager.memory_pool.remove_lora.return_value = None
    second = manager.rollback_lora_adapter(ref.lora_id)

    assert first.success and second.success
    assert manager.num_pinned_loras == 0
    assert set(manager.configs) == {"existing-id"}
    assert set(manager.loras) == {"existing-id"}
    assert set(manager.lora_refs) == {"existing-id"}
    manager._notify_lora_slots_updated.assert_called_once_with({1})


class _GatherSequence:
    def __init__(self, results):
        self.results = iter(results)
        self.local_values = []

    def all_gather_object(self, local_value):
        self.local_values.append(local_value)
        return next(self.results)


def _scheduler(load_result, gather_results):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tp_group = _GatherSequence(gather_results)
    scheduler.tp_worker = SimpleNamespace(
        load_lora_adapter=MagicMock(return_value=load_result),
        rollback_lora_adapter=MagicMock(
            return_value=_result(True, {"existing": "/adapters/existing"})
        ),
    )
    return scheduler


def _request():
    return LoadLoRAAdapterReqInput(
        lora_id="candidate-id",
        lora_name="candidate",
        lora_path="/adapters/candidate",
        pinned=False,
    )


def test_tp_load_success_requires_every_rank_and_matching_registries():
    committed = {"candidate": "/adapters/candidate"}
    rank_zero = _result(True, committed)
    scheduler = _scheduler(
        rank_zero,
        [[rank_zero, _result(True, committed)]],
    )

    result = scheduler.load_lora_adapter(_request())

    assert result is rank_zero
    scheduler.tp_worker.rollback_lora_adapter.assert_not_called()


def test_tp_partial_failure_rolls_back_successful_rank_before_reply():
    committed = {"candidate": "/adapters/candidate"}
    previous = {"existing": "/adapters/existing"}
    rank_zero = _result(True, committed)
    rollbacks = [_result(True, previous), _result(True, previous)]
    scheduler = _scheduler(
        rank_zero,
        [
            [rank_zero, _result(False, previous, "weights not visible")],
            rollbacks,
        ],
    )

    result = scheduler.load_lora_adapter(_request())

    assert not result.success
    assert "TP rank 1: weights not visible" in result.error_message
    assert result.loaded_adapters == previous
    scheduler.tp_worker.rollback_lora_adapter.assert_called_once_with("candidate-id")


def test_tp_registry_disagreement_is_a_failed_load():
    rank_zero = _result(True, {"candidate": "/adapters/candidate"})
    rank_one = _result(
        True,
        {
            "candidate": "/adapters/candidate",
            "stale": "/adapters/stale",
        },
    )
    previous = {"existing": "/adapters/existing"}
    scheduler = _scheduler(
        rank_zero,
        [
            [rank_zero, rank_one],
            [_result(True, previous), _result(True, previous)],
        ],
    )

    result = scheduler.load_lora_adapter(_request())

    assert not result.success
    assert "loaded adapter registries differ across TP ranks" in result.error_message
    scheduler.tp_worker.rollback_lora_adapter.assert_called_once_with("candidate-id")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
