from unittest.mock import Mock

import torch

from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_nccl_weight_update_group_is_initialized_eagerly(monkeypatch):
    captured = {}

    def fake_init_custom_process_group(**kwargs):
        captured.update(kwargs)
        return "process-group"

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        weight_updater,
        "init_custom_process_group",
        fake_init_custom_process_group,
    )
    updater = weight_updater.WeightUpdater(
        tp_rank=2,
        device="cuda",
        gpu_id=3,
        model_config=Mock(),
        custom_weight_loaders={},
        get_model=Mock(),
        update_model_fields=Mock(),
        recapture_cuda_graph=Mock(),
        get_model_runner=Mock(),
    )

    assert updater.init_weights_update_group(
        "10.0.0.1",
        12345,
        rank_offset=9,
        world_size=17,
        group_name="weight-sync",
    ) == (True, "Succeeded to initialize custom process group.")
    assert captured["rank"] == 11
    assert captured["world_size"] == 17
    assert captured["device_id"] == torch.device("cuda:3")
    assert updater._model_update_group["weight-sync"] == "process-group"


def test_non_nccl_weight_update_group_does_not_force_a_cuda_device(monkeypatch):
    captured = {}

    def fake_init_custom_process_group(**kwargs):
        captured.update(kwargs)
        return "process-group"

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        weight_updater,
        "init_custom_process_group",
        fake_init_custom_process_group,
    )
    updater = weight_updater.WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=Mock(),
        custom_weight_loaders={},
        get_model=Mock(),
        update_model_fields=Mock(),
        recapture_cuda_graph=Mock(),
        get_model_runner=Mock(),
    )

    assert updater.init_weights_update_group(
        "127.0.0.1",
        12345,
        rank_offset=1,
        world_size=2,
        group_name="weight-sync",
        backend="gloo",
    )[0]
    assert captured["device_id"] is None


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
