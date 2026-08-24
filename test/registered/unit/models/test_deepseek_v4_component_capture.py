import pytest
import torch
import torch.nn as nn

from sglang.srt.layers.attention.deepseek_v4_backend import (
    _referenced_cache_rows,
)
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_sampler_attention_component_capture_selects_position(tmp_path, monkeypatch):
    monkeypatch.setenv("XORL_DSV4_SAMPLER_ATTENTION_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("XORL_DSV4_ATTENTION_CAPTURE_LAYER", "2")
    monkeypatch.setenv("XORL_DSV4_ATTENTION_CAPTURE_POSITION", "64")
    attention = MQALayer.__new__(MQALayer)
    nn.Module.__init__(attention)
    attention.layer_id = 2
    value = torch.arange(3 * 4 * 5, dtype=torch.bfloat16).reshape(3, 4, 5)

    attention._maybe_capture_exact_attention_component(
        "q_pre_attention",
        value,
        torch.tensor([63, 64, 65]),
    )

    [capture_path] = list(tmp_path.glob("*.pt"))
    payload = torch.load(capture_path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "xorl.dsv4_sampler_attention_component.v1"
    assert payload["component"] == "q_pre_attention"
    assert payload["position"] == 64
    torch.testing.assert_close(payload["value"], value[1:2])


def test_sampler_operator_capture_extracts_split_plane_flashmla_rows():
    page_size = 64
    page_bytes = ((584 * page_size + 575) // 576) * 576
    storage = torch.zeros((2, page_bytes), dtype=torch.uint8)
    reference = page_size + 5
    storage[1, 5 * 576 : 6 * 576] = 17
    storage[1, page_size * 576 + 5 * 8 : page_size * 576 + 6 * 8] = 29
    kernel_view = (
        storage[:, : page_size * 584]
        .view(torch.float8_e4m3fn)
        .view(2, page_size, 1, 584)
    )

    references, rows = _referenced_cache_rows(
        kernel_view,
        torch.tensor([[[reference]]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    assert references.tolist() == [reference]
    assert rows.shape == (1, 584)
    assert (rows[0, :576] == 17).all()
    assert (rows[0, 576:] == 29).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
