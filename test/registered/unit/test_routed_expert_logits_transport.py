from types import SimpleNamespace

import numpy as np
import pybase64
import torch

from sglang.srt.state_capturer.base import BaseDeviceCache
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCaptureOutput,
    _routed_experts_device_cache_rows,
    extract_expert_logits_from_meta_info,
)


def test_routed_experts_cache_reconstructs_global_prefill_when_chunking_is_disabled():
    rows = _routed_experts_device_cache_rows(
        chunked_prefill_size=-1,
        max_prefill_tokens=32768,
        max_running_requests=27,
        dp_size=8,
    )
    assert rows == 262144

    cache = BaseDeviceCache(
        rows,
        num_layers=1,
        topk_size=8,
        device="cpu",
        name="routed_experts_test",
    )
    captured = torch.arange(49456 * 8, dtype=torch.int32).reshape(49456, 8)
    cache.capture(0, captured)
    torch.testing.assert_close(cache.buffer[:49456, 0], captured, rtol=0, atol=0)


def test_routed_experts_cache_reconstructs_global_chunk_and_decode_bounds():
    assert (
        _routed_experts_device_cache_rows(
            chunked_prefill_size=512,
            max_prefill_tokens=32768,
            max_running_requests=27,
            dp_size=8,
        )
        == 4096
    )
    assert (
        _routed_experts_device_cache_rows(
            chunked_prefill_size=128,
            max_prefill_tokens=32768,
            max_running_requests=256,
            dp_size=8,
        )
        == 2048
    )


def test_routed_experts_capture_output_finalizes_indices_and_weights():
    index_host = SimpleNamespace(buffer=torch.zeros((5, 2, 2), dtype=torch.int32))
    weight_host = SimpleNamespace(buffer=torch.zeros((5, 2, 2), dtype=torch.float32))
    output = RoutedExpertsCaptureOutput(
        out_cache_loc=torch.tensor([1, 3]),
        topk=torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int32),
        host_cache=index_host,
        expert_logits=torch.tensor(
            [[[0.75, 0.25], [0.6, 0.4]], [[0.9, 0.1], [0.8, 0.2]]],
            dtype=torch.float32,
        ),
        expert_logits_host_cache=weight_host,
    )

    output.map_device_tensors(lambda tensor: tensor.clone())
    output.finalize()

    torch.testing.assert_close(index_host.buffer[[1, 3]], output.topk)
    torch.testing.assert_close(
        weight_host.buffer[[1, 3]], output.expert_logits, rtol=0, atol=0
    )
    assert weight_host.buffer.dtype == torch.float32


def test_extract_expert_logits_preserves_float32_bytes():
    expected = np.asarray([0.75, 0.25, 0.6, 0.4], dtype=np.float32)
    encoded = pybase64.b64encode(expected.tobytes()).decode("ascii")
    actual = extract_expert_logits_from_meta_info(
        {"meta_info": {"expert_logits": encoded}}
    )
    np.testing.assert_array_equal(actual, expected)


def test_inkling_gate_captures_computed_routed_weights(monkeypatch):
    from sglang.srt.models.inkling_common import moe as inkling_moe

    gate = object.__new__(inkling_moe.InklingGate)
    torch.nn.Module.__init__(gate)
    gate.n_routed_experts = 3
    gate.n_shared_experts = 0
    gate.n_total_experts = 3
    gate.topk = 2
    gate.layer_id = 7
    gate.norm_after_topk = False
    gate.gate_activation = "softmax"
    gate.global_scale = None
    gate.route_scale = 1.25
    gate.bias = None
    gate.shared_expert_sink = False
    gate.weight = torch.nn.Parameter(torch.zeros(3, 2), requires_grad=False)

    logits = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32)
    topk_indices = torch.tensor([[1, 2]], dtype=torch.int32)
    captured = {}

    class _Capturer:
        def capture(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(inkling_moe, "linear_with_pad", lambda *_args: logits)
    monkeypatch.setattr(
        inkling_moe,
        "gate_topk",
        lambda *_args: (torch.empty_like(topk_indices), topk_indices),
    )
    monkeypatch.setattr(inkling_moe, "get_global_experts_capturer", lambda: _Capturer())

    routed_weights, actual_ids, _, _ = gate(torch.zeros(1, 2))

    assert captured["layer_id"] == 7
    torch.testing.assert_close(captured["topk_indices"], actual_ids)
    torch.testing.assert_close(captured["topk_weights"], routed_weights)
