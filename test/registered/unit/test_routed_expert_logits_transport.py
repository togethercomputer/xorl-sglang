from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
import torch

from sglang.srt.state_capturer.base import BaseDeviceCache
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCaptureOutput,
    RoutedExpertsCapturer,
    _routed_experts_device_cache_rows,
    add_routed_experts_to_pp_output,
    extract_expert_logits_from_meta_info,
    publish_routed_experts_from_pp_output,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
        req_pool_indices=torch.tensor([0, 2]),
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


def test_deepep_cp_capture_slices_local_rows_and_restores_logical_order(monkeypatch):
    from sglang.srt.layers.cp import base as cp_base
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    class _DeepEPBackend:
        @staticmethod
        def is_deepep():
            return True

    parallel = SimpleNamespace(attn_cp_size=4, attn_tp_size=4, attn_cp_rank=2)
    monkeypatch.setattr(
        routed_experts_module, "get_moe_a2a_backend", lambda: _DeepEPBackend()
    )
    monkeypatch.setattr(routed_experts_module, "get_parallel", lambda: parallel)

    metadata = SimpleNamespace(
        per_rank_logical_token=[3, 3, 3, 2],
        per_rank_actual_token=[4, 4, 4, 4],
        total_seq_lens=11,
    )
    forward_batch = SimpleNamespace(
        attn_cp_metadata=metadata,
        out_cache_loc=torch.arange(6),
    )
    forward_batch.out_cache_loc = torch.arange(11)
    assert RoutedExpertsCapturer._get_deepep_local_row_bounds(forward_batch) == (0, 16)

    physical_rows = torch.arange(16 * 2 * 2, dtype=torch.int32).reshape(16, 2, 2)
    logical_indices = torch.tensor([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10])
    expected = physical_rows.index_select(0, logical_indices)

    class _Strategy:
        name = "interleave"

    monkeypatch.setattr(cp_base, "get_cp_strategy", lambda: _Strategy())
    actual = RoutedExpertsCapturer._restore_deepep_cp_logical_rows(
        physical_rows,
        forward_batch,
        payload_name="routed-expert ids",
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deepep_cp_plan_is_bound_to_eager_owner_before_forward_end(monkeypatch):
    from sglang.srt.layers.cp import base as cp_base
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    class _DeepEPBackend:
        @staticmethod
        def is_deepep():
            return True

    parallel = SimpleNamespace(attn_cp_size=4, attn_tp_size=1, attn_cp_rank=2)
    monkeypatch.setattr(
        routed_experts_module, "get_moe_a2a_backend", lambda: _DeepEPBackend()
    )
    monkeypatch.setattr(routed_experts_module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        cp_base,
        "get_cp_strategy",
        lambda: SimpleNamespace(name="interleave"),
    )

    # This is the live 62-token failure geometry.  The execution copy has CP
    # metadata, but ModelRunner.on_forward_end receives the distinct owner.
    metadata = SimpleNamespace(
        per_rank_logical_token=[16, 16, 15, 15],
        per_rank_actual_token=[16, 16, 16, 16],
        total_seq_lens=62,
    )
    execution = SimpleNamespace(
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
        attn_cp_metadata=metadata,
        _original_num_tokens=62,
        input_ids=torch.arange(62),
        out_cache_loc=torch.arange(62),
    )
    owner = SimpleNamespace(
        attn_cp_metadata=None,
        _original_num_tokens=62,
        out_cache_loc=torch.arange(62),
    )

    capturer = object.__new__(RoutedExpertsCapturer)
    capturer.prepare_forward(execution, owner_forward_batch=owner)

    assert owner.attn_cp_metadata is metadata
    assert owner._deepep_route_capture_rank_rows == 16
    assert owner._deepep_route_capture_total_rows == 62

    # Rank-major CP padding occupies raw rows 47 and 63.  The exact historical
    # bug published raw row 47 as logical row 47.  A bound restoration plan
    # excludes both padding rows and maps logical row 47 from raw row 59.
    raw = torch.arange(64, dtype=torch.int32).view(64, 1, 1) + 1
    raw[47] = 0
    raw[63] = 0
    capturer.device_cache = SimpleNamespace(buffer=raw)
    capturer.topk_size = 1

    restored = capturer._get_local_slice(
        owner,
        can_run_graph=False,
        cuda_graph_batch=None,
    )
    assert restored.shape == (62, 1, 1)
    assert torch.count_nonzero(restored, dim=(1, 2)).bool().all()
    torch.testing.assert_close(restored[47], raw[59], rtol=0, atol=0)


def test_deepep_route_publication_drops_mlp_sync_padding(monkeypatch):
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    class _DeepEPBackend:
        @staticmethod
        def is_deepep():
            return True

    monkeypatch.setattr(
        routed_experts_module, "get_moe_a2a_backend", lambda: _DeepEPBackend()
    )
    monkeypatch.setattr(
        routed_experts_module,
        "get_parallel",
        lambda: SimpleNamespace(pp_size=1),
    )

    forward_batch = SimpleNamespace(
        _original_num_tokens=2,
        out_cache_loc=torch.tensor([3, 4, 0, 0]),
        req_pool_indices=torch.tensor([7]),
    )
    physical_rows = torch.tensor(
        [
            [[11, 12]],
            [[21, 22]],
            [[91, 92]],
            [[93, 94]],
        ],
        dtype=torch.int32,
    )
    physical_weights = physical_rows.to(torch.float32) / 100
    capturer = object.__new__(RoutedExpertsCapturer)
    capturer.host_cache = SimpleNamespace(
        buffer=torch.zeros((5, 1, 2), dtype=torch.int32)
    )
    capturer.expert_logits_host_cache = SimpleNamespace(
        buffer=torch.zeros((5, 1, 2), dtype=torch.float32)
    )
    capturer._get_local_slice = lambda *_args: physical_rows
    capturer._get_local_expert_logits_slice = lambda *_args: physical_weights
    capturer._transport_deepep_pp_layers = lambda indices, expert_logits, **kwargs: (
        indices,
        expert_logits,
        kwargs["out_cache_loc"],
        kwargs["req_pool_indices"],
    )

    actual = capturer.on_forward_end(
        forward_batch,
        can_run_graph=False,
        cuda_graph_batch=None,
        no_copy_to_cpu=False,
    )

    assert actual is None
    torch.testing.assert_close(
        capturer.host_cache.buffer[[3, 4]], physical_rows[:2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        capturer.expert_logits_host_cache.buffer[[3, 4]],
        physical_weights[:2],
        rtol=0,
        atol=0,
    )
    # Padded out_cache_loc entries are zero; they must never overwrite slot 0.
    assert torch.count_nonzero(capturer.host_cache.buffer[0]) == 0
    assert torch.count_nonzero(capturer.expert_logits_host_cache.buffer[0]) == 0


def test_deepep_prefill_capture_gathers_over_attention_cp(monkeypatch):
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    class _DeepEPBackend:
        @staticmethod
        def is_deepep():
            return True

    parallel = SimpleNamespace(attn_cp_size=4, attn_tp_size=1)
    monkeypatch.setattr(
        routed_experts_module, "get_moe_a2a_backend", lambda: _DeepEPBackend()
    )
    monkeypatch.setattr(routed_experts_module, "get_parallel", lambda: parallel)
    calls = []

    def cp_gather(output, local):
        calls.append("cp")
        for rank in range(4):
            output[rank * local.shape[0] : (rank + 1) * local.shape[0]].copy_(
                local + rank * 10
            )

    monkeypatch.setattr(
        routed_experts_module, "attn_cp_all_gather_into_tensor", cp_gather
    )
    monkeypatch.setattr(
        routed_experts_module,
        "attn_tp_all_gather_into_tensor",
        lambda *_args: pytest.fail("prefill CP capture used the attention-TP group"),
    )

    captured = {}
    capturer = object.__new__(RoutedExpertsCapturer)
    capturer.capture_topk_weights = False
    capturer._deepep_prefill_cp_active = True
    capturer.gather_buffer = torch.empty((8, 2), dtype=torch.int32)
    capturer.device_cache = SimpleNamespace(
        capture=lambda layer_id, values: captured.update(
            layer_id=layer_id, values=values.clone()
        )
    )
    local = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    capturer.capture(layer_id=7, topk_indices=local)

    assert calls == ["cp"]
    assert captured["layer_id"] == 7
    torch.testing.assert_close(
        captured["values"],
        torch.cat([local + rank * 10 for rank in range(4)]),
        rtol=0,
        atol=0,
    )


def test_deepep_decode_capture_uses_attention_tp_when_other_owner_extends(
    monkeypatch,
):
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    class _DeepEPBackend:
        @staticmethod
        def is_deepep():
            return True

    parallel = SimpleNamespace(attn_cp_size=4, attn_tp_size=2)
    monkeypatch.setattr(
        routed_experts_module, "get_moe_a2a_backend", lambda: _DeepEPBackend()
    )
    monkeypatch.setattr(routed_experts_module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        routed_experts_module,
        "attn_cp_all_gather_into_tensor",
        lambda *_args: pytest.fail("local decode capture used the attention-CP group"),
    )

    calls = []

    def tp_gather(output, local):
        calls.append("tp")
        output[: local.shape[0]].copy_(local)
        output[local.shape[0] :].copy_(local + 10)

    monkeypatch.setattr(
        routed_experts_module, "attn_tp_all_gather_into_tensor", tp_gather
    )
    captured = {}
    capturer = object.__new__(RoutedExpertsCapturer)
    capturer.capture_topk_weights = False
    # Another DP owner may make the synchronized extend flag true, but local
    # decode preparation leaves this CP plan inactive.
    capturer._deepep_prefill_cp_active = False
    capturer.gather_buffer = torch.empty((4, 2), dtype=torch.int32)
    capturer.device_cache = SimpleNamespace(
        capture=lambda layer_id, values: captured.update(
            layer_id=layer_id, values=values.clone()
        )
    )
    local = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    capturer.capture(layer_id=7, topk_indices=local)

    assert calls == ["tp"]
    torch.testing.assert_close(
        captured["values"], torch.cat((local, local + 10)), rtol=0, atol=0
    )


def test_deepep_prefill_capture_excludes_mlp_sync_suffix_before_cp_gather(
    monkeypatch,
):
    from sglang.srt.layers.cp import base as cp_base
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    class _DeepEPBackend:
        @staticmethod
        def is_deepep():
            return True

    parallel = SimpleNamespace(
        attn_cp_size=4,
        attn_tp_size=1,
        attn_cp_rank=2,
    )
    monkeypatch.setattr(
        routed_experts_module, "get_moe_a2a_backend", lambda: _DeepEPBackend()
    )
    monkeypatch.setattr(routed_experts_module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        cp_base,
        "get_cp_strategy",
        lambda: SimpleNamespace(name="interleave"),
    )

    gathered_inputs = []

    def cp_gather(output, local):
        gathered_inputs.append(local.clone())
        for rank in range(4):
            output[rank * local.shape[0] : (rank + 1) * local.shape[0]].copy_(
                local + rank * 100
            )

    monkeypatch.setattr(
        routed_experts_module, "attn_cp_all_gather_into_tensor", cp_gather
    )

    captured = {}
    capturer = object.__new__(RoutedExpertsCapturer)
    capturer.capture_topk_weights = True
    capturer.gather_buffer = torch.empty((16, 2), dtype=torch.int32)
    capturer.expert_logits_gather_buffer = torch.empty((16, 2), dtype=torch.float32)
    capturer.device_cache = SimpleNamespace(
        capture=lambda layer_id, values: captured.update(
            ids_layer=layer_id, ids=values.clone()
        )
    )
    capturer.expert_logits_device_cache = SimpleNamespace(
        capture=lambda layer_id, values: captured.update(
            weights_layer=layer_id, weights=values.clone()
        )
    )
    metadata = SimpleNamespace(
        # Real request length is 8.  DP/MLP sync pads the logical batch to 20,
        # then CP alignment gives every rank physical capacity 8.  Neither
        # metadata count is the real per-rank route count (2).
        per_rank_logical_token=[5, 5, 5, 5],
        per_rank_actual_token=[8, 8, 8, 8],
        total_seq_lens=20,
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
        attn_cp_metadata=metadata,
        _original_num_tokens=8,
        out_cache_loc=torch.arange(20),
    )
    capturer.prepare_forward(forward_batch)

    # The last six rows are DP/MLP-sync and CP-alignment padding.  They are
    # unstable-looking and must not affect either collective's rank stride.
    local_ids = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [901, 902],
            [903, 904],
            [905, 906],
            [907, 908],
            [909, 910],
            [911, 912],
        ],
        dtype=torch.int32,
    )
    local_weights = local_ids.to(torch.float32) / 10
    capturer.capture(
        layer_id=7,
        topk_indices=local_ids,
        topk_weights=local_weights,
    )

    assert captured["ids_layer"] == captured["weights_layer"] == 7
    assert len(gathered_inputs) == 2
    torch.testing.assert_close(gathered_inputs[0], local_ids[:2], rtol=0, atol=0)
    torch.testing.assert_close(gathered_inputs[1], local_weights[:2], rtol=0, atol=0)
    expected_ids = torch.cat([local_ids[:2] + rank * 100 for rank in range(4)])
    expected_weights = torch.cat([local_weights[:2] + rank * 100 for rank in range(4)])
    torch.testing.assert_close(captured["ids"], expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(captured["weights"], expected_weights, rtol=0, atol=0)

    logical_ids = RoutedExpertsCapturer._restore_deepep_cp_logical_rows(
        captured["ids"].unsqueeze(1),
        forward_batch,
        payload_name="routed-expert ids",
    ).squeeze(1)
    logical_weights = RoutedExpertsCapturer._restore_deepep_cp_logical_rows(
        captured["weights"].unsqueeze(1),
        forward_batch,
        payload_name="routed-expert weights",
    ).squeeze(1)
    logical_order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    torch.testing.assert_close(
        logical_ids, expected_ids.index_select(0, logical_order), rtol=0, atol=0
    )
    torch.testing.assert_close(
        logical_weights,
        expected_weights.index_select(0, logical_order),
        rtol=0,
        atol=0,
    )


def test_deepep_pp_capture_flows_downstream_without_collective(monkeypatch):
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    class _DeepEPBackend:
        @staticmethod
        def is_deepep():
            return True

    class _PPGroup:
        is_first_rank = True
        is_last_rank = False

    parallel = SimpleNamespace(pp_size=2, pp_group=_PPGroup())
    monkeypatch.setattr(
        routed_experts_module, "get_moe_a2a_backend", lambda: _DeepEPBackend()
    )
    monkeypatch.setattr(routed_experts_module, "get_parallel", lambda: parallel)

    first = object.__new__(RoutedExpertsCapturer)
    first.num_layers = 4
    first.pp_start_layer = 0
    first.pp_end_layer = 2
    first_indices = torch.tensor(
        [[[1, 2], [3, 4], [91, 92], [93, 94]]], dtype=torch.int32
    )
    first_weights = torch.tensor(
        [[[0.7, 0.3], [0.6, 0.4], [91.0, 92.0], [93.0, 94.0]]],
        dtype=torch.float32,
    )

    class _Proxy:
        def __init__(self):
            self.tensors = {}

        def __setitem__(self, key, value):
            self.tensors[key] = value

    downstream = _Proxy()

    actual = first._transport_deepep_pp_layers(
        first_indices,
        first_weights,
        out_cache_loc=torch.tensor([17]),
        req_pool_indices=torch.tensor([9]),
        incoming_pp_proxy_tensors=None,
        outgoing_pp_proxy_tensors=downstream,
    )
    assert actual == (None, None, None, None)
    assert len(downstream.tensors) == 4

    parallel.pp_group.is_first_rank = False
    parallel.pp_group.is_last_rank = True
    last = object.__new__(RoutedExpertsCapturer)
    last.num_layers = 4
    last.pp_start_layer = 2
    last.pp_end_layer = 4
    last_indices = torch.tensor(
        [[[81, 82], [83, 84], [5, 6], [7, 8]]], dtype=torch.int32
    )
    last_weights = torch.tensor(
        [[[81.0, 82.0], [83.0, 84.0], [0.9, 0.1], [0.8, 0.2]]],
        dtype=torch.float32,
    )

    (
        actual_indices,
        actual_weights,
        actual_out_cache_loc,
        actual_req_pool_indices,
    ) = last._transport_deepep_pp_layers(
        last_indices,
        last_weights,
        out_cache_loc=torch.tensor([81]),
        req_pool_indices=torch.tensor([19]),
        incoming_pp_proxy_tensors=downstream,
        outgoing_pp_proxy_tensors=None,
    )

    expected = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]], dtype=torch.int32)
    expected_weights = torch.tensor(
        [[[0.7, 0.3], [0.6, 0.4], [0.9, 0.1], [0.8, 0.2]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual_indices, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    assert actual_out_cache_loc.tolist() == [17]
    assert actual_req_pool_indices.tolist() == [9]


def test_deepep_pp_capture_returns_to_response_owner(monkeypatch):
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    index_host = SimpleNamespace(buffer=torch.zeros((5, 2, 2), dtype=torch.int32))
    weight_host = SimpleNamespace(buffer=torch.zeros((5, 2, 2), dtype=torch.float32))
    expected_indices = torch.tensor(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int32
    )
    expected_weights = torch.tensor(
        [[[0.7, 0.3], [0.6, 0.4]], [[0.9, 0.1], [0.8, 0.2]]],
        dtype=torch.float32,
    )
    terminal_output = RoutedExpertsCaptureOutput(
        out_cache_loc=torch.tensor([1, 3]),
        req_pool_indices=torch.tensor([9]),
        topk=expected_indices,
        host_cache=SimpleNamespace(buffer=None),
        expert_logits=expected_weights,
        expert_logits_host_cache=SimpleNamespace(buffer=None),
    )
    pp_output = {}
    add_routed_experts_to_pp_output(pp_output, terminal_output)

    owner = object.__new__(RoutedExpertsCapturer)
    owner.num_layers = 2
    owner.topk_size = 2
    owner.capture_topk_weights = True
    owner.host_cache = index_host
    owner.expert_logits_host_cache = weight_host
    monkeypatch.setattr(
        routed_experts_module, "get_global_experts_capturer", lambda: owner
    )

    publish_routed_experts_from_pp_output(pp_output)

    torch.testing.assert_close(
        index_host.buffer[[1, 3]], expected_indices, rtol=0, atol=0
    )
    torch.testing.assert_close(
        weight_host.buffer[[1, 3]], expected_weights, rtol=0, atol=0
    )


def test_deepep_pp_scheduler_round_trip_publishes_owner_cache(monkeypatch):
    from sglang.srt.managers.scheduler_pp_mixin import (
        PPBatchMetadata,
        SchedulerPPMixin,
    )
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    owner = object.__new__(RoutedExpertsCapturer)
    owner.num_layers = 2
    owner.topk_size = 2
    owner.capture_topk_weights = True
    owner.host_cache = SimpleNamespace(buffer=torch.zeros((5, 2, 2), dtype=torch.int32))
    owner.expert_logits_host_cache = SimpleNamespace(
        buffer=torch.zeros((5, 2, 2), dtype=torch.float32)
    )
    monkeypatch.setattr(
        routed_experts_module, "get_global_experts_capturer", lambda: owner
    )

    terminal_output = RoutedExpertsCaptureOutput(
        out_cache_loc=torch.tensor([1, 3]),
        req_pool_indices=torch.tensor([9]),
        topk=torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int32),
        host_cache=SimpleNamespace(buffer=None),
        expert_logits=torch.tensor(
            [[[0.7, 0.3], [0.6, 0.4]], [[0.9, 0.1], [0.8, 0.2]]],
            dtype=torch.float32,
        ),
        expert_logits_host_cache=SimpleNamespace(buffer=None),
    )
    terminal_result = SimpleNamespace(
        next_token_ids=torch.tensor([42]),
        routed_experts_output=terminal_output,
    )
    terminal_batch = SimpleNamespace(return_logprob=False)
    pp_dict = SchedulerPPMixin._pp_prepare_tensor_dict(
        None, terminal_result, terminal_batch
    )

    stashed = {}
    owner_scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True),
        future_map=SimpleNamespace(
            stash=lambda indices, payload: stashed.update(
                indices=indices, payload=payload
            )
        ),
    )
    owner_batch = SimpleNamespace(
        return_logprob=False,
        req_pool_indices=torch.tensor([9]),
        out_cache_loc=torch.tensor([1, 3]),
        input_ids=torch.tensor([10, 11]),
    )
    result = SchedulerPPMixin._pp_prep_batch_result(
        owner_scheduler,
        owner_batch,
        PPBatchMetadata(can_run_cuda_graph=False),
        PPProxyTensors(pp_dict),
    )

    torch.testing.assert_close(
        owner.host_cache.buffer[[1, 3]], terminal_output.topk, rtol=0, atol=0
    )
    torch.testing.assert_close(
        owner.expert_logits_host_cache.buffer[[1, 3]],
        terminal_output.expert_logits,
        rtol=0,
        atol=0,
    )
    assert result.next_token_ids.tolist() == [42]
    assert owner_batch.input_ids is None
    assert stashed["indices"].tolist() == [9]


def test_deepep_pp_scheduler_separates_route_and_token_ownership(monkeypatch):
    from sglang.srt.managers.scheduler_pp_mixin import (
        PPBatchMetadata,
        SchedulerPPMixin,
    )
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    owner = object.__new__(RoutedExpertsCapturer)
    owner.num_layers = 2
    owner.topk_size = 2
    owner.capture_topk_weights = True
    owner.host_cache = SimpleNamespace(buffer=torch.zeros((7, 2, 2), dtype=torch.int32))
    owner.expert_logits_host_cache = SimpleNamespace(
        buffer=torch.zeros((7, 2, 2), dtype=torch.float32)
    )
    monkeypatch.setattr(
        routed_experts_module, "get_global_experts_capturer", lambda: owner
    )

    expected_indices = torch.tensor(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int32
    )
    expected_weights = torch.tensor(
        [[[0.7, 0.3], [0.6, 0.4]], [[0.9, 0.1], [0.8, 0.2]]],
        dtype=torch.float32,
    )
    pp_outputs = PPProxyTensors(
        {
            "next_token_ids": torch.empty(0, dtype=torch.int64),
            "__sglang_pp_routed_expert_ids__": expected_indices,
            "__sglang_pp_routed_expert_weights__": expected_weights,
            "__sglang_pp_routed_expert_out_cache_loc__": torch.tensor([1, 5]),
            "__sglang_pp_routed_expert_req_pool_indices__": torch.tensor([9]),
        }
    )
    stashed = {}

    def stash(indices, payload):
        assert indices.numel() == payload.bonus_tokens.numel()
        stashed.update(indices=indices, payload=payload)

    scheduler = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True),
        future_map=SimpleNamespace(stash=stash),
    )
    # Routed-expert rows are gathered across attention/DP ranks, while sampled
    # tokens remain rank-local.  The PP route payload therefore owns different
    # cache/request rows from this rank's next-token relay.
    local_batch = SimpleNamespace(
        return_logprob=False,
        req_pool_indices=torch.empty(0, dtype=torch.int64),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        input_ids=torch.tensor([10, 11]),
    )
    launch_metadata = PPBatchMetadata(
        can_run_cuda_graph=False,
        req_pool_indices=torch.empty(0, dtype=torch.int64),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
    )

    SchedulerPPMixin._pp_prep_batch_result(
        scheduler,
        local_batch,
        launch_metadata,
        pp_outputs,
    )

    torch.testing.assert_close(
        owner.host_cache.buffer[[1, 5]], expected_indices, rtol=0, atol=0
    )
    torch.testing.assert_close(
        owner.expert_logits_host_cache.buffer[[1, 5]],
        expected_weights,
        rtol=0,
        atol=0,
    )
    assert stashed["indices"].tolist() == []


def test_pp_launch_snapshots_ownership_before_run_batch_mutates_it():
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    original_req_pool_indices = torch.tensor([9])
    original_out_cache_loc = torch.tensor([1, 5])
    batch = SimpleNamespace(
        reqs=[],
        req_pool_indices=original_req_pool_indices.clone(),
        out_cache_loc=original_out_cache_loc.clone(),
    )

    def run_batch(cur_batch, _pp_proxy_tensors):
        cur_batch.req_pool_indices = torch.empty(0, dtype=torch.int64)
        cur_batch.out_cache_loc = torch.empty(0, dtype=torch.int64)
        return SimpleNamespace(can_run_cuda_graph=False)

    event_calls = []
    event = SimpleNamespace(
        record=lambda _stream: event_calls.append("record"),
        synchronize=lambda: event_calls.append("synchronize"),
    )
    scheduler = SimpleNamespace(
        forward_stream_ctx=nullcontext(),
        forward_stream=SimpleNamespace(wait_stream=lambda _stream: None),
        schedule_stream=object(),
        run_batch=run_batch,
        device_module=SimpleNamespace(
            Event=lambda: event, current_stream=lambda: object()
        ),
        pp_group=SimpleNamespace(is_last_rank=False),
        ps=SimpleNamespace(dp_size=2),
        server_args=SimpleNamespace(moe_a2a_backend="deepep"),
    )
    metadata = [None]

    SchedulerPPMixin._pp_launch_batch(
        scheduler,
        0,
        batch,
        SimpleNamespace(),
        metadata,
        [],
    )

    torch.testing.assert_close(
        metadata[0].req_pool_indices, original_req_pool_indices, rtol=0, atol=0
    )
    torch.testing.assert_close(
        metadata[0].out_cache_loc, original_out_cache_loc, rtol=0, atol=0
    )
    assert event_calls == ["record", "synchronize"]


def test_deepep_cp_capture_rejects_stale_row_geometry(monkeypatch):
    from sglang.srt.layers.cp import base as cp_base
    from sglang.srt.state_capturer import routed_experts as routed_experts_module

    class _DeepEPBackend:
        @staticmethod
        def is_deepep():
            return True

    monkeypatch.setattr(
        routed_experts_module, "get_moe_a2a_backend", lambda: _DeepEPBackend()
    )
    monkeypatch.setattr(
        routed_experts_module,
        "get_parallel",
        lambda: SimpleNamespace(attn_cp_size=4, attn_tp_size=4, attn_cp_rank=0),
    )
    monkeypatch.setattr(
        cp_base,
        "get_cp_strategy",
        lambda: SimpleNamespace(name="interleave"),
    )
    forward_batch = SimpleNamespace(
        attn_cp_metadata=SimpleNamespace(
            per_rank_logical_token=[1, 1, 1, 1],
            per_rank_actual_token=[2, 2, 2, 2],
            total_seq_lens=4,
        ),
        out_cache_loc=torch.arange(4),
    )

    with pytest.raises(RuntimeError, match="wrong physical row count"):
        RoutedExpertsCapturer._restore_deepep_cp_logical_rows(
            torch.zeros((7, 2, 2), dtype=torch.float32),
            forward_batch,
            payload_name="routed-expert weights",
        )


@pytest.mark.parametrize("model_family", ["qwen2", "qwen3"])
def test_native_qwen_topk_publishes_routed_experts(monkeypatch, model_family):
    from sglang.srt.layers.moe import deepep_native_exact
    from sglang.srt.layers.moe import topk as topk_module

    hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
    router_logits = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    topk_ids = torch.tensor([[1, 4], [2, 5]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.75, 0.25], [0.625, 0.375]])
    captured = {}

    monkeypatch.setattr(
        deepep_native_exact,
        "native_exact_router_topk",
        lambda *_args, **_kwargs: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(
        topk_module,
        "capture_routed_experts_if_allowed",
        lambda config, layer_id, ids, weights: captured.update(
            config=config,
            layer_id=layer_id,
            ids=ids,
            weights=weights,
        ),
    )

    class Experts:
        moe_runner_config = SimpleNamespace(deepep_native_exact=True)

        def __call__(self, *, hidden_states, topk_output):
            captured["output"] = topk_output
            return hidden_states

    config = SimpleNamespace(renormalize=True)
    block = SimpleNamespace(
        experts=Experts(),
        top_k=2,
        topk=SimpleNamespace(topk_config=config, layer_id=9),
    )
    batch = SimpleNamespace()

    if model_family == "qwen2":
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        block._bi_router_logits = lambda _hidden: router_logits
        block._forward_shared_experts = lambda _hidden: None
        output = Qwen2MoeSparseMoeBlock._forward_deepep(block, hidden_states, batch)
    else:
        from sglang.srt.batch_invariant_ops import batch_invariant_ops
        from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock

        block.gate = SimpleNamespace(weight=torch.empty((6, 4)))
        monkeypatch.setattr(
            batch_invariant_ops,
            "bi_router_gemm",
            lambda *_args: router_logits,
        )
        output = Qwen3MoeSparseMoeBlock.forward_deepep(block, hidden_states, batch)

    assert output is hidden_states
    assert captured["config"] is config
    assert captured["layer_id"] == 9
    assert captured["ids"] is topk_ids
    assert captured["weights"] is topk_weights
    assert captured["output"].router_logits is router_logits


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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
