from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _CaptureSlot:
    def __init__(self, tensor: torch.Tensor, *, token_axis: bool = False):
        self.tensor = tensor
        self.token_axis = token_axis

    def slice_for(self, bs: int, num_tokens: int) -> torch.Tensor:
        size = num_tokens if self.token_axis else bs
        if self.tensor.ndim == 2 and self.token_axis:
            return self.tensor[:, :size]
        return self.tensor[:size]


class _CaptureRegistry:
    def __init__(self):
        self.slots = {
            "input_ids": _CaptureSlot(torch.arange(4), token_axis=True),
            "req_pool_indices": _CaptureSlot(torch.arange(4)),
            "seq_lens": _CaptureSlot(torch.ones(4, dtype=torch.int32)),
            "seq_lens_cpu": _CaptureSlot(torch.ones(4, dtype=torch.int32)),
            "out_cache_loc": _CaptureSlot(torch.arange(4), token_axis=True),
            "positions": _CaptureSlot(torch.arange(4), token_axis=True),
            "mrope_positions": _CaptureSlot(
                torch.zeros((3, 4), dtype=torch.int64), token_axis=True
            ),
        }

    def has_slot(self, name: str) -> bool:
        return name in self.slots

    def get_slot(self, name: str) -> _CaptureSlot:
        return self.slots[name]


def _make_dp2_decode_capture_runner() -> DecodeCudaGraphRunner:
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.buffers = SimpleNamespace(
        next_token_logits_buffer=torch.empty((4, 8)),
        rids_int=None,
        bootstrap_room_ids_int=None,
        num_token_non_padded=torch.zeros(1, dtype=torch.int32),
        pp_proxy_tensors={"hidden_states": torch.empty((4, 2))},
        global_num_tokens_gpu=torch.zeros(2, dtype=torch.int32),
        global_num_tokens_for_logprob_gpu=torch.zeros(2, dtype=torch.int32),
        ngram_embedding_info=None,
    )
    runner.buffer_registry = _CaptureRegistry()
    runner.captured_req_width = 1
    runner.require_gathered_buffer = True
    runner.enable_prefill_cp = False
    runner.pp_size = 2
    runner.require_mlp_tp_gather = True
    runner.require_attn_tp_gather = False
    runner.dp_size = 2
    runner.capture_hidden_mode = CaptureHiddenMode.NULL
    runner.capture_forward_mode = ForwardMode.DECODE
    runner.enable_pdmux = False
    runner.attn_backend = object()
    runner.get_spec_info = lambda _num_tokens: None
    runner.model_runner = SimpleNamespace(
        server_args=SimpleNamespace(enable_lora=False),
        spec_algorithm=None,
        hisparse_coordinator=None,
    )
    return runner


def test_dp2_decode_capture_carries_exact_per_dp_pruned_row_segments() -> None:
    """A bs4 graph bucket is two four-row DP segments, never one eight-row block."""

    runner = _make_dp2_decode_capture_runner()
    with patch(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "enable_num_token_non_padded",
        return_value=False,
    ):
        forward_batch, _, _ = runner.capture_prepare(4)

    assert forward_batch.global_num_tokens_cpu == [4, 4]
    assert forward_batch.global_num_tokens_for_logprob_cpu == [4, 4]
    assert forward_batch.global_num_tokens_for_logprob_gpu.tolist() == [4, 4]

    # Exercise the same metadata conversion and exact pruning guard used by the
    # terminal DSV4 head during capture on PP1.
    forward_batch.dsv4_exact_logits_rows_reconstructed = True
    forward_batch.dsv4_exact_logits_owner_rows = 4
    forward_batch.dsv4_exact_logits_dp_rank = 1
    metadata = LogitsMetadata.from_forward_batch(forward_batch)
    hidden = torch.arange(4, dtype=torch.float32).unsqueeze(1)

    pruned = LogitsProcessor._get_pruned_states(None, hidden, None, None, metadata)[0]

    assert torch.equal(pruned, hidden)


def test_exact_dsv4_eager_logits_still_reject_missing_dp_segments() -> None:
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        global_num_tokens_for_logprob_cpu=None,
        dsv4_exact_logits_rows_reconstructed=True,
        dsv4_exact_logits_owner_rows=4,
        dsv4_exact_logits_dp_rank=1,
    )

    with pytest.raises(RuntimeError, match="per-DP pruned-row ownership metadata"):
        LogitsProcessor._get_pruned_states(
            None, torch.zeros((4, 1)), None, None, metadata
        )


def test_ragged_prefill_next_token_pruning_uses_reconstructed_owner_rows() -> None:
    """One last row per request is selected after CP reconstruction."""

    hidden = torch.arange(15, dtype=torch.float32).unsqueeze(1)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([4, 5, 6], dtype=torch.int64),
        extend_seq_lens_cpu=[4, 5, 6],
        extend_return_logprob=False,
        global_num_tokens_for_logprob_cpu=[3],
        dsv4_exact_logits_rows_reconstructed=True,
        dsv4_exact_logits_owner_rows=15,
        dsv4_exact_logits_dp_rank=0,
    )

    pruned = LogitsProcessor._get_pruned_states(None, hidden, None, None, metadata)[0]

    assert pruned.flatten().tolist() == [3.0, 8.0, 14.0]


def test_ragged_prefill_input_rescoring_uses_reconstructed_owner_rows() -> None:
    """Input-token rows and each request's sampling row retain logical order."""

    hidden = torch.arange(15, dtype=torch.float32).unsqueeze(1)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([4, 5, 6], dtype=torch.int64),
        extend_seq_lens_cpu=[4, 5, 6],
        extend_logprob_start_lens_cpu=[0, 5, 3],
        extend_return_logprob=True,
        global_num_tokens_for_logprob_cpu=[8],
        dsv4_exact_logits_rows_reconstructed=True,
        dsv4_exact_logits_owner_rows=15,
        dsv4_exact_logits_dp_rank=0,
    )

    (
        pruned,
        _,
        _,
        sample_indices,
        input_logprob_indices,
        token_to_seq_idx,
    ) = LogitsProcessor._get_pruned_states(None, hidden, None, None, metadata)

    assert pruned.flatten().tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        8.0,
        12.0,
        13.0,
        14.0,
    ]
    assert sample_indices.tolist() == [3, 4, 7]
    assert input_logprob_indices.tolist() == [0, 1, 2, 3, 5, 6, 7]
    assert token_to_seq_idx == [0, 0, 0, 0, 1, 2, 2, 2]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
