import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
