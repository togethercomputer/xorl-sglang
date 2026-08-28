"""Unit tests for the DP-local slice the state capturers read.

A capturer has to find its own DP rank's rows inside a buffer that holds every
rank's rows concatenated.  Two different layouts reach this code, and the only
thing that distinguishes them is *which cuda graph ran*:

- The decode cuda graph replays into its own static buffer and returns from
  ``_forward_raw`` before ``_prepare_eager_forward_batch``, so
  ``global_num_tokens_cpu`` still holds raw per-rank counts that do not
  describe that buffer.  The offset must come from the graph's padded batch
  size.
- Everything else -- eager, and the *prefill* cuda graph -- runs
  ``prepare_mlp_sync_batch`` first, which rewrites ``global_num_tokens_cpu``
  to post-padding counts.  A running sum over those is then correct.

Conflating the two is not a shape mismatch that a length check would catch; it
silently hands a consumer another rank's expert routes.  The prefill cuda graph
also sets ``can_run_graph``, so keying on that flag reads the decode graph's
batch size for a forward the decode graph never ran -- ``None`` before its first
replay, and a stale wrong integer afterwards.
"""

import unittest
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

from sglang.srt.layers.dp_attention import get_dp_local_slice_cpu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _slice_at_rank(
    global_num_tokens: List[int],
    dp_rank: int,
    decode_graph_stride: Optional[int],
):
    """Call the seam as one DP rank sees it."""
    forward_batch = SimpleNamespace(global_num_tokens_cpu=list(global_num_tokens))
    with patch(
        "sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=dp_rank
    ):
        return get_dp_local_slice_cpu(forward_batch, decode_graph_stride)


class TestDpLocalSliceCpu(CustomTestCase):
    def test_prefill_graph_path_does_not_require_a_stride(self):
        """No stride must be needed when the decode graph did not run.

        This is the tpep regression: the prefill cuda graph sets
        ``can_run_graph``, so the old wiring passed that flag together with
        ``decode_cuda_graph_runner.bs``, which is ``None`` until the decode
        graph first replays -- ``dp_rank * None`` raised TypeError on every
        rank under --tp 4 --ep-size 4 --dp 2 --enable-dp-attention.
        """
        for dp_rank in range(4):
            with self.subTest(dp_rank=dp_rank):
                start, length = _slice_at_rank(
                    [8, 8, 8, 8], dp_rank, decode_graph_stride=None
                )
                self.assertEqual(start, dp_rank * 8)
                self.assertEqual(length, 8)

    def test_max_len_padding_makes_the_running_sum_equal_the_uniform_stride(self):
        """Under MAX_LEN the sum branch *is* the uniform-stride formula.

        ``prepare_mlp_sync_batch`` rewrites global_num_tokens to
        ``[max(...)] * dp_size`` under MAX_LEN.  Every entry being equal means
        ``sum(counts[:rank]) == rank * max``, so the prefill-graph path needs no
        stride parameter to reproduce the padded layout -- it already computes
        it.  If this equality ever broke, passing no stride would start
        returning a different offset than the buffer actually uses.
        """
        for dp_size in (2, 4, 8):
            padded = 12
            counts = [padded] * dp_size
            for dp_rank in range(dp_size):
                with self.subTest(dp_size=dp_size, dp_rank=dp_rank):
                    start, length = _slice_at_rank(
                        counts, dp_rank, decode_graph_stride=None
                    )
                    self.assertEqual(start, dp_rank * padded)
                    self.assertEqual(length, padded)

    def test_sum_len_padding_rejects_a_uniform_stride(self):
        """Under SUM_LEN a uniform stride is genuinely wrong, not just different.

        Ranks are packed at their true offsets, so only the running sum lands on
        rank-owned rows.  This is why the fix keys on *which graph ran* rather
        than unconditionally supplying some batch size: the prefill cuda graph
        keeps SUM_LEN whenever the batch does not fit a captured breakable
        graph, and a stride would silently mis-slice there.
        """
        counts = [3, 9, 4, 10]  # uneven -> SUM_LEN
        expected_starts = [0, 3, 12, 16]
        for dp_rank, expected_start in enumerate(expected_starts):
            with self.subTest(dp_rank=dp_rank):
                start, length = _slice_at_rank(
                    counts, dp_rank, decode_graph_stride=None
                )
                self.assertEqual(start, expected_start)
                self.assertEqual(length, counts[dp_rank])

        # A uniform stride would disagree for every rank past the first.
        stride = max(counts)
        for dp_rank in range(1, len(counts)):
            self.assertNotEqual(
                expected_starts[dp_rank],
                dp_rank * stride,
                "a uniform stride must not be mistaken for correct under SUM_LEN",
            )

    def test_decode_graph_uses_the_graph_stride_not_the_raw_counts(self):
        """When the decode graph replayed, its padded batch size wins.

        ``global_num_tokens_cpu`` holds raw counts on that path, so the running
        sum would point into the wrong rows of the graph's static buffer.
        """
        raw_counts = [1, 1, 1, 1]  # raw per-rank decode counts
        graph_stride = 16  # padded capture bucket
        for dp_rank in range(4):
            with self.subTest(dp_rank=dp_rank):
                start, length = _slice_at_rank(
                    raw_counts, dp_rank, decode_graph_stride=graph_stride
                )
                self.assertEqual(start, dp_rank * graph_stride)
                self.assertEqual(length, raw_counts[dp_rank])
                if dp_rank:
                    self.assertNotEqual(
                        start,
                        sum(raw_counts[:dp_rank]),
                        "raw counts must not drive the decode-graph offset",
                    )

    def test_slices_tile_the_buffer_without_overlap(self):
        """Whatever the layout, rank slices must not gap or overlap.

        Two ranks sharing a row means one of them reports the other's experts.
        """
        cases = [
            ([7, 7, 7, 7], None),  # MAX_LEN, prefill graph / eager
            ([3, 9, 4, 10], None),  # SUM_LEN, prefill graph / eager
            ([1, 1, 1, 1], 16),  # decode graph static buffer
        ]
        for counts, stride in cases:
            with self.subTest(counts=counts, stride=stride):
                spans = [
                    _slice_at_rank(counts, r, decode_graph_stride=stride)
                    for r in range(len(counts))
                ]
                previous_end = 0
                for start, length in spans:
                    self.assertGreaterEqual(
                        start, previous_end, "rank slices must not overlap"
                    )
                    previous_end = start + length


class TestModelRunnerOutputWiring(CustomTestCase):
    def test_stride_defaults_to_absent(self):
        """``can_run_graph`` alone must never imply a decode-graph stride.

        The prefill cuda graph builds its output with ``can_run_graph=True`` and
        nothing else, so the default has to be ``None`` -- that is exactly what
        routes it to the running-sum branch.  Giving this field a non-None
        default, or re-deriving it from ``can_run_graph``, reintroduces the bug.
        """
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput

        output = ModelRunnerOutput(logits_output=None, can_run_graph=True)
        self.assertIsNone(output.decode_graph_stride)


if __name__ == "__main__":
    unittest.main()
