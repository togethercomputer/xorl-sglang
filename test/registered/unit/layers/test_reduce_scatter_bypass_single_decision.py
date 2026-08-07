"""The post-MoE reduce-scatter bypass is one decision shared by both halves."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.communicator import (
    CommunicateSummableTensorPairFn,
    LayerCommunicator,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _FakeForwardBatch:
    def __init__(self, dp_padding_mode: DpPaddingMode):
        self.dp_padding_mode = dp_padding_mode


def _make_communicator(*, pinned: bool) -> LayerCommunicator:
    """Build only the state read by the block and scatter halves."""
    comm = object.__new__(LayerCommunicator)
    comm.allow_reduce_scatter = True
    comm.is_last_layer = False
    comm._communicate_summable_tensor_pair_fn = (
        CommunicateSummableTensorPairFn._scatter_hidden_states
    )
    comm._reduction_order_pinned = pinned
    comm._context = object()
    return comm


def _no_other_bypass_reasons():
    return (
        patch("sglang.srt.layers.communicator.dsa_use_prefill_cp", return_value=False),
        patch("sglang.srt.layers.communicator.mla_use_prefill_cp", return_value=False),
        patch("sglang.srt.layers.communicator.get_attn_tp_context"),
        patch(
            "sglang.srt.layers.communicator.get_parallel",
            return_value=SimpleNamespace(tp_size=8, attn_dp_size=8),
        ),
        patch("sglang.srt.layers.communicator.get_tp_group", return_value=object()),
    )


class TestBypassIsASingleDecision(unittest.TestCase):
    def _halves(self, *, pinned, mode):
        batch = _FakeForwardBatch(mode)
        dsa, mla, attn, parallel, tp_group = _no_other_bypass_reasons()
        with (
            dsa,
            mla,
            attn as attn_ctx,
            parallel,
            tp_group,
            patch("sglang.srt.layers.communicator.get_local_dp_buffer"),
            patch(
                "sglang.srt.layers.communicator.dp_reduce_scatter_tensor"
            ) as reduce_scatter,
            patch("sglang.srt.layers.communicator.dp_scatter") as scatter,
        ):
            attn_ctx.return_value.input_scattered = False
            comm = _make_communicator(pinned=pinned)
            block_half = comm.should_use_reduce_scatter(batch)
            comm.postprocess_layer(object(), object(), batch)

        self.assertEqual(
            reduce_scatter.call_count + scatter.call_count,
            1,
            "the scatter half must run exactly one collective",
        )
        return block_half, reduce_scatter.call_count == 1

    def test_both_halves_take_the_same_branch(self):
        for pinned in (False, True):
            for mode in (DpPaddingMode.MAX_LEN, DpPaddingMode.SUM_LEN):
                with self.subTest(pinned=pinned, dp_padding_mode=mode):
                    block_half, scatter_reduced = self._halves(pinned=pinned, mode=mode)
                    self.assertEqual(
                        scatter_reduced,
                        block_half,
                        "the block and scatter halves selected different sums",
                    )

    def test_pinned_max_len_slices_instead_of_reducing_again(self):
        batch = _FakeForwardBatch(DpPaddingMode.MAX_LEN)
        dsa, mla, attn, parallel, tp_group = _no_other_bypass_reasons()
        with (
            dsa,
            mla,
            attn as attn_ctx,
            parallel,
            tp_group,
            patch("sglang.srt.layers.communicator.get_local_dp_buffer"),
            patch(
                "sglang.srt.layers.communicator.dp_reduce_scatter_tensor"
            ) as reduce_scatter,
            patch("sglang.srt.layers.communicator.dp_scatter") as scatter,
        ):
            attn_ctx.return_value.input_scattered = False
            comm = _make_communicator(pinned=True)
            self.assertFalse(comm.should_use_reduce_scatter(batch))
            comm.postprocess_layer(object(), object(), batch)

        self.assertEqual(reduce_scatter.call_count, 0)
        self.assertEqual(scatter.call_count, 1)

    def test_unpinned_max_len_still_bypasses(self):
        batch = _FakeForwardBatch(DpPaddingMode.MAX_LEN)
        dsa, mla, attn, parallel, tp_group = _no_other_bypass_reasons()
        with (
            dsa,
            mla,
            attn as attn_ctx,
            parallel,
            tp_group,
            patch("sglang.srt.layers.communicator.get_local_dp_buffer"),
            patch(
                "sglang.srt.layers.communicator.dp_reduce_scatter_tensor"
            ) as reduce_scatter,
            patch("sglang.srt.layers.communicator.dp_scatter") as scatter,
        ):
            attn_ctx.return_value.input_scattered = False
            comm = _make_communicator(pinned=False)
            self.assertTrue(comm.should_use_reduce_scatter(batch))
            comm.postprocess_layer(object(), object(), batch)

        self.assertEqual(reduce_scatter.call_count, 1)
        self.assertEqual(scatter.call_count, 0)


if __name__ == "__main__":
    unittest.main()
