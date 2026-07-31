"""The post-MoE reduce-scatter bypass must honor the reduction-order pin."""

import unittest
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
    """Build only the state read by ``should_use_reduce_scatter``."""
    comm = object.__new__(LayerCommunicator)
    comm.allow_reduce_scatter = True
    comm.is_last_layer = False
    comm._communicate_summable_tensor_pair_fn = (
        CommunicateSummableTensorPairFn._scatter_hidden_states
    )
    comm._reduction_order_pinned = pinned
    return comm


class TestReduceScatterReductionOrderPin(unittest.TestCase):
    def test_max_len_uses_reduce_scatter_when_order_is_not_pinned(self):
        comm = _make_communicator(pinned=False)
        batch = _FakeForwardBatch(DpPaddingMode.MAX_LEN)
        self.assertTrue(comm.should_use_reduce_scatter(batch))

    def test_max_len_keeps_the_pinned_all_reduce(self):
        comm = _make_communicator(pinned=True)
        batch = _FakeForwardBatch(DpPaddingMode.MAX_LEN)
        self.assertFalse(comm.should_use_reduce_scatter(batch))

    def test_sum_len_never_uses_reduce_scatter(self):
        batch = _FakeForwardBatch(DpPaddingMode.SUM_LEN)
        for pinned in (False, True):
            comm = _make_communicator(pinned=pinned)
            with patch(
                "sglang.srt.layers.communicator.nsa_use_prefill_cp",
                return_value=False,
            ), patch("sglang.srt.layers.communicator.get_attn_tp_context") as attn_ctx:
                attn_ctx.return_value.input_scattered = False
                self.assertFalse(comm.should_use_reduce_scatter(batch))

    def test_allow_reduce_scatter_false_short_circuits(self):
        comm = _make_communicator(pinned=False)
        comm.allow_reduce_scatter = False
        batch = _FakeForwardBatch(DpPaddingMode.MAX_LEN)
        self.assertFalse(comm.should_use_reduce_scatter(batch))

    def test_nsa_prefill_cp_refuses_the_unpinned_collective(self):
        comm = _make_communicator(pinned=True)
        batch = _FakeForwardBatch(DpPaddingMode.SUM_LEN)
        with patch(
            "sglang.srt.layers.communicator.nsa_use_prefill_cp",
            return_value=True,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                comm.should_use_reduce_scatter(batch)
        self.assertIn("order-pinned", str(ctx.exception))

    def test_scattered_attn_tp_refuses_the_unpinned_collective(self):
        comm = _make_communicator(pinned=True)
        batch = _FakeForwardBatch(DpPaddingMode.SUM_LEN)
        with patch(
            "sglang.srt.layers.communicator.nsa_use_prefill_cp",
            return_value=False,
        ), patch("sglang.srt.layers.communicator.get_attn_tp_context") as attn_ctx:
            attn_ctx.return_value.input_scattered = True
            with self.assertRaises(RuntimeError) as ctx:
                comm.should_use_reduce_scatter(batch)
        self.assertIn("order-pinned", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
