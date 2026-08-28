"""MLP-sync padding must never reach the host cache.

``_pad_inputs_to_size`` appends dummy rows after the real ones and pads
``out_cache_loc`` with **zeros** (``_pad_tensor_to_size`` defaults to
``value=0``), so every padding row indexes KV slot 0.  The capturers write with
``host_cache.buffer[out_cache_loc] = topk``; writing the padded tensor stamps
padding-derived routes over whichever request currently owns slot 0.

That is silent corruption, not a crash: the victim still gets a correctly shaped
tensor of plausible expert ids, just not the ones its own forward computed.  It
is invisible to any single-response check -- ``concat(input, output) == legacy``
still holds, because both halves read the same corrupted row.

``post_forward_mlp_sync_batch`` already trims positions/seq_lens with
``_original_num_tokens`` but leaves ``out_cache_loc`` padded, so the capturers
have to do it themselves.  Both the overlap path (``TopkCaptureOutput.finalize``)
and the non-overlap path must be covered, since only one of them runs per
forward.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.state_capturer.base import BaseTopkCapturer
from sglang.srt.state_capturer.routed_experts import RoutedExpertsCapturer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 2
TOPK = 4
HOST_ROWS = 16

# Slot 0 is owned by an unrelated request; these are the rows it must keep.
VICTIM_ROW = 7


def _buffers(device_rows):
    host = SimpleNamespace(
        buffer=torch.zeros((HOST_ROWS, NUM_LAYERS, TOPK), dtype=torch.int32)
    )
    device = SimpleNamespace(
        buffer=torch.arange(device_rows * NUM_LAYERS * TOPK, dtype=torch.int32).reshape(
            device_rows, NUM_LAYERS, TOPK
        )
        + 100  # keep captured values distinct from the victim's
    )
    return host, device


def _forward_batch(out_cache_loc, original_num_tokens):
    return SimpleNamespace(
        out_cache_loc=out_cache_loc,
        _original_num_tokens=original_num_tokens,
    )


def _make(cls, host, device, **extra):
    """A real capturer instance without its CUDA-allocating __init__.

    BaseHostCache pins host memory, which needs a driver, so the buffers are
    injected instead.  The methods under test are the real ones.
    """
    obj = object.__new__(cls)
    obj.host_cache = host
    obj.device_cache = device
    obj.topk_size = TOPK
    for k, v in extra.items():
        setattr(obj, k, v)
    return obj


class TestNumRealRows(CustomTestCase):
    def test_none_means_no_padding_was_applied(self):
        """``None`` is "never padded", not "unknown".

        ``_original_num_tokens`` is only set by ``_pad_inputs_to_size``.  Paths
        that never pad -- no DP/MLP-sync, and the decode cuda graph, which
        returns before ``_prepare_eager_forward_batch`` -- leave it None and
        must write every row they have.
        """
        fb = _forward_batch(torch.tensor([3, 4, 5]), None)
        self.assertEqual(BaseTopkCapturer._num_real_rows(fb), 3)

    def test_padded_batch_reports_only_the_real_rows(self):
        fb = _forward_batch(torch.tensor([3, 4, 5, 0, 0]), 3)
        self.assertEqual(BaseTopkCapturer._num_real_rows(fb), 3)

    def test_clamped_to_the_tensor(self):
        """Never index past the tensor if the two ever disagree."""
        fb = _forward_batch(torch.tensor([3, 4]), 9)
        self.assertEqual(BaseTopkCapturer._num_real_rows(fb), 2)


class TestPaddingNeverReachesHostCache(CustomTestCase):
    """Both write sites, with a zero-padded tail that aims at slot 0."""

    def _run(self, capturer, fb, no_copy_to_cpu):
        out = capturer.on_forward_end(
            forward_batch=fb, decode_graph_stride=None, no_copy_to_cpu=no_copy_to_cpu
        )
        if no_copy_to_cpu:
            self.assertIsNotNone(out, "overlap path must return an output to finalize")
            out.finalize()
        return out

    def _check(self, capturer_factory, no_copy_to_cpu):
        host, device = _buffers(device_rows=5)
        capturer = capturer_factory(host, device)

        # 3 real tokens at slots 3,4,5; 2 MLP-sync padding rows, zero-filled.
        out_cache_loc = torch.tensor([3, 4, 5, 0, 0])
        fb = _forward_batch(out_cache_loc, original_num_tokens=3)

        sentinel = torch.full((NUM_LAYERS, TOPK), VICTIM_ROW, dtype=torch.int32)
        host.buffer[0] = sentinel

        self._run(capturer, fb, no_copy_to_cpu)

        torch.testing.assert_close(
            host.buffer[0],
            sentinel,
            msg="slot 0 belongs to another request; padding must not overwrite it",
        )
        # The real rows still land, so the trim is not simply dropping the write.
        for row, slot in enumerate((3, 4, 5)):
            torch.testing.assert_close(
                host.buffer[slot],
                device.buffer[row],
                msg=f"real token row {row} must reach slot {slot}",
            )

    def _base(self, host, device):
        return _make(BaseTopkCapturer, host, device)

    def _routed(self, host, device):
        # expert_logits_* are unused with capture_topk_weights False, but the
        # overlap path still names expert_logits_host_cache when it builds its
        # output, so the attribute has to exist.
        return _make(
            RoutedExpertsCapturer,
            host,
            device,
            capture_topk_weights=False,
            expert_logits_host_cache=None,
        )

    def test_base_capturer_non_overlap_path(self):
        self._check(self._base, no_copy_to_cpu=False)

    def test_base_capturer_overlap_path(self):
        """finalize() must inherit the trim rather than carry its own guard."""
        self._check(self._base, no_copy_to_cpu=True)

    def test_routed_experts_capturer_non_overlap_path(self):
        with patch(
            "sglang.srt.state_capturer.routed_experts.is_dp_attention_enabled",
            return_value=False,
        ):
            self._check(self._routed, no_copy_to_cpu=False)

    def test_routed_experts_capturer_overlap_path(self):
        with patch(
            "sglang.srt.state_capturer.routed_experts.is_dp_attention_enabled",
            return_value=False,
        ):
            self._check(self._routed, no_copy_to_cpu=True)

    def test_unpadded_batch_still_writes_every_row(self):
        """The trim must be a no-op when nothing was padded.

        Guards against "fixing" the corruption by dropping the last row on paths
        that never padded -- the decode cuda graph among them.
        """
        host, device = _buffers(device_rows=3)
        capturer = _make(BaseTopkCapturer, host, device)
        fb = _forward_batch(torch.tensor([3, 4, 5]), None)
        capturer.on_forward_end(
            forward_batch=fb, decode_graph_stride=None, no_copy_to_cpu=False
        )
        for row, slot in enumerate((3, 4, 5)):
            torch.testing.assert_close(host.buffer[slot], device.buffer[row])


if __name__ == "__main__":
    unittest.main()
