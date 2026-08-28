"""A captured route slice must not alias the shared device cache.

The overlap path returns a ``TopkCaptureOutput`` from ``on_forward_end`` and the
scheduler D2H-copies it on ``copy_stream``, which by design overlaps the *next*
forward on ``forward_stream``:

    copy_stream.wait_stream(forward_stream)      # copy waits for THIS forward
    with copy_stream_ctx:                        # ...but forward_stream never
        batch_result.copy_to_cpu(...)            #    waits for the copy

``_async_d2h`` calls ``record_stream`` on each source, which keeps the caching
allocator from recycling a block while the copy stream drains.  That covers
every other tensor on that path, because they are freshly allocated per forward.

It does not cover these rows.  ``BaseDeviceCache.buffer`` is allocated once and
``capture()`` writes it **in place** on every forward, so the allocator never
owns or recycles it and ``record_stream`` has nothing to hold.  Handing out a
view lets the next forward overwrite those rows mid-copy, and the request gets a
correctly shaped tensor carrying another forward's expert ids.

These tests express that as an ownership property rather than a timing one: after
``on_forward_end``, overwriting the shared buffer -- which is exactly what the
next forward does -- must not change what ``finalize()`` commits.  An ownership
assertion is stronger than a stream-timing assertion, since it holds regardless
of how the streams happen to interleave, and it runs without a GPU.
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
DEVICE_ROWS = 4

FIRST_FORWARD = 11
SECOND_FORWARD = 77


def _make(cls, **extra):
    """A real capturer without its CUDA-allocating __init__ (BaseHostCache pins
    host memory). The methods under test are the real ones."""
    obj = object.__new__(cls)
    obj.host_cache = SimpleNamespace(
        buffer=torch.zeros((HOST_ROWS, NUM_LAYERS, TOPK), dtype=torch.int32)
    )
    obj.device_cache = SimpleNamespace(
        buffer=torch.full(
            (DEVICE_ROWS, NUM_LAYERS, TOPK), FIRST_FORWARD, dtype=torch.int32
        )
    )
    obj.topk_size = TOPK
    for k, v in extra.items():
        setattr(obj, k, v)
    return obj


def _forward_batch(rows):
    # Unpadded: Defect A's trim is not what is under test here.
    return SimpleNamespace(
        out_cache_loc=torch.arange(2, 2 + rows), _original_num_tokens=None
    )


class TestCaptureSliceOwnership(CustomTestCase):
    def _capture_then_second_forward(self, capturer):
        """Capture, then let the next forward overwrite the shared buffer."""
        out = capturer.on_forward_end(
            forward_batch=_forward_batch(DEVICE_ROWS),
            decode_graph_stride=None,
            no_copy_to_cpu=True,
        )
        self.assertIsNotNone(out, "overlap path must return an output to finalize")
        # This is the next forward: BaseDeviceCache.capture() writes the shared
        # buffer in place, it is not reallocated.
        capturer.device_cache.buffer.fill_(SECOND_FORWARD)
        return out

    def _assert_committed_first_forward(self, capturer, out):
        out.finalize()
        committed = capturer.host_cache.buffer[2 : 2 + DEVICE_ROWS]
        self.assertTrue(
            bool((committed == FIRST_FORWARD).all()),
            "finalize() committed rows the next forward wrote; the capture slice "
            "still aliases the shared device cache",
        )

    def test_base_capturer_slice_survives_the_next_forward(self):
        capturer = _make(BaseTopkCapturer)
        out = self._capture_then_second_forward(capturer)
        self._assert_committed_first_forward(capturer, out)

    def test_base_capturer_slice_does_not_alias_the_shared_buffer(self):
        """Direct statement of the invariant, independent of finalize()."""
        capturer = _make(BaseTopkCapturer)
        out = capturer.on_forward_end(
            forward_batch=_forward_batch(DEVICE_ROWS),
            decode_graph_stride=None,
            no_copy_to_cpu=True,
        )
        shared = capturer.device_cache.buffer
        self.assertNotEqual(
            out.topk.data_ptr(),
            shared.data_ptr(),
            "captured rows must be privately owned, not a view of the cache",
        )
        self.assertFalse(
            out.topk.untyped_storage().data_ptr()
            == shared.untyped_storage().data_ptr(),
            "captured rows must not share storage with the cache",
        )

    def test_routed_experts_capturer_slice_survives_the_next_forward(self):
        capturer = _make(
            RoutedExpertsCapturer,
            capture_topk_weights=False,
            expert_logits_host_cache=None,
        )
        with patch(
            "sglang.srt.state_capturer.routed_experts.is_dp_attention_enabled",
            return_value=False,
        ):
            out = self._capture_then_second_forward(capturer)
        self._assert_committed_first_forward(capturer, out)

    def test_expert_logits_slice_survives_the_next_forward(self):
        """The logits plane aliases its own persistent cache and needs the same
        ownership, or a caller asking for expert_logits gets the same corruption."""
        capturer = _make(
            RoutedExpertsCapturer,
            capture_topk_weights=True,
            expert_logits_host_cache=SimpleNamespace(
                buffer=torch.zeros((HOST_ROWS, NUM_LAYERS, TOPK), dtype=torch.float32)
            ),
        )
        capturer.expert_logits_device_cache = SimpleNamespace(
            buffer=torch.full(
                (DEVICE_ROWS, NUM_LAYERS, TOPK),
                float(FIRST_FORWARD),
                dtype=torch.float32,
            )
        )
        with patch(
            "sglang.srt.state_capturer.routed_experts.is_dp_attention_enabled",
            return_value=False,
        ):
            out = capturer.on_forward_end(
                forward_batch=_forward_batch(DEVICE_ROWS),
                decode_graph_stride=None,
                no_copy_to_cpu=True,
            )
            capturer.expert_logits_device_cache.buffer.fill_(float(SECOND_FORWARD))
            self.assertIsNotNone(out.expert_logits)
            self.assertTrue(
                bool((out.expert_logits == FIRST_FORWARD).all()),
                "expert_logits still aliases its shared device cache",
            )

    def test_non_overlap_path_is_unaffected(self):
        """The synchronous path copies before returning, so it needs no snapshot
        -- and must keep committing the forward's own rows."""
        capturer = _make(BaseTopkCapturer)
        out = capturer.on_forward_end(
            forward_batch=_forward_batch(DEVICE_ROWS),
            decode_graph_stride=None,
            no_copy_to_cpu=False,
        )
        self.assertIsNone(out, "non-overlap path commits inline and returns None")
        capturer.device_cache.buffer.fill_(SECOND_FORWARD)
        committed = capturer.host_cache.buffer[2 : 2 + DEVICE_ROWS]
        self.assertTrue(bool((committed == FIRST_FORWARD).all()))


if __name__ == "__main__":
    unittest.main()
