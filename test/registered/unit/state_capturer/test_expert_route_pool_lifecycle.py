"""Unit tests for the token-slot lifetime of the expert-route sidecar.

The sidecar is not a per-request history: it is a host buffer indexed by the
same KV token slots the KV cache uses, and a request reaches its rows through
``req_to_token``.  Every property below follows from that indexing, and each one
would break silently -- returning rows that reshape fine but describe the wrong
tokens -- if the indexing were replaced by request-owned storage.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.state_capturer.base import BaseTopkCapturer
from sglang.srt.state_capturer.expert_route_selection import select_expert_routes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_TOKENS = 128
NUM_LAYERS = 8
TOP_K = 2
MAX_CONTEXT_LEN = 32


class ExpertRouteSidecarFixture:
    """A capturer + req_to_token pool wired the way the scheduler wires them."""

    def __init__(self):
        self.capturer = BaseTopkCapturer(
            num_tokens=NUM_TOKENS,
            max_batch_size=16,
            num_layers=NUM_LAYERS,
            topk_size=TOP_K,
            device="cpu",
            name="test_expert_routes",
        )
        self.pool = ReqToTokenPool(
            size=8,
            max_context_len=MAX_CONTEXT_LEN,
            device="cpu",
            enable_memory_saver=False,
        )

    def map_request(self, req_pool_idx, slots):
        """Point a request's token positions at the given KV slots."""
        self.pool.req_to_token[req_pool_idx][: len(slots)] = torch.tensor(
            slots, dtype=torch.int32
        )

    def run_forward(self, slots, tag):
        """Simulate a forward committing route rows into the given KV slots.

        Mirrors ``TopkCaptureOutput.finalize``: rows land at ``out_cache_loc``,
        i.e. the KV slots the forward just wrote.
        """
        rows = torch.stack(
            [
                torch.full((NUM_LAYERS, TOP_K), tag + j, dtype=torch.int32)
                for j in range(len(slots))
            ]
        )
        self.capturer.host_cache.buffer[torch.tensor(slots)] = rows

    def select(self, req_pool_idx, prompt_len, seqlen, want_input, want_output):
        return select_expert_routes(
            capturer=self.capturer,
            req_to_token_pool=self.pool,
            req_pool_idx=req_pool_idx,
            prompt_len=prompt_len,
            seqlen=seqlen,
            want_input=want_input,
            want_output=want_output,
        )

    def legacy(self, req_pool_idx, seqlen, start_len=0):
        return self.capturer.get_topk(
            req_pool_idx=req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.pool,
            start_len=start_len,
        )


class TestExpertRouteSidecarLifecycle(CustomTestCase):
    def setUp(self):
        self.fx = ExpertRouteSidecarFixture()
        # One 5-token prompt + 3 generated tokens occupying slots 10..17.
        self.prompt_len = 5
        self.seqlen = 8
        self.slots = list(range(10, 18))
        self.fx.map_request(1, self.slots)
        self.fx.run_forward(self.slots, tag=100)

    def test_both_partitions_concatenate_to_the_legacy_tensor(self):
        """Exact row equality, not just matching shapes: this is the property
        that lets an existing `return_routed_experts` consumer migrate to the
        partitioned flags without changing a single value it trains on."""
        legacy = self.fx.legacy(1, self.seqlen)
        result = self.fx.select(1, self.prompt_len, self.seqlen, True, True)
        self.assertTrue(
            torch.equal(
                torch.cat([result.input_rows, result.output_rows], dim=0), legacy
            )
        )

    def test_single_partition_matches_its_slice_of_the_full_history(self):
        """A partial gather must return exactly the rows a full gather would
        have yielded for that range -- the optimization that skips copying the
        whole history must not change a value."""
        legacy = self.fx.legacy(1, self.seqlen)
        boundary = self.prompt_len - 1

        only_in = self.fx.select(1, self.prompt_len, self.seqlen, True, False)
        self.assertIsNone(only_in.output_rows)
        self.assertIsNone(only_in.schema.output_num_rows)
        self.assertTrue(torch.equal(only_in.input_rows, legacy[:boundary]))

        only_out = self.fx.select(1, self.prompt_len, self.seqlen, False, True)
        self.assertIsNone(only_out.input_rows)
        self.assertIsNone(only_out.schema.input_num_rows)
        self.assertTrue(torch.equal(only_out.output_rows, legacy[boundary:]))

    def test_prefix_cache_hit_serves_the_original_producer_rows(self):
        """A request whose prompt hits the radix cache never re-runs those
        forwards. Its rows must still resolve, through the shared slots, to what
        the original producing forward wrote -- this is why capture is gated on
        server capability rather than on the request flags."""
        expected = self.fx.select(1, self.prompt_len, self.seqlen, True, False)

        reuse_slots = self.slots[: self.prompt_len] + [30, 31, 32]
        self.fx.map_request(2, reuse_slots)
        # Only the newly computed suffix runs a forward.
        self.fx.run_forward(reuse_slots[self.prompt_len :], tag=200)

        hit = self.fx.select(2, self.prompt_len, self.seqlen, True, False)
        self.assertTrue(torch.equal(hit.input_rows, expected.input_rows))

    def test_partial_prefix_hit_splices_cached_and_recomputed_rows(self):
        """A partial hit must read cached rows for the shared head and freshly
        computed rows for the divergent tail, with the seam at the right row."""
        shared = 3
        mixed_slots = self.slots[:shared] + [40, 41, 42, 43, 44]
        self.fx.map_request(3, mixed_slots)
        self.fx.run_forward(mixed_slots[shared:], tag=500)

        result = self.fx.select(3, self.prompt_len, self.seqlen, True, True)
        rows = torch.cat([result.input_rows, result.output_rows], dim=0)
        # Head rows still carry the original producer's tag ...
        self.assertTrue((rows[:shared] < 500).all())
        # ... and the tail carries the recomputed one.
        self.assertTrue((rows[shared:] >= 500).all())

    def test_reused_slot_never_serves_the_previous_occupants_rows(self):
        """Freeing a request and handing its slots to another must not leak the
        old routes. Nothing scrubs the sidecar on free -- correctness rests on
        the new occupant's forward overwriting the slot before anyone reads it,
        so a change that skipped that write would be caught here."""
        before = self.fx.select(1, self.prompt_len, self.seqlen, True, True)

        # Request 1 finishes, its slots are freed and reallocated to request 4,
        # whose forward writes the same slots.
        self.fx.map_request(4, self.slots)
        self.fx.run_forward(self.slots, tag=900)

        after = self.fx.select(4, self.prompt_len, self.seqlen, True, True)
        self.assertFalse(torch.equal(after.input_rows, before.input_rows))
        self.assertTrue((after.input_rows >= 900).all())
        self.assertTrue((after.output_rows >= 900).all())

    def test_empty_partition_does_not_alias_the_shared_host_buffer(self):
        """An empty gather must build a standalone tensor. Basic slicing would
        return a view onto the whole pinned host cache, which the IPC pickler
        would then serialize in full -- turning a zero-row response into a
        multi-gigabyte one."""
        self.fx.map_request(5, [50, 51])
        self.fx.run_forward([50, 51], tag=300)

        result = self.fx.select(5, 1, 2, True, True)
        self.assertEqual(result.input_rows.shape, (0, NUM_LAYERS, TOP_K))
        self.assertNotEqual(
            result.input_rows.untyped_storage().data_ptr(),
            self.fx.capturer.host_cache.buffer.untyped_storage().data_ptr(),
        )

    def test_gathered_rows_do_not_alias_the_shared_host_buffer(self):
        """A non-empty gather must copy too: a view would let a later forward
        writing the same slot mutate a response already handed to the streamer."""
        result = self.fx.select(1, self.prompt_len, self.seqlen, True, False)
        self.assertNotEqual(
            result.input_rows.untyped_storage().data_ptr(),
            self.fx.capturer.host_cache.buffer.untyped_storage().data_ptr(),
        )
        original = result.input_rows.clone()
        self.fx.run_forward(self.slots, tag=700)
        self.assertTrue(torch.equal(result.input_rows, original))

    def test_selection_requires_at_least_one_partition(self):
        """Negative-branch contract: a default request must never reach the
        gather path at all, so being called with both flags false is a bug in
        the caller and must not silently return an empty result."""
        with self.assertRaises(ValueError):
            self.fx.select(1, self.prompt_len, self.seqlen, False, False)

    def test_get_rows_rejects_malformed_ranges(self):
        with self.assertRaises(ValueError):
            self.fx.capturer.get_rows(
                req_pool_idx=1, start=-1, end=3, req_to_token_pool=self.fx.pool
            )
        with self.assertRaises(ValueError):
            self.fx.capturer.get_rows(
                req_pool_idx=1, start=4, end=2, req_to_token_pool=self.fx.pool
            )

    def test_captured_layer_ids_are_global_model_indices(self):
        """`make_layers` passes the *global* layer index even when a pipeline
        stage owns only a slice, so a plane index is a model layer index with no
        stage offset. If this ever became stage-local, every reported layer id
        would be silently wrong on ranks after the first.

        The stage-local failure mode is what the second assertion pins: a
        rebased implementation would report 0..n-1 for a tail slice that should
        read stage_start..NUM_LAYERS-1.
        """
        stage_start = NUM_LAYERS // 2
        expected = list(range(stage_start, NUM_LAYERS))
        # Guard against the case going vacuous if the fixture shrinks: an empty
        # slice would make both assertions trivially true.
        self.assertGreater(len(expected), 1, "fixture must give the stage a real slice")

        for layer in expected:
            self.fx.capturer.capture(layer, torch.zeros((2, TOP_K), dtype=torch.int32))

        self.assertEqual(self.fx.capturer.captured_layer_ids, expected)
        self.assertNotEqual(
            self.fx.capturer.captured_layer_ids,
            list(range(len(expected))),
            "layer ids look stage-local (rebased to 0) rather than global",
        )

    def test_captured_layer_ids_tracks_only_layers_that_routed(self):
        """Dense layers never call capture(), so their planes stay zero and are
        indistinguishable from expert id 0 on the wire. The recorded set is the
        only thing that maps a plane back to a model layer."""
        self.assertEqual(self.fx.capturer.captured_layer_ids, [])
        self.fx.capturer.capture(1, torch.zeros((2, TOP_K), dtype=torch.int32))
        self.fx.capturer.capture(3, torch.zeros((2, TOP_K), dtype=torch.int32))
        self.assertEqual(self.fx.capturer.captured_layer_ids, [1, 3])


if __name__ == "__main__":
    unittest.main()
