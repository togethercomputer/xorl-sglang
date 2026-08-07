import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.layers.attention.nsa.glm52_selector import (
    pack_selected_kv_dynamic,
    pack_selected_kv_static,
    select_canonical_logical_topk,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestGlm52SparseSelector(unittest.TestCase):
    def test_exact_ties_choose_lower_logical_index_then_canonicalize(self):
        scores = torch.tensor([[0.5, 2.0, 2.0, 1.0, 3.0]], dtype=torch.float32)

        selected = select_canonical_logical_topk(scores, torch.tensor([5]), topk=3)

        self.assertEqual(selected.dtype, torch.int32)
        self.assertEqual(selected.tolist(), [[1, 2, 4]])

    def test_row_start_returns_request_local_indices(self):
        scores = torch.tensor(
            [[99.0, 99.0, 1.0, 4.0, 4.0, 99.0, 99.0]], dtype=torch.float32
        )

        selected = select_canonical_logical_topk(
            scores,
            torch.tensor([3]),
            topk=4,
            row_starts=torch.tensor([2]),
        )

        self.assertEqual(selected.tolist(), [[0, 1, 2, -1]])

    def test_short_and_dead_rows_are_suffix_padded(self):
        scores = torch.tensor(
            [[1.0, 2.0, 9.0], [float("-inf"), 7.0, 8.0]],
            dtype=torch.float32,
        )

        selected = select_canonical_logical_topk(scores, torch.tensor([2, 0]), topk=4)

        self.assertEqual(selected.tolist(), [[0, 1, -1, -1], [-1, -1, -1, -1]])

    def test_nan_is_rejected_only_when_legal(self):
        illegal_nan = torch.tensor([[1.0, 2.0, float("nan")]], dtype=torch.float32)
        self.assertEqual(
            select_canonical_logical_topk(
                illegal_nan, torch.tensor([2]), topk=2
            ).tolist(),
            [[0, 1]],
        )

        legal_nan = torch.tensor([[1.0, float("nan"), 2.0]], dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            select_canonical_logical_topk(legal_nan, torch.tensor([3]), topk=2)

    def test_production_selection_width_is_a_strict_subset(self):
        scores = torch.arange(4096, dtype=torch.float32).unsqueeze(0)

        selected = select_canonical_logical_topk(
            scores, torch.tensor([4096]), topk=2048
        )

        self.assertEqual(selected.shape, (1, 2048))
        self.assertEqual(selected[0, 0].item(), 2048)
        self.assertEqual(selected[0, -1].item(), 4095)

    def test_production_boundary_tie_prefers_lower_key_and_ignores_dead_tail(self):
        scores = torch.zeros((1, 4112), dtype=torch.float32)
        scores[:, :2047] = 2.0
        scores[:, 2047:2049] = 1.0
        scores[:, 4099:] = 1000.0

        selected = select_canonical_logical_topk(
            scores, torch.tensor([4099]), topk=2048
        )

        self.assertEqual(selected.tolist(), [list(range(2048))])

    def test_trusted_selector_preserves_valid_input_bytes(self):
        scores = torch.tensor(
            [[3.0, 1.0, 3.0, 2.0, float("-inf")]], dtype=torch.float32
        )
        lengths = torch.tensor([4])

        checked = select_canonical_logical_topk(scores, lengths, topk=3)
        trusted = select_canonical_logical_topk(scores, lengths, topk=3, validate=False)

        self.assertTrue(torch.equal(checked, trusted))

    def test_threshold_repair_matches_lexicographic_reference(self):
        generator = torch.Generator().manual_seed(520052)
        scores = torch.randint(
            -3, 4, (31, 257), generator=generator, dtype=torch.int32
        ).to(torch.float32)
        starts = torch.randint(0, 31, (31,), generator=generator)
        lengths = torch.randint(0, 226, (31,), generator=generator)
        lengths = torch.minimum(lengths, scores.shape[1] - starts)

        selected = select_canonical_logical_topk(
            scores, lengths, topk=73, row_starts=starts
        )

        expected = []
        for row, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
            legal = list(range(start, start + length))
            ranked = sorted(
                legal, key=lambda column: (-scores[row, column].item(), column)
            )
            logical = sorted(column - start for column in ranked[:73])
            expected.append(logical + [-1] * (73 - len(logical)))
        self.assertEqual(selected.tolist(), expected)

    def test_selector_does_not_sort_the_full_score_rows(self):
        with mock.patch.object(
            torch, "argsort", side_effect=AssertionError("full sort used")
        ):
            selected = select_canonical_logical_topk(
                torch.zeros((2, 4112), dtype=torch.float32),
                torch.tensor([4099, 0]),
                topk=2048,
            )

        self.assertEqual(selected[0].tolist(), list(range(2048)))
        self.assertEqual(selected[1].tolist(), [-1] * 2048)

    @staticmethod
    def _backend():
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.use_fused_topk = False
        backend.hisparse_coordinator = None
        backend.dsa_topk_backend = DSATopKBackend.TORCH
        backend.forward_metadata = SimpleNamespace(
            paged_mqa_schedule_metadata=None,
            paged_mqa_ctx_lens_2d=None,
        )
        backend.get_topk_transform_method = lambda mode: TopkTransformMethod.PAGED
        return backend

    def test_current_dsa_metadata_routes_prefill_and_decode(self):
        backend = self._backend()

        prefill = backend.get_indexer_metadata(
            1, SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        )
        decode = backend.get_indexer_metadata(
            1, SimpleNamespace(forward_mode=ForwardMode.DECODE)
        )

        self.assertEqual(prefill.topk_transform_method, TopkTransformMethod.PAGED)
        self.assertEqual(decode.topk_transform_method, TopkTransformMethod.PAGED)
        self.assertEqual(prefill.topk_backend, DSATopKBackend.TORCH)
        self.assertTrue(prefill.force_unfused_topk)

    def test_hisparse_decode_forces_unfused_current_dsa_topk(self):
        backend = self._backend()
        backend.use_fused_topk = True
        backend.hisparse_coordinator = object()

        decode = backend.get_indexer_metadata(
            1, SimpleNamespace(
                forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True)
            ),
        )

        self.assertTrue(decode.force_unfused_topk)


class TestGlm52SparseSelectedKVPacker(unittest.TestCase):
    @staticmethod
    def _cache_and_table(physical_rows):
        cache = torch.full((12, 2), 255, dtype=torch.uint8)
        table = torch.tensor(physical_rows, dtype=torch.int32)
        logical_payloads = [
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[11, 12], [13, 14], [15, 16], [17, 18]],
        ]
        for request, row in enumerate(physical_rows):
            for logical, physical in enumerate(row):
                cache[physical] = torch.tensor(
                    logical_payloads[request][logical], dtype=torch.uint8
                )
        return cache, table

    def test_physical_layout_does_not_change_packed_bytes_or_indices(self):
        selected = torch.tensor([[0, 2, 3, -1], [1, 3, -1, -1]], dtype=torch.int32)
        first_cache, first_table = self._cache_and_table([[7, 3, 9, 1], [5, 2, 10, 4]])
        second_cache, second_table = self._cache_and_table([[2, 8, 1, 6], [9, 3, 7, 5]])

        first = pack_selected_kv_dynamic(first_cache, first_table, selected)
        second = pack_selected_kv_dynamic(second_cache, second_table, selected)

        self.assertTrue(torch.equal(first.kv, second.kv))
        self.assertTrue(torch.equal(first.compact_indices, second.compact_indices))
        self.assertEqual(
            first.compact_indices.tolist(),
            [[0, 1, 2, -1], [3, 4, -1, -1]],
        )
        self.assertEqual(first.selected_counts.tolist(), [3, 2])

    def test_bf16_payload_is_copied_bitwise(self):
        storage = torch.arange(24, dtype=torch.int16).view(8, 1, 3)
        cache = storage.view(torch.bfloat16)
        table = torch.tensor([[6, 1, 5]], dtype=torch.int32)
        selected = torch.tensor([[0, 2, -1]], dtype=torch.int32)

        packed = pack_selected_kv_dynamic(cache, table, selected)

        expected_bits = torch.stack([storage[6], storage[5]])
        self.assertTrue(torch.equal(packed.kv.view(torch.int16), expected_bits))

    def test_trusted_packer_preserves_valid_input_bytes_and_metadata(self):
        storage = torch.arange(24, dtype=torch.int16).view(8, 1, 3)
        cache = storage.view(torch.bfloat16)
        table = torch.tensor([[6, 1, 5]], dtype=torch.int32)
        selected = torch.tensor([[0, 2, -1]], dtype=torch.int32)

        checked = pack_selected_kv_dynamic(cache, table, selected)
        trusted = pack_selected_kv_dynamic(cache, table, selected, validate=False)

        self.assertTrue(
            torch.equal(checked.kv.view(torch.int16), trusted.kv.view(torch.int16))
        )
        self.assertTrue(torch.equal(checked.compact_indices, trusted.compact_indices))
        self.assertTrue(torch.equal(checked.physical_indices, trusted.physical_indices))
        self.assertTrue(torch.equal(checked.selected_counts, trusted.selected_counts))

    def test_static_packer_matches_dynamic_live_prefix_and_zeroes_tail(self):
        storage = torch.arange(36, dtype=torch.int16).view(12, 1, 3)
        cache = storage.view(torch.bfloat16)
        table = torch.tensor([[7, 3, 9, 1], [5, 2, 10, 4]], dtype=torch.int32)
        selected = torch.tensor([[0, 2, 3, -1], [1, 3, -1, -1]], dtype=torch.int32)

        dynamic = pack_selected_kv_dynamic(cache, table, selected)
        static = pack_selected_kv_static(cache, table, selected, max_selected_tokens=8)

        self.assertTrue(bool(static.contract_ok))
        self.assertTrue(
            torch.equal(
                static.kv[: dynamic.kv.shape[0]].view(torch.int16),
                dynamic.kv.view(torch.int16),
            )
        )
        self.assertTrue(
            torch.equal(
                static.kv[dynamic.kv.shape[0] :].view(torch.int16),
                torch.zeros_like(static.kv[dynamic.kv.shape[0] :].view(torch.int16)),
            )
        )
        self.assertTrue(torch.equal(static.compact_indices, dynamic.compact_indices))
        self.assertTrue(torch.equal(static.selected_counts, dynamic.selected_counts))

    def test_static_packer_reports_invalid_contract_without_oob_access(self):
        cache = torch.arange(8, dtype=torch.uint8).view(4, 2)
        table = torch.tensor([[0, -1, 2]], dtype=torch.int32)
        selected = torch.tensor([[0, 1, 8]], dtype=torch.int32)

        packed = pack_selected_kv_static(cache, table, selected, max_selected_tokens=2)

        self.assertFalse(bool(packed.contract_ok))
        self.assertEqual(packed.kv.shape, (2, 2))

    def test_noncanonical_or_invalid_rows_fail_closed(self):
        cache = torch.zeros((4, 2), dtype=torch.uint8)
        table = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        with self.assertRaisesRegex(RuntimeError, "unique and ascending"):
            pack_selected_kv_dynamic(
                cache, table, torch.tensor([[2, 1, -1]], dtype=torch.int32)
            )
        with self.assertRaisesRegex(RuntimeError, "suffix-only"):
            pack_selected_kv_dynamic(
                cache, table, torch.tensor([[0, -1, 2]], dtype=torch.int32)
            )
        with self.assertRaisesRegex(RuntimeError, "exactly -1"):
            pack_selected_kv_dynamic(
                cache, table, torch.tensor([[0, -2, -1]], dtype=torch.int32)
            )

    def test_noncontiguous_cache_refuses_implicit_full_copy(self):
        cache = torch.zeros((2, 4), dtype=torch.uint8).transpose(0, 1)
        table = torch.tensor([[0, 1]], dtype=torch.int32)
        selected = torch.tensor([[0, 1]], dtype=torch.int32)

        with self.assertRaisesRegex(RuntimeError, "implicit full-cache copy"):
            pack_selected_kv_dynamic(cache, table, selected)


if __name__ == "__main__":
    unittest.main()
