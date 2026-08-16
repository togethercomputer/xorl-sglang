import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _gather_bf16_sparse_paged_kv,
)
from sglang.srt.layers.attention.nsa.glm52_selector import (
    pack_selected_kv_dynamic,
    pack_selected_kv_static,
    select_canonical_logical_topk,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBf16SparsePagedKvGather(unittest.TestCase):
    def test_preserves_row_order_duplicates_and_bytes(self):
        kv_cache = (
            torch.arange(24, dtype=torch.float32)
            .mul_(0.25)
            .to(torch.bfloat16)
            .reshape(2, 3, 1, 4)
        )
        row_indices = torch.tensor([5, 1, 5, 0], dtype=torch.int64)

        actual = _gather_bf16_sparse_paged_kv(kv_cache, row_indices)
        rows = kv_cache.reshape(-1, kv_cache.shape[-1])
        expected = torch.stack((rows[5], rows[1], rows[5], rows[0])).unsqueeze(1)

        self.assertEqual(actual.shape, (4, 1, 4))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )

    def test_rejects_non_bf16_pool(self):
        with self.assertRaises(AssertionError):
            _gather_bf16_sparse_paged_kv(
                torch.zeros(2, 1, 4, dtype=torch.float32),
                torch.tensor([0], dtype=torch.int64),
            )


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
        backend.glm52_exact_mode = False
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
        self.assertFalse(prefill.glm52_exact_mode)

    @staticmethod
    def _exact_metadata(**overrides):
        values = {
            "attn_metadata": SimpleNamespace(
                cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
                dsa_seqlens_expanded=torch.tensor([3, 3], dtype=torch.int32),
                page_size=64,
                real_page_table=torch.tensor([[4], [7]], dtype=torch.int32),
                topk_indices_offset=torch.zeros(2, dtype=torch.int32),
            ),
            "topk_transform_method": TopkTransformMethod.PAGED,
            "topk_backend": DSATopKBackend.SGL_KERNEL,
            "force_unfused_topk": False,
            "glm52_exact_mode": True,
        }
        values.update(overrides)
        return DSAIndexerMetadata(**values)

    def test_exact_metadata_dispatches_canonical_paged_owner(self):
        metadata = self._exact_metadata()
        logits = torch.zeros((2, 8), dtype=torch.float32)
        expected = torch.tensor([[41, 99, -1], [7, -1, -1]], dtype=torch.int32)
        flags = torch.zeros(2, dtype=torch.int32)

        with (
            mock.patch(
                "sglang.srt.layers.attention.nsa.glm52_selector_fast."
                "select_canonical_paged_topk_fused",
                return_value=(expected, flags),
            ) as canonical,
            mock.patch.object(
                DSATopKBackend,
                "topk_transform",
                side_effect=AssertionError("generic top-k dispatched"),
            ),
        ):
            actual = metadata.topk_transform(logits, 2048)

        self.assertTrue(torch.equal(actual, expected))
        canonical.assert_called_once()
        kwargs = canonical.call_args.kwargs
        self.assertIs(kwargs["page_table"], metadata.attn_metadata.real_page_table)
        self.assertEqual(kwargs["page_size"], 64)

    def test_exact_metadata_rejects_noncanonical_routes(self):
        logits = torch.zeros((2, 8), dtype=torch.float32)
        cases = (
            (
                self._exact_metadata(topk_backend=DSATopKBackend.TORCH),
                "sgl-kernel",
            ),
            (
                self._exact_metadata(topk_transform_method=TopkTransformMethod.RAGGED),
                "requires PAGED",
            ),
            (self._exact_metadata(force_unfused_topk=True), "unfused"),
        )
        for metadata, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                metadata.topk_transform(logits, 2048)

    def test_exact_metadata_rejects_legal_nan_status(self):
        metadata = self._exact_metadata()
        logits = torch.zeros((2, 8), dtype=torch.float32)
        transformed = torch.full((2, 2048), -1, dtype=torch.int32)
        flags = torch.tensor([0, 1], dtype=torch.int32)

        with (
            mock.patch(
                "sglang.srt.layers.attention.nsa.glm52_selector_fast."
                "select_canonical_paged_topk_fused",
                return_value=(transformed, flags),
            ),
            self.assertRaisesRegex(RuntimeError, "NaN"),
        ):
            metadata.topk_transform(logits, 2048)

    def test_exact_dsa_metadata_preserves_physical_index_contract(self):
        backend = self._backend()
        backend.glm52_exact_mode = True
        backend.use_fused_topk = True
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL

        metadata = backend.get_indexer_metadata(
            1, SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        )

        self.assertTrue(metadata.glm52_exact_mode)
        self.assertFalse(metadata.force_unfused_topk)

    def test_hisparse_decode_forces_unfused_current_dsa_topk(self):
        backend = self._backend()
        backend.use_fused_topk = True
        backend.hisparse_coordinator = object()

        decode = backend.get_indexer_metadata(
            1,
            SimpleNamespace(
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
