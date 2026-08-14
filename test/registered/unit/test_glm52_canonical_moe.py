import importlib.util
import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_ref,
    transform_index_page_table_prefill_ref,
)
from sglang.srt.distributed.canonical_moe import (
    GLM52_CANONICAL_MOE_VERSION,
    CanonicalDistribution,
    CanonicalMoEOutput,
    CanonicalMoEWorkspace,
    CanonicalRowSlots,
    SamplerParallelPlan,
    canonical_moe_reference,
    canonicalize_glm52_local_partial,
    canonicalize_glm52_local_partial_v3,
)
from sglang.srt.layers.glm52_positions import align_glm52_moe_positions
from sglang.srt.models.glm52_index_share import (
    CanonicalLogicalIndices,
    Glm52IndexShareManager,
    Glm52IndexSharePlan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGlm52CanonicalMoE(unittest.TestCase):
    def test_logical_row_ownership_covers_every_dp_cp_factorization(self):
        from sglang.srt.layers.glm52_ownership import LogicalRowOwnership

        for dp_size, cp_size in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
            with self.subTest(dp_size=dp_size, cp_size=cp_size):
                ordinals = []
                for dp_rank in range(dp_size):
                    for cp_rank in range(cp_size):
                        ownership = LogicalRowOwnership(
                            dp_size, cp_size, dp_rank, cp_rank, 16
                        )
                        ordinals.append(ownership.source_ordinal)
                        self.assertEqual(
                            ownership.context_source_ordinals,
                            tuple(range(dp_rank * cp_size, (dp_rank + 1) * cp_size)),
                        )
                self.assertEqual(ordinals, list(range(16)))

    def test_logical_row_ownership_maps_prefill_and_decode_sources(self):
        from sglang.srt.layers.glm52_ownership import LogicalRowOwnership

        ownership = LogicalRowOwnership(4, 4, 2, 3, 16)
        self.assertEqual(
            ownership.local_source_slice(
                [8, 12, 16, 20], local_rows=4, context_sharded=True
            ),
            slice(32, 36),
        )
        self.assertEqual(
            ownership.local_source_slice(
                [2, 3, 4, 5], local_rows=4, context_sharded=False
            ),
            slice(5, 9),
        )
        self.assertEqual(
            ownership.select_dp_representatives(
                ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + [None] * 4
            ),
            ["a", "b", "c", None],
        )
        with self.assertRaisesRegex(ValueError, "CP replicas disagree"):
            ownership.select_dp_representatives(
                ["a"] * 4 + ["b"] * 3 + ["x"] + ["c"] * 4 + [None] * 4
            )

    def test_dp_owned_row_gather_uses_one_cp_representative_per_request(self):
        from sglang.srt.layers.communicator_dsa_cp import _gather_dp_owned_rows
        from sglang.srt.layers.glm52_ownership import LogicalRowOwnership

        local = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
        remote = local + 100
        for cp_rank in (0, 3):
            ownership = LogicalRowOwnership(2, 8, 1, cp_rank, 16)
            output = torch.empty((7, 2), dtype=torch.bfloat16)

            def fake_all_reduce(actual, *, group):
                self.assertIs(group, tp_group)
                self.assertTrue(torch.equal(actual[:3], torch.zeros_like(actual[:3])))
                expected_local = local if cp_rank == 0 else torch.zeros_like(local)
                self.assertTrue(torch.equal(actual[3:], expected_local))
                actual[:3].copy_(remote[:3])
                actual[3:].copy_(local)

            tp_group = object()
            with (
                patch(
                    "sglang.srt.layers.communicator_dsa_cp.get_dp_global_num_tokens",
                    return_value=[3, 4],
                ),
                patch(
                    "sglang.srt.layers.communicator_dsa_cp.get_parallel",
                    return_value=SimpleNamespace(
                        tp_group=SimpleNamespace(device_group=tp_group)
                    ),
                ),
                patch(
                    "sglang.srt.layers.communicator_dsa_cp.dist.all_reduce",
                    side_effect=fake_all_reduce,
                ),
            ):
                gathered = _gather_dp_owned_rows(
                    local, output=output, ownership=ownership
                )
            self.assertTrue(torch.equal(gathered[:3], remote[:3]))
            self.assertTrue(torch.equal(gathered[3:], local))

    def test_dp_owned_positions_follow_rank_major_gather(self):
        from sglang.srt.layers.communicator_dsa_cp import (
            align_glm52_moe_positions as align_runtime_positions,
        )

        def fake_dp_gather(local, *, output, ownership):
            self.assertEqual(ownership.dp_rank, 1)
            if output.dtype is torch.int64:
                self.assertTrue(torch.equal(local, torch.tensor([2, 3])))
                output.copy_(torch.tensor([0, 2, 3, 5, 6, -1]))
            else:
                output.copy_(torch.tensor([1, 1, 1, 1, 1, 0], dtype=output.dtype))
            return output

        with patch.multiple(
            "sglang.srt.layers.communicator_dsa_cp",
            dsa_use_prefill_cp=lambda *_args: False,
            mla_use_prefill_cp=lambda *_args: False,
            get_parallel=lambda: SimpleNamespace(
                attn_dp_size=3,
                attn_dp_rank=1,
                attn_cp_size=1,
                attn_cp_rank=0,
                tp_size=3,
            ),
            get_dp_global_num_tokens=lambda: [1, 2, 3],
            _gather_dp_owned_rows=fake_dp_gather,
        ):
            aligned = align_runtime_positions(
                torch.tensor([2, 3]),
                torch.empty((6, 4)),
                SimpleNamespace(),
            )

        self.assertTrue(torch.equal(aligned.values, torch.tensor([0, 2, 3, 5, 6, -1])))
        self.assertTrue(
            torch.equal(
                aligned.valid_mask,
                torch.tensor([True, True, True, True, True, False]),
            )
        )

    def test_mixed_dp_cp_positions_compose_context_and_owner_gathers(self):
        from sglang.srt.layers.communicator_dsa_cp import (
            align_glm52_moe_positions as align_runtime_positions,
        )

        cp_positions = torch.tensor([8, 9, -1, 11])
        cp_valid = torch.tensor([1, 1, 0, 1], dtype=torch.int32)
        expected_positions = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, 11, 12, 13, 14, 15]
        )
        expected_valid = expected_positions >= 0

        def fake_cp_gather(output, local):
            self.assertEqual(local.numel(), 1)
            output.copy_(cp_positions if output.dtype is torch.int64 else cp_valid)

        def fake_dp_gather(local, *, output, ownership):
            self.assertEqual((ownership.dp_rank, ownership.cp_rank), (2, 1))
            self.assertTrue(
                torch.equal(
                    local, cp_positions if output.dtype is torch.int64 else cp_valid
                )
            )
            output.copy_(
                expected_positions
                if output.dtype is torch.int64
                else expected_valid.to(output.dtype)
            )
            return output

        with patch.multiple(
            "sglang.srt.layers.communicator_dsa_cp",
            dsa_use_prefill_cp=lambda *_args: True,
            mla_use_prefill_cp=lambda *_args: False,
            get_parallel=lambda: SimpleNamespace(
                attn_dp_size=4,
                attn_dp_rank=2,
                attn_cp_size=4,
                attn_cp_rank=1,
                tp_size=16,
            ),
            get_dp_global_num_tokens=lambda: [4, 4, 4, 4],
            attn_cp_all_gather_into_tensor=fake_cp_gather,
            _gather_dp_owned_rows=fake_dp_gather,
        ):
            aligned = align_runtime_positions(
                torch.tensor([9]),
                torch.empty((16, 4)),
                SimpleNamespace(),
            )

        self.assertTrue(torch.equal(aligned.values, expected_positions))
        self.assertTrue(torch.equal(aligned.valid_mask, expected_valid))

    def test_ragged_mixed_dp_cp_uses_logical_cp_rows_and_round_trips_source(self):
        from sglang.srt.layers.communicator import ScatterMode
        from sglang.srt.layers.communicator_dsa_cp import (
            DSACPLayerCommunicator,
            DSAMLPOutputLayout,
            align_glm52_moe_positions as align_runtime_positions,
            gather_glm52_mlp_rows,
        )
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy

        cases = (
            (2, 8, [17, 29]),
            (4, 4, [7, 13, 19, 25]),
            (8, 2, [3, 5, 7, 9, 11, 13, 15, 17]),
        )
        for dp_size, cp_size, dp_lengths in cases:
            with self.subTest(dp_size=dp_size, cp_size=cp_size):
                dp_rank = dp_size - 1
                cp_rank = cp_size - 1
                local_logical_rows = dp_lengths[dp_rank]
                per_rank_logical = [
                    len(range(rank, local_logical_rows, cp_size))
                    for rank in range(cp_size)
                ]
                max_logical = max(per_rank_logical)
                physical_rows = (max_logical + cp_size - 1) // cp_size * cp_size
                metadata = SimpleNamespace(
                    total_seq_lens=local_logical_rows,
                    per_rank_logical_token=per_rank_logical,
                    per_rank_actual_token=[physical_rows] * cp_size,
                )
                forward_batch = SimpleNamespace(attn_cp_metadata=metadata)

                global_rows = sum(dp_lengths)
                global_hidden = torch.arange(
                    global_rows * 2, dtype=torch.bfloat16
                ).reshape(global_rows, 2)
                global_positions = torch.arange(global_rows, dtype=torch.int64)
                block_start = sum(dp_lengths[:dp_rank])
                block_hidden = global_hidden[
                    block_start : block_start + local_logical_rows
                ]
                block_positions = global_positions[
                    block_start : block_start + local_logical_rows
                ]

                def physical_shards(logical):
                    shards = []
                    for rank in range(cp_size):
                        shard = logical[rank::cp_size]
                        padded = logical.new_zeros((physical_rows, *logical.shape[1:]))
                        padded[: shard.shape[0]].copy_(shard)
                        shards.append(padded)
                    return shards

                hidden_shards = physical_shards(block_hidden)
                position_shards = physical_shards(block_positions)
                valid_shards = physical_shards(
                    torch.ones(local_logical_rows, dtype=torch.int32)
                )
                strategy = InterleaveCPStrategy(cp_size)
                tp_device_group = object()
                parallel = SimpleNamespace(
                    attn_dp_size=dp_size,
                    attn_dp_rank=dp_rank,
                    attn_cp_size=cp_size,
                    attn_cp_rank=cp_rank,
                    attn_tp_size=1,
                    tp_size=16,
                    tp_rank=dp_rank * cp_size + cp_rank,
                    tp_group=SimpleNamespace(device_group=tp_device_group),
                )
                cp_parallel = SimpleNamespace(
                    attn_cp_rank=cp_rank,
                    attn_cp_group=object(),
                )

                def fake_cp_all_gather(output, local):
                    if local.dtype is torch.bfloat16:
                        shards = hidden_shards
                    elif local.dtype is torch.int64:
                        shards = position_shards
                    else:
                        self.assertEqual(local.dtype, torch.int32)
                        shards = valid_shards
                    output.copy_(torch.cat(shards))

                def fake_dp_all_reduce(output, *, group):
                    self.assertIs(group, tp_device_group)
                    if output.dtype is torch.bfloat16:
                        output.copy_(global_hidden)
                    elif output.dtype is torch.int64:
                        output.copy_(global_positions)
                    else:
                        self.assertEqual(output.dtype, torch.int32)
                        output.fill_(1)

                with (
                    patch(
                        "sglang.srt.layers.cp.base.get_cp_strategy",
                        return_value=strategy,
                    ),
                    patch(
                        "sglang.srt.layers.cp.interleave.get_parallel",
                        return_value=cp_parallel,
                    ),
                    patch(
                        "sglang.srt.layers.cp.base.get_parallel",
                        return_value=cp_parallel,
                    ),
                    patch(
                        "sglang.srt.layers.cp.interleave.use_symmetric_memory",
                        side_effect=lambda *_args, **_kwargs: nullcontext(),
                    ),
                    patch(
                        "sglang.srt.layers.cp.interleave.is_allocation_symmetric",
                        return_value=False,
                    ),
                    patch(
                        "sglang.srt.layers.cp.interleave.attn_cp_all_gather_into_tensor",
                        side_effect=fake_cp_all_gather,
                    ),
                    patch(
                        "sglang.srt.layers.communicator_dsa_cp.get_parallel",
                        return_value=parallel,
                    ),
                    patch(
                        "sglang.srt.layers.communicator_dsa_cp.get_dp_global_num_tokens",
                        return_value=dp_lengths,
                    ),
                    patch(
                        "sglang.srt.layers.communicator_dsa_cp.get_global_dp_buffer",
                        side_effect=lambda _group: torch.empty_like(global_hidden),
                    ),
                    patch(
                        "sglang.srt.layers.communicator_dsa_cp.dist.all_reduce",
                        side_effect=fake_dp_all_reduce,
                    ),
                    patch(
                        "sglang.srt.layers.communicator_dsa_cp.dsa_use_prefill_cp",
                        return_value=True,
                    ),
                    patch(
                        "sglang.srt.layers.communicator_dsa_cp.mla_use_prefill_cp",
                        return_value=False,
                    ),
                ):
                    gathered = gather_glm52_mlp_rows(
                        hidden_shards[cp_rank], forward_batch
                    )
                    self.assertTrue(torch.equal(gathered, global_hidden))

                    aligned = align_runtime_positions(
                        position_shards[cp_rank], gathered, forward_batch
                    )
                    self.assertTrue(torch.equal(aligned.values, global_positions))
                    self.assertTrue(torch.all(aligned.valid_mask))

                    communicator = object.__new__(DSACPLayerCommunicator)
                    communicator.mlp_output_layout = DSAMLPOutputLayout.COMPLETE
                    communicator.layer_scatter_modes = SimpleNamespace(
                        mlp_mode=ScatterMode.FULL
                    )
                    communicator._context = parallel
                    residual = torch.zeros_like(hidden_shards[cp_rank])
                    local, returned_residual = communicator.postprocess_layer(
                        gathered, residual, forward_batch
                    )
                    self.assertTrue(torch.equal(local, hidden_shards[cp_rank]))
                    self.assertIs(returned_residual, residual)

    def test_ragged_dp1_cp_v2_preserves_consumer_sharded_physical_buckets(self):
        from sglang.srt.layers.communicator import ScatterMode
        from sglang.srt.layers.communicator_dsa_cp import (
            DSACPLayerCommunicator,
            DSAMLPOutputLayout,
            gather_glm52_mlp_rows,
        )

        cp_size = 8
        cp_rank = 5
        logical_rows = 17
        physical_rows = 8
        logical = torch.arange(logical_rows * 2, dtype=torch.bfloat16).reshape(
            logical_rows, 2
        )
        physical_shards = []
        per_rank_logical = []
        for rank in range(cp_size):
            shard = logical[rank::cp_size]
            per_rank_logical.append(shard.shape[0])
            padded = torch.full((physical_rows, 2), -100 - rank, dtype=torch.bfloat16)
            padded[: shard.shape[0]].copy_(shard)
            physical_shards.append(padded)
        expected_rank_major = torch.cat(physical_shards)
        gathered_buffer = torch.empty_like(expected_rank_major)
        metadata = SimpleNamespace(
            total_seq_lens=logical_rows,
            per_rank_logical_token=per_rank_logical,
            per_rank_actual_token=[physical_rows] * cp_size,
        )
        forward_batch = SimpleNamespace(attn_cp_metadata=metadata)
        parallel = SimpleNamespace(
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_cp_size=cp_size,
            attn_cp_rank=cp_rank,
            attn_tp_size=1,
            tp_size=cp_size,
            attn_cp_group=object(),
        )

        def fake_cp_all_gather(output, local):
            self.assertTrue(torch.equal(local, physical_shards[cp_rank]))
            output.copy_(expected_rank_major)

        with (
            patch(
                "sglang.srt.layers.communicator_dsa_cp.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.communicator_dsa_cp.get_local_dp_buffer",
                return_value=gathered_buffer,
            ),
            patch(
                "sglang.srt.layers.communicator_dsa_cp.attn_cp_all_gather_into_tensor",
                side_effect=fake_cp_all_gather,
            ),
            patch(
                "sglang.srt.layers.communicator_dsa_cp.dsa_use_prefill_cp",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communicator_dsa_cp.mla_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.cp.base.get_cp_strategy",
                side_effect=AssertionError(
                    "DP1 canonical-v3b prefill must keep physical rank-major rows"
                ),
            ),
        ):
            gathered = gather_glm52_mlp_rows(physical_shards[cp_rank], forward_batch)
            self.assertTrue(torch.equal(gathered, expected_rank_major))

            communicator = object.__new__(DSACPLayerCommunicator)
            communicator.mlp_output_layout = DSAMLPOutputLayout.COMPLETE
            communicator.layer_scatter_modes = SimpleNamespace(
                mlp_mode=ScatterMode.FULL
            )
            communicator._context = parallel
            consumer_shard = physical_shards[cp_rank] + 1000
            residual = torch.zeros_like(consumer_shard)
            local, returned_residual = communicator.postprocess_layer(
                consumer_shard, residual, forward_batch
            )

        self.assertTrue(torch.equal(local, consumer_shard))
        self.assertIs(returned_residual, residual)

    def test_cp_v2_logical_gather_rejects_invalid_strategy_row_count(self):
        from sglang.srt.layers.communicator_dsa_cp import (
            _gather_glm52_cp_logical_rows,
        )

        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(total_seq_lens=7)
        )
        strategy = SimpleNamespace(gather_hidden_states=lambda rows, _batch: rows[:3])
        with (
            patch(
                "sglang.srt.layers.cp.base.get_cp_strategy",
                return_value=strategy,
            ),
            self.assertRaisesRegex(RuntimeError, "wrong logical row count"),
        ):
            _gather_glm52_cp_logical_rows(torch.zeros(4, 2), forward_batch)

    def test_cp16_dp1_does_not_request_mlp_tp_gather(self):
        from sglang.srt.utils.common import require_mlp_sync, require_mlp_tp_gather

        parallel = SimpleNamespace(enable_dp_attention=True, dp_size=1)
        moe_utils = SimpleNamespace(
            get_moe_a2a_backend=lambda: self.fail(
                "DP1 must return before inspecting the MoE A2A backend"
            )
        )
        with (
            patch.dict("sys.modules", {"sglang.srt.layers.moe.utils": moe_utils}),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        ):
            self.assertFalse(require_mlp_tp_gather(SimpleNamespace(tp_size=16)))
            self.assertTrue(require_mlp_sync(SimpleNamespace(tp_size=16)))

    def test_exact_mode_selects_v3_or_v3b_without_transport_admission_gate(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")
        from sglang.srt.models.deepseek_v2 import (
            _resolve_glm52_canonical_transport,
            _select_glm52_canonical_transport,
        )

        self.assertEqual(
            _resolve_glm52_canonical_transport(SimpleNamespace(_glm52_exact_mode=True)),
            "auto",
        )
        for configured in ("auto", "dense_v1", "canonical_v3", "canonical_v3b"):
            with self.subTest(configured=configured):
                resolved = _resolve_glm52_canonical_transport(
                    SimpleNamespace(
                        _glm52_exact_mode=True,
                        _glm52_canonical_moe_transport=configured,
                    )
                )
                self.assertEqual(resolved, configured)
                if configured != "dense_v1":
                    self.assertEqual(
                        _select_glm52_canonical_transport(
                            resolved,
                            prefill_cp=True,
                        ),
                        "canonical_v3",
                    )
        self.assertEqual(
            _select_glm52_canonical_transport("auto", prefill_cp=False),
            "canonical_v3b",
        )
        self.assertEqual(
            _select_glm52_canonical_transport("canonical_v3", prefill_cp=False),
            "canonical_v3",
        )
        self.assertEqual(
            _select_glm52_canonical_transport("dense_v1", prefill_cp=False),
            "dense_v1",
        )
        with self.assertRaisesRegex(RuntimeError, "consumer-sharded"):
            _select_glm52_canonical_transport(
                "dense_v1",
                prefill_cp=True,
                consumer_sharded=True,
            )
        with self.assertRaisesRegex(RuntimeError, "must be auto"):
            _resolve_glm52_canonical_transport(
                SimpleNamespace(
                    _glm52_exact_mode=True,
                    _glm52_canonical_moe_transport="not-a-transport",
                )
            )

    def test_glm52_explicit_mlp_layer_types_treat_boundaries_as_absent(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        config = SimpleNamespace(
            num_hidden_layers=4,
            mlp_layer_types=["dense", "sparse", "dense", "sparse"],
        )
        decoder = SimpleNamespace(config=config, glm52_xorl_bi_contract=True)

        self.assertFalse(
            DeepseekV2DecoderLayer._is_layer_sparse(
                decoder, layer_id=-1, is_nextn=False
            )
        )
        self.assertFalse(
            DeepseekV2DecoderLayer._is_layer_sparse(decoder, layer_id=4, is_nextn=False)
        )
        self.assertFalse(
            DeepseekV2DecoderLayer._is_layer_sparse(decoder, layer_id=0, is_nextn=False)
        )
        self.assertTrue(
            DeepseekV2DecoderLayer._is_layer_sparse(decoder, layer_id=1, is_nextn=False)
        )

    def test_glm52_explicit_mlp_layer_types_validate_real_layers(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")
        from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer

        decoder = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=2,
                mlp_layer_types=["dense"],
            ),
            glm52_xorl_bi_contract=True,
        )
        with self.assertRaisesRegex(ValueError, "length does not match"):
            DeepseekV2DecoderLayer._is_layer_sparse(decoder, layer_id=2, is_nextn=False)

        decoder.config.mlp_layer_types = ["dense", "unknown"]
        with self.assertRaisesRegex(ValueError, "Unknown GLM-5.2 mlp layer type"):
            DeepseekV2DecoderLayer._is_layer_sparse(decoder, layer_id=1, is_nextn=False)

    def test_glm52_correction_bias_stays_fp32_under_fp8_quant_config(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")
        from sglang.srt.models.deepseek_v2 import MoEGate

        config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            topk_method="noaux_tc",
            indexer_types=["full"],
            _glm52_exact_mode=True,
        )
        quant_config = SimpleNamespace(get_name=lambda: "compressed_tensors")
        gate = MoEGate(config, quant_config)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)

    def test_glm52_bi_router_dispatches_to_pinned_kernel(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")
        from sglang.srt.models.deepseek_v2 import MoEGate

        config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            topk_method="noaux_tc",
            indexer_types=["full"],
            _glm52_exact_mode=True,
        )
        fake_hidden = SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            ndim=2,
            dtype=torch.bfloat16,
        )
        expected = object()
        with (
            patch(
                "sglang.srt.models.deepseek_v2.use_intel_amx_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(
                        enable_deterministic_inference=True,
                    ),
                ),
            ),
            patch(
                "sglang.srt.batch_invariant_ops.batch_invariant_ops.bi_router_gemm",
                return_value=expected,
            ) as router_gemm,
        ):
            gate = MoEGate(config, quant_config=None).to(dtype=torch.bfloat16)
            result = gate(fake_hidden)

        self.assertIs(result, expected)
        router_gemm.assert_called_once_with(fake_hidden, gate.weight)

    def test_glm52_bi_router_requires_deterministic_inference(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")
        from sglang.srt.models.deepseek_v2 import MoEGate

        config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            topk_method="noaux_tc",
            indexer_types=["full"],
            _glm52_exact_mode=True,
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v2.use_intel_amx_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(
                        enable_deterministic_inference=False,
                    ),
                ),
            ),
        ):
            gate = MoEGate(config, quant_config=None).to(dtype=torch.bfloat16)
            with self.assertRaisesRegex(RuntimeError, "deterministic inference"):
                gate(torch.zeros((1, 8), dtype=torch.bfloat16))

    def test_glm52_non_exact_gate_uses_the_standard_deterministic_path(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")
        from sglang.srt.models.deepseek_v2 import MoEGate

        config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            topk_method="noaux_tc",
            indexer_types=["full"],
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v2.use_intel_amx_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(
                        enable_deterministic_inference=True,
                    ),
                ),
            ),
        ):
            gate = MoEGate(config, quant_config=None).to(dtype=torch.bfloat16)
            hidden = torch.zeros((1, 8), dtype=torch.bfloat16)
            self.assertTrue(
                torch.equal(
                    gate(hidden), torch.nn.functional.linear(hidden, gate.weight)
                )
            )

    def test_non_glm_gate_ignores_glm52_exact_mode(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")
        from sglang.srt.models.deepseek_v2 import MoEGate

        config = SimpleNamespace(
            n_routed_experts=4,
            hidden_size=8,
            topk_method="noaux_tc",
            indexer_types=None,
            _glm52_exact_mode=True,
        )
        hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        with (
            patch(
                "sglang.srt.models.deepseek_v2.use_intel_amx_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(
                        enable_deterministic_inference=True,
                    ),
                ),
            ),
        ):
            gate = MoEGate(config, quant_config=None)
            gate.weight.data.copy_(
                torch.arange(32, dtype=torch.float32).reshape(4, 8) / 17
            )
            result = gate(hidden)

        self.assertTrue(
            torch.equal(result, torch.nn.functional.linear(hidden, gate.weight))
        )

    def test_official_index_share_schedule_and_context_lifecycle(self):
        full_layers = (0, 1, 2, *range(6, 75, 4))
        config = SimpleNamespace(
            num_hidden_layers=78,
            indexer_types=[
                "full" if layer in full_layers else "shared" for layer in range(78)
            ],
            index_topk_freq=4,
            index_skip_topk_offset=3,
            index_topk_pattern=None,
        )
        plan = Glm52IndexSharePlan.from_config(config)
        self.assertEqual(plan.full_layers, full_layers)
        self.assertEqual(len(plan.shared_layers), 57)
        self.assertEqual(plan.producer_by_layer[37], 34)
        self.assertEqual(plan.producer_by_layer[38], 38)
        plan.validate_pipeline_stage(0, 38)
        plan.validate_pipeline_stage(38, 78)
        with self.assertRaisesRegex(ValueError, "must begin on an IndexShare producer"):
            plan.validate_pipeline_stage(39, 78)

        manager = Glm52IndexShareManager(plan)
        with manager.invocation() as context:
            selected = CanonicalLogicalIndices(
                torch.tensor([[0, 1, -1]], dtype=torch.int32)
            )
            context.publish(34, selected)
            self.assertIs(context.consume(37, require_indices=True), selected)
            with self.assertRaisesRegex(TypeError, "typed canonical logical"):
                context.publish(38, selected.values)
            with self.assertRaisesRegex(RuntimeError, "one live"):
                manager.begin()
        self.assertTrue(context.closed)

    def test_index_share_keeps_logical_indices_and_consumers_transform_independently(
        self,
    ):
        plan = Glm52IndexSharePlan.from_config(
            SimpleNamespace(
                num_hidden_layers=2,
                indexer_types=["full", "shared"],
                index_topk_freq=2,
                index_skip_topk_offset=1,
                index_topk_pattern=[1, 0],
            )
        )
        logical = CanonicalLogicalIndices(torch.tensor([[0, 2, -1]], dtype=torch.int32))
        with Glm52IndexShareManager(plan).invocation() as context:
            context.publish(0, logical)
            reused = context.consume(1, require_indices=True)
            first_layout = transform_index_page_table_decode_ref(
                page_table=torch.tensor([[41, 7, 99]], dtype=torch.int32),
                topk_indices=reused.values,
            )
            second_layout = transform_index_page_table_decode_ref(
                page_table=torch.tensor([[5, 88, 13]], dtype=torch.int32),
                topk_indices=reused.values,
            )
        self.assertEqual(logical.values.tolist(), [[0, 2, -1]])
        self.assertEqual(first_layout.tolist(), [[41, 99, -1]])
        self.assertEqual(second_layout.tolist(), [[5, 13, -1]])

    def test_prefill_logical_index_transform_preserves_padded_query_rows(self):
        page_table = torch.tensor([[41, 7, 99, 13]], dtype=torch.int32)
        logical = torch.tensor(
            [[0, 2, -1], [1, 3, -1], [-1, -1, -1], [-1, -1, -1]],
            dtype=torch.int32,
        )

        transformed = transform_index_page_table_prefill_ref(
            page_table=page_table,
            topk_indices=logical,
            extend_lens_cpu=[2],
        )

        self.assertEqual(
            transformed.tolist(),
            [[41, 99, -1], [7, 13, -1], [-1, -1, -1], [-1, -1, -1]],
        )

    def test_prefill_logical_index_transform_masks_non_padding_suffix(self):
        page_table = torch.tensor([[41, 7, 99, 13]], dtype=torch.int32)
        logical = torch.tensor([[0, 2, -1], [1, 3, -1], [0, -1, -1]], dtype=torch.int32)

        transformed = transform_index_page_table_prefill_ref(
            page_table=page_table,
            topk_indices=logical,
            extend_lens_cpu=[2],
        )

        self.assertEqual(
            transformed.tolist(),
            [[41, 99, -1], [7, 13, -1], [-1, -1, -1]],
        )

    def test_prefill_logical_index_transform_rejects_short_topk_domain(self):
        page_table = torch.tensor([[41, 7, 99, 13]], dtype=torch.int32)
        logical = torch.tensor([[0, 2, -1]], dtype=torch.int32)

        with self.assertRaisesRegex(
            AssertionError,
            r"sum\(extend_lens_cpu\) \(2\) exceeds topk_indices rows \(1\)",
        ):
            transform_index_page_table_prefill_ref(
                page_table=page_table,
                topk_indices=logical,
                extend_lens_cpu=[2],
            )

    def test_prefill_positions_follow_full_cp_gather_and_decode_stays_replicated(self):
        rank_positions = torch.tensor(
            [[0, 8], [1, 9], [2, -1], [3, 11], [4, 12], [5, -1], [6, 14], [7, 15]],
            dtype=torch.int64,
        )
        rank_valid = rank_positions >= 0

        def fake_gather(output, _local):
            source = (
                rank_positions
                if output.dtype == torch.int64
                else rank_valid.to(torch.uint8)
            )
            output.copy_(source.reshape(-1))

        aligned = align_glm52_moe_positions(
            torch.tensor([0, 8]),
            torch.zeros((16, 4)),
            prefill_cp=True,
            cp_size=8,
            all_gather=fake_gather,
        )
        self.assertTrue(torch.equal(aligned.values, rank_positions.reshape(-1)))
        self.assertTrue(torch.equal(aligned.valid_mask, rank_valid.reshape(-1)))
        owners = aligned.values[aligned.valid_mask].remainder(8)
        self.assertEqual(set(owners.tolist()), set(range(8)))

        decode = align_glm52_moe_positions(
            torch.tensor([20, 21, 22]),
            torch.zeros((3, 4)),
            prefill_cp=False,
        )
        self.assertEqual(decode.values.tolist(), [20, 21, 22])
        self.assertTrue(bool(torch.all(decode.valid_mask)))

    def test_native_dsa_prefill_canonical_moe_selects_local_rows_without_reduce_scatter(
        self,
    ):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")

        from sglang.srt.layers.communicator import (
            CommunicateContext,
            LayerScatterModes,
            ScatterMode,
        )
        from sglang.srt.layers.communicator_dsa_cp import (
            DSACPLayerCommunicator,
            DSAMLPOutputLayout,
        )
        from sglang.srt.models.deepseek_v2 import (
            DeepseekV2DecoderLayer,
            DeepseekV2MoE,
        )

        class IdentityResidualNorm(torch.nn.Module):
            def forward(self, hidden_states, residual=None, *_args):
                if residual is None:
                    return hidden_states
                return hidden_states, residual

        class EchoAttention(torch.nn.Module):
            def maybe_use_decode_attn_tp(self, _forward_batch):
                return nullcontext()

            def forward(self, *, hidden_states, **_kwargs):
                return hidden_states

        class FakeGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.e_score_correction_bias = torch.zeros(8, dtype=torch.float32)
                self.dsa_enable_prefill_cp = True
                self.mla_enable_prefill_cp = False

            def forward(self, hidden_states, *_args):
                return hidden_states.new_zeros((hidden_states.shape[0], 8))

        class FakeTopK(torch.nn.Module):
            def forward(self, *_args, **_kwargs):
                return None

        class FakeExperts(torch.nn.Module):
            def __init__(self, partial):
                super().__init__()
                self.partial = partial
                self.quant_method = None
                self.moe_runner_config = SimpleNamespace(inplace=True)

            def forward(self, *_args):
                return self.partial.clone()

        class ZeroSharedExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = object()

            def forward(self, hidden_states, **_kwargs):
                return torch.zeros_like(hidden_states)

        cp_size = 8
        local_capacity = 2
        hidden_size = 4
        rank_positions = torch.tensor(
            [[0, 8], [1, 9], [2, -1], [3, 11], [4, 12], [5, -1], [6, 14], [7, 15]],
            dtype=torch.int64,
        )
        rank_valid = rank_positions >= 0
        full_positions = rank_positions.reshape(-1)
        full_valid = rank_valid.reshape(-1)
        self.assertEqual(
            set(full_positions[full_valid].remainder(cp_size).tolist()),
            set(range(cp_size)),
        )

        rank_hidden = torch.arange(
            cp_size * local_capacity * hidden_size,
            dtype=torch.bfloat16,
        ).reshape(cp_size, local_capacity, hidden_size)
        partials = torch.empty(
            (cp_size, cp_size * local_capacity, hidden_size),
            dtype=torch.bfloat16,
        )
        for contributor in range(cp_size):
            partials[contributor, :, 0] = contributor + 1
            partials[contributor, :, 1] = 4096.0 if contributor == 0 else -1.0
            partials[contributor, :, 2] = torch.arange(
                cp_size * local_capacity, dtype=torch.bfloat16
            )
            partials[contributor, :, 3] = contributor
        slots = CanonicalRowSlots.from_positions(
            full_positions,
            valid_mask=full_valid,
        )
        expected_full = canonical_moe_reference(partials, slots)

        modes = LayerScatterModes(
            layer_input_mode=ScatterMode.SCATTERED,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        forward_batch = SimpleNamespace()

        with self.subTest(cp_size=cp_size):
            for cp_rank in range(cp_size):
                context = CommunicateContext(
                    process_group_sizes={
                        ScatterMode.SCATTERED: 1,
                        ScatterMode.TP_ATTN_FULL: 1,
                        ScatterMode.FULL: cp_size,
                    },
                    attn_tp_rank=0,
                    attn_tp_size=1,
                    attn_dp_size=1,
                    attn_cp_rank=cp_rank,
                    attn_cp_size=cp_size,
                    tp_size=cp_size,
                    tp_rank=cp_rank,
                )
                full_hidden = rank_hidden.reshape(cp_size * local_capacity, hidden_size)

                def fake_cp_gather(output, _local):
                    if output.dtype == torch.bfloat16:
                        output.copy_(full_hidden)
                    elif output.dtype == torch.int64:
                        output.copy_(full_positions)
                    else:
                        output.copy_(full_valid.to(output.dtype))

                def fake_canonicalize(local_partial, actual_slots, **kwargs):
                    self.assertTrue(torch.equal(local_partial, partials[cp_rank]))
                    self.assertTrue(
                        torch.equal(actual_slots.absolute_positions, full_positions)
                    )
                    self.assertTrue(torch.equal(actual_slots.valid_mask, full_valid))
                    self.assertEqual(
                        kwargs["distribution"],
                        CanonicalDistribution.CONSUMER_SHARDED,
                    )
                    local_slots = CanonicalRowSlots.from_positions(
                        rank_positions[cp_rank],
                        valid_mask=rank_valid[cp_rank],
                    )
                    return CanonicalMoEOutput(
                        values=expected_full.narrow(
                            0, cp_rank * local_capacity, local_capacity
                        ).clone(),
                        owner_mask=local_slots.valid_mask,
                        slots=local_slots,
                        contract_status=torch.zeros((), dtype=torch.int32),
                    )

                with (
                    patch.object(CommunicateContext, "init_new", return_value=context),
                    patch.multiple(
                        "sglang.srt.layers.communicator",
                        get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
                        get_attn_tp_context=lambda: SimpleNamespace(
                            input_scattered=False
                        ),
                        is_enable_moe_cp_allgather=lambda: False,
                        is_dp_attention_enabled=lambda: False,
                        apply_flashinfer_allreduce_fusion=lambda _batch_size: False,
                    ),
                    patch.multiple(
                        "sglang.srt.layers.communicator_dsa_cp",
                        dsa_use_prefill_cp=lambda *_args, **_kwargs: True,
                        get_parallel=lambda: SimpleNamespace(
                            attn_dp_size=1,
                            attn_tp_size=1,
                            attn_cp_group=object(),
                            attn_cp_size=cp_size,
                            attn_cp_rank=cp_rank,
                        ),
                        get_local_dp_buffer=lambda _group: torch.empty_like(
                            full_hidden
                        ),
                        attn_cp_all_gather_into_tensor=fake_cp_gather,
                    ),
                    patch(
                        "sglang.srt.layers.communicator_dsa_cp.attn_cp_reduce_scatter_tensor",
                        side_effect=AssertionError(
                            "canonical output entered legacy reduce-scatter"
                        ),
                    ) as legacy_reduce_scatter,
                    patch(
                        "sglang.srt.models.deepseek_v2.CanonicalMoEV3Workspace.allocate",
                        return_value=object(),
                    ),
                    patch.multiple(
                        "sglang.srt.models.deepseek_v2",
                        canonicalize_glm52_local_partial_v3=fake_canonicalize,
                        get_parallel=lambda: SimpleNamespace(
                            tp_group=SimpleNamespace(device_group=object())
                        ),
                        get_is_capture_mode=lambda: False,
                        dsa_use_prefill_cp=lambda *_args, **_kwargs: True,
                        get_exec=lambda: SimpleNamespace(
                            moe=SimpleNamespace(enable_eplb=False),
                        ),
                        get_forward=lambda: SimpleNamespace(
                            fuse_mlp_allreduce=False,
                            mlp_reduce_scatter=False,
                            scoped=lambda **_kwargs: nullcontext(),
                        ),
                        get_server_args=lambda: SimpleNamespace(),
                        get_attn_tp_context=lambda: SimpleNamespace(
                            clear_attn_inputs=lambda: None
                        ),
                        use_intel_amx_backend=lambda *_args, **_kwargs: False,
                    ),
                ):
                    communicator = DSACPLayerCommunicator(
                        layer_scatter_modes=modes,
                        input_layernorm=IdentityResidualNorm(),
                        post_attention_layernorm=IdentityResidualNorm(),
                        allow_reduce_scatter=True,
                        mlp_output_layout=DSAMLPOutputLayout.COMPLETE,
                    )
                    self.assertEqual(
                        communicator.mlp_output_layout,
                        DSAMLPOutputLayout.COMPLETE,
                    )
                    self.assertFalse(
                        communicator.should_use_reduce_scatter(forward_batch)
                    )

                    moe = object.__new__(DeepseekV2MoE)
                    torch.nn.Module.__init__(moe)
                    moe._glm52_canonical_contract = True
                    moe._glm52_deferred_status_book = None
                    moe._glm52_canonical_transport = "canonical_v3b"
                    moe._canonical_v3_workspaces = {}
                    moe._enable_a2a_moe = False
                    moe._fuse_shared_experts_inside_sbo = False
                    moe._shared_expert_tp1 = False
                    moe._moe_quant_once = False
                    moe.is_nextn = False
                    moe.num_fused_shared_experts = 0
                    moe.shared_experts = ZeroSharedExperts()
                    moe.gate = FakeGate()
                    moe.topk = FakeTopK()
                    moe.experts = FakeExperts(partials[cp_rank])
                    moe.routed_scaling_factor = 1.0
                    moe.tp_size = cp_size
                    moe.moe_ep_size = cp_size
                    moe.glm52_parallel_plan = SamplerParallelPlan.glm52()
                    moe.layer_id = 7

                    decoder = object.__new__(DeepseekV2DecoderLayer)
                    torch.nn.Module.__init__(decoder)
                    decoder.layer_id = 7
                    decoder.dsa_enable_prefill_cp = True
                    decoder.mla_enable_prefill_cp = False
                    decoder.layer_scatter_modes = modes
                    decoder.layer_communicator = communicator
                    decoder.self_attn = EchoAttention()
                    decoder.mlp = moe
                    decoder.is_layer_sparse = True

                    local_hidden = rank_hidden[cp_rank].clone()
                    local_residual = (rank_hidden[cp_rank] + 100).clone()
                    output, residual, topk_indices = decoder(
                        rank_positions[cp_rank],
                        local_hidden,
                        forward_batch,
                        local_residual,
                        zero_allocator=None,
                    )

                expected_local = expected_full.narrow(
                    0, cp_rank * local_capacity, local_capacity
                )
                self.assertTrue(torch.equal(output, expected_local))
                self.assertTrue(torch.equal(residual, local_residual))
                self.assertIsNone(topk_indices)
                legacy_reduce_scatter.assert_not_called()

    def test_native_dsa_prefill_complete_dense_tp1_stays_rank_local_at_cp16(self):
        from sglang.srt.layers.communicator import (
            CommunicateContext,
            LayerScatterModes,
            ScatterMode,
        )
        from sglang.srt.layers.communicator_dsa_cp import (
            DSACPLayerCommunicator,
            DSAMLPOutputLayout,
        )

        modes = LayerScatterModes(
            layer_input_mode=ScatterMode.SCATTERED,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.SCATTERED,
            middle_residual_mode=ScatterMode.SCATTERED,
            layer_output_mode=ScatterMode.SCATTERED,
        )
        forward_batch = SimpleNamespace()
        for cp_rank in range(16):
            context = CommunicateContext(
                process_group_sizes={
                    ScatterMode.SCATTERED: 1,
                    ScatterMode.TP_ATTN_FULL: 1,
                    ScatterMode.FULL: 16,
                },
                attn_tp_rank=0,
                attn_tp_size=1,
                attn_dp_size=1,
                attn_cp_rank=cp_rank,
                attn_cp_size=16,
                tp_size=16,
                tp_rank=cp_rank,
            )
            with (
                patch.object(CommunicateContext, "init_new", return_value=context),
                patch(
                    "sglang.srt.layers.communicator.get_spec",
                    return_value=SimpleNamespace(speculative_algorithm=None),
                ),
                patch(
                    "sglang.srt.layers.communicator_dsa_cp.dsa_use_prefill_cp",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.layers.communicator_dsa_cp.attn_cp_reduce_scatter_tensor",
                    side_effect=AssertionError(
                        "complete dense output entered reduce-scatter"
                    ),
                ) as legacy_reduce_scatter,
            ):
                communicator = DSACPLayerCommunicator(
                    layer_scatter_modes=modes,
                    input_layernorm=torch.nn.Identity(),
                    post_attention_layernorm=torch.nn.Identity(),
                    allow_reduce_scatter=False,
                    mlp_output_layout=DSAMLPOutputLayout.COMPLETE,
                )
                local = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
                residual = local + cp_rank
                output, returned_residual = communicator.postprocess_layer(
                    local, residual, forward_batch
                )

            self.assertFalse(communicator.should_use_reduce_scatter(forward_batch))
            self.assertTrue(torch.equal(output, local))
            self.assertTrue(torch.equal(returned_residual, residual))
            legacy_reduce_scatter.assert_not_called()

    def test_canonical_dsa_layout_fails_closed_on_legacy_or_fused_reduction(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")

        from sglang.srt.layers.communicator import (
            CommunicateContext,
            LayerScatterModes,
            ScatterMode,
        )
        from sglang.srt.layers.communicator_dsa_cp import (
            DSACPLayerCommunicator,
            DSAMLPOutputLayout,
        )
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        context = CommunicateContext(
            process_group_sizes={
                ScatterMode.SCATTERED: 1,
                ScatterMode.TP_ATTN_FULL: 1,
                ScatterMode.FULL: 8,
            },
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_dp_size=1,
            attn_cp_rank=0,
            attn_cp_size=8,
            tp_size=8,
            tp_rank=0,
        )
        modes = LayerScatterModes(
            layer_input_mode=ScatterMode.SCATTERED,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        identity = torch.nn.Identity()
        with (
            patch.object(CommunicateContext, "init_new", return_value=context),
            patch(
                "sglang.srt.layers.communicator.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
        ):
            communicator = DSACPLayerCommunicator(
                layer_scatter_modes=modes,
                input_layernorm=identity,
                post_attention_layernorm=identity,
                allow_reduce_scatter=True,
                mlp_output_layout=DSAMLPOutputLayout.COMPLETE,
            )

        with self.assertRaisesRegex(RuntimeError, "legacy summed reduce-scatter"):
            communicator._communicate_summable_tensor_pair_fn()
        with (
            patch(
                "sglang.srt.layers.communicator_dsa_cp.dsa_use_prefill_cp",
                return_value=True,
            ),
            self.assertRaisesRegex(RuntimeError, "rank-local CP residual capacity"),
        ):
            communicator.postprocess_layer(
                torch.zeros((15, 4), dtype=torch.bfloat16),
                torch.zeros((2, 4), dtype=torch.bfloat16),
                SimpleNamespace(),
            )

        moe = object.__new__(DeepseekV2MoE)
        local_partial = torch.zeros((16, 4), dtype=torch.bfloat16)
        positions = torch.arange(16, dtype=torch.int64)
        for should_fuse, use_reduce_scatter in ((True, False), (False, True)):
            with self.assertRaisesRegex(RuntimeError, "cannot fuse or reduce-scatter"):
                DeepseekV2MoE._canonicalize_glm52_partial(
                    moe,
                    local_partial,
                    positions,
                    forward_batch=None,
                    fuse_mlp_allreduce=should_fuse,
                    mlp_reduce_scatter=use_reduce_scatter,
                )

    def test_decoder_constructor_declares_canonical_dsa_layout(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")

        from sglang.srt.layers.communicator import LayerScatterModes, ScatterMode
        from sglang.srt.layers.communicator_dsa_cp import DSAMLPOutputLayout
        from sglang.srt.models.deepseek_v2 import (
            DeepseekV2DecoderLayer,
            DeepseekV2MoE,
        )

        class FakeAttention(torch.nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

            def prepare_qkv_latent(self, *_args, **_kwargs):
                raise AssertionError("constructor test must not run attention")

        captured = {}

        class CaptureCommunicator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        def fake_moe_init(instance, *_args, **_kwargs):
            torch.nn.Module.__init__(instance)
            instance._glm52_canonical_contract = True

        modes = LayerScatterModes(
            layer_input_mode=ScatterMode.SCATTERED,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        config = SimpleNamespace(
            hidden_size=4,
            rope_theta=10000.0,
            rope_scaling=None,
            max_position_embeddings=128,
            num_attention_heads=1,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=4,
            kv_lora_rank=2,
            num_hidden_layers=3,
            mlp_layer_types=["sparse", "sparse", "sparse"],
            rms_norm_eps=1e-6,
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v2.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch(
                "sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA",
                FakeAttention,
            ),
            patch.object(DeepseekV2MoE, "__init__", fake_moe_init),
            patch(
                "sglang.srt.models.deepseek_v2.LayerScatterModes.init_new",
                return_value=modes,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.RMSNorm",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "sglang.srt.models.deepseek_v2.DSACPLayerCommunicator",
                CaptureCommunicator,
            ),
        ):
            decoder = DeepseekV2DecoderLayer(
                config,
                layer_id=1,
                dsa_enable_prefill_cp=True,
                glm52_xorl_bi_contract=True,
            )

        self.assertIsInstance(decoder.mlp, DeepseekV2MoE)
        self.assertFalse(captured["allow_reduce_scatter"])
        self.assertEqual(
            captured["mlp_output_layout"],
            DSAMLPOutputLayout.COMPLETE,
        )

    def test_decoder_constructor_declares_complete_dense_tp1_dsa_layout(self):
        if importlib.util.find_spec("sgl_kernel") is None:
            self.skipTest("sgl_kernel is required to import the serving model")

        from sglang.srt.layers.communicator import LayerScatterModes, ScatterMode
        from sglang.srt.layers.communicator_dsa_cp import DSAMLPOutputLayout
        from sglang.srt.models.deepseek_v2 import (
            DeepseekV2DecoderLayer,
            DeepseekV2MLP,
        )

        class FakeAttention(torch.nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

            def prepare_qkv_latent(self, *_args, **_kwargs):
                raise AssertionError("constructor test must not run attention")

        captured = {}

        class CaptureCommunicator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        def fake_mlp_init(instance, *_args, **_kwargs):
            torch.nn.Module.__init__(instance)

        modes = LayerScatterModes(
            layer_input_mode=ScatterMode.SCATTERED,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.SCATTERED,
            middle_residual_mode=ScatterMode.SCATTERED,
            layer_output_mode=ScatterMode.SCATTERED,
        )
        config = SimpleNamespace(
            hidden_size=4,
            intermediate_size=8,
            hidden_act="silu",
            rope_theta=10000.0,
            rope_scaling=None,
            max_position_embeddings=128,
            num_attention_heads=1,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=4,
            kv_lora_rank=2,
            num_hidden_layers=3,
            mlp_layer_types=["dense", "dense", "dense"],
            rms_norm_eps=1e-6,
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v2.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch(
                "sglang.srt.models.deepseek_v2.enable_moe_dense_fully_dp",
                return_value=True,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA",
                FakeAttention,
            ),
            patch.object(DeepseekV2MLP, "__init__", fake_mlp_init),
            patch(
                "sglang.srt.models.deepseek_v2.LayerScatterModes.init_new",
                return_value=modes,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.RMSNorm",
                return_value=torch.nn.Identity(),
            ),
            patch(
                "sglang.srt.models.deepseek_v2.DSACPLayerCommunicator",
                CaptureCommunicator,
            ),
        ):
            decoder = DeepseekV2DecoderLayer(
                config,
                layer_id=0,
                dsa_enable_prefill_cp=True,
                glm52_xorl_bi_contract=True,
            )

        self.assertIsInstance(decoder.mlp, DeepseekV2MLP)
        self.assertFalse(captured["allow_reduce_scatter"])
        self.assertEqual(
            captured["mlp_output_layout"],
            DSAMLPOutputLayout.COMPLETE,
        )

    def test_noncanonical_dsa_layout_keeps_legacy_reduce_scatter(self):
        from sglang.srt.layers.communicator import (
            CommunicateContext,
            LayerScatterModes,
            ScatterMode,
        )
        from sglang.srt.layers.communicator_dsa_cp import DSACPLayerCommunicator

        context = CommunicateContext(
            process_group_sizes={
                ScatterMode.SCATTERED: 1,
                ScatterMode.TP_ATTN_FULL: 1,
                ScatterMode.FULL: 8,
            },
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_dp_size=1,
            attn_cp_rank=3,
            attn_cp_size=8,
            tp_size=8,
            tp_rank=3,
        )
        modes = LayerScatterModes(
            layer_input_mode=ScatterMode.SCATTERED,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.SCATTERED,
            layer_output_mode=ScatterMode.SCATTERED,
        )
        with (
            patch.object(CommunicateContext, "init_new", return_value=context),
            patch(
                "sglang.srt.layers.communicator.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
        ):
            communicator = DSACPLayerCommunicator(
                layer_scatter_modes=modes,
                input_layernorm=torch.nn.Identity(),
                post_attention_layernorm=torch.nn.Identity(),
                allow_reduce_scatter=True,
            )

        full = torch.arange(16 * 4, dtype=torch.bfloat16).reshape(16, 4)
        residual = torch.zeros((2, 4), dtype=torch.bfloat16)
        with (
            patch(
                "sglang.srt.layers.communicator.dsa_use_prefill_cp",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communicator_dsa_cp.dsa_use_prefill_cp",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communicator_dsa_cp.get_parallel",
                return_value=SimpleNamespace(
                    attn_dp_size=1,
                    attn_tp_size=1,
                    attn_cp_size=8,
                    attn_cp_rank=3,
                ),
            ),
            patch(
                "sglang.srt.layers.communicator_dsa_cp.attn_cp_reduce_scatter_tensor"
            ) as legacy_reduce_scatter,
        ):
            self.assertTrue(communicator.should_use_reduce_scatter(SimpleNamespace()))
            local, returned_residual = communicator.postprocess_layer(
                full, residual, SimpleNamespace()
            )

        legacy_reduce_scatter.assert_called_once()
        self.assertTrue(torch.equal(local, full[6:8]))
        self.assertIs(returned_residual, residual)

    def test_balanced_reference_and_plan_for_power_of_two_contributors(self):
        for contributors in (2, 4, 8, 16):
            plan = SamplerParallelPlan.primitive(contributors)
            self.assertEqual(plan.version, GLM52_CANONICAL_MOE_VERSION)
            slots = CanonicalRowSlots.from_positions(
                torch.tensor([0, 3, 8], dtype=torch.int64),
                capacity=5,
            )
            partials = torch.zeros(
                (contributors, slots.capacity, 2), dtype=torch.bfloat16
            )
            for ordinal in range(contributors):
                partials[ordinal, :, 0] = [4096, -4096, 1, 0, 0][ordinal % 5]
                partials[ordinal, :, 1] = ordinal + 1
            expected = partials
            while expected.shape[0] > 1:
                expected = (expected[0::2] + expected[1::2]).to(torch.bfloat16)
            expected = expected[0]
            expected[~slots.valid_mask] = 0
            self.assertTrue(
                torch.equal(canonical_moe_reference(partials, slots), expected)
            )

        production = SamplerParallelPlan.glm52()
        permuted = replace(production, physical_to_logical=tuple(reversed(range(8))))
        self.assertNotEqual(permuted.identity, production.identity)

        with (
            patch("torch.distributed.get_world_size", return_value=8),
            patch(
                "torch.distributed.get_process_group_ranks",
                return_value=list(reversed(range(8))),
            ),
            self.assertRaisesRegex(RuntimeError, "physical rank order"),
        ):
            production.validate_runtime(
                group=object(),
                launcher_tp_size=8,
                effective_dense_tp=1,
                pp_size=1,
                ep_size=8,
                attention_cp_size=8,
            )

        production_16 = SamplerParallelPlan.glm52(contributors=16)
        self.assertEqual(production_16.contributor_count, 16)
        self.assertEqual(production_16.launcher_tp_size, 16)
        self.assertEqual(production_16.ep_size, 16)
        self.assertEqual(production_16.attention_cp_size, 16)
        self.assertEqual(production_16.attention_dp_size, 1)
        with (
            patch("torch.distributed.get_world_size", return_value=16),
            patch(
                "torch.distributed.get_process_group_ranks",
                return_value=list(range(16)),
            ),
        ):
            production_16.validate_runtime(
                group=object(),
                launcher_tp_size=16,
                effective_dense_tp=1,
                pp_size=1,
                ep_size=16,
                attention_cp_size=16,
            )

        production_dp16 = SamplerParallelPlan.glm52(
            contributors=16, attention_dp_size=16
        )
        self.assertEqual(production_dp16.attention_cp_size, 1)
        self.assertEqual(production_dp16.attention_dp_size, 16)
        production_dp16.validate_cuda_graph_policy(disable_cuda_graph=False)
        production_dp16.validate_cuda_graph_policy(disable_cuda_graph=True)
        with (
            patch("torch.distributed.get_world_size", return_value=16),
            patch(
                "torch.distributed.get_process_group_ranks",
                return_value=list(range(16)),
            ),
        ):
            production_dp16.validate_runtime(
                group=object(),
                launcher_tp_size=16,
                effective_dense_tp=1,
                pp_size=1,
                ep_size=16,
                attention_cp_size=1,
                attention_dp_size=16,
            )
        for dp_size, cp_size in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
            mixed = SamplerParallelPlan.glm52(
                contributors=16, attention_dp_size=dp_size
            )
            self.assertEqual(mixed.attention_cp_size, cp_size)
            self.assertEqual(mixed.attention_dp_size, dp_size)
        mixed = replace(
            production_dp16,
            attention_cp_size=4,
            attention_dp_size=4,
        )
        self.assertEqual(mixed.attention_cp_size * mixed.attention_dp_size, 16)

        with (
            patch("torch.distributed.get_world_size", return_value=8),
            self.assertRaisesRegex(RuntimeError, "Resolved GLM-5.2 sampler topology"),
        ):
            production.validate_runtime(
                group=object(),
                launcher_tp_size=4,
                effective_dense_tp=1,
                pp_size=1,
                ep_size=8,
                attention_cp_size=8,
            )

        stage_one = SamplerParallelPlan.glm52(
            pp_size=2,
            pp_rank=1,
            physical_ranks=tuple(range(8, 16)),
        )
        self.assertEqual(stage_one.global_world_size, 16)
        self.assertEqual(stage_one.physical_ranks, tuple(range(8, 16)))
        stage_one_bound = replace(stage_one, stage_layer_range=(38, 78))
        self.assertNotEqual(stage_one.identity, stage_one_bound.identity)
        stage_one_bound.validate_cuda_graph_policy(disable_cuda_graph=False)
        stage_one_bound.validate_cuda_graph_policy(disable_cuda_graph=True)
        SamplerParallelPlan.glm52().validate_cuda_graph_policy(disable_cuda_graph=False)
        with (
            patch(
                "torch.distributed.get_world_size",
                side_effect=lambda group=None: 16 if group is None else 8,
            ),
            patch(
                "torch.distributed.get_process_group_ranks",
                return_value=list(range(8, 16)),
            ),
        ):
            stage_one_bound.validate_runtime(
                group=object(),
                launcher_tp_size=8,
                effective_dense_tp=1,
                pp_size=2,
                ep_size=8,
                attention_cp_size=8,
            )
        noncontiguous = SamplerParallelPlan.glm52(
            pp_size=3,
            pp_rank=1,
            physical_ranks=(1, 4, 7, 10, 13, 16, 19, 22),
        )
        self.assertEqual(noncontiguous.pp_size, 3)
        self.assertEqual(noncontiguous.ep_size, 8)
        with self.assertRaisesRegex(ValueError, "EP must equal"):
            SamplerParallelPlan.glm52(ep_size=2)

    def test_mocked_raw_transport_owner_fold_and_replication(self):
        contributors = 8
        plan = SamplerParallelPlan.glm52()
        positions = torch.tensor([0, 7, 8, 15, 22], dtype=torch.int64)
        slots = CanonicalRowSlots.from_positions(positions, capacity=8)
        partials = torch.empty((contributors, slots.capacity, 3), dtype=torch.bfloat16)
        for source in range(contributors):
            partials[source, :, 0] = source + 1
            partials[source, :, 1] = 4096.0 if source == 0 else -1.0
            partials[source, :, 2] = torch.arange(slots.capacity)
        expected = canonical_moe_reference(partials, slots)
        owners = slots.owners(contributors)

        def fake_all_to_all(receive, _send, *, group):
            view = receive.view(contributors, slots.capacity, 3)
            view.zero_()
            for source in range(contributors):
                view[source, slots.valid_mask & (owners == 0)] = partials[
                    source, slots.valid_mask & (owners == 0)
                ]

        def fake_all_gather(gathered, _owner_values, *, group):
            view = gathered.view(contributors, slots.capacity, 3)
            view.zero_()
            for owner in range(contributors):
                view[owner, slots.valid_mask & (owners == owner)] = expected[
                    slots.valid_mask & (owners == owner)
                ]

        with (
            patch("torch.distributed.get_world_size", return_value=contributors),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "torch.distributed.get_process_group_ranks",
                return_value=list(range(contributors)),
            ),
            patch("torch.distributed.all_to_all_single", side_effect=fake_all_to_all),
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
        ):
            workspace = CanonicalMoEWorkspace.allocate(
                partials[0],
                plan=plan,
                group=object(),
            )
            output = canonicalize_glm52_local_partial(
                partials[0],
                slots,
                plan=plan,
                group=object(),
                layer_id=3,
                workspace=workspace,
            )
            output.raise_for_status()
        self.assertTrue(torch.equal(output.values, expected))
        self.assertTrue(
            torch.equal(output.owner_mask, slots.valid_mask & (owners == 0))
        )

    def test_mocked_v3_transports_preserve_dense_reference_bytes(self):
        contributors = 8
        capacity = 16
        local_capacity = capacity // contributors
        rank = 3
        plan = SamplerParallelPlan.glm52()
        slots = CanonicalRowSlots.from_positions(torch.arange(14), capacity=capacity)
        partials = torch.empty((contributors, capacity, 3), dtype=torch.bfloat16)
        for source in range(contributors):
            partials[source, :, 0] = source + 1
            partials[source, :, 1] = 4096.0 if source == 0 else -1.0
            partials[source, :, 2] = torch.arange(capacity) * (source + 1)
        expected = canonical_moe_reference(partials, slots)
        masked_rank = partials[rank].clone()
        masked_rank[~slots.valid_mask] = 0

        def fake_all_gather(gathered, local, *, group):
            self.assertTrue(torch.equal(local, masked_rank))
            gathered.view_as(partials).copy_(partials)

        def fake_all_to_all(received, local, *, group):
            self.assertTrue(torch.equal(local, masked_rank))
            start = rank * local_capacity
            end = start + local_capacity
            received.view(contributors, local_capacity, 3).copy_(partials[:, start:end])

        with (
            patch("torch.distributed.get_world_size", return_value=contributors),
            patch("torch.distributed.get_rank", return_value=rank),
            patch(
                "torch.distributed.get_process_group_ranks",
                return_value=list(range(contributors)),
            ),
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
            patch(
                "torch.distributed.all_to_all_single",
                side_effect=fake_all_to_all,
            ),
        ):
            replicated = canonicalize_glm52_local_partial_v3(
                partials[rank],
                slots,
                plan=plan,
                group=object(),
                layer_id=3,
                distribution=CanonicalDistribution.REPLICATED_CANONICAL,
            )
            sharded = canonicalize_glm52_local_partial_v3(
                partials[rank],
                slots,
                plan=plan,
                group=object(),
                layer_id=3,
                distribution=CanonicalDistribution.CONSUMER_SHARDED,
            )
            graph_replicated = canonicalize_glm52_local_partial_v3(
                partials[rank],
                slots,
                plan=plan,
                group=object(),
                layer_id=3,
                distribution=CanonicalDistribution.REPLICATED_CANONICAL,
                graph_capture=True,
            )

        replicated.raise_for_status()
        sharded.raise_for_status()
        start = rank * local_capacity
        end = start + local_capacity
        self.assertTrue(torch.equal(replicated.values, expected))
        self.assertTrue(torch.equal(sharded.values, expected[start:end]))
        self.assertTrue(torch.equal(graph_replicated.values, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph replay requires CUDA")
    def test_reference_cuda_graph_replay_has_stable_slots(self):
        slots = CanonicalRowSlots.from_positions(
            torch.tensor([0, 1, 2], device="cuda"),
            capacity=8,
        )
        partials = torch.randn((8, 8, 4), device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = canonical_moe_reference(partials, slots)
        first = result.clone()
        partials.add_(1)
        graph.replay()
        self.assertFalse(torch.equal(first, result))
        self.assertEqual(result.shape[0], slots.capacity)


if __name__ == "__main__":
    unittest.main()
