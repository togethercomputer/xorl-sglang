"""CPU/mock tests for GLM-5.2 exact architecture and MoE owners."""

import threading
import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import ModelConfig  # noqa: E402
from sglang.srt.distributed.canonical_moe import (  # noqa: E402
    CanonicalDistribution,
    CanonicalMoEV3ScratchPool,
    CanonicalMoEV3Workspace,
    CanonicalRowSlots,
    SamplerParallelPlan,
    canonicalize_glm52_local_partial_v3,
)
from sglang.srt.layers import layernorm as layernorm_module  # noqa: E402
from sglang.srt.layers.communicator import (  # noqa: E402
    CommunicateContext,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.communicator_dsa_cp import (  # noqa: E402
    DSACPLayerCommunicator,
    DSAMLPOutputLayout,
)
from sglang.srt.models import deepseek_v2  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlm52ExactOwners(CustomTestCase):
    @staticmethod
    def _decoder_config(*, exact: bool):
        config = SimpleNamespace(
            hidden_size=4,
            rope_theta=10000.0,
            rope_scaling=None,
            max_position_embeddings=128,
            num_attention_heads=1,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=4,
            q_lora_rank=2,
            kv_lora_rank=2,
            num_hidden_layers=2,
            mlp_layer_types=["sparse", "sparse"],
            indexer_types=["full", "full"],
            n_routed_experts=8,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            rms_norm_eps=1e-6,
            _glm52_exact_mode=exact,
        )
        return config

    def test_fresh_exact_config_runs_real_input_norm_through_dsa_communicator(self):
        class FakeAttention(torch.nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

            def prepare_qkv_latent(self, *_args, **_kwargs):
                raise AssertionError("constructor test must not run attention")

        def fake_moe_init(instance, *, config, **_kwargs):
            torch.nn.Module.__init__(instance)
            instance._glm52_canonical_contract = config._glm52_exact_mode

        config = self._decoder_config(exact=False)
        runner_config = SimpleNamespace(hf_config=config, hf_text_config=config)
        server_args = ServerArgs(model_path="dummy")
        server_args.glm52_exact_mode = True
        build_from_server_args = ModelConfig.from_server_args
        with patch(
            "sglang.srt.configs.model_config.ModelConfig",
            return_value=runner_config,
        ):
            fresh_config = build_from_server_args(server_args).hf_config

        modes = LayerScatterModes(
            layer_input_mode=ScatterMode.SCATTERED,
            attn_mode=ScatterMode.SCATTERED,
            mlp_mode=ScatterMode.SCATTERED,
            middle_residual_mode=ScatterMode.SCATTERED,
            layer_output_mode=ScatterMode.SCATTERED,
        )
        context = CommunicateContext(
            process_group_sizes={mode: 1 for mode in ScatterMode},
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_dp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            tp_size=1,
            tp_rank=0,
        )
        exact_contract = deepseek_v2._uses_glm52_exact_contract(fresh_config)
        with (
            patch.object(
                deepseek_v2,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch.object(deepseek_v2, "DeepseekV2AttentionMLA", FakeAttention),
            patch.object(deepseek_v2.DeepseekV2MoE, "__init__", fake_moe_init),
            patch.object(
                deepseek_v2.LayerScatterModes,
                "init_new",
                return_value=modes,
            ),
            patch.object(CommunicateContext, "init_new", return_value=context),
            patch(
                "sglang.srt.layers.communicator.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
        ):
            decoder = deepseek_v2.DeepseekV2DecoderLayer(
                fresh_config,
                layer_id=0,
                dsa_enable_prefill_cp=True,
                glm52_xorl_bi_contract=exact_contract,
            )

        self.assertTrue(exact_contract)
        self.assertIsInstance(decoder.layer_communicator, DSACPLayerCommunicator)
        self.assertEqual(
            decoder.input_layernorm.batch_invariant_family,
            "serving_no_residual",
        )
        self.assertEqual(
            decoder.post_attention_layernorm.batch_invariant_family,
            "serving_residual_tree",
        )
        self.assertEqual(
            decoder.layer_communicator.mlp_output_layout,
            DSAMLPOutputLayout.COMPLETE,
        )

        hidden_states = torch.zeros((1, 4), dtype=torch.bfloat16)
        decoder.input_layernorm.to(dtype=torch.bfloat16)
        decoder.input_layernorm._forward_method = decoder.input_layernorm.forward_cuda
        with (
            patch.object(
                layernorm_module,
                "is_batch_invariant_mode_enabled",
                return_value=True,
            ),
            patch.object(
                layernorm_module,
                "get_global_server_args",
                return_value=SimpleNamespace(glm52_exact_mode=True),
            ),
            patch.object(
                layernorm_module,
                "rms_norm_v2",
                return_value=hidden_states,
            ) as exact_norm,
        ):
            normalized, residual = decoder.layer_communicator.prepare_attn(
                hidden_states,
                None,
                SimpleNamespace(),
            )

        self.assertIs(normalized, hidden_states)
        self.assertIs(residual, hidden_states)
        exact_norm.assert_called_once()

    def test_decoder_declares_exact_norm_and_complete_cp_owners_only_in_exact_mode(
        self,
    ):
        norm_calls = []
        communicator_calls = []

        class FakeAttention(torch.nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

            def prepare_qkv_latent(self, *_args, **_kwargs):
                raise AssertionError("constructor test must not run attention")

        def fake_moe_init(instance, *, config, **_kwargs):
            torch.nn.Module.__init__(instance)
            instance._glm52_canonical_contract = config._glm52_exact_mode

        def fake_norm(*_args, **kwargs):
            norm_calls.append(kwargs.get("batch_invariant_family"))
            return torch.nn.Identity()

        class CaptureCommunicator:
            def __init__(self, **kwargs):
                communicator_calls.append(kwargs)

        modes = SimpleNamespace(mlp_mode=ScatterMode.FULL)
        with (
            patch.object(
                deepseek_v2,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch.object(deepseek_v2, "DeepseekV2AttentionMLA", FakeAttention),
            patch.object(deepseek_v2.DeepseekV2MoE, "__init__", fake_moe_init),
            patch.object(deepseek_v2.LayerScatterModes, "init_new", return_value=modes),
            patch.object(deepseek_v2, "RMSNorm", side_effect=fake_norm),
            patch.object(deepseek_v2, "DSACPLayerCommunicator", CaptureCommunicator),
        ):
            deepseek_v2.DeepseekV2DecoderLayer(
                self._decoder_config(exact=True),
                layer_id=1,
                dsa_enable_prefill_cp=True,
                glm52_xorl_bi_contract=True,
            )
            deepseek_v2.DeepseekV2DecoderLayer(
                self._decoder_config(exact=False),
                layer_id=1,
                dsa_enable_prefill_cp=True,
                glm52_xorl_bi_contract=False,
            )

        self.assertEqual(
            norm_calls,
            [
                "serving_residual_tree",
                "serving_residual_tree",
                None,
                None,
            ],
        )
        self.assertFalse(communicator_calls[0]["allow_reduce_scatter"])
        self.assertEqual(
            communicator_calls[0]["mlp_output_layout"],
            DSAMLPOutputLayout.COMPLETE,
        )
        self.assertTrue(communicator_calls[1]["allow_reduce_scatter"])
        self.assertEqual(
            communicator_calls[1]["mlp_output_layout"],
            DSAMLPOutputLayout.LEGACY_PARTIAL,
        )

    def test_canonical_owner_selects_decode_v3b_and_rejects_external_reduction(self):
        moe = object.__new__(deepseek_v2.DeepseekV2MoE)
        torch.nn.Module.__init__(moe)
        moe.tp_size = 8
        moe.moe_ep_size = 8
        moe.glm52_parallel_plan = SamplerParallelPlan.glm52()
        moe.gate = SimpleNamespace(
            e_score_correction_bias=torch.zeros(8, dtype=torch.float32),
            dsa_enable_prefill_cp=True,
            mla_enable_prefill_cp=False,
        )
        moe._glm52_canonical_transport = "canonical_v3b"
        moe._canonical_v3_workspaces = {}
        moe._glm52_deferred_status_book = None
        moe.layer_id = 7

        local_partial = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        positions = torch.arange(4, dtype=torch.int64)
        expected = local_partial + 1
        output = SimpleNamespace(
            values=expected,
            contract_status=torch.zeros((), dtype=torch.int32),
            raise_for_status=MagicMock(),
        )
        group = object()
        workspace = object()
        with (
            patch.object(
                deepseek_v2,
                "get_parallel",
                return_value=SimpleNamespace(
                    tp_group=SimpleNamespace(device_group=group)
                ),
            ),
            patch.object(deepseek_v2, "get_is_capture_mode", return_value=False),
            patch.object(
                deepseek_v2.CanonicalMoEV3Workspace,
                "allocate",
                return_value=workspace,
            ),
            patch.object(
                deepseek_v2,
                "canonicalize_glm52_local_partial_v3b",
                return_value=output,
            ) as decode_v3b,
            patch.object(
                deepseek_v2, "canonicalize_glm52_local_partial_v3"
            ) as prefill_v3,
        ):
            actual = moe._canonicalize_glm52_partial(
                local_partial,
                positions,
                forward_batch=None,
                fuse_mlp_allreduce=False,
                mlp_reduce_scatter=False,
            )

        self.assertIs(actual, expected)
        decode_v3b.assert_called_once()
        prefill_v3.assert_not_called()
        output.raise_for_status.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "cannot fuse or reduce-scatter"):
            moe._canonicalize_glm52_partial(
                local_partial,
                positions,
                forward_batch=None,
                fuse_mlp_allreduce=True,
                mlp_reduce_scatter=False,
            )

    def test_mixed_dp_cp_prefill_uses_model_shared_replicated_scratch(self):
        moe = object.__new__(deepseek_v2.DeepseekV2MoE)
        torch.nn.Module.__init__(moe)
        moe.tp_size = 16
        moe.moe_ep_size = 16
        moe.glm52_parallel_plan = SamplerParallelPlan.glm52(
            contributors=16,
            attention_dp_size=4,
        )
        moe.gate = SimpleNamespace(
            e_score_correction_bias=torch.zeros(16, dtype=torch.float32),
            dsa_enable_prefill_cp=True,
            mla_enable_prefill_cp=False,
        )
        moe._glm52_canonical_transport = "auto"
        moe._glm52_deferred_status_book = None
        moe.layer_id = 7

        local_partial = torch.arange(64, dtype=torch.bfloat16).reshape(16, 4)
        positions = torch.arange(16, dtype=torch.int64)
        output = SimpleNamespace(
            values=local_partial + 1,
            contract_status=torch.zeros((), dtype=torch.int32),
            raise_for_status=MagicMock(),
        )
        group = object()
        workspace = object()
        scratch_pool = MagicMock()
        scratch_pool.lease.return_value = nullcontext(workspace)
        moe._glm52_mixed_prefill_scratch_pool = scratch_pool
        forward_batch = object()
        with (
            patch.object(deepseek_v2, "dsa_use_prefill_cp", return_value=True),
            patch.object(deepseek_v2, "mla_use_prefill_cp", return_value=False),
            patch.object(
                deepseek_v2,
                "get_parallel",
                return_value=SimpleNamespace(
                    tp_group=SimpleNamespace(device_group=group)
                ),
            ),
            patch.object(deepseek_v2, "get_is_capture_mode", return_value=False),
            patch.object(
                moe,
                "_get_canonical_v3_workspace",
                side_effect=AssertionError(
                    "mixed eager prefill must use shared scratch"
                ),
            ) as get_workspace,
            patch.object(
                deepseek_v2,
                "canonicalize_glm52_local_partial_v3",
                return_value=output,
            ) as prefill_v3,
        ):
            actual = moe._canonicalize_glm52_partial(
                local_partial,
                positions,
                forward_batch=forward_batch,
                fuse_mlp_allreduce=False,
                mlp_reduce_scatter=False,
            )

        self.assertIs(actual, output.values)
        get_workspace.assert_not_called()
        scratch_pool.lease.assert_called_once_with(
            local_partial,
            plan=moe.glm52_parallel_plan,
            group=group,
            distribution=CanonicalDistribution.REPLICATED_CANONICAL,
        )
        prefill_v3.assert_called_once()
        output.raise_for_status.assert_called_once()

    def test_mixed_dp_cp_graph_capture_keeps_pinned_layer_workspace(self):
        moe = object.__new__(deepseek_v2.DeepseekV2MoE)
        torch.nn.Module.__init__(moe)
        moe.tp_size = 16
        moe.moe_ep_size = 16
        moe.glm52_parallel_plan = SamplerParallelPlan.glm52(
            contributors=16,
            attention_dp_size=4,
        )
        moe.gate = SimpleNamespace(
            e_score_correction_bias=torch.zeros(16, dtype=torch.float32),
            dsa_enable_prefill_cp=True,
            mla_enable_prefill_cp=False,
        )
        moe._glm52_canonical_transport = "auto"
        moe._glm52_deferred_status_book = None
        moe._glm52_mixed_prefill_scratch_pool = MagicMock()
        moe.layer_id = 7

        local_partial = torch.zeros((16, 4), dtype=torch.bfloat16)
        positions = torch.arange(16, dtype=torch.int64)
        output = SimpleNamespace(
            values=local_partial,
            contract_status=torch.zeros((), dtype=torch.int32),
            raise_for_status=MagicMock(),
        )
        workspace = object()
        group = object()
        with (
            patch.object(deepseek_v2, "dsa_use_prefill_cp", return_value=True),
            patch.object(deepseek_v2, "mla_use_prefill_cp", return_value=False),
            patch.object(
                deepseek_v2,
                "get_parallel",
                return_value=SimpleNamespace(
                    tp_group=SimpleNamespace(device_group=group)
                ),
            ),
            patch.object(deepseek_v2, "get_is_capture_mode", return_value=True),
            patch.object(
                moe,
                "_get_canonical_v3_workspace",
                return_value=workspace,
            ) as get_workspace,
            patch.object(
                deepseek_v2,
                "canonicalize_glm52_local_partial_v3",
                return_value=output,
            ),
        ):
            first = moe._canonicalize_glm52_partial(
                local_partial,
                positions,
                forward_batch=object(),
                fuse_mlp_allreduce=False,
                mlp_reduce_scatter=False,
            )
            second = moe._canonicalize_glm52_partial(
                local_partial,
                positions,
                forward_batch=object(),
                fuse_mlp_allreduce=False,
                mlp_reduce_scatter=False,
            )

        self.assertIs(first, output.values)
        self.assertIs(second, output.values)
        self.assertEqual(get_workspace.call_count, 2)
        self.assertTrue(
            all(call.kwargs["capture_mode"] for call in get_workspace.call_args_list)
        )
        moe._glm52_mixed_prefill_scratch_pool.lease.assert_not_called()
        output.raise_for_status.assert_not_called()

    def test_complete_cp_owner_selects_rows_without_legacy_sum_and_fails_closed(self):
        communicator = object.__new__(DSACPLayerCommunicator)
        communicator.mlp_output_layout = DSAMLPOutputLayout.COMPLETE
        communicator.layer_scatter_modes = SimpleNamespace(mlp_mode=ScatterMode.FULL)
        communicator._context = SimpleNamespace(attn_cp_size=4, attn_cp_rank=2)

        full = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2)
        residual = torch.zeros((2, 2), dtype=torch.bfloat16)
        forward_batch = SimpleNamespace()
        with (
            patch(
                "sglang.srt.layers.communicator_dsa_cp.dsa_use_prefill_cp",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communicator_dsa_cp.mla_use_prefill_cp",
                return_value=False,
            ),
        ):
            local, returned_residual = communicator.postprocess_layer(
                full, residual, forward_batch
            )
            with self.assertRaisesRegex(RuntimeError, "rank-local CP residual"):
                communicator.postprocess_layer(full[:-1], residual, forward_batch)

        self.assertTrue(torch.equal(local, full[4:6]))
        self.assertIs(returned_residual, residual)
        self.assertFalse(communicator.should_use_reduce_scatter(forward_batch))

        communicator.mlp_output_layout = DSAMLPOutputLayout.LEGACY_PARTIAL
        sentinel = (object(), object())
        with patch.object(
            LayerCommunicator, "postprocess_layer", return_value=sentinel
        ) as legacy:
            self.assertIs(
                communicator.postprocess_layer(full, residual, forward_batch), sentinel
            )
        legacy.assert_called_once_with(full, residual, forward_batch)


class TestCanonicalWorkspaceRetention(CustomTestCase):
    """Single-slot canonical-fold workspace cache for prefill-CP.

    Guarded bug: the workspace dict was keyed by CP-padded capacity and
    never evicted, so every distinct 256-row request bucket retained another
    workspace set. Eviction must be single-slot for eager CONSUMER_SHARDED
    entries only:
    capture-minted entries are replayed by address inside CUDA graphs and
    must survive every eviction attempt.
    """

    @staticmethod
    def _moe():
        moe = object.__new__(deepseek_v2.DeepseekV2MoE)
        torch.nn.Module.__init__(moe)
        moe._canonical_v3_workspaces = {}
        moe._canonical_v3_pinned_keys = set()
        moe.layer_id = 7
        return moe

    @staticmethod
    def _key(distribution, capacity):
        return (distribution.value, capacity, (4,), torch.bfloat16, "cpu")

    def _mint(
        self,
        moe,
        distribution,
        capacity,
        capture_mode,
        *,
        cache_eager=True,
    ):
        with patch.object(
            deepseek_v2.CanonicalMoEV3Workspace, "allocate", return_value=object()
        ) as allocate:
            workspace = moe._get_canonical_v3_workspace(
                self._key(distribution, capacity),
                torch.zeros(capacity, 4, dtype=torch.bfloat16),
                plan=object(),
                group=object(),
                distribution=distribution,
                capture_mode=capture_mode,
                cache_eager=cache_eager,
            )
        return workspace, allocate

    def test_eager_prefill_cp_workspace_is_single_slot(self):
        moe = self._moe()
        sharded = CanonicalDistribution.CONSUMER_SHARDED
        self._mint(moe, sharded, 2560, capture_mode=False)
        self._mint(moe, sharded, 512, capture_mode=False)
        self.assertEqual(list(moe._canonical_v3_workspaces), [self._key(sharded, 512)])

    def test_repeat_bucket_hits_cache_without_reallocation(self):
        moe = self._moe()
        sharded = CanonicalDistribution.CONSUMER_SHARDED
        first, _ = self._mint(moe, sharded, 2560, capture_mode=False)
        again, allocate = self._mint(moe, sharded, 2560, capture_mode=False)
        self.assertIs(again, first)
        allocate.assert_not_called()

    def test_eager_workspace_reused_during_capture_becomes_pinned(self):
        moe = self._moe()
        sharded = CanonicalDistribution.CONSUMER_SHARDED
        original_key = self._key(sharded, 256)

        first, _ = self._mint(moe, sharded, 256, capture_mode=False)
        captured, allocate = self._mint(moe, sharded, 256, capture_mode=True)

        self.assertIs(captured, first)
        allocate.assert_not_called()
        self.assertIn(original_key, moe._canonical_v3_pinned_keys)

        self._mint(moe, sharded, 512, capture_mode=False)
        self.assertIn(original_key, moe._canonical_v3_workspaces)

    def test_capture_minted_and_cached_replicated_entries_survive_eviction(self):
        moe = self._moe()
        sharded = CanonicalDistribution.CONSUMER_SHARDED
        replicated = CanonicalDistribution.REPLICATED_CANONICAL
        # Captured decode bucket and a captured sharded entry: pinned.
        self._mint(
            moe,
            replicated,
            16,
            capture_mode=True,
            cache_eager=False,
        )
        self._mint(moe, sharded, 256, capture_mode=True)
        # Eager replicated entry: outside the eviction scope.
        self._mint(moe, replicated, 32, capture_mode=False)
        # Two eager sharded mints: single-slot among themselves only.
        self._mint(moe, sharded, 2560, capture_mode=False)
        self._mint(moe, sharded, 512, capture_mode=False)
        self.assertEqual(
            set(moe._canonical_v3_workspaces),
            {
                self._key(replicated, 16),
                self._key(sharded, 256),
                self._key(replicated, 32),
                self._key(sharded, 512),
            },
        )

    def test_eager_replicated_prefill_workspace_is_transient(self):
        moe = self._moe()
        replicated = CanonicalDistribution.REPLICATED_CANONICAL

        first, first_allocate = self._mint(
            moe,
            replicated,
            1024,
            capture_mode=False,
            cache_eager=False,
        )
        second, second_allocate = self._mint(
            moe,
            replicated,
            1024,
            capture_mode=False,
            cache_eager=False,
        )

        self.assertIsNot(first, second)
        first_allocate.assert_called_once()
        second_allocate.assert_called_once()
        self.assertEqual(moe._canonical_v3_workspaces, {})


class TestMixedPrefillScratchPool(CustomTestCase):
    """Regression coverage for the 4096-token mixed-DP/CP prefill OOM.

    The failed production request kept contributor-sized scratch planes active
    across sparse layers. The shared slot must keep those planes O(1) across
    the complete 75-layer GLM sparse stack without retaining outputs.
    """

    @staticmethod
    def _plan():
        return SamplerParallelPlan.glm52(contributors=4, attention_dp_size=2)

    @staticmethod
    def _runtime(plan):
        return patch.multiple(
            "torch.distributed",
            get_world_size=MagicMock(return_value=plan.global_world_size),
            get_process_group_ranks=MagicMock(return_value=list(plan.physical_ranks)),
        )

    @staticmethod
    def _scratch_signature(workspace):
        names = (
            "masked_input",
            "collective",
            "ordered_sources",
            "folded",
            "zero",
            "logical_to_group",
            "status",
        )
        return tuple(
            getattr(workspace, name).untyped_storage().data_ptr() for name in names
        )

    def test_one_slot_serves_all_75_layers_without_retaining_results(self):
        plan = self._plan()
        pool = CanonicalMoEV3ScratchPool()
        local = torch.zeros((8, 4), dtype=torch.bfloat16)
        group = object()
        scratch_signatures = []
        results = []

        with self._runtime(plan):
            for _layer_id in range(75):
                with pool.lease(
                    local,
                    plan=plan,
                    group=group,
                    distribution=CanonicalDistribution.REPLICATED_CANONICAL,
                ) as workspace:
                    scratch_signatures.append(self._scratch_signature(workspace))
                    results.append(workspace.result)

        self.assertEqual(len(set(scratch_signatures)), 1)
        self.assertEqual(
            len({result.untyped_storage().data_ptr() for result in results}),
            75,
        )
        scratch_ptrs = set(scratch_signatures[0])
        self.assertTrue(
            all(
                result.untyped_storage().data_ptr() not in scratch_ptrs
                for result in results
            )
        )

    def test_model_binds_one_pool_to_every_local_sparse_layer(self):
        mlps = [SimpleNamespace() for _ in range(75)]
        mixed = self._plan()
        pool = deepseek_v2.DeepseekV2Model._bind_glm52_mixed_prefill_scratch_pool(
            mixed,
            mlps,
        )

        self.assertIsInstance(pool, CanonicalMoEV3ScratchPool)
        self.assertEqual(
            {id(mlp._glm52_mixed_prefill_scratch_pool) for mlp in mlps},
            {id(pool)},
        )

        for attention_dp_size in (1, 4):
            nonmixed = SamplerParallelPlan.glm52(
                contributors=4,
                attention_dp_size=attention_dp_size,
            )
            with self.subTest(attention_dp_size=attention_dp_size):
                self.assertIsNone(
                    deepseek_v2.DeepseekV2Model._bind_glm52_mixed_prefill_scratch_pool(
                        nonmixed,
                        mlps,
                    )
                )
                self.assertTrue(
                    all(mlp._glm52_mixed_prefill_scratch_pool is None for mlp in mlps)
                )

    def test_pool_serializes_callers_and_reuses_scratch_after_exception(self):
        plan = self._plan()
        pool = CanonicalMoEV3ScratchPool()
        local = torch.zeros((8, 4), dtype=torch.bfloat16)
        group = object()
        first_entered = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        errors = []
        pointers = []

        def first_caller():
            try:
                with pool.lease(
                    local,
                    plan=plan,
                    group=group,
                    distribution=CanonicalDistribution.REPLICATED_CANONICAL,
                ) as workspace:
                    pointers.append(self._scratch_signature(workspace))
                    first_entered.set()
                    second_started.wait(timeout=2)
                    if second_entered.is_set():
                        raise AssertionError(
                            "second caller entered a leased scratch slot"
                        )
                    release_first.wait(timeout=2)
                    raise RuntimeError("injected canonicalization failure")
            except RuntimeError as error:
                if str(error) != "injected canonicalization failure":
                    errors.append(error)
            except BaseException as error:  # pragma: no cover - thread handoff
                errors.append(error)

        def second_caller():
            try:
                first_entered.wait(timeout=2)
                second_started.set()
                with pool.lease(
                    local,
                    plan=plan,
                    group=group,
                    distribution=CanonicalDistribution.REPLICATED_CANONICAL,
                ) as workspace:
                    pointers.append(self._scratch_signature(workspace))
                    second_entered.set()
            except BaseException as error:  # pragma: no cover - thread handoff
                errors.append(error)

        with self._runtime(plan):
            first_thread = threading.Thread(target=first_caller)
            second_thread = threading.Thread(target=second_caller)
            first_thread.start()
            second_thread.start()
            self.assertTrue(first_entered.wait(timeout=2))
            self.assertTrue(second_started.wait(timeout=2))
            self.assertFalse(second_entered.is_set())
            release_first.set()
            first_thread.join(timeout=2)
            second_thread.join(timeout=2)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(second_entered.is_set())
        self.assertEqual(pointers[0], pointers[1])

    def test_event_handoff_is_recorded_even_when_canonicalization_raises(self):
        log = []

        class FakeStream:
            def __init__(self, name):
                self.name = name

            def wait_event(self, event):
                log.append(("wait", self.name, event.name))

        class FakeEvent:
            name = "ready"

            def record(self, stream):
                log.append(("record", stream.name, self.name))

        streams = iter((FakeStream("first"), FakeStream("second")))
        event = FakeEvent()

        class EventPool(CanonicalMoEV3ScratchPool):
            def _uses_cuda_event(self, _local_partial):
                return True

            def _current_stream(self, _device):
                return next(streams)

            def _new_event(self):
                return event

        plan = self._plan()
        pool = EventPool()
        local = torch.zeros((8, 4), dtype=torch.bfloat16)
        group = object()
        with self._runtime(plan):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with pool.lease(
                    local,
                    plan=plan,
                    group=group,
                    distribution=CanonicalDistribution.REPLICATED_CANONICAL,
                ):
                    raise RuntimeError("injected")
            with pool.lease(
                local,
                plan=plan,
                group=group,
                distribution=CanonicalDistribution.REPLICATED_CANONICAL,
            ):
                pass

        self.assertEqual(
            log,
            [
                ("record", "first", "ready"),
                ("wait", "second", "ready"),
                ("record", "second", "ready"),
            ],
        )

    def test_pooled_transport_is_byte_identical_and_result_does_not_alias(self):
        base_plan = self._plan()
        plan = replace(base_plan, physical_to_logical=(2, 0, 3, 1))
        group = object()
        slots = CanonicalRowSlots.from_positions(torch.arange(6), capacity=8)
        partials = torch.empty((4, 8, 4), dtype=torch.bfloat16)
        partials[0].fill_(4096)
        partials[1].fill_(1)
        partials[2].fill_(-4096)
        partials[3].copy_(torch.arange(32).reshape(8, 4))
        gathered_partials = partials.clone()
        gathered_partials[:, ~slots.valid_mask] = 0

        def fake_all_gather(output, _local, *, group):
            output.view_as(gathered_partials).copy_(gathered_partials)

        pool = CanonicalMoEV3ScratchPool()
        with (
            self._runtime(plan),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
        ):
            transient_workspace = CanonicalMoEV3Workspace.allocate(
                partials[0],
                plan=plan,
                group=group,
                distribution=CanonicalDistribution.REPLICATED_CANONICAL,
            )
            transient = canonicalize_glm52_local_partial_v3(
                partials[0],
                slots,
                plan=plan,
                group=group,
                layer_id=3,
                distribution=CanonicalDistribution.REPLICATED_CANONICAL,
                workspace=transient_workspace,
            )
            with pool.lease(
                partials[0],
                plan=plan,
                group=group,
                distribution=CanonicalDistribution.REPLICATED_CANONICAL,
            ) as pooled_workspace:
                pooled = canonicalize_glm52_local_partial_v3(
                    partials[0],
                    slots,
                    plan=plan,
                    group=group,
                    layer_id=3,
                    distribution=CanonicalDistribution.REPLICATED_CANONICAL,
                    workspace=pooled_workspace,
                )
                first_values = pooled.values
                first_bytes = first_values.view(torch.int16).clone()
                scratch_ptrs = set(self._scratch_signature(pooled_workspace))

            gathered_partials.neg_()
            with pool.lease(
                partials[0],
                plan=plan,
                group=group,
                distribution=CanonicalDistribution.REPLICATED_CANONICAL,
            ) as reused_workspace:
                reused = canonicalize_glm52_local_partial_v3(
                    partials[0],
                    slots,
                    plan=plan,
                    group=group,
                    layer_id=4,
                    distribution=CanonicalDistribution.REPLICATED_CANONICAL,
                    workspace=reused_workspace,
                )

        self.assertTrue(
            torch.equal(
                transient.values.view(torch.int16),
                first_values.view(torch.int16),
            )
        )
        self.assertTrue(torch.equal(first_values.view(torch.int16), first_bytes))
        self.assertNotEqual(
            first_values.untyped_storage().data_ptr(),
            reused.values.untyped_storage().data_ptr(),
        )
        self.assertNotIn(first_values.untyped_storage().data_ptr(), scratch_ptrs)


if __name__ == "__main__":
    unittest.main()
