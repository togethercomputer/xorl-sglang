"""CPU/mock tests for GLM-5.2 exact architecture and MoE owners."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import ModelConfig  # noqa: E402
from sglang.srt.distributed.canonical_moe import (  # noqa: E402
    SamplerParallelPlan,
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


if __name__ == "__main__":
    unittest.main()
