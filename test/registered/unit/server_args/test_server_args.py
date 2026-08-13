import importlib
import json
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups import pd_disaggregation_hook
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.entrypoints.sidecar import (
    SGLANG_GRPC_ENDPOINT_ENV,
    Sidecar,
    _run_sidecar,
    build_sidecar_endpoint,
    start_sidecar,
)
from sglang.srt.environ import envs
from sglang.srt.layers.cp.base import is_cp_enabled, is_interleave
from sglang.srt.lora.glm52 import GLM52_REQUIRED_TARGET_MODULES
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.server_args import PortArgs, ServerArgs, prepare_server_args
from sglang.srt.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=12, suite="base-c-test-cpu")

# Mock get_device() so all tests run on CPU-only CI runners
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestPrepareServerArgs(CustomTestCase):
    def test_return_hidden_states_mode_configuration(self):
        disabled = ServerArgs(model_path="dummy")
        self.assertFalse(disabled.enable_return_hidden_states)
        self.assertIsNone(disabled.return_hidden_states_mode)

        last = ServerArgs(
            model_path="dummy",
            return_hidden_states_mode="last",
        )
        self.assertTrue(last.enable_return_hidden_states)
        self.assertEqual(last.return_hidden_states_mode, "last")

        legacy_full = ServerArgs(
            model_path="dummy",
            enable_return_hidden_states=True,
        )
        self.assertTrue(legacy_full.enable_return_hidden_states)
        self.assertEqual(legacy_full.return_hidden_states_mode, "full")

        parsed_last = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--return-hidden-states-mode",
                "last",
            ]
        )
        self.assertTrue(parsed_last.enable_return_hidden_states)
        self.assertEqual(parsed_last.return_hidden_states_mode, "last")

        with self.assertRaisesRegex(
            ValueError,
            "return_hidden_states_mode must be one of",
        ):
            ServerArgs(
                model_path="dummy",
                return_hidden_states_mode="lst",
            )

    def test_config_nested_dict_args_are_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("mm-process-config:\n  image:\n    resize: 128\n")
            config_file = f.name

        try:
            parser = server_args_module.argparse.ArgumentParser()
            ServerArgs.add_cli_args(parser)
            merged = ConfigArgumentMerger(parser).merge_config_with_args(
                [
                    "--config",
                    config_file,
                    "--model-path",
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                ]
            )
            value = merged[merged.index("--mm-process-config") + 1]
            parsed = parser.parse_args(merged)

            self.assertEqual(json.loads(value), {"image": {"resize": 128}})
            self.assertEqual(parsed.mm_process_config, {"image": {"resize": 128}})
        finally:
            os.unlink(config_file)


class TestMmEncoderDataParallelLogging(CustomTestCase):
    def test_logs_when_encoder_dp_has_no_parallelism(self):
        server_args = ServerArgs(
            model_path="dummy", mm_enable_dp_encoder=True, tp_size=1
        )

        with self.assertLogs(server_args_module.logger, level="WARNING") as logs:
            server_args._handle_data_parallelism()

        self.assertIn("TP=1", logs.output[0])
        self.assertIn("no data-parallel work", logs.output[0])

    def test_logs_encoder_dp_tradeoff_for_tp(self):
        server_args = ServerArgs(
            model_path="dummy", mm_enable_dp_encoder=True, tp_size=4
        )

        with self.assertLogs(server_args_module.logger, level="INFO") as logs:
            server_args._handle_data_parallelism()

        self.assertIn("TP=4", logs.output[0])
        self.assertIn("high-resolution or multi-image", logs.output[0])


class TestMultimodalFeatureTransport(CustomTestCase):
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_cuda_ipc_is_explicit_and_bounded(self, _mock_is_cuda):
        server_args = ServerArgs(
            model_path="dummy",
            mm_feature_transport="cuda_ipc",
            tokenizer_worker_num=4,
            base_gpu_id=2,
        )

        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "0"}):
            with self.assertLogs(server_args_module.logger, level="INFO") as logs:
                server_args._handle_multimodal_feature_transport()

            self.assertEqual(server_args.mm_feature_transport, "cuda_ipc")
            self.assertTrue(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

        output = "\n".join(logs.output)
        self.assertIn("base GPU 2", output)
        self.assertIn("4 tokenizer worker", output)

    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_legacy_keep_flag_maps_to_cuda_ipc(self, _mock_is_cuda):
        server_args = ServerArgs(model_path="dummy", keep_mm_feature_on_device=True)

        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "0"}):
            with self.assertLogs(server_args_module.logger, level="WARNING") as logs:
                server_args._handle_multimodal_feature_transport()

            self.assertEqual(server_args.mm_feature_transport, "cuda_ipc")
            self.assertFalse(server_args.keep_mm_feature_on_device)
            self.assertTrue(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

        self.assertIn("deprecated", logs.output[0])

    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_explicit_cpu_overrides_legacy_environment(self, _mock_is_cuda):
        server_args = ServerArgs(model_path="dummy", mm_feature_transport="cpu")

        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "1"}):
            with self.assertLogs(server_args_module.logger, level="WARNING") as logs:
                server_args._handle_multimodal_feature_transport()

            self.assertEqual(server_args.mm_feature_transport, "cpu")
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

        self.assertIn("overrides", logs.output[0])

    def test_default_transport_is_cpu(self):
        server_args = ServerArgs(model_path="dummy")

        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "0"}):
            server_args._handle_multimodal_feature_transport()

            self.assertEqual(server_args.mm_feature_transport, "cpu")
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

    @patch("sglang.srt.server_args.is_cuda", return_value=False)
    def test_cuda_ipc_rejects_non_nvidia_platforms(self, _mock_is_cuda):
        server_args = ServerArgs(model_path="dummy", mm_feature_transport="cuda_ipc")

        with self.assertRaisesRegex(ValueError, "requires NVIDIA CUDA"):
            server_args._handle_multimodal_feature_transport()

    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_cuda_ipc_rejects_multi_node(self, _mock_is_cuda):
        server_args = ServerArgs(
            model_path="dummy", mm_feature_transport="cuda_ipc", nnodes=2
        )

        with self.assertRaisesRegex(ValueError, "single node"):
            server_args._handle_multimodal_feature_transport()


class TestMambaCacheStochasticRounding(unittest.TestCase):
    def test_rejects_fp32_ssm_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float32",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "--mamba-ssm-dtype float16"):
            server_args._handle_mamba_backend()

    @patch("sglang.srt.server_args.is_cuda", return_value=False)
    def test_rejects_non_cuda(self, _mock_is_cuda):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float16",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "NVIDIA CUDA"):
            server_args._handle_mamba_backend()

    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    @patch("sglang.srt.server_args.is_sm100_supported", return_value=False)
    def test_rejects_triton_without_sm100(self, _mock_sm100, _mock_is_cuda):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float16",
            mamba_backend="triton",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "requires SM100"):
            server_args._handle_mamba_backend()


class TestLoadBalanceMethod(unittest.TestCase):
    def _load_balance_args(self, **kwargs):
        server_args = ServerArgs(model_path="dummy", **kwargs)
        server_args._handle_pd_disaggregation()
        server_args._handle_load_balance_method()
        return server_args

    def test_non_pd_defaults_to_round_robin(self):
        server_args = self._load_balance_args(disaggregation_mode="null")
        self.assertEqual(server_args.load_balance_method, "round_robin")

    def test_pd_prefill_defaults_to_follow_bootstrap_room(self):
        server_args = self._load_balance_args(disaggregation_mode="prefill")
        self.assertEqual(server_args.load_balance_method, "follow_bootstrap_room")

    def test_pd_decode_defaults_to_round_robin(self):
        server_args = self._load_balance_args(disaggregation_mode="decode")
        self.assertEqual(server_args.load_balance_method, "round_robin")

    def test_pd_prefill_dcp_warns_about_performance(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            dcp_size=4,
        )
        with self.assertLogs(pd_disaggregation_hook.logger, level="WARNING") as logs:
            server_args._handle_pd_disaggregation()
        self.assertIn("without improving prefill performance", "\n".join(logs.output))

    def test_pd_decode_dcp_forces_chunk_cache(self):
        server_args = self._load_balance_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            dcp_size=4,
        )
        self.assertTrue(server_args.disable_radix_cache)

    def test_pd_decode_dcp_rejects_unsupported_transfer_backend(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="fake",
            dcp_size=4,
        )
        with self.assertRaisesRegex(ValueError, "mooncake or nixl"):
            server_args._handle_pd_disaggregation()

    def test_pd_decode_dcp_rejects_radix_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            disaggregation_decode_enable_radix_cache=True,
            dcp_size=4,
        )
        with self.assertRaisesRegex(ValueError, "currently requires chunk cache"):
            server_args._handle_pd_disaggregation()

    def test_pd_decode_dcp_rejects_hierarchical_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            enable_hierarchical_cache=True,
            dcp_size=4,
        )
        with self.assertRaisesRegex(ValueError, "--enable-hierarchical-cache"):
            server_args._handle_pd_disaggregation()

    def test_pd_decode_radix_cache_rejects_hisparse(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="nixl",
            enable_hisparse=True,
        )
        with self.assertRaises(ValueError) as context:
            server_args._handle_pd_disaggregation()

        self.assertIn(
            "--disaggregation-decode-enable-radix-cache is incompatible with "
            "--enable-hisparse",
            str(context.exception),
        )

    def test_pd_decode_radix_cache_rejects_fake_backend(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="fake",
        )
        with self.assertRaises(ValueError) as context:
            server_args._handle_pd_disaggregation()

        self.assertIn(
            "--disaggregation-decode-enable-radix-cache is incompatible "
            "with --disaggregation-transfer-backend fake",
            str(context.exception),
        )

    def test_pd_decode_radix_cache_allows_mooncake_tcp(self):
        server_args = self._load_balance_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake_tcp",
        )

        self.assertFalse(server_args.disable_radix_cache)
        self.assertEqual(server_args.disaggregation_transfer_backend, "mooncake")


class TestSkipTokenizerInit(unittest.TestCase):
    def test_skip_tokenizer_worker_counts(self):
        server_args = ServerArgs(
            model_path="dummy",
            skip_tokenizer_init=True,
            tokenizer_worker_num=4,
            detokenizer_worker_num=3,
        )

        server_args._handle_tokenizer_batching()

        # Tokenizer fanout preserved; detokenizer coerced to 1 (no decode work).
        self.assertEqual(server_args.tokenizer_worker_num, 4)
        self.assertEqual(server_args.detokenizer_worker_num, 1)


class TestHiSparseDsaBackendPolicy(unittest.TestCase):
    # The backend selection moved to the resolution pipeline; these policy
    # tests drive the pass through its read-only view.
    @staticmethod
    def _resolve(kv_cache_dtype, **kw):
        from types import SimpleNamespace

        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _dsa_split_backend_resolution,
        )

        hf = SimpleNamespace(architectures=["DeepseekV32ForCausalLM"])
        defaults = dict(
            kv_cache_dtype=kv_cache_dtype,
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            enable_hisparse=True,
        )
        defaults.update(kw)
        view = ResolvedView(
            SimpleNamespace(
                get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
            )
        )
        with (
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch("sglang.srt.arg_groups.overrides.is_npu", return_value=False),
            patch("sglang.srt.arg_groups.overrides.is_xpu", return_value=False),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            declared = _dsa_split_backend_resolution(view)
        return {
            "dsa_prefill_backend": declared.get(
                "dsa_prefill_backend", defaults["dsa_prefill_backend"]
            ),
            "dsa_decode_backend": declared.get(
                "dsa_decode_backend", defaults["dsa_decode_backend"]
            ),
        }

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_defaults_to_flashmla_sparse_on_cuda_bfloat16(self, _mock_is_hip):
        resolved = self._resolve("bfloat16")

        self.assertEqual(resolved["dsa_prefill_backend"], "flashmla_sparse")
        self.assertEqual(resolved["dsa_decode_backend"], "flashmla_sparse")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_defaults_to_flashmla_kv_on_cuda_fp8(self, _mock_is_hip):
        resolved = self._resolve("fp8_e4m3")

        self.assertEqual(resolved["dsa_prefill_backend"], "flashmla_kv")
        self.assertEqual(resolved["dsa_decode_backend"], "flashmla_kv")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_defaults_to_tilelang_on_rocm(self, _mock_is_hip):
        resolved = self._resolve("bfloat16")

        self.assertEqual(resolved["dsa_prefill_backend"], "tilelang")
        self.assertEqual(resolved["dsa_decode_backend"], "tilelang")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_preserves_rocm_user_backend_and_defaults_missing_side(
        self, _mock_is_hip
    ):
        resolved = self._resolve("bfloat16", dsa_prefill_backend="tilelang")

        self.assertEqual(resolved["dsa_prefill_backend"], "tilelang")
        self.assertEqual(resolved["dsa_decode_backend"], "tilelang")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_accepts_aiter_backend_on_rocm(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="aiter",
            dsa_decode_backend="aiter",
        )

        server_args._validate_hisparse_dsa_backend("dsa_prefill_backend", "prefill")
        server_args._validate_hisparse_dsa_backend("dsa_decode_backend", "decode")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_rejects_cuda_backend_on_rocm(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="flashmla_sparse",
        )

        with self.assertRaisesRegex(ValueError, "tilelang"):
            server_args._validate_hisparse_dsa_backend("dsa_prefill_backend", "prefill")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_rejects_rocm_backend_on_cuda(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_decode_backend="tilelang",
        )

        with self.assertRaisesRegex(ValueError, "flashmla_sparse"):
            server_args._validate_hisparse_dsa_backend("dsa_decode_backend", "decode")

    def test_hisparse_accepts_bfloat16_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
        )

        server_args._validate_hisparse_kv_cache_dtype()

    def test_hisparse_accepts_fp8_e4m3_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="fp8_e4m3",
        )

        server_args._validate_hisparse_kv_cache_dtype()

    def test_hisparse_rejects_unsupported_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="float16",
        )

        with self.assertRaisesRegex(ValueError, r"fp8_e4m3"):
            server_args._validate_hisparse_kv_cache_dtype()


class TestFa4PageSizeAutoForce(CustomTestCase):
    """FA4 requires page_size 128 for non-MLA models on SM100. The auto-force
    must trigger for `--attention-backend fa4` (combined) too, not only for the
    explicit `--prefill-attention-backend fa4` path."""

    def _make_args(self, attention_backend, prefill=None, decode=None, page_size=1):
        args = ServerArgs(model_path="dummy")
        args.attention_backend = attention_backend
        args.prefill_attention_backend = prefill
        args.decode_attention_backend = decode
        args.page_size = page_size
        # Short-circuit get_model_config(): the fa4 page_size branch only needs
        # use_mla_backend() (mocked) and is_sm100_supported() (mocked), not a
        # real model_config. Pre-set the attribute so get_model_config returns
        # early without touching ModelConfig.from_server_args.
        args.model_config = MagicMock()
        args.model_config.hf_config.dual_chunk_attention_config = None
        return args

    @patch("sglang.srt.arg_groups.overrides.is_sm100_supported", return_value=True)
    @patch("sglang.srt.server_args.ServerArgs.use_mla_backend", return_value=False)
    def test_combined_attention_backend_fa4_forces_page_size_128(
        self, _mock_mla, _mock_sm100
    ):
        # `--attention-backend fa4` (combined): prefill/decode fields stay None.
        args = self._make_args(attention_backend="fa4")

        args._handle_attention_backend_compatibility()

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(args.page_size, 1)  # dual-apply retired: pristine
        self.assertEqual(resolved_view(args).page_size, 128)

    @patch("sglang.srt.arg_groups.overrides.is_sm100_supported", return_value=True)
    @patch("sglang.srt.server_args.ServerArgs.use_mla_backend", return_value=False)
    def test_explicit_prefill_fa4_forces_page_size_128(self, _mock_mla, _mock_sm100):
        # `--prefill-attention-backend fa4`: the previously-covered path.
        args = self._make_args(attention_backend=None, prefill="fa4", page_size=1)

        args._handle_attention_backend_compatibility()

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(args.page_size, 1)  # dual-apply retired: pristine
        self.assertEqual(resolved_view(args).page_size, 128)


class TestContextParallelServerArgs(CustomTestCase):
    def setUp(self):
        self.parser = server_args_module.argparse.ArgumentParser()
        ServerArgs.add_cli_args(self.parser)

    def _new_cp_args(self, **overrides):
        server_args = object.__new__(ServerArgs)
        defaults = dict(
            enable_prefill_context_parallel=False,
            enable_dsa_prefill_context_parallel=False,
            enable_prefill_cp=False,
            cp_strategy=None,
            model_path="instance://127.0.0.1:8000/dummy",
            dsa_prefill_cp_mode="round-robin-split",
            prefill_cp_mode="in-seq-split",
            attn_cp_size=1,
            tp_size=1,
            dp_size=1,
            moe_dp_size=1,
            ep_size=1,
            pp_size=1,
            enable_aiter_allreduce_fusion=False,
        )
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(server_args, key, value)
        return server_args

    def test_canonical_prefill_cp_requires_strategy(self):
        args = self.parser.parse_args(["--model", "dummy", "--enable-prefill-cp"])

        self.assertTrue(args.enable_prefill_cp)
        self.assertIsNone(args.cp_strategy)

        server_args = self._new_cp_args(
            enable_prefill_cp=args.enable_prefill_cp,
            cp_strategy=args.cp_strategy,
        )
        with self.assertRaisesRegex(ValueError, "--cp-strategy"):
            server_args._handle_context_parallelism()

    def test_deprecated_dsa_cp_mode_maps_to_unified_strategy(self):
        args = self.parser.parse_args(
            [
                "--model",
                "dummy",
                "--enable-dsa-prefill-context-parallel",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
            ]
        )
        server_args = self._new_cp_args(
            enable_dsa_prefill_context_parallel=(
                args.enable_dsa_prefill_context_parallel
            ),
            dsa_prefill_cp_mode=args.dsa_prefill_cp_mode,
        )

        server_args._handle_legacy_cp_arguments()

        self.assertTrue(server_args.enable_prefill_cp)
        self.assertEqual(server_args.cp_strategy, "interleave")
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")

    def test_canonical_interleave_cp_mirrors_to_dsa_runtime_aliases(self):
        server_args = self._new_cp_args(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attention_backend="dsa",
        )

        server_args._handle_legacy_cp_arguments()
        server_args._handle_context_parallelism()

        self.assertTrue(server_args.enable_dsa_prefill_context_parallel)
        self.assertFalse(server_args.enable_prefill_context_parallel)
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")
        self.assertEqual(server_args.prefill_cp_mode, "round-robin-split")

    def test_context_parallel_handler_initializes_cp_strategy(self):
        server_args = self._new_cp_args(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=2,
            tp_size=2,
        )

        server_args._handle_context_parallelism()

        self.assertTrue(is_cp_enabled())
        self.assertTrue(is_interleave())

    def test_registered_cp_legacy_args_map_to_unified_strategy(self):
        cases = [
            (
                "deepseek_v3_mla_cp",
                dict(enable_prefill_context_parallel=True),
                "zigzag",
                "in-seq-split",
                False,
                True,
            ),
            (
                "qwen3_gqa_cp",
                dict(
                    enable_prefill_context_parallel=True,
                    tp_size=4,
                    attn_cp_size=2,
                ),
                "zigzag",
                "in-seq-split",
                False,
                True,
            ),
            (
                "deepseek_v32_dsa_in_seq_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    dsa_prefill_cp_mode="in-seq-split",
                    tp_size=8,
                    dp_size=2,
                    attn_cp_size=4,
                ),
                "zigzag",
                "in-seq-split",
                True,
                False,
            ),
            (
                "deepseek_v32_dsa_round_robin_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    tp_size=8,
                    attn_cp_size=8,
                ),
                "interleave",
                "round-robin-split",
                True,
                False,
            ),
            (
                "deepseek_v4_flash_fp4_b200_dsa_round_robin_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    dsa_prefill_cp_mode="round-robin-split",
                    tp_size=4,
                    attn_cp_size=4,
                ),
                "interleave",
                "round-robin-split",
                True,
                False,
            ),
        ]

        for name, overrides, strategy, mode, expect_dsa, expect_generic in cases:
            with self.subTest(name=name):
                server_args = self._new_cp_args(**overrides)

                server_args._handle_legacy_cp_arguments()
                server_args._handle_context_parallelism()

                self.assertTrue(server_args.enable_prefill_cp)
                self.assertEqual(server_args.cp_strategy, strategy)
                self.assertEqual(server_args.dsa_prefill_cp_mode, mode)
                self.assertEqual(server_args.prefill_cp_mode, mode)
                self.assertEqual(
                    server_args.enable_dsa_prefill_context_parallel, expect_dsa
                )
                self.assertEqual(
                    server_args.enable_prefill_context_parallel, expect_generic
                )


class TestDeterministicGlmDsa(unittest.TestCase):
    @staticmethod
    def _server_args(prefill_backend="flashmla_sparse", decode_backend="fa3"):
        server_args = ServerArgs(model_path="dummy")
        server_args.model_path = "/tmp/glm-5.2"
        server_args.enable_deterministic_inference = True
        server_args.attention_backend = "dsa"
        server_args.dsa_prefill_backend = prefill_backend
        server_args.dsa_decode_backend = decode_backend
        return server_args

    @staticmethod
    def _glm_model_config():
        fp8_unquantized_modules = set()
        for layer in range(79):
            prefix = f"model.layers.{layer}"
            fp8_unquantized_modules.update(
                {
                    f"{prefix}.input_layernorm",
                    f"{prefix}.post_attention_layernorm",
                    f"{prefix}.self_attn.q_a_layernorm",
                    f"{prefix}.self_attn.kv_a_layernorm",
                }
            )
        for layer in range(3, 79):
            prefix = f"model.layers.{layer}.mlp.gate"
            fp8_unquantized_modules.update(
                {prefix, f"{prefix}.e_score_correction_bias"}
            )
        for layer in (0, 1, 2, *range(6, 79, 4)):
            prefix = f"model.layers.{layer}.self_attn"
            fp8_unquantized_modules.update(
                {
                    f"{prefix}.indexers_proj",
                    f"{prefix}.indexer.k_norm",
                    f"{prefix}.indexer.k_norm.bias",
                }
            )
        fp8_unquantized_modules.update(
            {
                "model.embed_tokens",
                "model.norm",
                "lm_head",
                "model.layers.78.shared_head.norm",
                "model.layers.78.hnorm",
                "model.layers.78.enorm",
                "model.layers.78.eh_proj",
            }
        )
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            hidden_size=6144,
            intermediate_size=12288,
            moe_intermediate_size=2048,
            moe_layer_freq=1,
            num_hidden_layers=78,
            vocab_size=154880,
            num_attention_heads=64,
            num_key_value_heads=64,
            first_k_dense_replace=3,
            mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
            rms_norm_eps=1e-5,
            rope_parameters={
                "rope_theta": 8_000_000,
                "rope_type": "default",
                "type": "default",
            },
            max_position_embeddings=1_048_576,
            hidden_act="silu",
            n_routed_experts=256,
            n_shared_experts=1,
            num_experts_per_tok=8,
            index_topk=2048,
            index_topk_freq=4,
            index_skip_topk_offset=3,
            index_head_dim=128,
            index_n_heads=32,
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            indexer_rope_interleave=True,
            rope_interleave=True,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
            topk_method="noaux_tc",
            tie_word_embeddings=False,
            quantization_config={
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": sorted(fp8_unquantized_modules),
            },
            indexer_types=[
                "full" if i < 3 or i % 4 == 2 else "shared" for i in range(78)
            ],
        )
        model_config = MagicMock()
        model_config.hf_config = hf_config
        return model_config

    def test_accepts_pinned_glm_dsa_pair_with_radix(self):
        server_args = self._server_args()
        server_args.get_model_config = MagicMock(return_value=self._glm_model_config())

        server_args._handle_deterministic_inference()

        self.assertEqual(server_args.attention_backend, "dsa")
        self.assertEqual(server_args.dsa_prefill_backend, "flashmla_sparse")
        self.assertEqual(server_args.dsa_decode_backend, "fa3")
        self.assertFalse(server_args.disable_radix_cache)

    def test_glm52_xorl_resolves_the_exact_defaults_without_environment_flags(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.rl_on_policy_target = "xorl"
        server_args.nnodes = 2
        server_args.tp_size = 16
        server_args.cuda_graph_config = CudaGraphConfig()
        server_args.cuda_graph_config.decode.bs = list(range(1, 513))
        server_args.cuda_graph_config.decode.max_bs = 512
        server_args._cuda_graph_config_locked = set()
        hf_config = self._glm_model_config().hf_config

        server_args._resolve_glm52_exact_contract(
            hf_config,
            model_arch="GlmMoeDsaForCausalLM",
            is_dsa_model=True,
        )

        self.assertTrue(server_args.glm52_exact_mode)
        self.assertTrue(hf_config._glm52_exact_mode)
        self.assertEqual(server_args.dtype, "bfloat16")
        self.assertEqual(server_args.quantization, "fp8")
        self.assertEqual(server_args.kv_cache_dtype, "bfloat16")
        self.assertEqual(server_args.attention_backend, "dsa")
        self.assertEqual(server_args.dsa_prefill_backend, "flashmla_sparse")
        self.assertEqual(server_args.dsa_decode_backend, "flashmla_sparse")
        self.assertEqual(server_args.dsa_paged_mqa_logits_backend, "deepgemm")
        self.assertEqual(server_args.dsa_topk_backend, "sgl-kernel")
        self.assertEqual(server_args.moe_runner_backend, "triton")
        self.assertEqual(server_args.fp8_gemm_runner_backend, "triton")
        self.assertEqual(server_args.moe_dense_tp_size, 1)
        self.assertEqual(server_args.ep_size, 16)
        self.assertEqual(server_args.attn_cp_size, 16)
        self.assertEqual(server_args.dp_size, 1)
        self.assertEqual(server_args.moe_dp_size, 1)
        self.assertEqual(server_args.tp_size, 16)
        self.assertEqual(server_args.pp_size, 1)
        self.assertTrue(server_args.enable_dsa_prefill_context_parallel)
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")
        self.assertTrue(server_args.disable_shared_experts_fusion)
        self.assertTrue(server_args.disable_custom_all_reduce)
        self.assertTrue(server_args.disable_overlap_schedule)
        self.assertTrue(server_args.disable_piecewise_cuda_graph)
        self.assertTrue(server_args.disable_radix_cache)
        self.assertEqual(server_args.cuda_graph_bs_decode, [16])
        self.assertEqual(server_args.cuda_graph_max_bs_decode, 16)
        self.assertEqual(server_args.chunked_prefill_size, -1)
        self.assertEqual(server_args.max_prefill_tokens, 8192)
        self.assertEqual(server_args.prefill_max_requests, 1)
        self.assertEqual(server_args.max_total_tokens, 8192)
        self.assertEqual(server_args.max_running_requests, 16)
        self.assertEqual(server_args.mem_fraction_static, 0.82)
        self.assertEqual(server_args.model_impl, "sglang")
        self.assertEqual(server_args.max_lora_rank, 1)
        self.assertEqual(server_args.lora_backend, "triton")
        self.assertTrue(server_args.experts_shared_outer_loras)
        self.assertFalse(server_args.enable_lora_overlap_loading)
        self.assertFalse(server_args.lora_use_virtual_experts)
        self.assertTrue(server_args.lora_strict_loading)
        self.assertEqual(server_args.dcp_size, 1)
        self.assertFalse(server_args.enable_cp_decode_attn_tp)
        self.assertEqual(server_args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertEqual(server_args.cuda_graph_config.decode.bs, [16])
        self.assertEqual(server_args.cuda_graph_config.decode.max_bs, 16)
        self.assertEqual(
            server_args.cuda_graph_config.prefill.backend, Backend.DISABLED
        )
        self.assertIn((Phase.DECODE, "backend"), server_args._cuda_graph_config_locked)
        self.assertIn((Phase.DECODE, "bs"), server_args._cuda_graph_config_locked)
        self.assertIn((Phase.DECODE, "max_bs"), server_args._cuda_graph_config_locked)
        self.assertIn((Phase.PREFILL, "backend"), server_args._cuda_graph_config_locked)

        server_args.page_size = 64
        server_args.enable_dp_attention = True
        with (
            patch.object(envs.SGLANG_ENABLE_CP_V2, "get", return_value=True),
            patch.object(
                envs.SGLANG_DISABLE_DSA_INDEXER_FUSION,
                "get",
                return_value=False,
            ),
        ):
            server_args._validate_glm52_exact_resolved_contract()
            server_args.cuda_graph_config.decode.backend = Backend.DISABLED
            with self.assertRaisesRegex(ValueError, "drifted.*decode.backend"):
                server_args._validate_glm52_exact_resolved_contract()

        server_args.cuda_graph_config.decode.backend = Backend.FULL
        with (
            patch.object(envs.SGLANG_ENABLE_CP_V2, "get", return_value=False),
            self.assertRaisesRegex(ValueError, "SGLANG_ENABLE_CP_V2"),
        ):
            server_args._validate_glm52_exact_resolved_contract()
        with (
            patch.object(envs.SGLANG_ENABLE_CP_V2, "get", return_value=True),
            patch.object(
                envs.SGLANG_DISABLE_DSA_INDEXER_FUSION,
                "get",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "SGLANG_DISABLE_DSA_INDEXER_FUSION"),
        ):
            server_args._validate_glm52_exact_resolved_contract()

        disabled_envs = (
            envs.SGLANG_SIMULATE_UNIFORM_EXPERTS,
            envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS,
            envs.SGLANG_OPT_MOE_QUANT_ONCE,
            envs.SGLANG_SHARED_EXPERT_TP1,
        )
        for setting in disabled_envs:
            with (
                self.subTest(setting=setting.name),
                patch.object(envs.SGLANG_ENABLE_CP_V2, "get", return_value=True),
                patch.object(
                    envs.SGLANG_DISABLE_DSA_INDEXER_FUSION,
                    "get",
                    return_value=False,
                ),
                patch.object(setting, "get", return_value=True),
                self.assertRaisesRegex(ValueError, setting.name),
            ):
                server_args._validate_glm52_exact_resolved_contract()
        with (
            patch.object(envs.SGLANG_ENABLE_CP_V2, "get", return_value=True),
            patch.object(
                envs.SGLANG_DISABLE_DSA_INDEXER_FUSION,
                "get",
                return_value=False,
            ),
            patch.object(
                envs.SGLANG_FP8_IGNORED_LAYERS,
                "get",
                return_value="model.embed_tokens",
            ),
            self.assertRaisesRegex(ValueError, "SGLANG_FP8_IGNORED_LAYERS"),
        ):
            server_args._validate_glm52_exact_resolved_contract()

    def test_glm52_xorl_resolves_dp16_cp1_as_eager_dp_owned_lane(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.rl_on_policy_target = "xorl"
        server_args.nnodes = 2
        server_args.tp_size = 16
        server_args.dp_size = 16
        server_args.attn_cp_size = 1
        server_args.cuda_graph_config = CudaGraphConfig()
        server_args._cuda_graph_config_locked = set()

        server_args._resolve_glm52_exact_contract(
            self._glm_model_config().hf_config,
            model_arch="GlmMoeDsaForCausalLM",
            is_dsa_model=True,
        )

        self.assertEqual(server_args.ep_size, 16)
        self.assertEqual(server_args.dp_size, 16)
        self.assertEqual(server_args.attn_cp_size, 1)
        self.assertFalse(server_args.enable_prefill_cp)
        self.assertFalse(server_args.enable_dsa_prefill_context_parallel)
        self.assertIsNone(server_args.cp_strategy)
        self.assertTrue(server_args.enable_dp_lm_head)
        self.assertEqual(server_args.max_loras_per_batch, 2)
        self.assertEqual(server_args.max_loaded_loras, 2)
        self.assertTrue(server_args.disable_cuda_graph)
        self.assertTrue(server_args.disable_cuda_graph_padding)
        self.assertIsNone(server_args.cuda_graph_bs_decode)
        self.assertIsNone(server_args.cuda_graph_max_bs_decode)
        self.assertEqual(server_args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(
            server_args.cuda_graph_config.prefill.backend, Backend.DISABLED
        )
        self.assertTrue(server_args.disable_radix_cache)

        server_args.page_size = 64
        server_args.enable_dp_attention = True
        with patch.object(
            envs.SGLANG_DISABLE_DSA_INDEXER_FUSION, "get", return_value=False
        ):
            server_args._validate_glm52_exact_resolved_contract()

        server_args.attn_cp_size = 16
        with (
            patch.object(
                envs.SGLANG_DISABLE_DSA_INDEXER_FUSION, "get", return_value=False
            ),
            self.assertRaisesRegex(ValueError, "drifted.*attn_cp_size"),
        ):
            server_args._validate_glm52_exact_resolved_contract()

    def test_glm52_xorl_preserves_arbitrary_positive_max_lora_rank(self):
        for rank in (1, 3, 7, 16, 31, 64):
            with self.subTest(rank=rank):
                server_args = ServerArgs(model_path="dummy")
                server_args.rl_on_policy_target = "xorl"
                server_args.nnodes = 2
                server_args.tp_size = 16
                server_args.max_lora_rank = rank
                server_args.cuda_graph_config = CudaGraphConfig()
                server_args._cuda_graph_config_locked = set()

                server_args._resolve_glm52_exact_contract(
                    self._glm_model_config().hf_config,
                    model_arch="GlmMoeDsaForCausalLM",
                    is_dsa_model=True,
                )

                self.assertEqual(server_args.max_lora_rank, rank)
                server_args.page_size = 64
                server_args.enable_dp_attention = True
                with (
                    patch.object(envs.SGLANG_ENABLE_CP_V2, "get", return_value=True),
                    patch.object(
                        envs.SGLANG_DISABLE_DSA_INDEXER_FUSION,
                        "get",
                        return_value=False,
                    ),
                ):
                    server_args._validate_glm52_exact_resolved_contract()

    def test_glm52_xorl_preserves_admitted_large_kv_capacity(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.rl_on_policy_target = "xorl"
        server_args.nnodes = 2
        server_args.tp_size = 16
        server_args.max_total_tokens = 32768
        server_args.cuda_graph_config = CudaGraphConfig()
        server_args._cuda_graph_config_locked = set()

        server_args._resolve_glm52_exact_contract(
            self._glm_model_config().hf_config,
            model_arch="GlmMoeDsaForCausalLM",
            is_dsa_model=True,
        )

        self.assertEqual(server_args.max_total_tokens, 32768)
        server_args.page_size = 64
        server_args.enable_dp_attention = True
        with (
            patch.object(envs.SGLANG_ENABLE_CP_V2, "get", return_value=True),
            patch.object(
                envs.SGLANG_DISABLE_DSA_INDEXER_FUSION,
                "get",
                return_value=False,
            ),
        ):
            server_args._validate_glm52_exact_resolved_contract()

    def test_glm52_xorl_rejects_explicit_incompatible_programs(self):
        incompatible = {
            "dtype": "float16",
            "quantization": "int8",
            "kv_cache_dtype": "fp8_e4m3",
            "attention_backend": "fa3",
            "dsa_prefill_backend": "fa3",
            "dsa_decode_backend": "flashmla_kv",
            "dsa_paged_mqa_logits_backend": "cutedsl",
            "dsa_topk_backend": "torch",
            "prefill_attention_backend": "fa3",
            "decode_attention_backend": "fa3",
            "moe_runner_backend": "flashinfer_trtllm",
            "fp8_gemm_runner_backend": "deep_gemm",
            "moe_dense_tp_size": 2,
            "moe_a2a_backend": "deepep",
            "ep_num_redundant_experts": 1,
            "ep_dispatch_algorithm": "dynamic",
            "init_expert_location": "random",
            "enable_eplb": True,
            "tp_size": 8,
            "ep_size": 8,
            "pp_size": 2,
            "dp_size": 2,
            "moe_dp_size": 2,
            "attn_cp_size": 8,
            "dsa_prefill_cp_mode": "in-seq-split",
            "enable_hisparse": True,
            "page_size": 32,
            "nnodes": 1,
            "disaggregation_mode": "prefill",
            "sampling_backend": "flashinfer",
            "chunked_prefill_size": 4096,
            "max_prefill_tokens": 4096,
            "prefill_max_requests": 2,
            "max_total_tokens": 4096,
            "max_running_requests": 32,
            "model_impl": "transformers",
            "device": "cpu",
            "is_embedding": True,
            "debug_cuda_graph": True,
            "enable_torch_compile": True,
            "enable_two_batch_overlap": True,
            "enable_single_batch_overlap": True,
            "debug_tensor_dump_output_folder": "/tmp/glm52-debug",
            "msprobe_dump_config": {"enabled": True},
            "lora_backend": "torch_native",
            "experts_shared_outer_loras": False,
            "enable_lora_overlap_loading": True,
            "lora_use_virtual_experts": True,
            "dcp_size": 2,
            "enable_cp_decode_attn_tp": True,
            "enable_dp_lm_head": True,
            "disable_cuda_graph_padding": True,
        }
        for name, value in incompatible.items():
            with self.subTest(name=name, value=value):
                server_args = ServerArgs(model_path="dummy")
                server_args.rl_on_policy_target = "xorl"
                server_args.nnodes = 2
                server_args.tp_size = 16
                setattr(server_args, name, value)
                with self.assertRaisesRegex(ValueError, "exact GLM-5.2 XORL"):
                    server_args._resolve_glm52_exact_contract(
                        self._glm_model_config().hf_config,
                        model_arch="GlmMoeDsaForCausalLM",
                        is_dsa_model=True,
                    )

    def test_glm52_xorl_requires_or_materializes_the_complete_lora_target_set(
        self,
    ):
        incomplete = ServerArgs(model_path="dummy")
        incomplete.rl_on_policy_target = "xorl"
        incomplete.nnodes = 2
        incomplete.tp_size = 16
        incomplete.lora_target_modules = {"lm_head"}
        with self.assertRaisesRegex(ValueError, "complete LoRA target set"):
            incomplete._resolve_glm52_exact_contract(
                self._glm_model_config().hf_config,
                model_arch="GlmMoeDsaForCausalLM",
                is_dsa_model=True,
            )

        dynamic = ServerArgs(model_path="dummy", enable_lora=True)
        dynamic.rl_on_policy_target = "xorl"
        dynamic.nnodes = 2
        dynamic.tp_size = 16
        dynamic._resolve_glm52_exact_contract(
            self._glm_model_config().hf_config,
            model_arch="GlmMoeDsaForCausalLM",
            is_dsa_model=True,
        )

        self.assertEqual(
            set(dynamic.lora_target_modules), GLM52_REQUIRED_TARGET_MODULES
        )

    def test_glm52_xorl_rejects_explicit_incompatible_graph_programs(self):
        cases = (
            (Phase.DECODE, "backend", Backend.DISABLED),
            (Phase.DECODE, "bs", [8, 16]),
            (Phase.DECODE, "max_bs", 32),
            (Phase.PREFILL, "backend", Backend.BREAKABLE),
        )
        for phase, name, value in cases:
            with self.subTest(phase=phase, name=name, value=value):
                server_args = ServerArgs(model_path="dummy")
                server_args.rl_on_policy_target = "xorl"
                server_args.nnodes = 2
                server_args.tp_size = 16
                server_args.cuda_graph_config = CudaGraphConfig()
                setattr(getattr(server_args.cuda_graph_config, phase), name, value)
                server_args._cuda_graph_config_locked = {(phase, name)}

                with self.assertRaisesRegex(ValueError, "cuda_graph_config"):
                    server_args._resolve_glm52_exact_contract(
                        self._glm_model_config().hf_config,
                        model_arch="GlmMoeDsaForCausalLM",
                        is_dsa_model=True,
                    )

    def _glm_exact_args(self, **overrides):
        server_args = ServerArgs(model_path="dummy")
        server_args.rl_on_policy_target = "xorl"
        server_args.nnodes = 2
        server_args.tp_size = 16
        server_args.cuda_graph_config = CudaGraphConfig()
        server_args._cuda_graph_config_locked = set()
        for name, value in overrides.items():
            setattr(server_args, name, value)
        return server_args

    def test_glm52_xorl_exact_radix_is_env_gated_and_fail_closed(self):
        """The radix path stays opt-in and rejects unsupported combinations.

        With SGLANG_ENABLE_GLM52_EXACT_RADIX unset, the resolver keeps the
        default radix-disabled envelope. The opt-in path rejects cache and
        adapter combinations outside its contract.
        """
        resolve = lambda args: args._resolve_glm52_exact_contract(
            self._glm_model_config().hf_config,
            model_arch="GlmMoeDsaForCausalLM",
            is_dsa_model=True,
        )

        with envs.SGLANG_ENABLE_GLM52_EXACT_RADIX.override(False):
            default_args = self._glm_exact_args()
            resolve(default_args)
            self.assertTrue(default_args.disable_radix_cache)
            self.assertEqual(default_args.mem_fraction_static, 0.82)
            self.assertEqual(default_args.max_prefill_tokens, 8192)

        with envs.SGLANG_ENABLE_GLM52_EXACT_RADIX.override(True):
            radix_args = self._glm_exact_args()
            resolve(radix_args)
            self.assertFalse(radix_args.disable_radix_cache)
            # Capacity-only changes reserve prefill scratch headroom.
            self.assertEqual(radix_args.mem_fraction_static, 0.80)
            self.assertEqual(radix_args.max_prefill_tokens, 4864)
            with self.assertRaisesRegex(ValueError, "max_prefill_tokens"):
                resolve(self._glm_exact_args(max_prefill_tokens=5376))
            with self.assertRaisesRegex(ValueError, "max_prefill_tokens"):
                resolve(self._glm_exact_args(max_prefill_tokens=8192))
            with self.assertRaisesRegex(ValueError, "mem_fraction_static"):
                stale_fraction = self._glm_exact_args(mem_fraction_static=0.82)
                stale_fraction._mem_fraction_static_user_supplied = True
                resolve(stale_fraction)

            # The late validator accepts the opt-in path and still detects
            # drift back to a disabled cache.
            radix_args.page_size = 64
            radix_args.enable_dp_attention = True
            with patch.object(envs.SGLANG_ENABLE_CP_V2, "get", return_value=True):
                radix_args._validate_glm52_exact_resolved_contract()
                radix_args.disable_radix_cache = True
                with self.assertRaisesRegex(ValueError, "disable_radix_cache"):
                    radix_args._validate_glm52_exact_resolved_contract()

            with self.assertRaisesRegex(ValueError, "conflicts"):
                resolve(self._glm_exact_args(disable_radix_cache=True))
            with self.assertRaisesRegex(ValueError, "in-device radix tree"):
                resolve(self._glm_exact_args(enable_hierarchical_cache=True))
            with self.assertRaisesRegex(ValueError, "in-device radix tree"):
                resolve(self._glm_exact_args(enable_lmcache=True))
            with self.assertRaisesRegex(ValueError, "adapter-keyed"):
                resolve(self._glm_exact_args(enable_lora=True))

    def test_glm52_xorl_accepts_transformers_normalized_rope_parameters(self):
        hf_config = self._glm_model_config().hf_config
        hf_config.rope_parameters.pop("type")
        server_args = ServerArgs(model_path="dummy")
        server_args.rl_on_policy_target = "xorl"
        server_args.nnodes = 2
        server_args.tp_size = 16

        server_args._resolve_glm52_exact_contract(
            hf_config,
            model_arch="GlmMoeDsaForCausalLM",
            is_dsa_model=True,
        )

        self.assertTrue(server_args.glm52_exact_mode)

    def test_glm52_xorl_rejects_architecture_alias_with_unqualified_geometry(self):
        cases = {
            "num_hidden_layers": 92,
            "intermediate_size": 4096,
            "moe_intermediate_size": 1024,
            "moe_layer_freq": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "first_k_dense_replace": 4,
            "mlp_layer_types": ["dense"] * 78,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10_000, "rope_type": "default"},
            "max_position_embeddings": 131_072,
            "hidden_act": "gelu",
            "norm_topk_prob": False,
            "n_group": 8,
            "topk_group": 4,
            "scoring_func": "softmax",
            "routed_scaling_factor": 1.0,
            "topk_method": "greedy",
            "tie_word_embeddings": True,
            "swiglu_limit": 7.0,
            "llama_4_scaling": {"beta": 0.1},
            "index_skip_topk_offset": 2,
            "index_topk_pattern": [True] * 78,
            "index_head_dim": 64,
            "index_n_heads": 16,
            "q_lora_rank": 1024,
            "kv_lora_rank": 256,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 32,
            "v_head_dim": 128,
            "indexer_rope_interleave": False,
            "rope_interleave": False,
            "indexer_types": ["full"] * 78,
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                hf_config = self._glm_model_config().hf_config
                setattr(hf_config, name, value)
                server_args = ServerArgs(model_path="dummy")
                server_args.rl_on_policy_target = "xorl"
                server_args.nnodes = 2
                server_args.tp_size = 16

                with self.assertRaisesRegex(
                    ValueError, f"qualified model geometry.*{name}"
                ):
                    server_args._resolve_glm52_exact_contract(
                        hf_config,
                        model_arch="GlmMoeDsaForCausalLM",
                        is_dsa_model=True,
                    )

    def test_glm52_xorl_rejects_unqualified_fp8_layout(self):
        cases = {
            "quant_method": "int8",
            "activation_scheme": "static",
            "weight_block_size": [64, 128],
            "ignored_layers": ["model.embed_tokens"],
            "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]},
            "kv_cache_quant_algo": "fp8",
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                hf_config = self._glm_model_config().hf_config
                hf_config.quantization_config[name] = value
                server_args = ServerArgs(model_path="dummy")
                server_args.rl_on_policy_target = "xorl"
                server_args.nnodes = 2
                server_args.tp_size = 16

                with self.assertRaisesRegex(ValueError, "qualified FP8"):
                    server_args._resolve_glm52_exact_contract(
                        hf_config,
                        model_arch="GlmMoeDsaForCausalLM",
                        is_dsa_model=True,
                    )

        hf_config = self._glm_model_config().hf_config
        hf_config.quantization_config["modules_to_not_convert"].pop()
        server_args = ServerArgs(model_path="dummy")
        server_args.rl_on_policy_target = "xorl"
        server_args.nnodes = 2
        server_args.tp_size = 16
        with self.assertRaisesRegex(ValueError, "modules_to_not_convert"):
            server_args._resolve_glm52_exact_contract(
                hf_config,
                model_arch="GlmMoeDsaForCausalLM",
                is_dsa_model=True,
            )

    def test_glm52_xorl_rejects_alternate_index_and_router_programs(self):
        for name, value in (("cli_factor", 2), ("num_hash_layers", 1)):
            with self.subTest(name=name):
                hf_config = self._glm_model_config().hf_config
                setattr(hf_config, name, value)
                server_args = ServerArgs(model_path="dummy")
                server_args.rl_on_policy_target = "xorl"
                server_args.nnodes = 2
                server_args.tp_size = 16
                with self.assertRaisesRegex(ValueError, name):
                    server_args._resolve_glm52_exact_contract(
                        hf_config,
                        model_arch="GlmMoeDsaForCausalLM",
                        is_dsa_model=True,
                    )

    def test_glm52_xorl_rejects_eager_and_non_bs16_graphs(self):
        cases = (
            ("disable_cuda_graph", True, False),
            ("cuda_graph_bs_decode", [8, 16], True),
        )
        for name, value, explicit_graph_bs in cases:
            with self.subTest(name=name, value=value):
                server_args = ServerArgs(model_path="dummy")
                server_args.rl_on_policy_target = "xorl"
                server_args.nnodes = 2
                server_args.tp_size = 16
                setattr(server_args, name, value)
                server_args._cuda_graph_bs_user_supplied = explicit_graph_bs
                with self.assertRaisesRegex(ValueError, "exact GLM-5.2 XORL"):
                    server_args._resolve_glm52_exact_contract(
                        self._glm_model_config().hf_config,
                        model_arch="GlmMoeDsaForCausalLM",
                        is_dsa_model=True,
                    )

    def test_non_glm_xorl_does_not_enable_the_glm_numerical_family(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.rl_on_policy_target = "xorl"
        hf_config = MagicMock()

        server_args._resolve_glm52_exact_contract(
            hf_config,
            model_arch="Qwen3ForCausalLM",
            is_dsa_model=False,
        )

        self.assertFalse(server_args.glm52_exact_mode)
        self.assertFalse(hf_config._glm52_exact_mode)
        self.assertIsNone(server_args.dsa_prefill_backend)
        self.assertEqual(server_args.moe_runner_backend, "auto")

    def test_model_specific_adjustments_resolves_qwen35_through_real_caller(self):
        """Exercise the production caller so helper-signature drift fails here."""
        hf_config = SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            text_config=SimpleNamespace(
                hidden_size=2048,
                num_hidden_layers=40,
                num_attention_heads=16,
                num_key_value_heads=2,
                vocab_size=248320,
                num_experts=256,
                num_experts_per_tok=8,
                linear_num_key_heads=16,
                linear_num_value_heads=32,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
                full_attention_interval=4,
                layer_types=[
                    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                    for i in range(40)
                ],
            ),
        )
        model_config = MagicMock(hf_config=hf_config)
        server_args = ServerArgs(model_path="dummy")
        server_args.rl_on_policy_target = "xorl"
        server_args.tp_size = 8
        server_args.ep_size = 1
        server_args.get_model_config = MagicMock(return_value=model_config)

        with patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa", return_value=False
        ):
            server_args._handle_model_specific_adjustments()

        self.assertTrue(server_args.qwen35_gdn_exact_mode)
        self.assertTrue(hf_config._qwen35_gdn_exact_mode)
        self.assertEqual(server_args.tp_size, 8)
        self.assertEqual(server_args.dp_size, 8)
        self.assertEqual(server_args.ep_size, 8)
        self.assertTrue(server_args.enable_dp_attention)

    def test_rejects_unpinned_glm_dsa_pair(self):
        server_args = self._server_args(decode_backend="flashmla_kv")
        server_args.get_model_config = MagicMock(return_value=self._glm_model_config())

        with self.assertRaisesRegex(ValueError, "Deterministic GLM DSA requires"):
            server_args._handle_deterministic_inference()

    def test_accepts_exact_sparse_decode_only_for_xorl_contract(self):
        server_args = self._server_args(decode_backend="flashmla_sparse")
        server_args.rl_on_policy_target = "xorl"
        server_args.glm52_exact_mode = True
        server_args.get_model_config = MagicMock(return_value=self._glm_model_config())

        server_args._handle_deterministic_inference()

        self.assertEqual(server_args.dsa_decode_backend, "flashmla_sparse")
        self.assertFalse(server_args.disable_radix_cache)

    def test_rejects_exact_sparse_decode_without_xorl_contract(self):
        server_args = self._server_args(decode_backend="flashmla_sparse")
        server_args.get_model_config = MagicMock(return_value=self._glm_model_config())

        with self.assertRaisesRegex(ValueError, "Deterministic GLM DSA requires"):
            server_args._handle_deterministic_inference()

    def test_top_level_fa3_remains_a_distinct_non_dsa_path(self):
        server_args = self._server_args()
        server_args.attention_backend = "fa3"
        server_args.get_model_config = MagicMock(return_value=self._glm_model_config())

        server_args._handle_deterministic_inference()

        self.assertEqual(server_args.attention_backend, "fa3")


class TestPortArgs(unittest.TestCase):
    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_standard_case(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = False

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.scheduler_input_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("ipc://"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_builds_decoupled_spec_ipc_config(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        server_args.decoupled_spec_role = "verifier"
        server_args.decoupled_spec_bind_endpoint = "ipc:///tmp/v"
        server_args.decoupled_spec_connect_endpoints = ["ipc:///tmp/d"]
        server_args.decoupled_spec_rank = 0

        port_args = PortArgs.init_new(server_args)

        self.assertIsNotNone(port_args.decoupled_spec_ipc_config)
        self.assertEqual(port_args.decoupled_spec_ipc_config.rank, 0)
        self.assertEqual(
            port_args.decoupled_spec_ipc_config.bind_endpoint, "ipc:///tmp/v"
        )
        self.assertEqual(
            port_args.decoupled_spec_ipc_config.connect_endpoints, ("ipc:///tmp/d",)
        )

    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_no_decoupled_config_when_role_null(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        # decoupled_spec_role defaults to "null"

        port_args = PortArgs.init_new(server_args)

        self.assertIsNone(port_args.decoupled_spec_ipc_config)

    def test_init_new_decoupled_role_requires_endpoints(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        server_args.decoupled_spec_role = "drafter"
        # endpoints intentionally left as their None defaults

        with self.assertRaises(ValueError):
            PortArgs.init_new(server_args)

    def test_init_new_with_single_node_dp_attention(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = None

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_dp_rank(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = "192.168.1.1:25000"

        worker_ports = [25006, 25007, 25008, 25009]
        port_args = PortArgs.init_new(server_args, dp_rank=2, worker_ports=worker_ports)

        self.assertTrue(port_args.scheduler_input_ipc_name.endswith(":25008"))

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://192.168.1.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_malformed_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("Missing port", str(context.exception))

    def test_init_new_with_malformed_ipv4_address_invalid_port(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:abc"

        with self.assertRaises(ValueError):
            PortArgs.init_new(server_args)


class TestSSLArgs(unittest.TestCase):
    def _validate_ssl(self, **kwargs):
        server_args = ServerArgs(model_path="dummy", **kwargs)
        server_args._handle_ssl_validation()
        return server_args

    def test_ssl_keyfile_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(ssl_keyfile="key.pem")
        self.assertIn("--ssl-certfile", str(context.exception))

    def test_ssl_certfile_without_keyfile_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(ssl_certfile="cert.pem")
        self.assertIn("--ssl-keyfile", str(context.exception))

    def test_url_returns_http_without_ssl(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertTrue(server_args.url().startswith("http://"))

    def test_url_rewrites_all_interfaces_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="0.0.0.0")
        self.assertEqual(server_args.url(), "http://127.0.0.1:30000")

    def test_url_rewrites_empty_host_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="")
        self.assertEqual(server_args.url(), "http://127.0.0.1:30000")

    @patch("os.path.isfile", return_value=True)
    def test_url_returns_https_with_ssl(self, _mock_isfile):
        server_args = self._validate_ssl(ssl_keyfile="key.pem", ssl_certfile="cert.pem")
        self.assertTrue(server_args.url().startswith("https://"))

    def test_ssl_verify_without_ssl(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertIs(server_args.ssl_verify(), True)

    @patch("os.path.isfile", return_value=True)
    def test_ssl_verify_with_ssl_no_ca(self, _mock_isfile):
        server_args = self._validate_ssl(ssl_keyfile="key.pem", ssl_certfile="cert.pem")
        self.assertIs(server_args.ssl_verify(), False)

    @patch("os.path.isfile", return_value=True)
    def test_ssl_verify_with_ssl_and_ca(self, _mock_isfile):
        server_args = self._validate_ssl(
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            ssl_ca_certs="ca.pem",
        )
        self.assertEqual(server_args.ssl_verify(), "ca.pem")

    def test_ssl_ca_certs_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(ssl_ca_certs="ca.pem")
        self.assertIn("--ssl-ca-certs", str(context.exception))

    def test_ssl_keyfile_password_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(ssl_keyfile_password="secret")
        self.assertIn("--ssl-keyfile-password", str(context.exception))

    def test_ssl_keyfile_not_found_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(
                ssl_keyfile="/nonexistent/key.pem",
                ssl_certfile="/nonexistent/cert.pem",
            )
        self.assertIn("not found", str(context.exception))

    def test_ssl_certfile_not_found_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as keyfile:
            with self.assertRaises(ValueError) as context:
                self._validate_ssl(
                    ssl_keyfile=keyfile.name,
                    ssl_certfile="/nonexistent/cert.pem",
                )
            self.assertIn("SSL certificate file not found", str(context.exception))

    def test_ssl_ca_certs_not_found_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as keyfile:
            with tempfile.NamedTemporaryFile(suffix=".pem") as certfile:
                with self.assertRaises(ValueError) as context:
                    self._validate_ssl(
                        ssl_keyfile=keyfile.name,
                        ssl_certfile=certfile.name,
                        ssl_ca_certs="/nonexistent/ca.pem",
                    )
                self.assertIn(
                    "SSL CA certificates file not found", str(context.exception)
                )

    def test_enable_ssl_refresh_without_ssl_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(enable_ssl_refresh=True)
        self.assertIn("--enable-ssl-refresh", str(context.exception))
        self.assertIn("--ssl-certfile", str(context.exception))

    @patch("os.path.isfile", return_value=True)
    def test_enable_ssl_refresh_with_ssl_accepted(self, _mock_isfile):
        server_args = self._validate_ssl(
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            enable_ssl_refresh=True,
        )
        self.assertTrue(server_args.enable_ssl_refresh)


class TestHiCacheArgs(unittest.TestCase):
    def _make_args(self, **overrides) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _assert_hicache_fields(
        self,
        args: ServerArgs,
        *,
        expected_io_backend: str,
        expected_mem_layout: str,
        expected_decode_backend: str | None = None,
    ):
        self.assertEqual(args.hicache_io_backend, expected_io_backend)
        self.assertEqual(args.hicache_mem_layout, expected_mem_layout)
        if expected_decode_backend is not None:
            self.assertEqual(args.decode_attention_backend, expected_decode_backend)

    def test_hicache_io_backend_and_mem_layout_compatibility(self):
        cases = [
            {
                "name": "default_kernel_page_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                },
                "expected_io_backend": "kernel",
                "expected_mem_layout": "page_first",
            },
            {
                "name": "kernel_with_page_first_direct",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first_direct",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "direct_with_page_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "page_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "mooncake_with_layer_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_storage_backend": "mooncake",
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "layer_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "fa3_kernel_with_explicit_decode_backend",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first",
                    "attention_backend": "triton",
                    "decode_attention_backend": "fa3",
                },
                "expected_io_backend": "kernel",
                "expected_mem_layout": "page_first",
                "expected_decode_backend": "fa3",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                args = self._make_args(**case["overrides"])
                args._handle_hicache()
                self._assert_hicache_fields(
                    args,
                    expected_io_backend=case["expected_io_backend"],
                    expected_mem_layout=case["expected_mem_layout"],
                    expected_decode_backend=case.get("expected_decode_backend"),
                )

    def test_hicache_kernel_keeps_implicit_fa3_decode_backend(self):
        args = self._make_args(
            enable_hierarchical_cache=True,
            hicache_io_backend="kernel",
            attention_backend="fa3",
            decode_attention_backend=None,
        )

        args._handle_hicache()

        self.assertEqual(args.hicache_io_backend, "kernel")
        self.assertEqual(args.hicache_mem_layout, "page_first")
        self.assertIsNone(args.decode_attention_backend)


class TestNgramExternalSamArgs(CustomTestCase):
    def _make_dummy_ngram_args(self, **overrides):
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "NGRAM"
        args.speculative_num_draft_tokens = 12
        args.device = "cuda"
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_external_sam_budget_must_fit_draft_budget(self):
        args = self._make_dummy_ngram_args(
            speculative_num_draft_tokens=4,
            speculative_ngram_external_corpus_path="/tmp/ngram-corpus.jsonl",
            speculative_ngram_external_sam_budget=4,
        )
        with self.assertRaises(ValueError) as context:
            handle_speculative_decoding(args)
        self.assertIn("speculative_num_draft_tokens - 1", str(context.exception))

    def test_external_corpus_max_tokens_must_be_positive(self):
        args = self._make_dummy_ngram_args(
            speculative_ngram_external_corpus_path="/tmp/ngram-corpus.jsonl",
            speculative_ngram_external_sam_budget=2,
            speculative_ngram_external_corpus_max_tokens=0,
        )
        with self.assertRaises(ValueError) as context:
            handle_speculative_decoding(args)
        self.assertIn("external-corpus-max-tokens", str(context.exception))


class TestDecoupledSpecArgs(CustomTestCase):
    """Decoupled speculative-decoding CLI flags.

    These flags are auto-derived from the ``A[...]`` field metadata on
    ``ServerArgs``; a bare annotation is silently skipped by
    ``add_cli_args_from_dataclass``. This guards against the regression where
    the flags went missing (e.g. after rebasing onto the auto-gen
    ``add_cli_args``), which the direct-attribute ``PortArgs`` tests cannot
    catch because they never exercise the CLI.
    """

    def test_decoupled_spec_cli_flags_round_trip(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--decoupled-spec-role",
                "verifier",
                "--decoupled-spec-bind-endpoint",
                "ipc:///tmp/v",
                "--decoupled-spec-connect-endpoints",
                '["ipc:///tmp/d"]',
                "--decoupled-spec-rank",
                "0",
                "--spec-trace-dir",
                "/tmp/tr",
            ]
        )
        self.assertEqual(server_args.decoupled_spec_role, "verifier")
        self.assertEqual(server_args.decoupled_spec_bind_endpoint, "ipc:///tmp/v")
        self.assertEqual(server_args.decoupled_spec_connect_endpoints, ["ipc:///tmp/d"])
        self.assertEqual(server_args.decoupled_spec_rank, 0)
        self.assertEqual(server_args.spec_trace_dir, "/tmp/tr")

    def test_decoupled_spec_role_rejects_invalid_choice(self):
        with self.assertRaises(SystemExit):
            prepare_server_args(
                ["--model-path", "dummy", "--decoupled-spec-role", "bogus"]
            )


class TestAdaptiveSpecArgs(CustomTestCase):
    def test_adaptive_defaults_to_config_step_when_spec_params_omitted(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 3, 5]},
                    "8": {"candidate_steps": [1]},
                },
                f,
            )
            f.flush()

            args = ServerArgs(model_path="dummy")
            args.speculative_algorithm = "EAGLE"
            args.speculative_adaptive = True
            args.speculative_adaptive_config = f.name
            args.device = "cuda"
            args.get_model_config = lambda: SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["LlamaForCausalLM"],
                    get_text_config=lambda: SimpleNamespace(),
                )
            )

            handle_speculative_decoding(args)

        self.assertTrue(args.speculative_adaptive)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_steps, 3)
        self.assertEqual(args.speculative_num_draft_tokens, 4)


class TestWaterfillArgs(CustomTestCase):
    def test_waterfill_enforces_shared_experts_fusion(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_waterfill=True,
            disable_shared_experts_fusion=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        from sglang.srt.arg_groups.overrides import resolved_view

        # dual-apply retired: the fields stay pristine, the declarations win
        self.assertTrue(server_args.disable_shared_experts_fusion)
        self.assertFalse(resolved_view(server_args).disable_shared_experts_fusion)
        self.assertTrue(server_args.enforce_shared_experts_fusion)

    def test_waterfill_overrides_moe_a2a_backend_to_deepep(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="none",
            enable_waterfill=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(server_args.moe_a2a_backend, "none")  # pristine
        self.assertEqual(resolved_view(server_args).moe_a2a_backend, "deepep")
        self.assertTrue(server_args.enforce_shared_experts_fusion)

    def test_waterfill_keeps_megamoe_backend(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="megamoe",
            enable_waterfill=True,
            disable_shared_experts_fusion=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(resolved_view(server_args).moe_a2a_backend, "megamoe")
        self.assertFalse(resolved_view(server_args).disable_shared_experts_fusion)
        self.assertTrue(server_args.enforce_shared_experts_fusion)

    def test_waterfill_supports_deepep_low_latency_mode(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_waterfill=True,
            deepep_mode="low_latency",
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        self.assertEqual(server_args.deepep_mode, "low_latency")
        self.assertFalse(server_args.disable_cuda_graph)
        self.assertTrue(server_args.enforce_shared_experts_fusion)


class TestPrefillOnlyDisableKvCache(unittest.TestCase):
    """Validation for --prefill-only-disable-kv-cache.

    The flag wires NoOpMHATokenToKVPool, which is only safe when:
      - the engine is in embedding mode (fa_skip_kv_cache active in FA backend),
      - chunked_prefill_size == -1 (no inter-chunk K/V reuse),
      - disable_radix_cache (radix cache otherwise indexes empty pool slots),
      - no context-parallel attention (CP writes to the pool via set_kv_buffer),
      - no HiSparse (uses a different pool family),
      - kv_cache_dtype is not nvfp4/fp4_mx_block16 (FP4 pool is a separate allocation path).
    All other configurations must be rejected before model load.
    """

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            model_path="dummy",
            is_embedding=True,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
            prefill_only_disable_kv_cache=True,
        )
        kwargs.update(overrides)
        return kwargs

    def _validate_prefill_only_args(self, **overrides):
        sa = ServerArgs(**self._base_kwargs(**overrides))
        sa._handle_legacy_cp_arguments()
        sa._validate_prefill_only_disable_kv_cache_args()
        return sa

    def test_valid_minimal_config_constructs(self):
        sa = self._validate_prefill_only_args()
        self.assertTrue(sa.prefill_only_disable_kv_cache)

    def test_rejects_when_not_embedding(self):
        with self.assertRaisesRegex(ValueError, "requires --is-embedding"):
            self._validate_prefill_only_args(is_embedding=False)

    def test_rejects_when_chunked_prefill_size_not_minus_one(self):
        with self.assertRaisesRegex(ValueError, "--chunked-prefill-size=-1"):
            self._validate_prefill_only_args(chunked_prefill_size=8192)

    def test_rejects_when_radix_cache_enabled(self):
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            self._validate_prefill_only_args(disable_radix_cache=False)

    def test_rejects_attn_cp_size_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "--attn-cp-size"):
            self._validate_prefill_only_args(attn_cp_size=2, tp_size=2)

    def test_rejects_prefill_context_parallel(self):
        with self.assertRaisesRegex(ValueError, "--enable-prefill-cp"):
            self._validate_prefill_only_args(enable_prefill_context_parallel=True)

    def test_rejects_hisparse(self):
        with self.assertRaisesRegex(ValueError, "--enable-hisparse"):
            self._validate_prefill_only_args(enable_hisparse=True)

    def test_rejects_fp4_kv_cache(self):
        for kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                with self.assertRaisesRegex(ValueError, "nvfp4.*fp4_mx_block16"):
                    self._validate_prefill_only_args(kv_cache_dtype=kv_cache_dtype)


class TestCudaGraphConfigDataclassAccess(CustomTestCase):
    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_tc_piecewise_build_config_reads_phase_config_dataclass(
        self, mock_get_moe_a2a_backend
    ):
        from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
            TcPiecewiseCudaGraphBackend,
        )

        mock_backend = mock_get_moe_a2a_backend.return_value
        mock_backend.is_deepep.return_value = False
        mock_backend.is_mooncake.return_value = False
        server_args = SimpleNamespace(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(
                    backend=Backend.TC_PIECEWISE,
                    bs=[32, 64],
                    tc_compiler="eager",
                )
            ),
            enable_torch_compile_debug_mode=False,
        )

        config = TcPiecewiseCudaGraphBackend.build_compilation_config(server_args)

        self.assertEqual(config.get_capture_sizes(), [32, 64])
        self.assertEqual(config.compiler, "eager")


class TestCudaGraphDisaggregationRoles(CustomTestCase):
    def _handled_args(self, **overrides):
        args = ServerArgs(model_path="dummy", **overrides)
        args.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=False,
            is_multimodal_piecewise_cuda_graph_supported=False,
        )
        with (
            patch("sglang.srt.utils.is_cuda", return_value=True),
            patch.object(ServerArgs, "use_mla_backend", return_value=False),
        ):
            args._handle_cuda_graph_config()
        return args

    def test_cuda_graph_prefill_role_defaults_disable_decode_graph(self):
        args = self._handled_args(disaggregation_mode="prefill")

        self.assertFalse(args.disable_cuda_graph)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_cuda_graph_decode_role_defaults_disable_prefill_graph(self):
        args = self._handled_args(disaggregation_mode="decode")

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        self.assertNotEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)

    def test_cuda_graph_global_disable_still_disables_both_phases_for_all_roles(self):
        for disaggregation_mode in ("prefill", "decode", "null"):
            with self.subTest(disaggregation_mode=disaggregation_mode):
                args = self._handled_args(
                    disaggregation_mode=disaggregation_mode,
                    disable_cuda_graph=True,
                )

                self.assertEqual(
                    args.cuda_graph_config.decode.backend, Backend.DISABLED
                )
                self.assertEqual(
                    args.cuda_graph_config.prefill.backend, Backend.DISABLED
                )

    def test_cuda_graph_explicit_decode_backend_survives_prefill_role(self):
        args = self._handled_args(
            disaggregation_mode="prefill",
            cuda_graph_backend_decode=Backend.FULL,
        )

        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertIn((Phase.DECODE, "backend"), args._cuda_graph_config_locked)


class TestPrefillCudaGraphLoRACompatibility(CustomTestCase):
    """LoRA no longer auto-disables the breakable prefill CUDA graph; guards
    test_bcg_with_lora.py against a rule re-disabling it (vacuous pass)."""

    def _handled_args(self, **overrides):
        args = ServerArgs(model_path="dummy", **overrides)
        args.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=False,
            is_multimodal_piecewise_cuda_graph_supported=False,
        )
        with (
            patch("sglang.srt.utils.is_cuda", return_value=True),
            patch.object(ServerArgs, "use_mla_backend", return_value=False),
        ):
            args._handle_cuda_graph_config()
        return args

    def test_enable_lora_keeps_breakable_prefill_graph(self):
        args = self._handled_args(enable_lora=True)

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_lora_paths_keep_breakable_prefill_graph(self):
        args = self._handled_args(lora_paths=["dummy/lora-adapter"])

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_lora_still_disables_tc_piecewise_prefill_graph(self):
        # Pin the tc_piecewise LoRA rule itself, with the hardware rule
        # neutralized so this runs on CPU-only CI.
        args = ServerArgs(model_path="dummy", enable_lora=True)
        args.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=False,
            is_multimodal_piecewise_cuda_graph_supported=False,
        )
        args.cuda_graph_config = CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE)
        )
        with (
            patch("sglang.srt.server_args.is_hip", return_value=False),
            patch("sglang.srt.server_args.is_npu", return_value=False),
            patch("sglang.srt.server_args.is_cpu", return_value=False),
            patch("sglang.srt.server_args.is_mps", return_value=False),
            patch("sglang.srt.server_args.is_xpu", return_value=False),
        ):
            args._disable_tc_piecewise_cudagraph_if_incompatible()

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)


class TestBreakableCudaGraphMultimodalAllowlist(CustomTestCase):
    """The BCG "multimodal model" rule exempts archs on the BCG multimodal
    opt-in allowlist (multimodal_breakable_cuda_graph_supported_model_archs)."""

    def _handled_args(self, *, architectures, is_multimodal, allowlisted):
        args = ServerArgs(model_path="dummy")
        args.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=is_multimodal,
            is_multimodal_piecewise_cuda_graph_supported=False,
            is_multimodal_breakable_cuda_graph_supported=allowlisted,
        )
        with (
            patch("sglang.srt.utils.is_cuda", return_value=True),
            patch.object(ServerArgs, "use_mla_backend", return_value=False),
        ):
            args._handle_cuda_graph_config()
        return args

    def test_multimodal_arch_disables_prefill_breakable(self):
        args = self._handled_args(
            architectures=["Qwen3VLForConditionalGeneration"],
            is_multimodal=True,
            allowlisted=False,
        )
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)

    def test_allowlisted_multimodal_arch_keeps_prefill_breakable(self):
        args = self._handled_args(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            is_multimodal=True,
            allowlisted=True,
        )
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_allowlist_membership(self):
        from sglang.srt.configs.model_config import (
            is_multimodal_breakable_cuda_graph_supported,
        )

        self.assertTrue(
            is_multimodal_breakable_cuda_graph_supported(
                ["Qwen3_5MoeForConditionalGeneration"]
            )
        )
        self.assertTrue(
            is_multimodal_breakable_cuda_graph_supported(
                ["Qwen3_5ForConditionalGeneration"]
            )
        )
        self.assertFalse(
            is_multimodal_breakable_cuda_graph_supported(
                ["Qwen3VLForConditionalGeneration"]
            )
        )


class TestCutedslMoeMaxNumTokens(CustomTestCase):
    """The shared CuteDSL MoE per-forward token bound. Fields are set directly
    to exercise the math independently of __post_init__ resolution.

    cg-refactor: the legacy disable_piecewise_cuda_graph /
    piecewise_cuda_graph_max_tokens / cuda_graph_max_bs fields were
    consolidated into cuda_graph_config; the helper accepts the legacy
    kwarg names for test readability and translates them to the per-phase
    dataclasses.
    """

    def _args(self, **overrides):
        server_args = ServerArgs(model_path="dummy")
        fields = dict(
            speculative_algorithm=None,
            speculative_num_draft_tokens=None,
            max_prefill_tokens=16384,
            disable_piecewise_cuda_graph=False,
            piecewise_cuda_graph_max_tokens=2048,
            cuda_graph_max_bs=512,
        )
        fields.update(overrides)
        disable_piecewise = fields.pop("disable_piecewise_cuda_graph")
        piecewise_max = fields.pop("piecewise_cuda_graph_max_tokens")
        cg_max_bs = fields.pop("cuda_graph_max_bs")
        for key, value in fields.items():
            setattr(server_args, key, value)
        server_args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, max_bs=cg_max_bs),
            prefill=PhaseConfig(
                backend=(
                    Backend.DISABLED if disable_piecewise else Backend.TC_PIECEWISE
                ),
                max_bs=piecewise_max,
                tc_compiler="eager",
            ),
        )
        return server_args

    def test_prefill_dominates_in_default_config(self):
        self.assertEqual(self._args().cutedsl_moe_max_num_tokens(), 16384)

    def test_speculative_decoding_scales_decode_bound(self):
        # decode bound 512 * 8 dominates the small prefill/piecewise bounds
        args = self._args(
            max_prefill_tokens=512,
            piecewise_cuda_graph_max_tokens=512,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=8,
        )
        self.assertEqual(args.cutedsl_moe_max_num_tokens(), 4096)

    def test_piecewise_bound_excluded_when_disabled(self):
        args = self._args(
            max_prefill_tokens=512,
            disable_piecewise_cuda_graph=True,
            cuda_graph_max_bs=64,
        )
        self.assertEqual(args.cutedsl_moe_max_num_tokens(), 512)


class TestSamplingBackendTokenOracleEnvGate(CustomTestCase):
    """The 'token_oracle' choice is gated on SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.

    The choice set is built once at server_args.py import time, so each subtest
    reloads the module with the env var set to the desired value.
    """

    def _reload_server_args_with_env(self, *, enabled: bool):
        previous = os.environ.get("SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE")
        os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1" if enabled else "0"
        try:
            return importlib.reload(server_args_module)
        finally:
            if previous is None:
                os.environ.pop("SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE", None)
            else:
                os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = previous

    def test_token_oracle_rejected_when_env_disabled(self):
        reloaded = self._reload_server_args_with_env(enabled=False)
        self.assertNotIn("token_oracle", reloaded.SAMPLING_BACKEND_CHOICES)

        with self.assertRaises(SystemExit):
            reloaded.prepare_server_args(
                [
                    "--model-path",
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                    "--sampling-backend",
                    "token_oracle",
                ]
            )

    def test_token_oracle_accepted_when_env_enabled(self):
        reloaded = self._reload_server_args_with_env(enabled=True)
        self.assertIn("token_oracle", reloaded.SAMPLING_BACKEND_CHOICES)

        parsed = reloaded.prepare_server_args(
            [
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                "--sampling-backend",
                "token_oracle",
                # Explicit device so ServerArgs.__post_init__ does not call
                # get_device() (fails on CPU-only CI runners) and does not run
                # _handle_cpu_backends (which would override sampling_backend
                # to "pytorch", masking what we want to verify).
                "--device",
                "cuda",
            ]
        )
        self.assertEqual(parsed.sampling_backend, "token_oracle")


class TestHandleCrashDumpEnv(CustomTestCase):
    _COREDUMP_ENV_KEYS = (
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
        "CUDA_ENABLE_USER_TRIGGERED_COREDUMP",
        "CUDA_COREDUMP_SHOW_PROGRESS",
        "CUDA_COREDUMP_GENERATION_FLAGS",
        "CUDA_COREDUMP_FILE",
        "CUDA_COREDUMP_PIPE",
    )

    def _run_handler(self, crash_dump_folder, preset_env=None):
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.crash_dump_folder = crash_dump_folder
        with patch.dict(os.environ, preset_env or {}):
            for key in self._COREDUMP_ENV_KEYS:
                if key not in (preset_env or {}):
                    os.environ.pop(key, None)
            ServerArgs._handle_crash_dump_env(server_args)

    def test_creates_coredump_dir_when_auto_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_handler(tmp)
            self.assertTrue(
                os.path.isdir(os.path.join(tmp, socket.gethostname())),
                "coredump dir not created for auto-set CUDA_COREDUMP_FILE",
            )

    def test_creates_coredump_dir_when_env_preset(self):
        # Regression test: when CUDA_COREDUMP_FILE is preset, the coredump
        # directory must still be created up front.
        with tempfile.TemporaryDirectory() as tmp:
            preset_dir = os.path.join(tmp, "preset-location")
            self._run_handler(
                tmp,
                preset_env={"CUDA_COREDUMP_FILE": f"{preset_dir}/%h/core.cuda.%t.%p"},
            )
            self.assertTrue(
                os.path.isdir(os.path.join(preset_dir, socket.gethostname())),
                "coredump dir not created for preset CUDA_COREDUMP_FILE",
            )


class TestGrpcServerArgs(CustomTestCase):
    """Native gRPC is enabled by --grpc-port (or SGLANG_GRPC_PORT) and runs
    alongside HTTP; --smg-grpc-mode (and the deprecated --grpc-mode) select the
    legacy SMG server. Worker-threads / max-prefill-tokens are env-only knobs.

    The gRPC setup lives in ServerArgs._handle_deprecated_args, which
    __post_init__ skips for dummy models, so these tests build a dummy
    ServerArgs and invoke that handler directly (mirroring the real flow for a
    concrete model path).
    """

    @staticmethod
    def _args(**kwargs):
        return ServerArgs(model_path="dummy", **kwargs)

    def test_http_only_high_port_does_not_derive_grpc_port(self):
        sa = self._args(port=56000)
        sa._handle_deprecated_args()
        self.assertIsNone(sa.grpc_port)

    def test_grpc_port_enables_native_and_env_knobs(self):
        sa = self._args(grpc_port=50051)
        with envs.SGLANG_GRPC_WORKER_THREADS.override(8):
            sa._handle_deprecated_args()
        self.assertEqual(sa.grpc_port, 50051)
        self.assertEqual(sa.grpc_worker_threads, 8)

    def test_env_grpc_port_enables_native(self):
        sa = self._args(port=30000)
        with envs.SGLANG_GRPC_PORT.override(45000):
            sa._handle_deprecated_args()
        self.assertEqual(sa.grpc_port, 45000)

    @staticmethod
    def _sidecar_parser():
        parser = server_args_module.argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser

    def test_sidecar_builds_loopback_grpc_endpoints(self):
        self.assertEqual(
            build_sidecar_endpoint(SimpleNamespace(host="0.0.0.0", grpc_port=50051)),
            "http://127.0.0.1:50051",
        )
        self.assertEqual(
            build_sidecar_endpoint(SimpleNamespace(host="::", grpc_port=50051)),
            "http://[::1]:50051",
        )
        self.assertEqual(
            build_sidecar_endpoint(SimpleNamespace(host="[::]", grpc_port=50051)),
            "http://[::1]:50051",
        )

    def test_sidecar_args_parse_as_exact_json_argv(self):
        argv = ["--flag", "value"]
        parsed = self._sidecar_parser().parse_args(
            ["--model-path", "dummy", "--sidecar-args", json.dumps(argv)]
        )
        self.assertEqual(parsed.sidecar_args, argv)

    def test_start_sidecar_passes_endpoint_and_provider_argv_separately(self):
        server_args = SimpleNamespace(
            sidecar="example.sidecar",
            sidecar_args=[
                "--sidecar-shutdown-timeout",
                "42",
                "--grpc-connections",
                "2",
            ],
            host="127.0.0.1",
            grpc_port=50051,
        )
        with (
            patch("sglang.srt.entrypoints.sidecar.mp.get_context") as get_context,
            patch("sglang.srt.entrypoints.sidecar.Sidecar") as sidecar_class,
        ):
            start_sidecar(server_args)

        process_kwargs = get_context.return_value.Process.call_args.kwargs
        self.assertEqual(process_kwargs["name"], "sglang_sidecar_example.sidecar")
        self.assertEqual(process_kwargs["target"], _run_sidecar)
        self.assertEqual(
            process_kwargs["args"],
            (
                "example.sidecar",
                ["--grpc-connections", "2"],
                "http://127.0.0.1:50051",
            ),
        )
        sidecar_class.assert_called_once_with(
            get_context.return_value.Process.return_value,
            "example.sidecar",
            shutdown_timeout=42.0,
        )

    def test_sidecar_requires_native_grpc(self):
        sa = self._args(sidecar="example.sidecar")
        with self.assertRaisesRegex(ValueError, "requires --grpc-port"):
            sa._handle_deprecated_args()

    def test_sidecar_rejects_legacy_grpc(self):
        sa = self._args(sidecar="example.sidecar", smg_grpc_mode=True)
        with self.assertRaisesRegex(ValueError, "native gRPC server"):
            sa._handle_deprecated_args()

    def test_sidecar_rejects_empty_value(self):
        sa = self._args(sidecar="", grpc_port=50051)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            sa._handle_deprecated_args()

    def test_sidecar_sets_endpoint_env_before_import_and_calls_main(self):
        main = MagicMock()

        def import_module(module_name):
            self.assertEqual(module_name, "example.sidecar")
            self.assertEqual(
                os.environ[SGLANG_GRPC_ENDPOINT_ENV],
                "http://127.0.0.1:50051",
            )
            self.assertEqual(os.environ["DYN_NAMESPACE"], "pluh")
            return SimpleNamespace(main=main)

        with (
            patch.dict(
                os.environ,
                {
                    SGLANG_GRPC_ENDPOINT_ENV: "http://stale.example:1",
                    "DYN_NAMESPACE": "pluh",
                },
            ),
            patch("sglang.srt.entrypoints.sidecar.kill_itself_when_parent_died"),
            patch(
                "sglang.srt.entrypoints.sidecar.importlib.import_module",
                side_effect=import_module,
            ),
        ):
            _run_sidecar(
                "example.sidecar",
                ["--provider-flag", "value"],
                "http://127.0.0.1:50051",
            )

        main.assert_called_once_with(["--provider-flag", "value"])

    def test_sidecar_stop_uses_configured_shutdown_timeout(self):
        proc = MagicMock(pid=1234)
        proc.is_alive.side_effect = [True, True]
        sidecar = Sidecar(
            proc,
            "example.sidecar",
            shutdown_timeout=42.0,
        )

        with patch("sglang.srt.entrypoints.sidecar.kill_process_tree") as kill_tree:
            sidecar.stop()

        proc.terminate.assert_called_once_with()
        proc.join.assert_called_once_with(timeout=42.0)
        kill_tree.assert_called_once_with(1234, wait_timeout=42.0)

    def test_legacy_smg_derives_grpc_port_from_http_port(self):
        sa = self._args(port=30000, smg_grpc_mode=True)
        sa._handle_deprecated_args()
        self.assertEqual(sa.grpc_port, 40000)

    def test_grpc_mode_is_deprecated_alias_for_smg_grpc_mode(self):
        sa = self._args(grpc_mode=True)
        with self.assertLogs(server_args_module.logger, level="WARNING") as cm:
            sa._handle_deprecated_args()
        self.assertTrue(sa.smg_grpc_mode)
        self.assertTrue(any("--grpc-mode is deprecated" in line for line in cm.output))

    def test_legacy_smg_takes_precedence_over_grpc_port(self):
        sa = self._args(grpc_port=50051, smg_grpc_mode=True)
        sa._handle_deprecated_args()
        self.assertTrue(sa.smg_grpc_mode)
        self.assertEqual(sa.grpc_port, 50051)

    def test_native_grpc_rejects_multi_tokenizer(self):
        sa = self._args(grpc_port=40000, tokenizer_worker_num=2)
        with self.assertRaises(ValueError):
            sa._handle_deprecated_args()

    def test_native_grpc_rejects_http_auth(self):
        sa = self._args(grpc_port=40000, api_key="secret")
        with self.assertRaises(ValueError):
            sa._handle_deprecated_args()

    def test_invalid_grpc_worker_threads_rejected(self):
        sa = self._args(grpc_port=40000)
        with envs.SGLANG_GRPC_WORKER_THREADS.override(0):
            with self.assertRaises(ValueError):
                sa._handle_deprecated_args()

    def test_start_server_call_site_matches_native_signature(self):
        """Regression for the startup blocker: the native start_server binding
        only accepts (host, port, runtime_handle, worker_threads, ...). The
        arg-parsing tests above never call start_server, so a stray kwarg (e.g.
        the removed max_prefill_tokens) would only surface as a TypeError at
        launch. This mocks the native extension and locks the kwarg set."""
        import sys

        from sglang.srt.entrypoints import http_server

        fake_core = SimpleNamespace(start_server=MagicMock(return_value="handle"))
        fake_bridge = SimpleNamespace(RuntimeHandle=MagicMock(return_value="rt"))
        server_args = SimpleNamespace(
            host="127.0.0.1", grpc_port=50051, grpc_worker_threads=4
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.grpc": SimpleNamespace(_core=fake_core),
                "sglang.srt.grpc._core": fake_core,
                "sglang.srt.entrypoints.grpc_bridge": fake_bridge,
            },
        ):
            handle = http_server._start_native_grpc_server_for_runtime(
                server_args=server_args,
                tokenizer_manager=MagicMock(),
                template_manager=MagicMock(),
                scheduler_info={},
            )

        self.assertEqual(handle, "handle")
        _, kwargs = fake_core.start_server.call_args
        self.assertEqual(
            set(kwargs), {"host", "port", "runtime_handle", "worker_threads"}
        )
        self.assertNotIn("max_prefill_tokens", kwargs)


class TestTwoBatchOverlapBackend(CustomTestCase):
    """Non-EP DP two-batch-overlap backend requirement.

    With no EP a2a backend (moe_a2a_backend='none'), --enable-two-batch-overlap
    is only valid on the DeepSeek-V4 non-EP DP TP-MoE path (overlapping the DP
    all_gatherv / reduce_scatterv with the other ubatch's compute), which
    requires --enable-dp-attention. This replaced the removed opt-in
    SGLANG_ENABLE_DP_TBO env: enabling DP TBO now needs no extra flag.

    dummy-model short-circuits __post_init__, so the guard handler is invoked
    directly (same pattern as TestWaterfillArgs)."""

    def _args(self, **overrides):
        args = ServerArgs(model_path="dummy")
        args.enable_two_batch_overlap = True
        args.moe_a2a_backend = "none"
        args.enable_dp_attention = False
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_no_a2a_without_dp_attention_raises(self):
        args = self._args(enable_dp_attention=False)
        with self.assertRaisesRegex(ValueError, "enable-dp-attention"):
            args._check_two_batch_overlap()

    def test_no_a2a_with_dp_attention_ok(self):
        # DP TBO path is valid: --enable-dp-attention + --enable-two-batch-overlap
        # with a2a backend 'none' must NOT raise (no SGLANG_ENABLE_DP_TBO needed).
        args = self._args(enable_dp_attention=True)
        args._check_two_batch_overlap()

    def test_ep_a2a_backend_ok_without_dp_attention(self):
        # EP a2a path (e.g. deepep) overlaps dispatch/combine; the guard does not
        # require dp-attention there.
        args = self._args(moe_a2a_backend="deepep", enable_dp_attention=False)
        args._check_two_batch_overlap()


if __name__ == "__main__":
    unittest.main()
