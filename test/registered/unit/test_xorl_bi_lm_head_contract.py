import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.batch_invariant_ops import bi_families_v2
from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    get_batch_invariant_ops,
)
from sglang.srt.layers.xorl_batch_invariant import (
    BI_FAMILIES_V2_CONTRACT,
    XORL_GLM52_REQUIRED_BI_OPS,
    log_xorl_bi_contract_plan_once,
    record_xorl_bi_engagement,
    record_xorl_glm52_pipeline_stage_receipt,
    resolve_or_validate_xorl_bi_family,
    resolve_xorl_bi_family,
    validate_xorl_glm52_norm_envelope,
    validate_xorl_bi_logit_transforms,
    xorl_glm52_norm_site_family,
    xorl_bi_lm_head,
    xorl_bi_sample_and_score,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.server_args import (
    RL_ON_POLICY_TARGET_CHOICES,
    XORL_RL_TARGET,
    ServerArgs,
    is_batch_invariant_rl_target,
    is_glm52_exact_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _module_stub(name, **attributes):
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


@contextmanager
def _paired_family(family):
    value = "1" if family == "v2" else "0"
    with patch.dict(os.environ, {}, clear=False):
        os.environ["XORL_FAMILIES_V2"] = value
        os.environ["SGLANG_FAMILIES_V2"] = value
        yield


def _load_rms_norm_class():
    sgl_kernel = _module_stub(
        "sgl_kernel",
        fused_add_rmsnorm=lambda *args, **kwargs: None,
        gemma_fused_add_rmsnorm=lambda *args, **kwargs: None,
        gemma_rmsnorm=lambda *args, **kwargs: None,
        rmsnorm=lambda *args, **kwargs: None,
    )
    source = Path(__file__).parents[3] / "python/sglang/srt/layers/layernorm.py"
    spec = importlib.util.spec_from_file_location("_xorl_test_layernorm", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"sgl_kernel": sgl_kernel}):
        spec.loader.exec_module(module)
    return module, module.RMSNorm


def _load_scheduler_output_processor_mixin():
    """Load the production mixin without importing unrelated GPU kernels."""
    empty_type = type("_UnusedSchedulerDependency", (), {})
    dependency_stubs = {
        "sglang.srt.disaggregation.utils": _module_stub(
            "sglang.srt.disaggregation.utils",
            DisaggregationMode=SimpleNamespace(DECODE="decode"),
        ),
        "sglang.srt.layers.logits_processor": _module_stub(
            "sglang.srt.layers.logits_processor",
            LogitsProcessorOutput=empty_type,
        ),
        "sglang.srt.layers.moe.routed_experts_capturer": _module_stub(
            "sglang.srt.layers.moe.routed_experts_capturer",
            get_global_experts_capturer=lambda: None,
        ),
        "sglang.srt.managers.io_struct": _module_stub(
            "sglang.srt.managers.io_struct",
            AbortReq=empty_type,
            BatchEmbeddingOutput=empty_type,
            BatchTokenIDOutput=empty_type,
        ),
        "sglang.srt.managers.schedule_batch": _module_stub(
            "sglang.srt.managers.schedule_batch",
            BaseFinishReason=empty_type,
            Req=empty_type,
            ScheduleBatch=empty_type,
        ),
        "sglang.srt.mem_cache.common": _module_stub(
            "sglang.srt.mem_cache.common",
            release_kv_cache=lambda *args: None,
        ),
    }
    source = (
        Path(__file__).parents[3]
        / "python/sglang/srt/managers/scheduler_output_processor_mixin.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_xorl_test_scheduler_output_processor_mixin", source
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependency_stubs):
        spec.loader.exec_module(module)
    return module.SchedulerOutputProcessorMixin


SchedulerOutputProcessorMixin = _load_scheduler_output_processor_mixin()


class _NoSpecAlgorithm:
    @staticmethod
    def is_none():
        return True


class _SchedulerResponseHarness(SchedulerOutputProcessorMixin):
    """Run the production response methods while isolating external side effects."""

    def __init__(self):
        self.is_generation = True
        self.enable_overlap = False
        self.enable_metrics = False
        self.num_generated_tokens = 0
        self.forward_ct_decode = 0
        self.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            multi_item_scoring_delimiter=None,
        )
        self.tree_cache = SimpleNamespace(cache_unfinished_req=lambda req: None)
        self.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=lambda: None,
            free_group_end=lambda: None,
        )

    def stream_output(self, *args, **kwargs):
        pass

    def report_prefill_stats(self, *args, **kwargs):
        pass

    def report_decode_stats(self, *args, **kwargs):
        pass


class TestXorlBatchInvariantTarget(unittest.TestCase):
    def test_xorl_exact_mode_is_resolved_from_the_model(self):
        self.assertIn(XORL_RL_TARGET, RL_ON_POLICY_TARGET_CHOICES)
        self.assertFalse(is_batch_invariant_rl_target(XORL_RL_TARGET))
        self.assertFalse(is_glm52_exact_mode(SimpleNamespace()))
        self.assertTrue(is_glm52_exact_mode(SimpleNamespace(glm52_exact_mode=True)))

    def test_standard_unfiltered_request_sets_no_rejected_sampling_flag(self):
        params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
        )

        self.assertEqual(params.top_k, TOP_K_ALL)
        self.assertFalse(params.top_k <= 1)
        self.assertFalse(params.top_p != 1.0)
        self.assertFalse(params.top_k != TOP_K_ALL)
        self.assertFalse(params.min_p > 0)

    def test_glm52_exact_mode_rejects_speculative_and_draft_modes(self):
        for kwargs in (
            {"speculative_algorithm": "EAGLE"},
            {"speculative_draft_model_path": "draft-model"},
            {"enable_multi_layer_eagle": True},
        ):
            with self.subTest(kwargs=kwargs):
                args = ServerArgs(
                    model_path="dummy",
                    rl_on_policy_target=XORL_RL_TARGET,
                    **kwargs,
                )
                args.glm52_exact_mode = True
                with self.assertRaisesRegex(
                    ValueError, "does not support speculative or draft decoding"
                ):
                    args._validate_glm52_exact_contract()

        args = ServerArgs(model_path="dummy", speculative_algorithm="EAGLE")
        self.assertEqual(args.speculative_algorithm, "EAGLE")

    def test_v2_family_exports_semantic_contract(self):
        self.assertEqual(BI_FAMILIES_V2_CONTRACT, "xorl_batch_invariant_families_v2")

    def test_family_receipt_defaults_v2_supports_paired_rollback_and_rejects_mismatch(
        self,
    ):
        for flags, expected in (
            ({}, "v2"),
            ({"XORL_FAMILIES_V2": "1", "SGLANG_FAMILIES_V2": "true"}, "v2"),
            ({"XORL_FAMILIES_V2": "0", "SGLANG_FAMILIES_V2": "false"}, "v1"),
        ):
            with self.subTest(flags=flags, expected=expected):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("XORL_FAMILIES_V2", None)
                    os.environ.pop("SGLANG_FAMILIES_V2", None)
                    os.environ.update(flags)
                    self.assertEqual(resolve_xorl_bi_family(), expected)

        for flags in (
            {"XORL_FAMILIES_V2": "0", "SGLANG_FAMILIES_V2": "1"},
            {"XORL_FAMILIES_V2": "1", "SGLANG_FAMILIES_V2": "0"},
            {"XORL_FAMILIES_V2": "0"},
        ):
            with self.subTest(flags=flags):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("XORL_FAMILIES_V2", None)
                    os.environ.pop("SGLANG_FAMILIES_V2", None)
                    os.environ.update(flags)
                    with self.assertRaisesRegex(RuntimeError, "family flags disagree"):
                        resolve_xorl_bi_family()

        with patch.dict(
            os.environ,
            {"XORL_FAMILIES_V2": "maybe", "SGLANG_FAMILIES_V2": "1"},
        ):
            with self.assertRaisesRegex(RuntimeError, "is invalid"):
                resolve_xorl_bi_family()

        with _paired_family("v2"):
            self.assertEqual(resolve_or_validate_xorl_bi_family("v2"), "v2")
            with self.assertRaisesRegex(RuntimeError, "process-wide contract"):
                resolve_or_validate_xorl_bi_family("v1")

    def test_whole_family_plan_is_once_and_declares_peer_requirement(self):
        receipt_logger = MagicMock()
        with (
            _paired_family("v2"),
            patch(
                "sglang.srt.layers.xorl_batch_invariant._CONTRACT_PLAN_LOGGED",
                False,
            ),
        ):
            kwargs = dict(
                use_qk_norm=False,
                speculative_decode=False,
                mtp_decode=False,
                legacy_bi_ops=("addmm", "bmm", "log_softmax", "mean", "mm"),
            )
            self.assertEqual(
                log_xorl_bi_contract_plan_once(receipt_logger, **kwargs), "v2"
            )
            self.assertEqual(
                log_xorl_bi_contract_plan_once(receipt_logger, **kwargs), "v2"
            )

        receipt_logger.info.assert_called_once()
        message = receipt_logger.info.call_args.args[0]
        self.assertIn("resolved_use_qk_norm=false", message)
        self.assertIn("speculative_decode=false mtp_decode=false", message)
        self.assertIn("legacy_bi_ops=%s glm52_bi_router=true", message)
        self.assertIn("required_peer_trainer_rmsnorm_mode=sglang_fused", message)
        self.assertNotIn("observed", message)
        self.assertEqual(receipt_logger.info.call_args.args[2], BI_FAMILIES_V2_CONTRACT)

        invalid_plans = (
            {**kwargs, "use_qk_norm": True},
            {**kwargs, "speculative_decode": True},
            {**kwargs, "mtp_decode": True},
            {**kwargs, "legacy_bi_ops": ("mm",)},
        )
        for invalid in invalid_plans:
            with self.subTest(invalid=invalid), _paired_family("v2"):
                with self.assertRaises(RuntimeError):
                    log_xorl_bi_contract_plan_once(MagicMock(), **invalid)

    def test_exact_contract_selects_its_ops_without_an_environment_override(self):
        disable_batch_invariant_mode()
        with (
            patch.dict(os.environ, {"SGLANG_BATCH_INVARIANT_OPS": "all"}),
            patch("torch.library.Library"),
        ):
            try:
                enable_batch_invariant_mode(ops=XORL_GLM52_REQUIRED_BI_OPS)
                self.assertEqual(
                    get_batch_invariant_ops(),
                    XORL_GLM52_REQUIRED_BI_OPS,
                )
            finally:
                disable_batch_invariant_mode()

    def test_observation_receipt_separates_template_paths_from_sampled_boundary(self):
        receipt_logger = MagicMock()
        fresh_counts = {
            "rmsnorm": 0,
            "lm_head": 0,
            "bi_router_gemm": 0,
            "canonical_moe": 0,
            "sampler_score": 0,
        }
        with (
            patch.dict(
                "sglang.srt.layers.xorl_batch_invariant._ENGAGEMENT_COUNTS",
                fresh_counts,
                clear=True,
            ),
            patch(
                "sglang.srt.layers.xorl_batch_invariant._ENGAGEMENT_RECEIPT_LOGGED",
                False,
            ),
        ):
            for component in (
                "rmsnorm",
                "lm_head",
                "bi_router_gemm",
                "canonical_moe",
            ):
                record_xorl_bi_engagement(component, receipt_logger=receipt_logger)
            receipt_logger.info.assert_not_called()
            record_xorl_bi_engagement(
                "sampler_score",
                require_complete=True,
                receipt_logger=receipt_logger,
            )
            record_xorl_bi_engagement(
                "sampler_score",
                require_complete=True,
                receipt_logger=receipt_logger,
            )

        receipt_logger.info.assert_called_once()
        message = receipt_logger.info.call_args.args[0]
        self.assertIn("numerical observation receipt", message)
        self.assertIn("template_or_eager_path_observed=%s", message)
        self.assertIn("real_sampled_boundary_observed=sampler_score", message)
        self.assertIn("cuda_graph_replay_python_instrumentation=false", message)
        self.assertEqual(
            receipt_logger.info.call_args.args[1],
            "rmsnorm,lm_head,bi_router_gemm,canonical_moe",
        )

        with patch.dict(
            "sglang.srt.layers.xorl_batch_invariant._ENGAGEMENT_COUNTS",
            {component: 0 for component in fresh_counts},
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "missing_template_or_eager_path=.*"
                "cuda_graph_replay_python_instrumentation=false",
            ):
                record_xorl_bi_engagement("sampler_score", require_complete=True)

    def test_non_final_pipeline_stage_receipt_is_exact_and_one_shot(self):
        receipt_logger = MagicMock()
        stage_counts = {
            "rmsnorm": 152,
            "lm_head": 0,
            "bi_router_gemm": 35,
            "canonical_moe": 1,
            "sampler_score": 0,
        }
        with (
            patch.dict(
                "sglang.srt.layers.xorl_batch_invariant._ENGAGEMENT_COUNTS",
                stage_counts,
                clear=True,
            ),
            patch(
                "sglang.srt.layers.xorl_batch_invariant._PIPELINE_STAGE_RECEIPT_LOGGED",
                False,
            ),
        ):
            for _ in range(2):
                record_xorl_glm52_pipeline_stage_receipt(
                    pp_rank=0,
                    start_layer=0,
                    end_layer=38,
                    moe_layer_count=35,
                    receipt_logger=receipt_logger,
                )

        receipt_logger.info.assert_called_once()
        self.assertIn(
            "pipeline-stage observation receipt", receipt_logger.info.call_args.args[0]
        )
        self.assertEqual(receipt_logger.info.call_args.args[1:4], (0, 0, 38))
        self.assertEqual(
            receipt_logger.info.call_args.args[4],
            "rmsnorm:152,lm_head:0,bi_router_gemm:35,canonical_moe:1,sampler_score:0",
        )

        for bad_counts in (
            {**stage_counts, "rmsnorm": 151},
            {**stage_counts, "lm_head": 1},
            {**stage_counts, "sampler_score": 1},
        ):
            with (
                self.subTest(bad_counts=bad_counts),
                patch.dict(
                    "sglang.srt.layers.xorl_batch_invariant._ENGAGEMENT_COUNTS",
                    bad_counts,
                    clear=True,
                ),
                patch(
                    "sglang.srt.layers.xorl_batch_invariant."
                    "_PIPELINE_STAGE_RECEIPT_LOGGED",
                    False,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "disagree with the stage contract"
                ),
            ):
                record_xorl_glm52_pipeline_stage_receipt(
                    pp_rank=0,
                    start_layer=0,
                    end_layer=38,
                    moe_layer_count=35,
                )

    def test_official_glm52_norm_site_map_and_qk_envelope(self):
        self.assertEqual(xorl_glm52_norm_site_family("q_a"), "serving_no_residual")
        self.assertEqual(xorl_glm52_norm_site_family("kv_a"), "serving_no_residual")
        self.assertEqual(
            xorl_glm52_norm_site_family("input", layer_id=0),
            "serving_no_residual",
        )
        self.assertEqual(
            xorl_glm52_norm_site_family("input", layer_id=1),
            "serving_residual_tree",
        )
        self.assertEqual(
            xorl_glm52_norm_site_family("post_attention"),
            "serving_residual_tree",
        )
        self.assertEqual(xorl_glm52_norm_site_family("final"), "serving_residual_tree")
        validate_xorl_glm52_norm_envelope(use_qk_norm=False)
        with self.assertRaisesRegex(RuntimeError, "does not certify use_qk_norm"):
            validate_xorl_glm52_norm_envelope(use_qk_norm=True)

        deepseek_source = (
            Path(__file__).parents[3] / "python/sglang/srt/models/deepseek_v2.py"
        ).read_text()
        for site in ("q_a", "kv_a", "input", "post_attention", "final"):
            self.assertIn(f'xorl_glm52_norm_site_family("{site}"', deepseek_source)
        self.assertIn("validate_xorl_glm52_norm_envelope(", deepseek_source)


class TestXorlBatchInvariantRMSNormDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layernorm, cls.rms_norm_class = _load_rms_norm_class()

    @staticmethod
    def _server_args():
        return SimpleNamespace(
            rl_on_policy_target=XORL_RL_TARGET,
            glm52_exact_mode=True,
        )

    def _norm(self, family):
        return self.rms_norm_class(
            16,
            weight_dtype=torch.bfloat16,
            batch_invariant_family=family,
        )

    def test_actual_rmsnorm_dispatches_v2_for_no_residual_and_residual_sites(self):
        x = torch.zeros((8, 16), dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        no_residual = self._norm("serving_no_residual")
        residual_tree = self._norm("serving_residual_tree")
        no_residual_out = object()
        residual_out = (object(), object())

        with (
            _paired_family("v2"),
            patch.object(
                self.layernorm, "is_batch_invariant_mode_enabled", return_value=True
            ),
            patch.object(self.layernorm, "get_global_server_args", self._server_args),
            patch.object(
                self.layernorm,
                "rms_norm_v2",
                side_effect=[no_residual_out, residual_out],
            ) as v2,
            patch.object(
                self.layernorm,
                "bi_rms_norm",
                side_effect=AssertionError("v1 no-residual norm used"),
            ),
            patch.object(
                self.layernorm,
                "bi_fused_add_rms_norm",
                side_effect=AssertionError("v1 residual norm used"),
            ),
            patch.object(self.layernorm, "record_xorl_bi_engagement"),
        ):
            self.assertIs(no_residual.forward_cuda(x), no_residual_out)
            self.assertIs(
                residual_tree.forward_cuda(x, residual),
                residual_out,
            )

        self.assertEqual(v2.call_count, 2)
        self.assertIsNone(v2.call_args_list[0].kwargs["residual"])
        self.assertIs(v2.call_args_list[1].kwargs["residual"], residual)

    def test_actual_rmsnorm_dispatches_paired_v1_site_families(self):
        x = torch.zeros((1, 16), dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        no_residual = self._norm("serving_no_residual")
        residual_tree = self._norm("serving_residual_tree")
        no_residual_out = object()
        residual_out = (object(), object())

        with (
            _paired_family("v1"),
            patch.object(
                self.layernorm, "is_batch_invariant_mode_enabled", return_value=True
            ),
            patch.object(self.layernorm, "get_global_server_args", self._server_args),
            patch.object(
                self.layernorm,
                "rms_norm_v2",
                side_effect=AssertionError("v2 norm used"),
            ),
            patch.object(
                self.layernorm, "bi_rms_norm", return_value=no_residual_out
            ) as v1_single,
            patch.object(
                self.layernorm,
                "bi_fused_add_rms_norm",
                return_value=residual_out,
            ) as v1_residual,
            patch.object(self.layernorm, "record_xorl_bi_engagement"),
        ):
            self.assertIs(no_residual.forward_cuda(x), no_residual_out)
            self.assertIs(
                residual_tree.forward_cuda(x, residual),
                residual_out,
            )

        self.assertEqual(v1_single.call_args.kwargs["family"], "serving_no_residual")
        self.assertEqual(
            v1_residual.call_args.kwargs["family"], "serving_residual_tree"
        )

    def test_actual_rmsnorm_fails_closed_on_undeclared_or_flipped_site(self):
        x = torch.zeros((1, 16), dtype=torch.bfloat16)
        with (
            _paired_family("v2"),
            patch.object(
                self.layernorm, "is_batch_invariant_mode_enabled", return_value=True
            ),
            patch.object(self.layernorm, "get_global_server_args", self._server_args),
        ):
            undeclared = self.rms_norm_class(16, weight_dtype=torch.bfloat16)
            with self.assertRaisesRegex(RuntimeError, "without an explicit"):
                undeclared.forward_cuda(x)

            no_residual = self._norm("serving_no_residual")
            with self.assertRaisesRegex(RuntimeError, "received a residual"):
                no_residual.forward_cuda(x, torch.ones_like(x))


class TestXorlBatchInvariantHeadAndSampler(unittest.TestCase):
    @staticmethod
    def _sampling_info(n: int):
        return SimpleNamespace(
            temperatures=torch.ones((n, 1), dtype=torch.float32),
            sampling_seed=torch.arange(1, n + 1, dtype=torch.int64),
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            has_custom_logit_processor=False,
            acc_linear_penalties=None,
            penalizer_orchestrator=None,
            vocab_mask=None,
            logit_bias=None,
        )

    @staticmethod
    def _response_req():
        return SimpleNamespace(
            finished=lambda: False,
            is_retracted=False,
            is_chunked=0,
            is_prefill_only=False,
            time_stats=SimpleNamespace(
                set_prefill_finished_time=lambda: None,
                set_last_decode_finish_time=lambda: None,
            ),
            output_ids=[],
            check_finished=lambda *args: None,
            return_logprob=True,
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            top_logprobs_num=0,
            token_ids_logprob=None,
            return_hidden_states=False,
            grammar=None,
            origin_input_ids=[1, 2],
            mamba_ping_pong_track_buffer=None,
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None,
            input_token_ids_logprobs_idx=None,
        )

    def test_bi_head_engages_versioned_family_for_n1_and_n8_without_matmul(self):
        for family in ("v1", "v2"):
            for n in (1, 8):
                with self.subTest(family=family, n=n), _paired_family(family):
                    hidden = torch.zeros((n, 16), dtype=torch.bfloat16)
                    weight = torch.zeros((32, 16), dtype=torch.bfloat16)
                    expected = torch.zeros((n, 32), dtype=torch.float32)
                    expected_lse = torch.zeros(n, dtype=torch.float32)

                    with (
                        patch(
                            "sglang.srt.layers.xorl_batch_invariant."
                            "bi_lm_head_full_logits",
                            return_value=expected,
                        ) as v1_head,
                        patch(
                            "sglang.srt.layers.xorl_batch_invariant."
                            "head_v2_full_logits_with_lse",
                            return_value=(expected, expected_lse),
                        ) as v2_head,
                        patch(
                            "torch.matmul",
                            side_effect=AssertionError("ordinary matmul used"),
                        ),
                        patch(
                            "sglang.srt.layers.xorl_batch_invariant."
                            "record_xorl_bi_engagement"
                        ),
                    ):
                        actual = xorl_bi_lm_head(
                            hidden,
                            SimpleNamespace(weight=weight),
                            use_fp32_lm_head=False,
                            family=family,
                        )

                    used_head = v2_head if family == "v2" else v1_head
                    unused_head = v1_head if family == "v2" else v2_head
                    self.assertIs(actual, expected)
                    self.assertIs(used_head.call_args.args[0], hidden)
                    self.assertIs(used_head.call_args.args[1], weight)
                    unused_head.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_v2_tp8_head_composition_matches_global_trainer_contract(self):
        generator = torch.Generator(device="cuda").manual_seed(1234)
        hidden_size = 64
        vocab_size = 8192
        tp_size = 8
        weight = torch.randn(
            (vocab_size, hidden_size),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)

        for n in (1, 8):
            with self.subTest(n=n), _paired_family("v2"), torch.no_grad():
                hidden = torch.randn(
                    (n, hidden_size),
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                ).to(torch.bfloat16)
                token_ids = torch.tensor(
                    (
                        [7000]
                        if n == 1
                        else [0, 1023, 1024, 2047, 4095, 4096, 7000, 8191]
                    ),
                    device="cuda",
                    dtype=torch.int64,
                )

                trainer_lp, trainer_lse, trainer_selected = (
                    bi_families_v2.head_v2_selected_logprob(
                        hidden,
                        weight,
                        token_ids,
                    )
                )
                serving_logits = torch.cat(
                    [
                        bi_families_v2.head_v2_full_logits_with_lse(
                            hidden,
                            shard,
                        )[0]
                        for shard in weight.tensor_split(tp_size, dim=0)
                    ],
                    dim=1,
                )
                serving_lp, serving_lse, serving_selected = (
                    bi_families_v2.head_v2_selected_logprob_from_logits(
                        serving_logits,
                        token_ids,
                    )
                )

                self.assertTrue(torch.equal(trainer_lp, serving_lp))
                self.assertTrue(torch.equal(trainer_lse, serving_lse))
                self.assertTrue(torch.equal(trainer_selected, serving_selected))

    def test_bi_head_and_post_head_transforms_fail_closed(self):
        hidden = torch.zeros((1, 16), dtype=torch.bfloat16)
        weight = torch.zeros((32, 16), dtype=torch.bfloat16)

        with self.assertRaisesRegex(RuntimeError, "enable-fp32-lm-head"):
            xorl_bi_lm_head(
                hidden,
                SimpleNamespace(weight=weight),
                use_fp32_lm_head=True,
            )

        lora_head = SimpleNamespace(
            weight=weight,
            set_lora=lambda: None,
            apply_lora=lambda: None,
        )
        with self.assertRaisesRegex(RuntimeError, "LoRA-wrapped"):
            xorl_bi_lm_head(hidden, lora_head, use_fp32_lm_head=False)

        with self.assertRaisesRegex(RuntimeError, "embedding bias"):
            xorl_bi_lm_head(
                hidden,
                SimpleNamespace(weight=weight),
                use_fp32_lm_head=False,
                embedding_bias=torch.zeros(32, dtype=torch.bfloat16),
            )

        for bad_hidden, bad_weight in (
            (hidden.float(), weight),
            (hidden, weight.float()),
        ):
            with self.subTest(hidden=bad_hidden.dtype, weight=bad_weight.dtype):
                with self.assertRaisesRegex(RuntimeError, "requires BF16"):
                    xorl_bi_lm_head(
                        bad_hidden,
                        SimpleNamespace(weight=bad_weight),
                        use_fp32_lm_head=False,
                    )

        with self.assertRaisesRegex(RuntimeError, "does not support logit_scale"):
            validate_xorl_bi_logit_transforms(0.5, None)
        with self.assertRaisesRegex(
            RuntimeError, "does not support final_logit_softcapping"
        ):
            validate_xorl_bi_logit_transforms(None, 30.0)

    def test_sample_score_and_sync_use_same_logits_for_both_families(self):
        for family in ("v1", "v2"):
            for n in (1, 8):
                with self.subTest(family=family, n=n), _paired_family(family):
                    logits = torch.arange(n * 32, dtype=torch.float32).reshape(n, 32)
                    output = SimpleNamespace(
                        next_token_logits=logits,
                        next_token_logprobs=None,
                    )
                    initial_ids = torch.zeros(n, dtype=torch.int32)
                    selected_logprobs = -torch.arange(1, n + 1, dtype=torch.float32)
                    events = []

                    def sample_from_logprobs(actual_logits, sampling_info, positions):
                        events.append(("sample", actual_logits, positions))
                        return initial_ids

                    def sync_token_ids(token_ids, sampling_info):
                        events.append(("sync", token_ids.clone()))
                        token_ids.add_(1)

                    def selected_logprob(actual_logits, token_ids):
                        events.append(("score", actual_logits, token_ids.clone()))
                        return selected_logprobs, object(), object()

                    with (
                        patch(
                            "sglang.srt.layers.xorl_batch_invariant."
                            "bi_lm_head_selected_logprob_from_logits",
                            side_effect=selected_logprob,
                        ) as v1_score,
                        patch(
                            "sglang.srt.layers.xorl_batch_invariant."
                            "head_v2_selected_logprob_from_logits",
                            side_effect=selected_logprob,
                        ) as v2_score,
                        patch(
                            "torch.log_softmax",
                            side_effect=AssertionError("ordinary log_softmax used"),
                        ),
                        patch(
                            "torch.nn.functional.log_softmax",
                            side_effect=AssertionError("ordinary log_softmax used"),
                        ),
                        patch(
                            "sglang.srt.layers.xorl_batch_invariant."
                            "record_xorl_bi_engagement"
                        ) as engagement,
                    ):
                        actual_ids = xorl_bi_sample_and_score(
                            output,
                            self._sampling_info(n),
                            return_logprob=True,
                            top_logprobs_nums=[0] * n,
                            token_ids_logprobs=[None] * n,
                            positions=torch.arange(n),
                            sample_from_logprobs=sample_from_logprobs,
                            sync_token_ids=sync_token_ids,
                            enable_deterministic=True,
                            return_original_logprob=False,
                            family=family,
                        )

                    used_score = v2_score if family == "v2" else v1_score
                    unused_score = v1_score if family == "v2" else v2_score
                    used_score.assert_called_once()
                    unused_score.assert_not_called()
                    engagement.assert_called_once_with(
                        "sampler_score", require_complete=True
                    )
                    self.assertIs(actual_ids, initial_ids)
                    self.assertEqual(
                        [event[0] for event in events], ["sample", "sync", "score"]
                    )
                    self.assertIs(events[0][1], logits)
                    self.assertIs(events[2][1], logits)
                    torch.testing.assert_close(
                        events[1][1], torch.zeros(n, dtype=torch.int32)
                    )
                    torch.testing.assert_close(
                        events[2][2], torch.ones(n, dtype=torch.int32)
                    )
                    self.assertIs(output.next_token_logprobs, selected_logprobs)

    def test_extend_response_consumes_sampled_id_logprob_pair(self):
        scheduler = _SchedulerResponseHarness()
        req = self._response_req()
        output = SimpleNamespace(
            next_token_logits=torch.zeros((1, 8), dtype=torch.float32),
            next_token_logprobs=torch.tensor([-0.75], dtype=torch.float32),
            input_token_logprobs=None,
            hidden_states=None,
            customized_info=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=True,
            decoding_reqs=[],
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            logits_output=output,
            next_token_ids=torch.tensor([17], dtype=torch.int32),
            extend_input_len_per_req=[0],
            extend_logprob_start_len_per_req=[0],
            can_run_cuda_graph=False,
        )

        scheduler.process_batch_result_prefill(batch, result)

        self.assertEqual(req.output_ids, [17])
        self.assertEqual(req.output_token_logprobs_idx, [17])
        self.assertEqual(req.output_token_logprobs_val, [-0.75])

    def test_kv_decode_response_consumes_sampled_id_logprob_pair(self):
        scheduler = _SchedulerResponseHarness()
        req = self._response_req()
        output = SimpleNamespace(
            next_token_logits=torch.zeros((1, 8), dtype=torch.float32),
            next_token_logprobs=torch.tensor([-1.25], dtype=torch.float32),
            hidden_states=None,
            customized_info=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=True,
            spec_algorithm=_NoSpecAlgorithm(),
            is_spec_v2=False,
        )
        result = SimpleNamespace(
            copy_done=None,
            logits_output=output,
            next_token_ids=torch.tensor([29], dtype=torch.int32),
            can_run_cuda_graph=True,
            num_accepted_tokens=1,
        )

        scheduler.process_batch_result_decode(batch, result)

        self.assertEqual(req.output_ids, [29])
        self.assertEqual(req.output_token_logprobs_idx, [29])
        self.assertEqual(req.output_token_logprobs_val, [-1.25])

    def test_sampler_fails_closed_on_unsupported_inputs(self):
        cases = {
            "wrong-logits-dtype": (
                "output",
                SimpleNamespace(
                    next_token_logits=torch.zeros((1, 8), dtype=torch.bfloat16),
                    next_token_logprobs=None,
                ),
                "requires FP32 logits",
            ),
            "deterministic": ("enable_deterministic", False, "deterministic inference"),
            "temperature": (
                "temperatures",
                torch.tensor([[0.5]]),
                "temperature == 1",
            ),
            "top-p": ("need_top_p_sampling", True, "does not support top-p"),
            "top-k": ("need_top_k_sampling", True, "does not support top-p"),
            "min-p": ("need_min_p_sampling", True, "does not support top-p"),
            "penalty": (
                "penalizer_orchestrator",
                SimpleNamespace(is_required=True),
                "does not support penalties",
            ),
        }

        for name, (field, value, message) in cases.items():
            with self.subTest(name=name):
                output = SimpleNamespace(
                    next_token_logits=torch.zeros((1, 8)),
                    next_token_logprobs=None,
                )
                sampling_info = self._sampling_info(1)
                enable_deterministic = True
                if field == "output":
                    output = value
                elif field == "enable_deterministic":
                    enable_deterministic = value
                else:
                    setattr(sampling_info, field, value)

                with self.assertRaisesRegex(RuntimeError, message):
                    xorl_bi_sample_and_score(
                        output,
                        sampling_info,
                        return_logprob=True,
                        top_logprobs_nums=[0],
                        token_ids_logprobs=[None],
                        positions=torch.tensor([0]),
                        sample_from_logprobs=lambda *args: torch.tensor([0]),
                        sync_token_ids=lambda *args: None,
                        enable_deterministic=enable_deterministic,
                        return_original_logprob=False,
                    )


if __name__ == "__main__":
    unittest.main()
