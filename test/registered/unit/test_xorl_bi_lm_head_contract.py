import importlib.util
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
    resolve_or_validate_xorl_bi_family,
    resolve_xorl_bi_family,
    validate_xorl_bi_logit_transforms,
    validate_xorl_glm52_norm_envelope,
    xorl_bi_lm_head,
    xorl_bi_sample_and_score,
    xorl_glm52_norm_site_family,
)
from sglang.srt.lora.layers import ParallelLMHeadWithLoRA
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.server_args import (
    RL_ON_POLICY_TARGET_CHOICES,
    XORL_RL_TARGET,
    ServerArgs,
    is_batch_invariant_rl_target,
    is_glm52_exact_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _module_stub(name, **attributes):
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


class _TestParallelLMHeadWithLoRA(ParallelLMHeadWithLoRA):
    def __init__(
        self,
        *,
        weight,
        backend_name="triton",
        batch_info=object(),
        set_lora=True,
        a_buffer=None,
        b_buffer=None,
        callback=None,
    ):
        torch.nn.Module.__init__(self)
        self.weight = weight
        self.set_lora = set_lora
        self.lora_backend = SimpleNamespace(
            name=backend_name,
            batch_info=batch_info,
            _glm52_exact_batch_certified=True,
        )
        if a_buffer is not None:
            self.lm_head_A_buffer = a_buffer
        if b_buffer is not None:
            self.lm_head_B_buffer = b_buffer
        self.callback = callback

    def apply_lora(self, base_output, hidden_states):
        if self.callback is None:
            raise AssertionError("test LoRA callback was not configured")
        return self.callback(base_output, hidden_states)


@contextmanager
def _paired_family(family):
    with patch("sglang.srt.layers.xorl_batch_invariant._FORCED_FAMILY", family):
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


class _NoSpecAlgorithm:
    @staticmethod
    def is_none():
        return True


def _scheduler_response_processor():
    metrics_reporter = SimpleNamespace(
        num_generated_tokens=0,
        forward_ct_decode=0,
        report_prefill_stats=lambda *args, **kwargs: None,
        report_decode_stats=lambda *args, **kwargs: None,
        update_spec_metrics=lambda *args, **kwargs: None,
    )
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(return_hidden_states_mode="none"),
        model_config=SimpleNamespace(think_end_ids=None, vocab_size=8),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_group_begin=lambda: None,
            free_group_end=lambda: None,
        ),
        tree_cache=SimpleNamespace(),
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=SimpleNamespace(),
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=SimpleNamespace(),
        logprob_result_processor=SchedulerLogprobResultProcessor(
            model_config=SimpleNamespace(vocab_size=8)
        ),
        output_streamer=SimpleNamespace(
            stream_output=lambda *args, **kwargs: None,
        ),
        abort_request=lambda *args, **kwargs: None,
    )


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

    def test_family_is_structural_and_validates_explicit_callers(self):
        self.assertEqual(resolve_xorl_bi_family(), "v2")
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
        with patch("torch.library.Library"):
            try:
                enable_batch_invariant_mode(ops=XORL_GLM52_REQUIRED_BI_OPS)
                self.assertEqual(
                    get_batch_invariant_ops(),
                    XORL_GLM52_REQUIRED_BI_OPS,
                )
            finally:
                disable_batch_invariant_mode()

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
        return SamplingBatchInfo(
            temperatures=torch.ones((n, 1), dtype=torch.float32),
            top_ps=torch.ones(n, dtype=torch.float32),
            top_ks=torch.full((n,), TOP_K_ALL, dtype=torch.int32),
            min_ps=torch.zeros(n, dtype=torch.float32),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=32,
            sampling_seed=torch.arange(1, n + 1, dtype=torch.int64),
            device="cpu",
        )

    @staticmethod
    def _response_req():
        return SimpleNamespace(
            finished=lambda: False,
            is_retracted=False,
            inflight_middle_chunks=0,
            is_prefill_only=False,
            time_stats=SimpleNamespace(
                set_prefill_finished_time=lambda: None,
                set_last_decode_finish_time=lambda: None,
            ),
            output_ids=[],
            update_finish_state=lambda *args: None,
            return_logprob=True,
            logprob=SimpleNamespace(
                output_token_logprobs_val=[],
                output_token_logprobs_idx=[],
                top_logprobs_num=0,
                token_ids_logprob=None,
                input_token_logprobs_val=None,
                input_token_logprobs_idx=None,
                input_top_logprobs_val=None,
                input_top_logprobs_idx=None,
                input_token_ids_logprobs_val=None,
                input_token_ids_logprobs_idx=None,
            ),
            return_hidden_states=False,
            return_sampling_mask=False,
            grammar=None,
            require_reasoning=False,
            customized_info=None,
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

    def test_bi_head_composes_literal_rank_one_lora_after_exact_base(self):
        hidden = torch.zeros((2, 16), dtype=torch.bfloat16)
        weight = torch.zeros((32, 16), dtype=torch.bfloat16)
        base_logits = torch.arange(64, dtype=torch.float32).reshape(2, 32)
        captures = {}

        def apply_lora(actual_base, actual_hidden):
            captures["base"] = actual_base
            captures["hidden"] = actual_hidden
            actual_base.add_(2.0)
            return actual_base

        active_head = _TestParallelLMHeadWithLoRA(
            weight=weight,
            a_buffer=torch.zeros((2, 1, 16), dtype=torch.bfloat16),
            b_buffer=torch.zeros((2, 32, 1), dtype=torch.bfloat16),
            callback=apply_lora,
        )

        with (
            _paired_family("v2"),
            patch(
                "sglang.srt.layers.xorl_batch_invariant.head_v2_full_logits_with_lse",
                return_value=(base_logits, torch.zeros(2)),
            ) as exact_base,
            patch("torch.matmul", side_effect=AssertionError("ordinary matmul used")),
        ):
            actual = xorl_bi_lm_head(
                hidden,
                active_head,
                use_fp32_lm_head=False,
            )

        self.assertIs(actual, base_logits)
        self.assertIs(captures["base"], base_logits)
        self.assertIs(captures["hidden"], hidden)
        self.assertTrue(
            torch.equal(
                actual,
                torch.arange(64, dtype=torch.float32).reshape(2, 32) + 2,
            )
        )
        exact_base.assert_called_once_with(hidden, weight)

    def test_bi_head_keeps_exact_base_for_inactive_lora_wrapper(self):
        hidden = torch.zeros((1, 16), dtype=torch.bfloat16)
        weight = torch.zeros((32, 16), dtype=torch.bfloat16)
        base_logits = torch.zeros((1, 32), dtype=torch.float32)
        inactive = _TestParallelLMHeadWithLoRA(
            weight=weight,
            set_lora=False,
            batch_info=None,
            callback=MagicMock(side_effect=AssertionError("inactive LoRA ran")),
        )

        with (
            _paired_family("v2"),
            patch(
                "sglang.srt.layers.xorl_batch_invariant.head_v2_full_logits_with_lse",
                return_value=(base_logits, torch.zeros(1)),
            ),
        ):
            actual = xorl_bi_lm_head(
                hidden,
                inactive,
                use_fp32_lm_head=False,
            )

        self.assertIs(actual, base_logits)
        inactive.callback.assert_not_called()

    def test_bi_head_and_post_head_transforms_fail_closed(self):
        hidden = torch.zeros((1, 16), dtype=torch.bfloat16)
        weight = torch.zeros((32, 16), dtype=torch.bfloat16)

        with self.assertRaisesRegex(RuntimeError, "enable-fp32-lm-head"):
            xorl_bi_lm_head(
                hidden,
                SimpleNamespace(weight=weight),
                use_fp32_lm_head=True,
            )

        lora_head = _TestParallelLMHeadWithLoRA(
            weight=weight,
            backend_name="csgmv",
            a_buffer=torch.zeros((2, 1, 16), dtype=torch.bfloat16),
            b_buffer=torch.zeros((2, 32, 1), dtype=torch.bfloat16),
            callback=lambda *_args: None,
        )
        with (
            _paired_family("v2"),
            patch(
                "sglang.srt.layers.xorl_batch_invariant.head_v2_full_logits_with_lse",
                return_value=(
                    torch.zeros((1, 32), dtype=torch.float32),
                    torch.zeros(1),
                ),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Triton LoRA backend"):
                xorl_bi_lm_head(hidden, lora_head, use_fp32_lm_head=False)

            zero_rank_head = _TestParallelLMHeadWithLoRA(
                weight=weight,
                a_buffer=torch.zeros((2, 0, 16), dtype=torch.bfloat16),
                b_buffer=torch.zeros((2, 32, 0), dtype=torch.bfloat16),
                callback=lambda *_args: None,
            )
            with self.assertRaisesRegex(RuntimeError, "physical buffers"):
                xorl_bi_lm_head(hidden, zero_rank_head, use_fp32_lm_head=False)

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

                    def selected_logprob(actual_logits, token_ids, *, temperature):
                        events.append(("score", actual_logits, token_ids.clone()))
                        self.assertIsNone(temperature)
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
                    self.assertIs(actual_ids, initial_ids)
                    self.assertEqual(
                        [event[0] for event in events], ["sample", "sync", "score"]
                    )
                    self.assertTrue(torch.equal(events[0][1], logits))
                    self.assertTrue(torch.equal(events[2][1], logits))
                    self.assertIs(events[0][1], events[2][1])
                    torch.testing.assert_close(
                        events[1][1], torch.zeros(n, dtype=torch.int32)
                    )
                    torch.testing.assert_close(
                        events[2][2], torch.ones(n, dtype=torch.int32)
                    )
                    self.assertIs(output.next_token_logprobs, selected_logprobs)

    def test_mixed_temperature_is_applied_once_before_sampling_and_scoring(self):
        logits = torch.tensor(
            [[1.25, -0.5, 0.75], [-1.0, 2.0, 0.25]],
            dtype=torch.float32,
        )
        temperatures = torch.tensor([[0.7], [1.3]], dtype=torch.float32)
        sampling_info = self._sampling_info(2)
        sampling_info.temperatures = temperatures
        output = SimpleNamespace(next_token_logits=logits, next_token_logprobs=None)
        sampled = torch.tensor([2, 1], dtype=torch.int32)
        seen = {}

        def sample_from_logprobs(transformed, *_args):
            seen["sample"] = transformed
            return sampled

        def score(transformed, token_ids, *, temperature):
            seen["score"] = transformed
            self.assertIsNone(temperature)
            self.assertTrue(torch.equal(token_ids, sampled))
            return torch.tensor([-0.5, -0.75]), None, None

        with (
            _paired_family("v2"),
            patch(
                "sglang.srt.layers.xorl_batch_invariant."
                "head_v2_selected_logprob_from_logits",
                side_effect=score,
            ),
        ):
            xorl_bi_sample_and_score(
                output,
                sampling_info,
                return_logprob=True,
                top_logprobs_nums=[0, 0],
                token_ids_logprobs=[None, None],
                positions=torch.arange(2),
                sample_from_logprobs=sample_from_logprobs,
                sync_token_ids=lambda *_args: None,
                enable_deterministic=True,
                return_original_logprob=False,
                family="v2",
            )

        expected = logits * (1.0 / temperatures)
        self.assertTrue(torch.equal(seen["sample"], expected))
        self.assertIs(seen["sample"], seen["score"])

    def test_sampler_accepts_absent_optional_logprob_lists(self):
        output = SimpleNamespace(
            next_token_logits=torch.zeros((1, 32), dtype=torch.float32),
            next_token_logprobs=None,
        )
        expected = torch.tensor([3], dtype=torch.int32)

        actual = xorl_bi_sample_and_score(
            output,
            self._sampling_info(1),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            positions=torch.tensor([0]),
            sample_from_logprobs=lambda *_args: expected,
            sync_token_ids=lambda *_args: None,
            enable_deterministic=True,
            return_original_logprob=False,
        )

        self.assertIs(actual, expected)

    def test_sampling_without_logprob_accepts_absent_logprob_metadata(self):
        logits = torch.arange(32, dtype=torch.float32).reshape(1, 32)
        output = SimpleNamespace(
            next_token_logits=logits,
            next_token_logprobs=None,
        )
        sampled = torch.tensor([3], dtype=torch.int32)

        with (
            _paired_family("v2"),
            patch(
                "sglang.srt.layers.xorl_batch_invariant."
                "head_v2_selected_logprob_from_logits"
            ) as score,
        ):
            actual = xorl_bi_sample_and_score(
                output,
                self._sampling_info(1),
                return_logprob=False,
                top_logprobs_nums=None,
                token_ids_logprobs=None,
                positions=torch.tensor([0]),
                sample_from_logprobs=lambda *_args: sampled,
                sync_token_ids=lambda *_args: None,
                enable_deterministic=True,
                return_original_logprob=False,
            )

        self.assertIs(actual, sampled)
        self.assertIsNone(output.next_token_logprobs)
        score.assert_not_called()

    def test_extend_response_consumes_sampled_id_logprob_pair(self):
        scheduler = _scheduler_response_processor()
        req = self._response_req()
        output = SimpleNamespace(
            next_token_logits=torch.zeros((1, 8), dtype=torch.float32),
            next_token_logprobs=torch.tensor([-0.75], dtype=torch.float32),
            input_token_logprobs=None,
            next_token_top_logprobs_val=None,
            next_token_top_logprobs_idx=None,
            next_token_token_ids_logprobs_val=None,
            next_token_token_ids_logprobs_idx=None,
            hidden_states=None,
            customized_info=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=True,
            return_hidden_states=False,
            return_hidden_states_mode=0,
            spec_info=None,
            decoding_reqs=[],
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=output,
            next_token_ids=torch.tensor([17], dtype=torch.int32),
            extend_input_len_per_req=[0],
            extend_logprob_start_len_per_req=[0],
            can_run_cuda_graph=False,
            grammar_advanced=False,
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.batch_result_processor."
                "maybe_cache_unfinished_req",
                lambda *args, **kwargs: None,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.batch_result_processor."
                "get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.logprob_result_processor."
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_mis=False)
                ),
            ),
            patch.object(
                SchedulerBatchResultProcessor,
                "_get_prefill_hidden_capture_mode",
                return_value=None,
            ),
        ):
            scheduler.process_batch_result_prefill(batch, result)

        self.assertEqual(req.output_ids, [17])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [17])
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.75])

    def test_kv_decode_response_consumes_sampled_id_logprob_pair(self):
        scheduler = _scheduler_response_processor()
        req = self._response_req()
        output = SimpleNamespace(
            next_token_logits=torch.zeros((1, 8), dtype=torch.float32),
            next_token_logprobs=torch.tensor([-1.25], dtype=torch.float32),
            next_token_top_logprobs_val=None,
            next_token_top_logprobs_idx=None,
            next_token_token_ids_logprobs_val=None,
            next_token_token_ids_logprobs_idx=None,
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
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=output,
            next_token_ids=torch.tensor([29], dtype=torch.int32),
            can_run_cuda_graph=True,
            num_accepted_tokens=1,
            num_correct_drafts=None,
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.batch_result_processor."
                "get_observability",
                return_value=SimpleNamespace(enable_metrics=False),
            ),
            patch.object(
                SchedulerBatchResultProcessor,
                "_handle_finish_state_updated_req",
                return_value=None,
            ),
        ):
            scheduler.process_batch_result_decode(batch, result)

        self.assertEqual(req.output_ids, [29])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [29])
        self.assertEqual(req.logprob.output_token_logprobs_val, [-1.25])

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
            "top-p": ("need_top_p_sampling", True, "does not support top-p"),
            "top-k": ("need_top_k_sampling", True, "does not support top-p"),
            "min-p": ("need_min_p_sampling", True, "does not support top-p"),
            "penalty": (
                "penalizer_orchestrator",
                SimpleNamespace(is_required=True),
                "does not support penalties",
            ),
            "additive-penalty": (
                "acc_additive_penalties",
                torch.zeros((1, 8)),
                "does not support penalties",
            ),
            "scaling-penalty": (
                "acc_scaling_penalties",
                torch.ones((1, 8)),
                "does not support penalties",
            ),
            "grammar": ("grammars", [object()], "does not support penalties"),
            "grammar-mask": (
                "grammar_mask",
                object(),
                "does not support penalties",
            ),
            "logit-bias": (
                "logit_bias",
                torch.zeros((1, 8)),
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
