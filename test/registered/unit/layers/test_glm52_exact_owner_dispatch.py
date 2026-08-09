"""Production-owner dispatch tests for the GLM-5.2 exact sampler and LM head."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import logits_processor as logits_module
from sglang.srt.layers import sampler as sampler_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestGlm52ExactOwnerDispatch(CustomTestCase):
    @staticmethod
    def _make_sampler(exact_mode):
        server_args = SimpleNamespace(glm52_exact_mode=exact_mode)
        execution = SimpleNamespace(
            deterministic=SimpleNamespace(
                rl_on_policy_target="xorl" if exact_mode else None,
                enable_deterministic_inference=exact_mode,
            ),
            kernel=SimpleNamespace(sampling_backend="pytorch"),
        )
        with (
            patch.object(
                sampler_module,
                "is_glm52_exact_mode",
                return_value=exact_mode,
            ) as selector,
            patch.object(
                sampler_module,
                "get_server_args",
                return_value=server_args,
            ),
            patch.object(
                sampler_module,
                "get_tp_group",
                return_value=SimpleNamespace(device_group=object()),
            ),
            patch.object(
                sampler_module,
                "is_dp_attention_enabled",
                return_value=False,
            ),
            patch.object(sampler_module, "get_exec", return_value=execution),
        ):
            sampler = sampler_module.Sampler()
        return sampler, selector, server_args

    @staticmethod
    def _make_logits_processor(
        exact_mode,
        *,
        logit_scale=None,
        final_logit_softcapping=None,
    ):
        server_args = SimpleNamespace(glm52_exact_mode=exact_mode)
        execution = SimpleNamespace(
            features=SimpleNamespace(
                enable_fp32_lm_head=False,
                enable_mis=False,
            ),
            deterministic=SimpleNamespace(
                rl_on_policy_target="xorl" if exact_mode else None,
            ),
        )
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            tp_size=1,
            attn_dp_size=1,
        )
        config = SimpleNamespace(vocab_size=32)
        if final_logit_softcapping is not None:
            config.final_logit_softcapping = final_logit_softcapping
        with (
            patch.object(
                logits_module,
                "is_glm52_exact_mode",
                return_value=exact_mode,
            ) as selector,
            patch.object(
                logits_module,
                "get_server_args",
                return_value=server_args,
            ),
            patch.object(logits_module, "get_exec", return_value=execution),
            patch.object(logits_module, "get_parallel", return_value=parallel),
            patch.object(logits_module.triton_symm_mem_ag, "MultimemAllGatherer"),
            patch.object(logits_module, "InputLogprobProcessor"),
        ):
            processor = logits_module.LogitsProcessor(
                config,
                logit_scale=logit_scale,
            )
        return processor, selector, server_args

    def test_exact_sampler_dispatches_before_generic_preprocessing(self):
        """Exact GLM sampling must not silently re-enter stock logit handling."""
        sampler, selector, server_args = self._make_sampler(True)
        selector.assert_called_once_with(server_args)

        logits_output = SimpleNamespace(
            next_token_logits=torch.zeros((1, 32), dtype=torch.float32),
            next_token_logprobs=None,
        )
        sampling_info = SimpleNamespace()
        positions = torch.tensor([7], dtype=torch.int64)
        expected_ids = torch.tensor([11], dtype=torch.int32)
        sampler._preprocess_logits = MagicMock(
            side_effect=AssertionError("generic preprocessing ran")
        )

        with patch.object(
            sampler_module,
            "xorl_bi_sample_and_score",
            return_value=expected_ids,
        ) as exact_sample:
            actual_ids = sampler(
                logits_output,
                sampling_info,
                return_logprob=True,
                top_logprobs_nums=[0],
                token_ids_logprobs=[None],
                positions=positions,
            )

        self.assertIs(actual_ids, expected_ids)
        sampler._preprocess_logits.assert_not_called()
        exact_sample.assert_called_once_with(
            logits_output,
            sampling_info,
            return_logprob=True,
            top_logprobs_nums=[0],
            token_ids_logprobs=[None],
            positions=positions,
            sample_from_logprobs=sampler._sample_from_logprobs,
            sync_token_ids=sampler._sync_token_ids_across_tp,
            enable_deterministic=True,
            return_original_logprob=sampler_module.SGLANG_RETURN_ORIGINAL_LOGPROB,
        )

    def test_non_exact_sampler_keeps_stock_greedy_path(self):
        """A false exact-mode bit must leave ordinary sampler dispatch intact."""
        sampler, selector, server_args = self._make_sampler(False)
        selector.assert_called_once_with(server_args)

        logits = torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.float32)
        logits_output = SimpleNamespace(next_token_logits=logits)
        sampling_info = SimpleNamespace(
            is_all_greedy=True,
            return_sampling_masks=[],
            grammars=[],
        )
        sampler._preprocess_logits = MagicMock(return_value=logits)

        with patch.object(
            sampler_module,
            "xorl_bi_sample_and_score",
        ) as exact_sample:
            actual_ids = sampler(
                logits_output,
                sampling_info,
                return_logprob=False,
                top_logprobs_nums=[0],
                token_ids_logprobs=[None],
                positions=torch.tensor([0]),
            )

        self.assertTrue(torch.equal(actual_ids, torch.tensor([1])))
        sampler._preprocess_logits.assert_called_once_with(logits, sampling_info)
        exact_sample.assert_not_called()

    def test_exact_lm_head_dispatches_before_lora_quant_and_matmul(self):
        """Exact GLM logits must not fall through to any stock LM-head branch."""
        processor, selector, server_args = self._make_logits_processor(True)
        selector.assert_called_once_with(server_args)

        hidden_states = torch.zeros((1, 16), dtype=torch.bfloat16)
        lm_head = MagicMock(spec=["set_lora", "apply_lora"])
        expected_logits = torch.zeros((1, 32), dtype=torch.float32)
        with (
            patch.object(
                logits_module,
                "xorl_bi_lm_head",
                return_value=expected_logits,
            ) as exact_head,
            patch.object(
                logits_module,
                "should_apply_lm_head_quant_method",
                side_effect=AssertionError("quantized head ran"),
            ) as quantized_head,
            patch.object(
                torch,
                "matmul",
                side_effect=AssertionError("generic matmul ran"),
            ) as generic_matmul,
        ):
            actual_logits = processor._compute_lm_head(hidden_states, lm_head)

        self.assertIs(actual_logits, expected_logits)
        exact_head.assert_called_once_with(
            hidden_states,
            lm_head,
            use_fp32_lm_head=False,
            embedding_bias=None,
        )
        lm_head.assert_not_called()
        quantized_head.assert_not_called()
        generic_matmul.assert_not_called()

    def test_exact_lm_head_fails_closed_on_scale_and_softcap(self):
        """Unsupported post-head transforms cannot bypass exact dispatch."""
        for kwargs, message in (
            ({"logit_scale": 0.5}, "does not support logit_scale"),
            (
                {"final_logit_softcapping": 30.0},
                "does not support final_logit_softcapping",
            ),
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self._make_logits_processor(True, **kwargs)

    def test_non_exact_lm_head_keeps_stock_dispatch(self):
        """A false exact-mode bit must preserve LoRA, quant, and dense heads."""
        processor, selector, server_args = self._make_logits_processor(
            False,
            logit_scale=0.5,
            final_logit_softcapping=30.0,
        )
        selector.assert_called_once_with(server_args)
        hidden_states = torch.zeros((1, 16), dtype=torch.bfloat16)

        with patch.object(
            logits_module,
            "xorl_bi_lm_head",
        ) as exact_head:
            lora_logits = torch.ones((1, 32))
            lora_head = MagicMock(spec=["set_lora", "apply_lora"])
            lora_head.return_value = lora_logits
            self.assertIs(
                processor._compute_lm_head(hidden_states, lora_head),
                lora_logits,
            )

            quant_logits = torch.full((1, 32), 2.0)
            quant_method = SimpleNamespace(apply=MagicMock(return_value=quant_logits))
            quant_head = SimpleNamespace(
                weight=torch.zeros((32, 16)),
                quant_method=quant_method,
            )
            with patch.object(
                logits_module,
                "should_apply_lm_head_quant_method",
                return_value=True,
            ):
                self.assertIs(
                    processor._compute_lm_head(hidden_states, quant_head),
                    quant_logits,
                )

            dense_logits = torch.full((1, 32), 3.0)
            dense_head = SimpleNamespace(
                weight=torch.zeros((32, 16), dtype=torch.bfloat16)
            )
            with patch.object(torch, "matmul", return_value=dense_logits) as matmul:
                self.assertIs(
                    processor._compute_lm_head(hidden_states, dense_head),
                    dense_logits,
                )
                matmul.assert_called_once()

        exact_head.assert_not_called()
        self.assertEqual(processor.logit_scale, 0.5)
        self.assertEqual(processor.final_logit_softcapping, 30.0)


if __name__ == "__main__":
    unittest.main()
