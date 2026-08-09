"""Unit tests for model configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_embedding_gemma,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )


class TestEmbeddingGemmaConfig(CustomTestCase):
    def test_detects_bidirectional_gemma3_text_config(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=True
        )
        self.assertTrue(is_embedding_gemma(config))

    def test_does_not_misclassify_causal_gemma3(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False
        )
        self.assertFalse(is_embedding_gemma(config))


class TestExactRuntimeContractConfig(CustomTestCase):
    def test_fresh_runner_config_carries_the_resolved_glm52_target_contract(self):
        build_from_server_args = ModelConfig.from_server_args

        for exact_mode, is_draft_model, expected in (
            (True, False, True),
            (True, True, False),
            (False, False, False),
        ):
            with self.subTest(
                exact_mode=exact_mode,
                is_draft_model=is_draft_model,
            ):
                server_args = ServerArgs(model_path="dummy")
                server_args.glm52_exact_mode = exact_mode
                runner_config = SimpleNamespace(
                    hf_config=SimpleNamespace(),
                    hf_text_config=SimpleNamespace(),
                )

                with patch(
                    "sglang.srt.configs.model_config.ModelConfig",
                    return_value=runner_config,
                ):
                    result = build_from_server_args(
                        server_args,
                        is_draft_model=is_draft_model,
                    )

                self.assertIs(result, runner_config)
                self.assertEqual(
                    result.hf_config._glm52_exact_mode,
                    expected,
                )
                self.assertEqual(
                    result.hf_text_config._glm52_exact_mode,
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
