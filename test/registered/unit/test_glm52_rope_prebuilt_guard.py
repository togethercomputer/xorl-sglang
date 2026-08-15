import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.utils.common import reserve_rope_cache_for_long_sequences
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _exec_config(
    *,
    rl_on_policy_target=None,
    glm52_exact_mode=False,
    qwen35_gdn_exact_mode=False,
):
    return SimpleNamespace(
        deterministic=SimpleNamespace(
            rl_on_policy_target=rl_on_policy_target,
            glm52_exact_mode=glm52_exact_mode,
            qwen35_gdn_exact_mode=qwen35_gdn_exact_mode,
        )
    )


def _exec_patch(**kwargs):
    # RoPE reads the resolved RuntimeContext namespace, not legacy globals.
    return patch(
        "sglang.srt.layers.rotary_embedding.base.get_exec",
        return_value=_exec_config(**kwargs),
    )


def _rope(max_positions: int = 128) -> RotaryEmbedding:
    with _exec_patch():
        return RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=max_positions,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        )


class _Host(torch.nn.Module):
    def __init__(self, rope: RotaryEmbedding):
        super().__init__()
        self.rope = rope


class TestGlm52RopePrebuiltGuard(unittest.TestCase):
    def test_cpu_cache_provenance_is_per_architecture(self):
        # GLM-5.2's certified table provenance is split: CPU inverse
        # frequencies, CUDA outer product and cos/sin — so its table pin is
        # None (ambient). Qwen3.5-family exact serving evaluates the full table
        # on CPU, matching its trainer.
        rope = _rope(128)
        cases = (
            ({"glm52_exact_mode": True, "rl_on_policy_target": "xorl"}, None),
            (
                {"qwen35_gdn_exact_mode": True, "rl_on_policy_target": "xorl"},
                torch.device("cpu"),
            ),
        )
        for exec_config, expected in cases:
            with (
                self.subTest(exec_config=exec_config),
                _exec_patch(**exec_config),
            ):
                self.assertEqual(rope._cos_sin_cache_device(), expected)

    def test_glm_inv_freq_is_cpu_computed_then_moved(self):
        rope = _rope(128)
        with _exec_patch(rl_on_policy_target="xorl", glm52_exact_mode=True):
            inv_freq = rope._cos_sin_cache_inv_freq()
        expected = 1.0 / (
            10000 ** (torch.arange(0, 64, 2, dtype=torch.float, device="cpu") / 64)
        )
        self.assertEqual(inv_freq.device.type, rope._cos_sin_cache_out_device().type)
        self.assertTrue(torch.equal(inv_freq.cpu(), expected))

    def test_application_class_is_per_architecture(self):
        from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

        # GLM applies rotation through the compiled Class-B expression;
        # Qwen3.5-family exact serving stays on the eager Class-A expression.
        compiled_apply = object()
        with (
            _exec_patch(rl_on_policy_target="xorl", glm52_exact_mode=True),
            patch(
                "sglang.srt.layers.rotary_embedding.base.torch.compile",
                return_value=lambda fn: compiled_apply,
            ) as compile_mock,
            patch.object(
                RotaryEmbedding,
                "_cos_sin_cache_out_device",
                return_value=torch.device("cpu"),
            ),
        ):
            compiled_rope = RotaryEmbedding(
                head_size=64,
                rotary_dim=64,
                max_position_embeddings=128,
                base=10000,
                is_neox_style=True,
                dtype=torch.float32,
            )
        compile_mock.assert_called_once()
        self.assertIs(compiled_rope._apply_rotary_emb_wrapped, compiled_apply)
        with (
            _exec_patch(rl_on_policy_target="xorl", qwen35_gdn_exact_mode=True),
            patch(
                "sglang.srt.layers.rotary_embedding.base.torch.compile"
            ) as compile_mock,
            patch.object(
                RotaryEmbedding,
                "_cos_sin_cache_out_device",
                return_value=torch.device("cpu"),
            ),
        ):
            eager_rope = RotaryEmbedding(
                head_size=64,
                rotary_dim=64,
                max_position_embeddings=128,
                base=10000,
                is_neox_style=True,
                dtype=torch.float32,
            )
        compile_mock.assert_not_called()
        self.assertIs(eager_rope._apply_rotary_emb_wrapped, apply_rotary_emb)

    def test_default_path_still_grows_incrementally(self):
        rope = _rope(128)
        with _exec_patch():
            rope._ensure_cos_sin_cache_length(4096)
        self.assertGreater(int(rope.cos_sin_cache.shape[0]), 4096)

    def test_exact_mode_admits_positions_within_the_prebuilt_cache(self):
        rope = _rope(128)
        rope.glm52_exact_prebuilt_only = True
        before = rope.cos_sin_cache.clone()
        rope._ensure_cos_sin_cache_length(64)
        self.assertTrue(torch.equal(before, rope.cos_sin_cache))

    def test_exact_mode_fails_closed_on_growth(self):
        rope = _rope(128)
        rope.glm52_exact_prebuilt_only = True
        with self.assertRaisesRegex(RuntimeError, "prebuilt RoPE cache"):
            rope._ensure_cos_sin_cache_length(4096)

    def test_startup_reserve_fails_closed_in_exact_glm52_mode(self):
        model = _Host(_rope(128))
        server_args = SimpleNamespace(
            context_length=8192,
            rl_on_policy_target="xorl",
            glm52_exact_mode=True,
            speculative_num_steps=0,
            speculative_num_draft_tokens=0,
        )
        model_config = SimpleNamespace(
            context_len=8192,
            hf_config=SimpleNamespace(indexer_types=("full",)),
            hf_text_config=SimpleNamespace(max_position_embeddings=128),
        )
        with self.assertRaisesRegex(RuntimeError, "prebuilt RoPE cache"):
            reserve_rope_cache_for_long_sequences(model, server_args, model_config)

    def test_startup_reserve_marks_and_passes_when_the_cache_covers_context(self):
        model = _Host(_rope(8192))
        server_args = SimpleNamespace(
            context_length=8192,
            rl_on_policy_target="xorl",
            glm52_exact_mode=True,
            speculative_num_steps=0,
            speculative_num_draft_tokens=0,
        )
        model_config = SimpleNamespace(
            context_len=8192,
            hf_config=SimpleNamespace(indexer_types=("full",)),
            hf_text_config=SimpleNamespace(max_position_embeddings=8192),
        )
        before = model.rope.cos_sin_cache.clone()
        reserve_rope_cache_for_long_sequences(model, server_args, model_config)
        self.assertTrue(model.rope.glm52_exact_prebuilt_only)
        self.assertTrue(torch.equal(before, model.rope.cos_sin_cache))

    def test_startup_reserve_still_grows_outside_exact_mode(self):
        model = _Host(_rope(128))
        server_args = SimpleNamespace(
            context_length=8192,
            rl_on_policy_target=None,
            glm52_exact_mode=False,
            speculative_num_steps=0,
            speculative_num_draft_tokens=0,
        )
        model_config = SimpleNamespace(
            context_len=8192,
            hf_config=SimpleNamespace(indexer_types=None),
            hf_text_config=SimpleNamespace(max_position_embeddings=128),
        )
        with _exec_patch():
            reserve_rope_cache_for_long_sequences(model, server_args, model_config)
        self.assertGreater(int(model.rope.cos_sin_cache.shape[0]), 8192)


if __name__ == "__main__":
    unittest.main()
