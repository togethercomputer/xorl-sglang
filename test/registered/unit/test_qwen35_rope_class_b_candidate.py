from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.rotary_embedding import base, mrope


def _deterministic(*, candidate: bool) -> SimpleNamespace:
    return SimpleNamespace(
        rl_on_policy_target="xorl",
        glm52_exact_mode=False,
        qwen35_gdn_exact_mode=True,
        qwen35_rope_class_b_candidate=candidate,
    )


def test_qwen35_class_b_candidate_keeps_cpu_fp32_table_provenance():
    with (
        patch.object(base, "_is_cuda", True),
        patch.object(
            base,
            "get_exec",
            return_value=SimpleNamespace(deterministic=_deterministic(candidate=True)),
        ),
    ):
        rope = base.RotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=16,
            base=500000,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )

    inv_freq = 1.0 / (
        500000
        ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8)
    )
    positions = torch.arange(16, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    assert rope.cos_sin_cache.dtype is torch.float32
    assert torch.equal(rope.cos_sin_cache.cpu(), expected)


def test_qwen35_class_b_candidate_rejects_multimodal_mrope_positions():
    rope = mrope.MRotaryEmbedding.__new__(mrope.MRotaryEmbedding)
    torch.nn.Module.__init__(rope)
    with patch.object(
        mrope,
        "get_exec",
        return_value=SimpleNamespace(deterministic=_deterministic(candidate=True)),
    ):
        with pytest.raises(RuntimeError, match="does not support multimodal"):
            rope.forward_native(
                torch.tensor([[0, 1], [0, 2], [0, 3]], dtype=torch.long),
                torch.zeros((2, 8), dtype=torch.bfloat16),
                torch.zeros((2, 8), dtype=torch.bfloat16),
            )
