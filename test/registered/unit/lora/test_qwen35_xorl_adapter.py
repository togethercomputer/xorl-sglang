from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_manager import _expand_qwen35_logical_lora_targets
from sglang.srt.lora.utils import get_normalized_target_modules
from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_qwen35_split_target_manifest_selects_attention_and_gdn_packs() -> None:
    assert get_normalized_target_modules(
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "g_proj",
            "o_proj",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    ) == {
        "qkv_proj",
        "in_proj_qkvz",
        "o_proj",
        "out_proj",
        "gate_up_proj",
        "down_proj",
    }


def test_qwen35_trainer_split_gdn_factors_normalize_before_attention_qkv() -> None:
    prefix = "base_model.model.model.layers.0.linear_attn"
    weights = {}
    output_sizes = {"q_proj": 2, "k_proj": 2, "v_proj": 4, "g_proj": 4}
    for index, (name, output_size) in enumerate(output_sizes.items(), start=1):
        weights[f"{prefix}.{name}.lora_A.weight"] = torch.full((2, 3), float(index))
        weights[f"{prefix}.{name}.lora_B.weight"] = torch.full(
            (output_size, 2), float(index * 10)
        )

    adapter = LoRAAdapter.__new__(LoRAAdapter)
    adapter._normalize_in_proj_qkvz(weights)

    a_name = f"{prefix}.in_proj_qkvz.lora_A.weight"
    b_name = f"{prefix}.in_proj_qkvz.lora_B.weight"
    assert set(weights) == {a_name, b_name}
    assert weights[a_name].shape == (8, 3)
    assert weights[b_name].shape == (12, 2)
    torch.testing.assert_close(
        weights[a_name][:, 0],
        torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.float32),
    )
    torch.testing.assert_close(
        weights[b_name][:, 0],
        torch.tensor(
            [10, 10, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40],
            dtype=torch.float32,
        ),
    )


def test_qwen35_trainer_linear_output_maps_without_touching_attention() -> None:
    linear_name = "base_model.model.model.layers.0.linear_attn.o_proj.lora_A.weight"
    attention_name = "base_model.model.model.layers.3.self_attn.o_proj.lora_A.weight"
    weights = {
        linear_name: torch.ones((2, 3)),
        attention_name: torch.full((2, 3), 2.0),
    }

    adapter = LoRAAdapter.__new__(LoRAAdapter)
    adapter._normalize_linear_attn_out_proj(weights)

    normalized = linear_name.replace(".linear_attn.o_proj.", ".linear_attn.out_proj.")
    assert set(weights) == {normalized, attention_name}
    torch.testing.assert_close(weights[normalized], torch.ones((2, 3)))
    torch.testing.assert_close(weights[attention_name], torch.full((2, 3), 2.0))


def test_qwen35_explicit_gdn_targets_are_cli_supported() -> None:
    assert {"g_proj", "out_proj", "in_proj_qkvz"} <= set(SUPPORTED_LORA_TARGET_MODULES)


def test_qwen35_logical_o_proj_allocates_attention_and_gdn_outputs() -> None:
    assert _expand_qwen35_logical_lora_targets(
        {"qkv_proj", "o_proj"},
        SimpleNamespace(model_type="qwen3_5_moe_text"),
    ) == {"qkv_proj", "o_proj", "out_proj"}


def test_non_qwen35_logical_o_proj_does_not_allocate_gdn_output() -> None:
    assert _expand_qwen35_logical_lora_targets(
        {"qkv_proj", "o_proj"},
        SimpleNamespace(model_type="qwen3_moe"),
    ) == {"qkv_proj", "o_proj"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
