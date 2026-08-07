"""GLM-5.2 contract for XoRL shared-outer adapters.

The serving implementation continues to use SGLang's ordinary LoRA adapter
loader and memory pool. This module only describes the complete GLM-5.2
product target and validates exported tensors before the generic loader packs
them into fused runtime targets. A server initialized without an adapter must
still select ``--experts-shared-outer-loras`` at launch because the ordinary
LoRA memory-pool layout is fixed before later POST loads. The certified target
also requires ``--disable-shared-experts-fusion`` so its unfused shared-expert
factors have real runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

GLM52_ARCHITECTURE = "GlmMoeDsaForCausalLM"
GLM52_LOGICAL_FACTOR_COUNT = 1700
GLM52_REQUIRED_TARGET_MODULES = frozenset(
    {
        "down_proj",
        "gate_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "lm_head",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "up_proj",
    }
)

_ADAPTER_PREFIX = "base_model.model."


@dataclass(frozen=True)
class Glm52LoRAFactorSpec:
    """One trainer-logical factor and its SGLang load destination."""

    role: str
    layer_id: Optional[int]
    factor: str
    export_key: str
    load_key: str
    physical_target: str
    export_shape: tuple[int, ...]
    export_dtype: torch.dtype
    orientation: str
    expert_layout: str
    physical_slice: str


def _config_architectures(config) -> set[str]:
    return set(getattr(config, "architectures", None) or ())


def is_glm52_xorl_shared_outer_adapter(config, adapter_config: dict) -> bool:
    """Whether an adapter should take the strict GLM-5.2 XoRL path."""

    return (
        GLM52_ARCHITECTURE in _config_architectures(config)
        and adapter_config.get("_sglang_lora_format") == "shared_outer"
    )


def _require_positive_int(config, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"GLM-5.2 shared-outer LoRA requires positive integer "
            f"config.{name}; got {value!r}."
        )
    return value


def _validate_official_geometry(config, adapter_config: dict) -> dict[str, int]:
    if adapter_config.get("moe_hybrid_shared_lora") is not True:
        raise ValueError(
            "GLM-5.2 XoRL shared-outer adapters require " "moe_hybrid_shared_lora=true."
        )

    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, list):
        raise ValueError(
            "GLM-5.2 XoRL shared-outer adapters require an explicit "
            "target_modules list."
        )
    target_set = set(target_modules)
    if target_set != GLM52_REQUIRED_TARGET_MODULES:
        missing = sorted(GLM52_REQUIRED_TARGET_MODULES - target_set)
        extra = sorted(target_set - GLM52_REQUIRED_TARGET_MODULES)
        raise ValueError(
            "GLM-5.2 XoRL shared-outer target_modules mismatch: "
            f"missing={missing}, extra={extra}."
        )

    geometry = {
        name: _require_positive_int(config, name)
        for name in (
            "first_k_dense_replace",
            "hidden_size",
            "intermediate_size",
            "kv_lora_rank",
            "moe_intermediate_size",
            "moe_layer_freq",
            "n_routed_experts",
            "n_shared_experts",
            "num_attention_heads",
            "num_hidden_layers",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "vocab_size",
        )
    }
    expected = {
        "first_k_dense_replace": 3,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "kv_lora_rank": 512,
        "moe_intermediate_size": 2048,
        "moe_layer_freq": 1,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_hidden_layers": 78,
        "q_lora_rank": 2048,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "vocab_size": 154880,
    }
    mismatches = {
        name: (geometry[name], expected_value)
        for name, expected_value in expected.items()
        if geometry[name] != expected_value
    }
    if mismatches:
        detail = ", ".join(
            f"{name}={actual} (expected {expected_value})"
            for name, (actual, expected_value) in sorted(mismatches.items())
        )
        raise ValueError(f"Not the certified GLM-5.2 adapter geometry: {detail}.")

    rank = adapter_config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(
            f"GLM-5.2 shared-outer LoRA requires a positive integer rank; "
            f"got {rank!r}."
        )
    geometry["rank"] = rank
    return geometry


def _factor_suffix(factor: str) -> str:
    return f"lora_{factor}.weight"


def _add_linear_specs(
    specs: list[Glm52LoRAFactorSpec],
    *,
    role: str,
    layer_id: Optional[int],
    export_target: str,
    physical_target: str,
    input_dim: int,
    output_dim: int,
    rank: int,
    load_target: Optional[str] = None,
    a_slice: str = "full",
    b_slice: str = "full",
    expert_layout: str = "ordinary",
) -> None:
    load_target = load_target or export_target
    for factor, shape, orientation, physical_slice in (
        ("A", (rank, input_dim), "[rank,in]", a_slice),
        ("B", (output_dim, rank), "[out,rank]", b_slice),
    ):
        suffix = _factor_suffix(factor)
        specs.append(
            Glm52LoRAFactorSpec(
                role=role,
                layer_id=layer_id,
                factor=factor,
                export_key=f"{_ADAPTER_PREFIX}{export_target}.{suffix}",
                load_key=f"{_ADAPTER_PREFIX}{load_target}.{suffix}",
                physical_target=physical_target,
                export_shape=shape,
                export_dtype=torch.bfloat16,
                orientation=orientation,
                expert_layout=expert_layout,
                physical_slice=physical_slice,
            )
        )


def _add_routed_specs(
    specs: list[Glm52LoRAFactorSpec],
    *,
    layer_id: int,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    rank: int,
) -> None:
    for slot, role, physical_target, input_dim, output_dim, shared_factor in (
        (
            "w1",
            "routed_expert.gate_proj",
            f"{prefix}.experts.gate_up_proj",
            hidden_size,
            intermediate_size,
            "A",
        ),
        (
            "w3",
            "routed_expert.up_proj",
            f"{prefix}.experts.gate_up_proj",
            hidden_size,
            intermediate_size,
            "A",
        ),
        (
            "w2",
            "routed_expert.down_proj",
            f"{prefix}.experts.down_proj",
            intermediate_size,
            hidden_size,
            "B",
        ),
    ):
        load_target = (
            f"{prefix}.experts.gate_up_proj"
            if slot in {"w1", "w3"}
            else f"{prefix}.experts.down_proj"
        )
        for factor in ("A", "B"):
            shared = factor == shared_factor
            expert_dim = 1 if shared else num_experts
            if factor == "A":
                shape = (expert_dim, rank, input_dim)
                orientation = "[expert_or_1,rank,in]"
                physical_slice = (
                    "gate_rank[0:r]"
                    if slot == "w1"
                    else "up_rank[r:2r]" if slot == "w3" else "full"
                )
            else:
                shape = (expert_dim, output_dim, rank)
                orientation = "[expert_or_1,out,rank]"
                physical_slice = (
                    "gate_output[0:intermediate]"
                    if slot == "w1"
                    else (
                        "up_output[intermediate:2*intermediate]"
                        if slot == "w3"
                        else "full"
                    )
                )
            suffix = _factor_suffix(factor)
            specs.append(
                Glm52LoRAFactorSpec(
                    role=role,
                    layer_id=layer_id,
                    factor=factor,
                    export_key=f"{_ADAPTER_PREFIX}{prefix}.experts.{slot}.{suffix}",
                    load_key=f"{_ADAPTER_PREFIX}{load_target}.{suffix}",
                    physical_target=physical_target,
                    export_shape=shape,
                    export_dtype=torch.bfloat16,
                    orientation=orientation,
                    expert_layout=("shared_outer" if shared else "expert_local_inner"),
                    physical_slice=physical_slice,
                )
            )


def build_glm52_xorl_shared_outer_inventory(
    config, adapter_config: dict
) -> tuple[Glm52LoRAFactorSpec, ...]:
    """Build the exact 1,700-row sampler-side adapter inventory."""

    geometry = _validate_official_geometry(config, adapter_config)
    hidden = geometry["hidden_size"]
    rank = geometry["rank"]
    q_rank = geometry["q_lora_rank"]
    kv_rank = geometry["kv_lora_rank"]
    qk_rope = geometry["qk_rope_head_dim"]
    qk_nope = geometry["qk_nope_head_dim"]
    v_head = geometry["v_head_dim"]
    num_heads = geometry["num_attention_heads"]
    dense_inter = geometry["intermediate_size"]
    moe_inter = geometry["moe_intermediate_size"]
    num_experts = geometry["n_routed_experts"]
    num_shared = geometry["n_shared_experts"]

    specs: list[Glm52LoRAFactorSpec] = []
    for layer_id in range(geometry["num_hidden_layers"]):
        layer = f"model.layers.{layer_id}"
        attn = f"{layer}.self_attn"
        fused_a = f"{attn}.fused_qkv_a_proj_with_mqa"
        _add_linear_specs(
            specs,
            role="attention.q_a_proj",
            layer_id=layer_id,
            export_target=f"{attn}.q_a_proj",
            load_target=fused_a,
            physical_target=fused_a,
            input_dim=hidden,
            output_dim=q_rank,
            rank=rank,
            a_slice="rank[0:r]",
            b_slice="output[0:q_lora_rank]",
        )
        _add_linear_specs(
            specs,
            role="attention.kv_a_proj_with_mqa",
            layer_id=layer_id,
            export_target=f"{attn}.kv_a_proj_with_mqa",
            load_target=fused_a,
            physical_target=fused_a,
            input_dim=hidden,
            output_dim=kv_rank + qk_rope,
            rank=rank,
            a_slice="rank[r:2r]",
            b_slice="output[q_lora_rank:]",
        )
        _add_linear_specs(
            specs,
            role="attention.q_b_proj",
            layer_id=layer_id,
            export_target=f"{attn}.q_b_proj",
            physical_target=f"{attn}.q_b_proj",
            input_dim=q_rank,
            output_dim=num_heads * (qk_nope + qk_rope),
            rank=rank,
        )
        _add_linear_specs(
            specs,
            role="attention.kv_b_proj",
            layer_id=layer_id,
            export_target=f"{attn}.kv_b_proj",
            physical_target=f"{attn}.kv_b_proj",
            input_dim=kv_rank,
            output_dim=num_heads * (qk_nope + v_head),
            rank=rank,
            expert_layout="absorbed_mla_correction",
        )
        _add_linear_specs(
            specs,
            role="attention.o_proj",
            layer_id=layer_id,
            export_target=f"{attn}.o_proj",
            physical_target=f"{attn}.o_proj",
            input_dim=num_heads * v_head,
            output_dim=hidden,
            rank=rank,
        )

        mlp = f"{layer}.mlp"
        if layer_id < geometry["first_k_dense_replace"]:
            mlp_prefix = mlp
            mlp_role = "dense_mlp"
            intermediate = dense_inter
        else:
            mlp_prefix = f"{mlp}.shared_experts"
            mlp_role = "shared_expert"
            intermediate = moe_inter * num_shared

        gate_up = f"{mlp_prefix}.gate_up_proj"
        _add_linear_specs(
            specs,
            role=f"{mlp_role}.gate_proj",
            layer_id=layer_id,
            export_target=f"{mlp_prefix}.gate_proj",
            load_target=gate_up,
            physical_target=gate_up,
            input_dim=hidden,
            output_dim=intermediate,
            rank=rank,
            a_slice="rank[0:r]",
            b_slice="output[0:intermediate]",
        )
        _add_linear_specs(
            specs,
            role=f"{mlp_role}.up_proj",
            layer_id=layer_id,
            export_target=f"{mlp_prefix}.up_proj",
            load_target=gate_up,
            physical_target=gate_up,
            input_dim=hidden,
            output_dim=intermediate,
            rank=rank,
            a_slice="rank[r:2r]",
            b_slice="output[intermediate:2*intermediate]",
        )
        _add_linear_specs(
            specs,
            role=f"{mlp_role}.down_proj",
            layer_id=layer_id,
            export_target=f"{mlp_prefix}.down_proj",
            physical_target=f"{mlp_prefix}.down_proj",
            input_dim=intermediate,
            output_dim=hidden,
            rank=rank,
        )

        if layer_id >= geometry["first_k_dense_replace"]:
            _add_routed_specs(
                specs,
                layer_id=layer_id,
                prefix=mlp,
                hidden_size=hidden,
                intermediate_size=moe_inter,
                num_experts=num_experts,
                rank=rank,
            )

    for factor, shape, orientation in (
        ("A", (rank, hidden), "[rank,in]"),
        ("B", (geometry["vocab_size"], rank), "[out,rank]"),
    ):
        specs.append(
            Glm52LoRAFactorSpec(
                role="output.lm_head",
                layer_id=None,
                factor=factor,
                export_key=f"{_ADAPTER_PREFIX}lm_head.lora_embedding_{factor}",
                load_key=f"{_ADAPTER_PREFIX}lm_head.lora_embedding_{factor}",
                physical_target="lm_head",
                export_shape=shape,
                export_dtype=torch.bfloat16,
                orientation=orientation,
                expert_layout="ordinary_bf16_base",
                physical_slice="full",
            )
        )

    if len(specs) != GLM52_LOGICAL_FACTOR_COUNT:
        raise AssertionError(
            "Internal GLM-5.2 LoRA inventory error: "
            f"expected {GLM52_LOGICAL_FACTOR_COUNT} factors, got {len(specs)}."
        )
    export_keys = {spec.export_key for spec in specs}
    if len(export_keys) != len(specs):
        raise AssertionError("GLM-5.2 LoRA inventory contains duplicate export keys.")
    return tuple(specs)


class Glm52XorlSharedOuterValidator:
    """Streaming fail-closed validation for one complete adapter directory."""

    def __init__(self, config, adapter_config: dict):
        self.specs = build_glm52_xorl_shared_outer_inventory(config, adapter_config)
        self._expected = {spec.export_key: spec for spec in self.specs}
        self._seen: set[str] = set()

    def observe(self, name: str, tensor: torch.Tensor) -> None:
        spec = self._expected.get(name)
        if spec is None:
            raise ValueError(
                "Unexpected tensor in GLM-5.2 XoRL shared-outer adapter: " f"{name!r}."
            )
        if name in self._seen:
            raise ValueError(
                f"Duplicate tensor in GLM-5.2 XoRL shared-outer adapter: {name!r}."
            )
        actual_shape = tuple(tensor.shape)
        if actual_shape != spec.export_shape:
            raise ValueError(
                f"GLM-5.2 XoRL shared-outer tensor {name!r} has shape "
                f"{actual_shape}; expected {spec.export_shape} "
                f"({spec.orientation})."
            )
        if tensor.dtype != spec.export_dtype:
            raise ValueError(
                f"GLM-5.2 XoRL shared-outer tensor {name!r} has dtype "
                f"{tensor.dtype}; expected {spec.export_dtype}."
            )
        self._seen.add(name)

    def finalize(self) -> None:
        missing = sorted(set(self._expected) - self._seen)
        if missing:
            raise ValueError(
                "Incomplete GLM-5.2 XoRL shared-outer adapter: "
                f"missing {len(missing)} of {len(self._expected)} tensors; "
                f"first missing keys={missing[:8]}."
            )


def maybe_create_glm52_validator(
    config, adapter_config: dict
) -> Optional[Glm52XorlSharedOuterValidator]:
    if not is_glm52_xorl_shared_outer_adapter(config, adapter_config):
        return None
    return Glm52XorlSharedOuterValidator(config, adapter_config)


def summarize_glm52_factor_roles(
    specs: Iterable[Glm52LoRAFactorSpec],
) -> dict[str, int]:
    """Return role counts used by the trainer/sampler inventory join."""

    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.role] = counts.get(spec.role, 0) + 1
    return counts
