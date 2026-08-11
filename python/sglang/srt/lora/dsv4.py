"""Fail-closed DSV4-Flash active-LoRA adapter inventory.

The first exact lane intentionally keeps the DSV4 query/KV-A projections
unfused.  This makes every exported logical factor map to one physical serving
module without introducing an unqualified concatenation boundary.  Routed
expert factors remain packed as one three-dimensional tensor per projection
and factor, preserving the complete expert bank while avoiding 256 separate
checkpoint keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


DSV4_FLASH_ARCHITECTURE = "DeepseekV4ForCausalLM"
DSV4_FLASH_LOGICAL_FACTOR_COUNT = 948
DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT = 345
DSV4_FLASH_ROUTED_BANK_COUNT = 43
DSV4_FLASH_LORA_FORMAT = "dsv4_expert_banks"
DSV4_FLASH_EXACT_ROUTED_DP_RANK = 0
DSV4_FLASH_REQUIRED_TARGET_MODULES = frozenset(
    {
        "down_proj",
        "gate_proj",
        "lm_head",
        "self_attn.wq_b",
        "up_proj",
        "wkv",
        "wo_a",
        "wo_b",
        "wq_a",
    }
)
DSV4_FLASH_COMPRESS_RATIOS = (
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0,
)

_ADAPTER_PREFIX = "base_model.model."


def validate_dsv4_flash_exact_request_routing(
    routed_dp_rank: Optional[int],
) -> None:
    """Require the one qualified MAX_LEN row placement for the first lane."""

    if routed_dp_rank != DSV4_FLASH_EXACT_ROUTED_DP_RANK:
        raise ValueError(
            "The exact DSV4-Flash lane requires routed_dp_rank=0 so MAX_LEN "
            "DP-attention row placement is invariant; unpinned or other-rank "
            f"requests are not qualified (got {routed_dp_rank!r})."
        )


@dataclass(frozen=True)
class Dsv4FlashLoRAFactorSpec:
    """One trainer-logical factor and its physical SGLang destination."""

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


def is_dsv4_flash_exact_adapter(config, adapter_config: dict) -> bool:
    """Return whether this is the explicitly identified exact DSV4 format."""

    return (
        DSV4_FLASH_ARCHITECTURE in _config_architectures(config)
        and adapter_config.get("_sglang_lora_format") == DSV4_FLASH_LORA_FORMAT
    )


def _require_positive_int(config, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"DSV4-Flash exact LoRA requires positive integer config.{name}; "
            f"got {value!r}."
        )
    return value


def _validate_official_geometry(config, adapter_config: dict) -> dict[str, int]:
    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, list):
        raise ValueError(
            "DSV4-Flash exact LoRA requires an explicit target_modules list."
        )
    target_set = set(target_modules)
    if target_set != DSV4_FLASH_REQUIRED_TARGET_MODULES:
        missing = sorted(DSV4_FLASH_REQUIRED_TARGET_MODULES - target_set)
        extra = sorted(target_set - DSV4_FLASH_REQUIRED_TARGET_MODULES)
        raise ValueError(
            "DSV4-Flash exact target_modules mismatch: "
            f"missing={missing}, extra={extra}."
        )

    if adapter_config.get("moe_hybrid_shared_lora", False) is not False:
        raise ValueError(
            "DSV4-Flash exact LoRA requires per-expert A and B banks; "
            "moe_hybrid_shared_lora must be false."
        )

    geometry = {
        name: _require_positive_int(config, name)
        for name in (
            "head_dim",
            "hidden_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "o_groups",
            "o_lora_rank",
            "q_lora_rank",
            "qk_rope_head_dim",
            "vocab_size",
        )
    }
    expected = {
        "head_dim": 512,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_hidden_layers": 43,
        "num_key_value_heads": 1,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "vocab_size": 129280,
    }
    mismatches = {
        name: (geometry[name], expected_value)
        for name, expected_value in expected.items()
        if geometry[name] != expected_value
    }
    for name, expected_value in (
        ("expert_dtype", "fp4"),
        ("hidden_act", "silu"),
        ("scoring_func", "sqrtsoftplus"),
        ("topk_method", "noaux_tc"),
    ):
        actual = getattr(config, name, None)
        if actual != expected_value:
            mismatches[name] = (actual, expected_value)
    if getattr(config, "swiglu_limit", None) != 10.0:
        mismatches["swiglu_limit"] = (getattr(config, "swiglu_limit", None), 10.0)
    if mismatches:
        detail = ", ".join(
            f"{name}={actual!r} (expected {expected_value!r})"
            for name, (actual, expected_value) in sorted(mismatches.items())
        )
        raise ValueError(f"Not the official DSV4-Flash adapter geometry: {detail}.")

    rank = adapter_config.get("r")
    alpha = adapter_config.get("lora_alpha")
    dropout = adapter_config.get("lora_dropout", 0.0)
    exact_mismatches = []
    if rank != 1 or isinstance(rank, bool):
        exact_mismatches.append(f"r={rank!r} (expected 1)")
    if alpha != 1 or isinstance(alpha, bool):
        exact_mismatches.append(f"lora_alpha={alpha!r} (expected 1)")
    if dropout != 0 or isinstance(dropout, bool):
        exact_mismatches.append(f"lora_dropout={dropout!r} (expected 0)")
    for name, default in (
        ("bias", "none"),
        ("fan_in_fan_out", False),
        ("use_dora", False),
        ("use_rslora", False),
    ):
        value = adapter_config.get(name, default)
        if value != default:
            exact_mismatches.append(f"{name}={value!r} (expected {default!r})")
    for name in ("alpha_pattern", "rank_pattern"):
        if adapter_config.get(name) not in (None, {}):
            exact_mismatches.append(f"{name} must be empty")
    if exact_mismatches:
        raise ValueError(
            "The exact DSV4-Flash active-LoRA contract requires the complete "
            "rank-1/alpha-1 factor-only adapter: " + ", ".join(exact_mismatches)
        )
    geometry["rank"] = rank
    return geometry


def _factor_suffix(factor: str) -> str:
    return f"lora_{factor}.weight"


def _add_linear_specs(
    specs: list[Dsv4FlashLoRAFactorSpec],
    *,
    role: str,
    layer_id: Optional[int],
    target: str,
    input_dim: int,
    output_dim: int,
    rank: int,
) -> None:
    for factor, shape, orientation in (
        ("A", (rank, input_dim), "[rank,in]"),
        ("B", (output_dim, rank), "[out,rank]"),
    ):
        key = f"{_ADAPTER_PREFIX}{target}.{_factor_suffix(factor)}"
        specs.append(
            Dsv4FlashLoRAFactorSpec(
                role=role,
                layer_id=layer_id,
                factor=factor,
                export_key=key,
                load_key=key,
                physical_target=target,
                export_shape=shape,
                export_dtype=torch.bfloat16,
                orientation=orientation,
                expert_layout="ordinary",
                physical_slice="full",
            )
        )


def _add_routed_specs(
    specs: list[Dsv4FlashLoRAFactorSpec],
    *,
    layer_id: int,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    rank: int,
) -> None:
    for slot, projection, input_dim, output_dim, physical_target in (
        ("w1", "gate_proj", hidden_size, intermediate_size, "gate_up_proj"),
        ("w3", "up_proj", hidden_size, intermediate_size, "gate_up_proj"),
        ("w2", "down_proj", intermediate_size, hidden_size, "down_proj"),
    ):
        target = f"{prefix}.experts.{slot}"
        for factor, shape, orientation, physical_slice in (
            (
                "A",
                (num_experts, rank, input_dim),
                "[expert,rank,in]",
                "gate_rank[0:r]" if slot == "w1" else "up_rank[r:2r]" if slot == "w3" else "full",
            ),
            (
                "B",
                (num_experts, output_dim, rank),
                "[expert,out,rank]",
                "gate_output[0:intermediate]"
                if slot == "w1"
                else "up_output[intermediate:2*intermediate]"
                if slot == "w3"
                else "full",
            ),
        ):
            key = f"{_ADAPTER_PREFIX}{target}.{_factor_suffix(factor)}"
            specs.append(
                Dsv4FlashLoRAFactorSpec(
                    role=f"routed_expert.{projection}",
                    layer_id=layer_id,
                    factor=factor,
                    export_key=key,
                    load_key=key,
                    physical_target=f"{prefix}.experts.{physical_target}",
                    export_shape=shape,
                    export_dtype=torch.bfloat16,
                    orientation=orientation,
                    expert_layout="per_expert_bank",
                    physical_slice=physical_slice,
                )
            )


def build_dsv4_flash_exact_inventory(
    config, adapter_config: dict
) -> tuple[Dsv4FlashLoRAFactorSpec, ...]:
    """Build the complete 948-row sampler-side adapter inventory."""

    geometry = _validate_official_geometry(config, adapter_config)
    hidden = geometry["hidden_size"]
    rank = geometry["rank"]
    q_rank = geometry["q_lora_rank"]
    head_dim = geometry["head_dim"]
    heads = geometry["num_attention_heads"]
    groups = geometry["o_groups"]
    o_rank = geometry["o_lora_rank"]
    intermediate = geometry["moe_intermediate_size"]
    num_experts = geometry["n_routed_experts"]

    specs: list[Dsv4FlashLoRAFactorSpec] = []
    for layer_id in range(geometry["num_hidden_layers"]):
        layer = f"model.layers.{layer_id}"
        attn = f"{layer}.self_attn"
        for projection, input_dim, output_dim in (
            ("wq_a", hidden, q_rank),
            ("wq_b", q_rank, heads * head_dim),
            ("wkv", hidden, head_dim),
            ("wo_a", heads * head_dim // groups, groups * o_rank),
            ("wo_b", groups * o_rank, hidden),
        ):
            _add_linear_specs(
                specs,
                role=f"attention.{projection}",
                layer_id=layer_id,
                target=f"{attn}.{projection}",
                input_dim=input_dim,
                output_dim=output_dim,
                rank=rank,
            )

        shared = f"{layer}.mlp.shared_experts"
        for projection, input_dim, output_dim in (
            ("gate_proj", hidden, intermediate),
            ("up_proj", hidden, intermediate),
            ("down_proj", intermediate, hidden),
        ):
            _add_linear_specs(
                specs,
                role=f"shared_expert.{projection}",
                layer_id=layer_id,
                target=f"{shared}.{projection}",
                input_dim=input_dim,
                output_dim=output_dim,
                rank=rank,
            )

        _add_routed_specs(
            specs,
            layer_id=layer_id,
            prefix=f"{layer}.mlp",
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            rank=rank,
        )

    for factor, shape, orientation in (
        ("A", (rank, hidden), "[rank,in]"),
        ("B", (geometry["vocab_size"], rank), "[out,rank]"),
    ):
        key = f"{_ADAPTER_PREFIX}lm_head.lora_embedding_{factor}"
        specs.append(
            Dsv4FlashLoRAFactorSpec(
                role="output.lm_head",
                layer_id=None,
                factor=factor,
                export_key=key,
                load_key=key,
                physical_target="lm_head",
                export_shape=shape,
                export_dtype=torch.bfloat16,
                orientation=orientation,
                expert_layout="ordinary",
                physical_slice="full",
            )
        )

    if len(specs) != DSV4_FLASH_LOGICAL_FACTOR_COUNT:
        raise AssertionError(
            "Internal DSV4-Flash LoRA inventory error: expected "
            f"{DSV4_FLASH_LOGICAL_FACTOR_COUNT} factors, got {len(specs)}."
        )
    if len({spec.export_key for spec in specs}) != len(specs):
        raise AssertionError("DSV4-Flash LoRA inventory contains duplicate export keys.")
    routed_banks = {spec.layer_id for spec in specs if spec.role.startswith("routed_expert.")}
    if len(routed_banks) != DSV4_FLASH_ROUTED_BANK_COUNT:
        raise AssertionError(
            "Internal DSV4-Flash routed-bank inventory error: expected "
            f"{DSV4_FLASH_ROUTED_BANK_COUNT}, got {len(routed_banks)}."
        )
    non_routed = sum(not spec.role.startswith("routed_expert.") for spec in specs) // 2
    if non_routed != DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT:
        raise AssertionError(
            "Internal DSV4-Flash non-routed projection inventory error: expected "
            f"{DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT}, got {non_routed}."
        )
    return tuple(specs)


class Dsv4FlashExactValidator:
    """Streaming validation for one complete DSV4-Flash adapter directory."""

    def __init__(self, config, adapter_config: dict):
        self.specs = build_dsv4_flash_exact_inventory(config, adapter_config)
        self._expected = {spec.export_key: spec for spec in self.specs}
        self._seen: set[str] = set()
        self.all_zero = True

    def observe(self, name: str, tensor: torch.Tensor) -> None:
        spec = self._expected.get(name)
        if spec is None:
            raise ValueError(f"Unexpected tensor in DSV4-Flash exact adapter: {name!r}.")
        if name in self._seen:
            raise ValueError(f"Duplicate tensor in DSV4-Flash exact adapter: {name!r}.")
        if tuple(tensor.shape) != spec.export_shape:
            raise ValueError(
                f"DSV4-Flash exact tensor {name!r} has shape {tuple(tensor.shape)}; "
                f"expected {spec.export_shape} ({spec.orientation})."
            )
        if tensor.dtype != spec.export_dtype:
            raise ValueError(
                f"DSV4-Flash exact tensor {name!r} has dtype {tensor.dtype}; "
                f"expected {spec.export_dtype}."
            )
        if self.all_zero and bool(torch.count_nonzero(tensor)):
            self.all_zero = False
        self._seen.add(name)

    def finalize(self) -> None:
        missing = sorted(set(self._expected) - self._seen)
        if missing:
            raise ValueError(
                "Incomplete DSV4-Flash exact adapter: "
                f"missing {len(missing)} of {len(self._expected)} tensors; "
                f"first missing keys={missing[:8]}."
            )


def maybe_create_dsv4_flash_validator(
    config, adapter_config: dict
) -> Optional[Dsv4FlashExactValidator]:
    exact = bool(
        getattr(config, "_dsv4_flash_exact_mode", False)
        and DSV4_FLASH_ARCHITECTURE in _config_architectures(config)
    )
    identified = is_dsv4_flash_exact_adapter(config, adapter_config)
    if exact and not identified:
        raise ValueError(
            "The exact DSV4-Flash active-LoRA contract requires "
            f"_sglang_lora_format={DSV4_FLASH_LORA_FORMAT!r}; ordinary, missing, "
            "or partially identified adapter formats are not admitted."
        )
    return Dsv4FlashExactValidator(config, adapter_config) if identified else None


def summarize_dsv4_flash_factor_roles(
    specs: Iterable[Dsv4FlashLoRAFactorSpec],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.role] = counts.get(spec.role, 0) + 1
    return counts
