import re
from typing import Dict, Iterable, Tuple

import torch

MOE_GATE_UP_A = "gate_up_A"
MOE_GATE_B = "gate_B"
MOE_UP_B = "up_B"
MOE_DOWN_A = "down_A"
MOE_DOWN_B = "down_B"

MOE_LOGICAL_WEIGHT_NAMES = {
    MOE_GATE_UP_A,
    MOE_GATE_B,
    MOE_UP_B,
    MOE_DOWN_A,
    MOE_DOWN_B,
}

_EXPERT_ID_RE = re.compile(r"(?:^|\.|_)experts?(?:\.|\[)(\d+)(?:\]|\.|_|$)")
_PROJECTION_RE = re.compile(
    r"(?:^|\.|_)(w1|w2|w3|gate_proj|down_proj|up_proj)(?:\.|_|$)"
)


def is_moe_lora_weight_name(name: str) -> bool:
    return "experts" in name or any(token in name for token in MOE_LOGICAL_WEIGHT_NAMES)


def _get_expert_id(name: str) -> int | None:
    match = _EXPERT_ID_RE.search(name)
    return int(match.group(1)) if match else None


def _get_projection_name(name: str) -> str | None:
    match = _PROJECTION_RE.search(name)
    if match is None:
        return None
    proj_name = match.group(1)
    return {
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
    }.get(proj_name, proj_name)


def _validate_shared(
    shared_name: str,
    tensors: Iterable[torch.Tensor | None],
) -> torch.Tensor | None:
    shared_tensor = None
    for tensor in tensors:
        if tensor is None:
            continue
        if shared_tensor is None:
            shared_tensor = tensor
        elif not torch.equal(shared_tensor, tensor):
            raise ValueError(
                f"MoE LoRA expects shared '{shared_name}' weights to be identical across experts."
            )
    return shared_tensor


def _stack_or_zero(
    values: Dict[int, torch.Tensor],
    num_experts: int,
    like: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for expert_id in range(num_experts):
        if expert_id in values:
            rows.append(values[expert_id])
        else:
            rows.append(torch.zeros_like(like))
    return torch.stack(rows, dim=0)


def _normalize_doc_native_moe_weights(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    logical: Dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
        matched_name = next(
            (logical_name for logical_name in MOE_LOGICAL_WEIGHT_NAMES if logical_name in name),
            None,
        )
        if matched_name is None:
            continue

        expert_id = _get_expert_id(name)
        if matched_name in {MOE_GATE_B, MOE_UP_B, MOE_DOWN_A}:
            if expert_id is None:
                raise ValueError(
                    f"MoE LoRA weight '{name}' is missing an expert index for '{matched_name}'."
                )
            logical.setdefault(matched_name, {})[expert_id] = tensor
        else:
            if expert_id is not None:
                raise ValueError(
                    f"MoE LoRA weight '{name}' should be shared across experts for '{matched_name}'."
                )
            logical[matched_name] = tensor

    if not logical:
        return {}

    required_names = {
        MOE_GATE_UP_A,
        MOE_GATE_B,
        MOE_UP_B,
        MOE_DOWN_A,
        MOE_DOWN_B,
    }
    if set(logical.keys()) != required_names:
        missing = sorted(required_names - set(logical.keys()))
        raise ValueError(
            f"Doc-native MoE LoRA adapter is missing logical tensors: {missing}"
        )

    gate_b = logical[MOE_GATE_B]
    up_b = logical[MOE_UP_B]
    down_a = logical[MOE_DOWN_A]
    expert_ids = set(gate_b.keys()) | set(up_b.keys()) | set(down_a.keys())
    if not expert_ids:
        raise ValueError("Doc-native MoE LoRA adapter does not contain any experts.")
    num_experts = max(expert_ids) + 1

    return {
        MOE_GATE_UP_A: logical[MOE_GATE_UP_A],
        MOE_GATE_B: _stack_or_zero(gate_b, num_experts, next(iter(gate_b.values()))),
        MOE_UP_B: _stack_or_zero(up_b, num_experts, next(iter(up_b.values()))),
        MOE_DOWN_A: _stack_or_zero(down_a, num_experts, next(iter(down_a.values()))),
        MOE_DOWN_B: logical[MOE_DOWN_B],
    }


def normalize_moe_lora_weights(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not weights:
        return {}

    if any(logical_name in name for name in weights for logical_name in MOE_LOGICAL_WEIGHT_NAMES):
        return _normalize_doc_native_moe_weights(weights)

    per_expert: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
    expert_ids = set()
    for name, tensor in weights.items():
        expert_id = _get_expert_id(name)
        proj_name = _get_projection_name(name)
        if expert_id is None or proj_name is None:
            continue
        if "lora_A" in name:
            lora_side = "lora_A"
        elif "lora_B" in name:
            lora_side = "lora_B"
        else:
            continue
        per_expert.setdefault((proj_name, lora_side), {})[expert_id] = tensor
        expert_ids.add(expert_id)

    if not per_expert:
        return {}

    num_experts = max(expert_ids) + 1

    gate_a = _validate_shared(
        "gate_proj.lora_A",
        (per_expert.get(("gate_proj", "lora_A"), {})).values(),
    )
    up_a = _validate_shared(
        "up_proj.lora_A",
        (per_expert.get(("up_proj", "lora_A"), {})).values(),
    )
    down_b = _validate_shared(
        "down_proj.lora_B",
        (per_expert.get(("down_proj", "lora_B"), {})).values(),
    )

    gate_b = per_expert.get(("gate_proj", "lora_B"), {})
    up_b = per_expert.get(("up_proj", "lora_B"), {})
    down_a = per_expert.get(("down_proj", "lora_A"), {})

    like_gate_b = next(iter(gate_b.values()), None)
    like_up_b = next(iter(up_b.values()), None)
    like_down_a = next(iter(down_a.values()), None)

    if gate_a is None and up_a is None and down_b is None and like_down_a is None:
        return {}

    if gate_a is None and up_a is not None:
        gate_a = torch.zeros_like(up_a)
    if up_a is None and gate_a is not None:
        up_a = torch.zeros_like(gate_a)

    if gate_a is None or up_a is None:
        raise ValueError(
            "MoE LoRA gate/up weights must provide either shared gate/up A weights or an expert-local expert format that can be normalized."
        )

    if like_gate_b is None and like_up_b is not None:
        like_gate_b = torch.zeros_like(like_up_b)
    if like_up_b is None and like_gate_b is not None:
        like_up_b = torch.zeros_like(like_gate_b)

    if like_gate_b is None or like_up_b is None:
        raise ValueError(
            "MoE LoRA gate/up weights must provide at least one expert-local B tensor."
        )

    if down_b is None or like_down_a is None:
        raise ValueError(
            "MoE LoRA down weights must provide both expert-local down A and shared down B tensors."
        )

    return {
        MOE_GATE_UP_A: torch.cat((gate_a, up_a), dim=0),
        MOE_GATE_B: _stack_or_zero(gate_b, num_experts, like_gate_b),
        MOE_UP_B: _stack_or_zero(up_b, num_experts, like_up_b),
        MOE_DOWN_A: _stack_or_zero(down_a, num_experts, like_down_a),
        MOE_DOWN_B: down_b,
    }


def build_chunked_compound_segments_cpu(
    expert_ids: torch.Tensor,
    lora_ids: torch.Tensor,
    max_loras: int,
    chunk_size: int,
):
    if expert_ids.device.type != "cpu" or lora_ids.device.type != "cpu":
        raise ValueError("build_chunked_compound_segments_cpu expects CPU tensors.")
    if expert_ids.shape != lora_ids.shape:
        raise ValueError("expert_ids and lora_ids must have the same shape.")

    sort_key = expert_ids.to(torch.int64) * max_loras + lora_ids.to(torch.int64)
    permutation = torch.argsort(sort_key, stable=True)
    sorted_keys = sort_key[permutation]
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)

    seg_weight_indices = []
    seg_lens = []
    for weight_idx, count in zip(unique_keys.tolist(), counts.tolist()):
        while count > chunk_size:
            seg_weight_indices.append(weight_idx)
            seg_lens.append(chunk_size)
            count -= chunk_size
        seg_weight_indices.append(weight_idx)
        seg_lens.append(count)

    seg_weight_indices_tensor = torch.tensor(seg_weight_indices, dtype=torch.int32)
    seg_indptr = torch.empty((len(seg_lens) + 1,), dtype=torch.int32)
    seg_indptr[0] = 0
    seg_indptr[1:] = torch.tensor(seg_lens, dtype=torch.int32).cumsum(dim=0)
    return permutation.to(torch.int32), seg_weight_indices_tensor, seg_indptr
