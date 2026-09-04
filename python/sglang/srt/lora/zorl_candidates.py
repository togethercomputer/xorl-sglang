"""Seed-transport ZORL candidates on top of the current MultiLoRA runtime.

This module intentionally owns only the sampler-side boundary: register an
externally planned candidate population, deterministically materialize its
LoRA factors from seeds, and forget it after scoring.  The parameter server
continues to own optimization, folding, and weight synchronization.

The supported scientific contract is ``philox_subseed_v2``.  It is shared
bit-for-bit with XoRL's ZORL fold and makes each tensor stream independently
addressable, which is required for EP-local materialization.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.zorl_philox import (
    PhiloxSegmentGeometry,
    ZORL_NOISE_LAYOUT_V2,
    zorl_param_subseed,
)


class ZORLCandidateStore:
    """Virtual seeded candidates backed by an existing ``LoRAManager``."""

    def __init__(self, manager):
        self.manager = manager
        self.virtual: Dict[str, Dict[str, Any]] = {}
        self._geometry_cache: Dict[tuple, PhiloxSegmentGeometry] = {}

    @staticmethod
    def _mode(value: str) -> str:
        mode = str(value or "b_only")
        if mode not in ("b_only", "a_and_b", "fresh_ab"):
            raise ValueError(f"Unsupported ZORL perturbation_mode {mode!r}")
        return mode

    @staticmethod
    def _require_noise_contract() -> None:
        layout = os.environ.get("XORL_ZORL_NOISE_LAYOUT", "").strip()
        if layout != ZORL_NOISE_LAYOUT_V2:
            raise ValueError(
                "ZORL seed transport on the OSS MultiLoRA path requires "
                f"XORL_ZORL_NOISE_LAYOUT={ZORL_NOISE_LAYOUT_V2!r}; got {layout!r}"
            )

    @staticmethod
    def _config_targets(adapter: LoRAAdapter) -> set[str]:
        targets = adapter.config.target_modules
        if isinstance(targets, str):
            return {targets}
        return {str(target) for target in targets}

    @staticmethod
    def _trainer_key(name: str) -> str:
        if name.startswith("base_model.model."):
            name = name[len("base_model.model.") :]
        if name.endswith(".weight"):
            name = name[: -len(".weight")]
        return name.replace("lora_embedding_A", "lora_A").replace(
            "lora_embedding_B", "lora_B"
        )

    @staticmethod
    def _gdn_dims(adapter: LoRAAdapter) -> tuple[int, int]:
        config = adapter.base_hf_config
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        key_dim = int(config.linear_key_head_dim) * int(config.linear_num_key_heads)
        value_dim = int(config.linear_value_head_dim) * int(
            config.linear_num_value_heads
        )
        return key_dim, value_dim

    @staticmethod
    def _shared_tail_key(name: str, adapter: LoRAAdapter) -> Optional[str]:
        config = adapter.base_hf_config
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        marker = f".mlp.experts.{int(config.num_experts)}."
        if marker not in name:
            return None
        return name.replace(marker, ".mlp.shared_expert.")

    def _raw_gdn_entries(
        self,
        name: str,
        weight: torch.Tensor,
        adapter: LoRAAdapter,
    ):
        if ".linear_attn.in_proj_qkvz." in name:
            split_dim = weight.dim() - 2
            key_dim, value_dim = self._gdn_dims(adapter)
            is_a = "lora_A" in name
            if is_a:
                rank = int(adapter.config.r)
                first, second = 3 * rank, rank
            else:
                first, second = 2 * key_dim + value_dim, value_dim
            if int(weight.shape[split_dim]) != first + second:
                raise ValueError(
                    f"Unexpected GDN stacked shape for {name!r}: "
                    f"axis {split_dim} has {weight.shape[split_dim]}, "
                    f"expected {first + second}"
                )
            key = self._trainer_key(name)
            first_shape = list(weight.shape)
            first_shape[split_dim] = first
            second_shape = list(weight.shape)
            second_shape[split_dim] = second
            return [
                (
                    key.replace("in_proj_qkvz", "qkv_proj"),
                    tuple(first_shape),
                    False,
                ),
                (
                    key.replace("in_proj_qkvz", "g_proj"),
                    tuple(second_shape),
                    False,
                ),
            ], split_dim
        if ".linear_attn.out_proj." in name:
            return [
                (
                    self._trainer_key(name).replace(
                        ".linear_attn.out_proj.", ".linear_attn.o_proj."
                    ),
                    tuple(weight.shape),
                    False,
                )
            ], None
        return None

    def _raw_b_entries(self, name: str, weight: torch.Tensor, adapter: LoRAAdapter):
        targets = self._config_targets(adapter)

        gdn = self._raw_gdn_entries(name, weight, adapter)
        if gdn is not None:
            return gdn

        shared_key = self._shared_tail_key(name, adapter)
        if shared_key is not None and "gate_up_proj" in name:
            split_dim = weight.dim() - 2
            if weight.shape[split_dim] % 2:
                raise ValueError(f"Odd stacked shared gate_up LoRA-B shape for {name!r}")
            half = weight.shape[split_dim] // 2
            shape = list(weight.shape)
            shape[split_dim] = half
            key = self._trainer_key(shared_key)
            return [
                (key.replace("gate_up_proj", "gate_proj"), tuple(shape), False),
                (key.replace("gate_up_proj", "up_proj"), tuple(shape), False),
            ], split_dim
        if shared_key is not None:
            return [(self._trainer_key(shared_key), tuple(weight.shape), False)], None

        if ".mlp.experts." in name and "gate_up_proj" in name:
            split_dim = weight.dim() - 2
            if weight.shape[split_dim] % 2:
                raise ValueError(f"Odd stacked gate_up LoRA-B shape for {name!r}")
            half = weight.shape[split_dim] // 2
            trainer_shape = list(weight.shape)
            trainer_shape[split_dim] = weight.shape[-1]
            trainer_shape[-1] = half
            key = self._trainer_key(name)
            return [
                (
                    key.replace("gate_up_proj.lora_B", "gate_proj_lora_B"),
                    tuple(trainer_shape),
                    True,
                ),
                (
                    key.replace("gate_up_proj.lora_B", "up_proj_lora_B"),
                    tuple(trainer_shape),
                    True,
                ),
            ], split_dim

        if ".mlp.experts." in name and "down_proj" in name:
            trainer_shape = list(weight.shape)
            trainer_shape[-2], trainer_shape[-1] = (
                trainer_shape[-1],
                trainer_shape[-2],
            )
            key = self._trainer_key(name)
            return [
                (
                    key.replace("down_proj.lora_B", "down_proj_lora_B"),
                    tuple(trainer_shape),
                    True,
                )
            ], None

        if "gate_up_proj" in name and "gate_up_proj" not in targets:
            split_dim = weight.dim() - 2
            if weight.shape[split_dim] % 2:
                raise ValueError(f"Odd stacked gate_up LoRA-B shape for {name!r}")
            half = weight.shape[split_dim] // 2
            shape = list(weight.shape)
            shape[split_dim] = half
            key = self._trainer_key(name)
            return [
                (key.replace("gate_up_proj", "gate_proj"), tuple(shape), False),
                (key.replace("gate_up_proj", "up_proj"), tuple(shape), False),
            ], split_dim

        return [(self._trainer_key(name), tuple(weight.shape), False)], None

    def _b_layout(self, adapter: LoRAAdapter):
        raw_entries = []
        assemble = []
        seen = set()
        for layer_id, layer in enumerate(adapter.layers):
            for name, weight in layer.weights.items():
                if "lora_B" not in name:
                    continue
                parts, split_dim = self._raw_b_entries(name, weight, adapter)
                assemble.append(
                    (
                        ("layer", layer_id, name),
                        [(raw_name, transpose) for raw_name, _shape, transpose in parts],
                        split_dim,
                    )
                )
                for raw_name, shape, _transpose in parts:
                    if raw_name not in seen:
                        raw_entries.append((raw_name, shape))
                        seen.add(raw_name)
        return sorted(raw_entries), assemble

    def _raw_a_entries(self, name: str, weight: torch.Tensor, adapter: LoRAAdapter):
        gdn = self._raw_gdn_entries(name, weight, adapter)
        if gdn is not None:
            return gdn
        shared_key = self._shared_tail_key(name, adapter)
        if shared_key is not None:
            return [(self._trainer_key(shared_key), tuple(weight.shape), False)], None
        return [(self._trainer_key(name), tuple(weight.shape), False)], None

    def _a_layout(self, adapter: LoRAAdapter):
        raw_entries = []
        assemble = []
        seen = set()
        for layer_id, layer in enumerate(adapter.layers):
            for name, weight in layer.weights.items():
                if "lora_A" not in name:
                    continue
                parts, split_dim = self._raw_a_entries(name, weight, adapter)
                assemble.append(
                    (
                        ("layer", layer_id, name),
                        [(raw_name, transpose) for raw_name, _shape, transpose in parts],
                        split_dim,
                    )
                )
                for raw_name, shape, _transpose in parts:
                    if raw_name not in seen:
                        raw_entries.append((raw_name, shape))
                        seen.add(raw_name)
        return sorted(raw_entries), assemble

    def _raw_noises(self, sorted_raw, *, seed: int, device) -> Dict[str, torch.Tensor]:
        names = [name for name, _shape in sorted_raw]
        numels = [int(torch.Size(shape).numel()) for _name, shape in sorted_raw]
        key = (tuple(names), tuple(numels), str(device))
        geometry = self._geometry_cache.get(key)
        if geometry is None:
            geometry = self._geometry_cache[key] = PhiloxSegmentGeometry(numels, device)
        flat = geometry.draw([zorl_param_subseed(int(seed), name) for name in names])
        return {
            name: flat[offset : offset + numel].view(shape)
            for (name, shape), numel, offset in zip(
                sorted_raw, numels, geometry.out_offsets, strict=True
            )
        }

    @staticmethod
    def _reassemble(raw, assemble):
        result = {}
        for normalized_key, parts, split_dim in assemble:
            pieces = []
            for raw_name, transpose in parts:
                tensor = raw[raw_name]
                if transpose:
                    tensor = tensor.transpose(-2, -1).contiguous()
                pieces.append(tensor)
            result[normalized_key] = (
                pieces[0] if split_dim is None else torch.cat(pieces, dim=split_dim)
            )
        return result

    def _candidate_adapter(
        self,
        parent: LoRAAdapter,
        *,
        lora_id: str,
        b_seed: int,
        a_seed: Optional[int],
        direction: str,
        b_sigma: float,
        perturbation_mode: str,
    ) -> LoRAAdapter:
        mode = self._mode(perturbation_mode)
        if direction not in ("positive", "negative"):
            raise ValueError(f"Unsupported ZORL direction {direction!r}")
        if mode in ("a_and_b", "fresh_ab") and a_seed is None:
            raise ValueError(f"a_seed is required for {mode!r}")
        if parent.embedding_layers:
            raise ValueError("ZORL seed transport does not support embedding LoRA")

        manager = self.manager
        device = manager.device if manager.device.type == "cuda" else torch.device("cpu")
        b_raw, b_assemble = self._b_layout(parent)
        b_noises = self._reassemble(
            self._raw_noises(b_raw, seed=b_seed, device=device), b_assemble
        )
        a_noises = {}
        if mode in ("a_and_b", "fresh_ab"):
            a_raw, a_assemble = self._a_layout(parent)
            a_noises = self._reassemble(
                self._raw_noises(a_raw, seed=int(a_seed), device=device),
                a_assemble,
            )

        candidate = LoRAAdapter(
            lora_id,
            parent.config,
            manager.base_hf_config,
            manager.load_config,
            manager.lora_backend,
            base_model=manager.base_model,
        )
        sign = 1.0 if direction == "positive" else -1.0

        def replace(weight, noise, scale):
            value = (noise.to(torch.float32) * float(scale)).to(weight.dtype)
            return value.cpu().contiguous() if value.is_cuda else value.contiguous()

        def perturb(weight, noise):
            base = weight.detach().to(device=noise.device, dtype=torch.float32)
            value = base.add(noise, alpha=sign * float(b_sigma)).to(weight.dtype)
            return value.cpu().contiguous() if value.is_cuda else value.contiguous()

        for layer_id, parent_layer in enumerate(parent.layers):
            candidate_layer = candidate.layers[layer_id]
            for name, weight in parent_layer.weights.items():
                key = ("layer", layer_id, name)
                if mode == "fresh_ab":
                    if "lora_B" in name:
                        candidate_layer.weights[name] = replace(
                            weight, b_noises[key], sign * float(b_sigma)
                        )
                    elif "lora_A" in name:
                        candidate_layer.weights[name] = replace(
                            weight, a_noises[key], 1.0
                        )
                    else:
                        candidate_layer.weights[name] = weight
                elif "lora_B" in name:
                    candidate_layer.weights[name] = perturb(weight, b_noises[key])
                elif mode == "a_and_b" and "lora_A" in name:
                    candidate_layer.weights[name] = perturb(weight, a_noises[key])
                else:
                    candidate_layer.weights[name] = weight
        return candidate

    @staticmethod
    def _validate_fresh_parent(parent: LoRAAdapter) -> None:
        for layer_id, layer in enumerate(parent.layers):
            for name, weight in layer.weights.items():
                if "lora_B" in name and bool(torch.any(weight != 0)):
                    raise ValueError(
                        "fresh_ab requires a zero-B parent; "
                        f"layer {layer_id} weight {name!r} is nonzero"
                    )

    def create(
        self,
        *,
        parent_lora_id: str,
        candidate_specs: list[Dict[str, Any]],
        b_sigma: float,
        perturbation_mode: str,
        preload_candidates: bool,
    ):
        self._require_noise_contract()
        manager = self.manager
        parent = manager.loras.get(parent_lora_id)
        if parent is None:
            return manager.create_lora_update_result(
                False, f"Parent LoRA adapter id {parent_lora_id!r} is not loaded"
            )
        mode = self._mode(perturbation_mode)
        if mode == "fresh_ab":
            self._validate_fresh_parent(parent)

        created = []
        try:
            for spec in candidate_specs:
                candidate_mode = self._mode(spec.get("perturbation_mode", mode))
                if candidate_mode != mode:
                    raise ValueError("Mixed perturbation modes in one generation")
                if candidate_mode in ("a_and_b", "fresh_ab") and spec.get("a_seed") is None:
                    raise ValueError(f"a_seed is required for {candidate_mode!r}")
                declared_rank = spec.get("rank")
                if declared_rank is not None and int(declared_rank) != int(parent.config.r):
                    raise ValueError(
                        f"Candidate rank {declared_rank} != parent rank {parent.config.r}"
                    )
                ref = LoRARef(
                    lora_id=str(spec["lora_id"]),
                    lora_name=str(spec["lora_name"]),
                    lora_path="__zorl_seed__",
                    pinned=bool(spec.get("pinned", False)),
                )
                if ref.lora_id in manager.configs:
                    raise ValueError(f"LoRA adapter id {ref.lora_id!r} is already loaded")
                manager.validate_new_adapter(parent.config, ref)
                manager.configs[ref.lora_id] = parent.config
                manager.lora_refs[ref.lora_id] = ref
                manager.num_pinned_loras += int(ref.pinned)
                self.virtual[ref.lora_id] = {
                    "parent_lora_id": parent_lora_id,
                    "b_seed": int(spec["b_seed"]),
                    "a_seed": None if spec.get("a_seed") is None else int(spec["a_seed"]),
                    "direction": str(spec["direction"]),
                    "b_sigma": float(spec.get("b_sigma", b_sigma)),
                    "perturbation_mode": candidate_mode,
                }
                created.append(ref.lora_id)

            if preload_candidates:
                if len(created) > manager.max_loras_per_batch:
                    raise ValueError(
                        f"Cannot preload {len(created)} candidates with "
                        f"max_loras_per_batch={manager.max_loras_per_batch}"
                    )
                manager.fetch_new_loras(set(created))
                if torch.cuda.is_available():
                    torch.cuda.synchronize(manager.device)
        except Exception as error:
            for lora_id in created:
                self.forget(lora_id)
                manager.rollback_lora_adapter(lora_id)
            return manager.create_lora_update_result(False, str(error))

        result = manager.create_lora_update_result(True)
        preloaded_slot_signature = []
        if preload_candidates:
            preloaded_slot_signature = sorted(
                (
                    "<base>" if uid is None else str(uid),
                    int(manager.memory_pool.uid_to_buffer_id[uid]),
                )
                for uid in (None, *created)
                if uid in manager.memory_pool.uid_to_buffer_id
            )
        result.metadata = {
            "created_candidate_count": len(created),
            "virtual_candidate_count": len(created),
            "preloaded_candidate_count": len(created) if preload_candidates else 0,
            "noise_layout": ZORL_NOISE_LAYOUT_V2,
            "fused_shared_expert_lora": False,
            "preloaded_slot_signature": preloaded_slot_signature,
        }
        return result

    def materialize(self, lora_id: str) -> None:
        existing = self.manager.loras.get(lora_id)
        if existing is not None and not getattr(existing, "_zorl_seeded_stub", False):
            return
        spec = self.virtual.get(lora_id)
        if spec is None:
            return
        parent = self.manager.loras.get(spec["parent_lora_id"])
        if parent is None:
            raise ValueError(f"Missing ZORL parent {spec['parent_lora_id']!r}")
        self.manager.loras[lora_id] = self._candidate_adapter(
            parent,
            lora_id=lora_id,
            b_seed=spec["b_seed"],
            a_seed=spec["a_seed"],
            direction=spec["direction"],
            b_sigma=spec["b_sigma"],
            perturbation_mode=spec["perturbation_mode"],
        )

    def strip_materialized(self, uids: Iterable[Optional[str]]) -> None:
        for uid in uids:
            if uid is None or uid not in self.virtual:
                continue
            adapter = self.manager.loras.get(uid)
            if adapter is None or getattr(adapter, "_zorl_seeded_stub", False):
                continue
            for layer in adapter.layers:
                layer.weights = {}
            adapter.embedding_layers = {}
            adapter.added_tokens_embeddings = {}
            adapter._zorl_seeded_stub = True

    def forget(self, lora_id: str) -> None:
        self.virtual.pop(lora_id, None)
