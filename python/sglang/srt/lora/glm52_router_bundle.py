"""Transactional receiver for XoRL exact GLM-5.2 router sidecars."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import torch
from safetensors import safe_open

GLM52_ROUTER_BUNDLE_SCHEMA = "xorl.glm52_router_bundle.v1"
# This must not be a root-level safetensors file: the generic LoRA loader
# scans those files and would consume router keys as adapter factors.
GLM52_ROUTER_TENSORS = "xorl_router/xorl_glm52_router.safetensors"
GLM52_ROUTER_MANIFEST = "xorl_glm52_router.json"
_ROUTER_MODULE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
logger = logging.getLogger(__name__)


def _expected_router_modules(base_model, config) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {}
    for name, module in base_model.named_modules():
        if module.__class__.__name__ != "MoEGate" or not getattr(
            module, "is_glm52", False
        ):
            continue
        match = _ROUTER_MODULE.search(name)
        if match is None:
            raise RuntimeError(
                f"Cannot derive GLM-5.2 layer id from serving router {name!r}"
            )
        key = f"layer.{int(match.group(1))}.weight"
        if key in modules:
            raise RuntimeError(f"Duplicate serving GLM-5.2 router key {key!r}")
        modules[key] = module

    expected_ids = list(
        range(int(config.first_k_dense_replace), int(config.num_hidden_layers))
    )
    actual_ids = sorted(int(key.split(".")[1]) for key in modules)
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Serving GLM-5.2 router inventory is incomplete: actual={actual_ids}, expected={expected_ids}"
        )
    return modules


def apply_glm52_router_bundle_transaction(
    base_model, config, adapter_config: dict, adapter_path: str
):
    """Validate all bytes first, then apply routers and return rollback snapshots."""

    marker = adapter_config.get("_xorl_glm52_router_bundle")
    if marker is None:
        return None
    if not getattr(config, "_glm52_exact_mode", False):
        raise RuntimeError("A GLM-5.2 router sidecar requires exact serving mode")
    if (
        not isinstance(marker, dict)
        or marker.get("schema") != GLM52_ROUTER_BUNDLE_SCHEMA
    ):
        raise RuntimeError("Unsupported or malformed GLM-5.2 router sidecar marker")

    directory = Path(adapter_path)
    manifest = json.loads((directory / GLM52_ROUTER_MANIFEST).read_text())
    if manifest.get("schema") != GLM52_ROUTER_BUNDLE_SCHEMA:
        raise RuntimeError("GLM-5.2 router manifest schema mismatch")
    for field in ("tensor_file", "sha256", "router_count", "layer_ids", "weight_step"):
        if manifest.get(field) != marker.get(field):
            raise RuntimeError(f"GLM-5.2 router marker/manifest {field} mismatch")
    if manifest.get("tensor_file") != GLM52_ROUTER_TENSORS:
        raise RuntimeError("GLM-5.2 router manifest names an unsupported tensor file")
    tensor_path = directory / GLM52_ROUTER_TENSORS
    digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    if digest != manifest.get("sha256") or digest != marker.get("sha256"):
        raise RuntimeError("GLM-5.2 router sidecar SHA256 mismatch")

    modules = _expected_router_modules(base_model, config)
    expected_keys = set(modules)
    expected_ids = sorted(int(key.split(".")[1]) for key in expected_keys)
    if (
        manifest.get("router_count") != len(expected_keys)
        or manifest.get("layer_ids") != expected_ids
    ):
        raise RuntimeError(
            "GLM-5.2 router manifest does not match the serving router inventory"
        )
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                "GLM-5.2 router sidecar inventory mismatch: "
                f"missing={sorted(expected_keys - actual_keys)}, extra={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(actual_keys):
            tensor = handle.get_tensor(key)
            destination = modules[key].weight
            if tensor.dtype is not torch.bfloat16 or tuple(tensor.shape) != tuple(
                destination.shape
            ):
                raise RuntimeError(
                    f"GLM-5.2 router {key!r} has dtype/shape {tensor.dtype}/{tuple(tensor.shape)}, "
                    f"expected {destination.dtype}/{tuple(destination.shape)}"
                )
            tensors[key] = tensor

    snapshots = {key: module.weight.detach().clone() for key, module in modules.items()}
    try:
        with torch.no_grad():
            for key, module in modules.items():
                module.weight.copy_(tensors[key].to(device=module.weight.device))
    except BaseException:
        restore_glm52_router_snapshot(base_model, config, snapshots)
        raise
    logger.info(
        "Applied exact GLM-5.2 router sidecar: routers=%d weight_step=%s sha256=%s",
        len(modules),
        manifest["weight_step"],
        digest,
    )
    return snapshots


def restore_glm52_router_snapshot(base_model, config, snapshots) -> None:
    if snapshots is None:
        return
    modules = _expected_router_modules(base_model, config)
    if set(modules) != set(snapshots):
        raise RuntimeError(
            "Cannot restore GLM-5.2 routers from an incompatible snapshot"
        )
    with torch.no_grad():
        for key, module in modules.items():
            module.weight.copy_(snapshots[key])


__all__ = ["apply_glm52_router_bundle_transaction", "restore_glm52_router_snapshot"]
