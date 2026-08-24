import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.srt.lora.glm52_router_bundle import (
    GLM52_ROUTER_BUNDLE_SCHEMA,
    GLM52_ROUTER_MANIFEST,
    GLM52_ROUTER_TENSORS,
    apply_glm52_router_bundle_transaction,
    restore_glm52_router_snapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class MoEGate(torch.nn.Module):
    is_glm52 = True

    def __init__(self, value: float):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.full((3, 4), value, dtype=torch.bfloat16)
        )


class _MLP(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.gate = MoEGate(value)


class _Layer(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.mlp = _MLP(value)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(1.0), _Layer(2.0)])


def _write_bundle(directory):
    state = {
        "layer.0.weight": torch.full((3, 4), 11.0, dtype=torch.bfloat16),
        "layer.1.weight": torch.full((3, 4), 12.0, dtype=torch.bfloat16),
    }
    tensor_path = directory / GLM52_ROUTER_TENSORS
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, tensor_path)
    digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest = {
        "schema": GLM52_ROUTER_BUNDLE_SCHEMA,
        "tensor_file": GLM52_ROUTER_TENSORS,
        "sha256": digest,
        "router_count": 2,
        "layer_ids": [0, 1],
        "weight_step": 7,
    }
    (directory / GLM52_ROUTER_MANIFEST).write_text(json.dumps(manifest))
    return {"_xorl_glm52_router_bundle": dict(manifest)}


def _config():
    return SimpleNamespace(
        _glm52_exact_mode=True, first_k_dense_replace=0, num_hidden_layers=2
    )


def test_router_bundle_apply_and_restore_are_transactional(tmp_path):
    model = _Model()
    originals = [layer.mlp.gate.weight.detach().clone() for layer in model.layers]
    adapter_config = _write_bundle(tmp_path)

    snapshot = apply_glm52_router_bundle_transaction(
        model, _config(), adapter_config, str(tmp_path)
    )
    assert torch.all(model.layers[0].mlp.gate.weight == 11)
    assert torch.all(model.layers[1].mlp.gate.weight == 12)

    restore_glm52_router_snapshot(model, _config(), snapshot)
    assert torch.equal(model.layers[0].mlp.gate.weight, originals[0])
    assert torch.equal(model.layers[1].mlp.gate.weight, originals[1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_corrupt_router_bundle_fails_before_mutation(tmp_path):
    model = _Model()
    originals = [layer.mlp.gate.weight.detach().clone() for layer in model.layers]
    adapter_config = _write_bundle(tmp_path)
    adapter_config["_xorl_glm52_router_bundle"]["sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="marker/manifest sha256 mismatch"):
        apply_glm52_router_bundle_transaction(
            model, _config(), adapter_config, str(tmp_path)
        )
    assert torch.equal(model.layers[0].mlp.gate.weight, originals[0])
    assert torch.equal(model.layers[1].mlp.gate.weight, originals[1])
