import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.lora.dsv4 import (
    DSV4_FLASH_LOGICAL_FACTOR_COUNT,
    DSV4_FLASH_LORA_FORMAT,
    DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT,
    DSV4_FLASH_REQUIRED_TARGET_MODULES,
    DSV4_FLASH_ROUTED_BANK_COUNT,
    Dsv4FlashExactValidator,
    build_dsv4_flash_exact_inventory,
    maybe_create_dsv4_flash_validator,
    summarize_dsv4_flash_factor_roles,
    validate_dsv4_flash_exact_request_routing,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _load_lora_utils_for_cpu_contract_test():
    """Load pure shape helpers without importing the optional kernel stack."""

    dependency_names = (
        "sglang.srt.model_executor.forward_batch_info",
        "sglang.srt.utils.hf_transformers_utils",
    )
    previous = {name: sys.modules.get(name) for name in dependency_names}
    forward_stub = ModuleType(dependency_names[0])
    forward_stub.ForwardBatch = object
    transformers_stub = ModuleType(dependency_names[1])
    transformers_stub.AutoConfig = object
    sys.modules[dependency_names[0]] = forward_stub
    sys.modules[dependency_names[1]] = transformers_stub
    try:
        path = Path(__file__).parents[4] / "python/sglang/srt/lora/utils.py"
        spec = importlib.util.spec_from_file_location("_dsv4_test_lora_utils", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


_LORA_UTILS = _load_lora_utils_for_cpu_contract_test()


def _official_config(*, exact: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        model_type="deepseek_v4",
        head_dim=512,
        hidden_size=4096,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_attention_heads=64,
        num_hidden_layers=43,
        num_key_value_heads=1,
        o_groups=8,
        o_lora_rank=1024,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        vocab_size=129280,
        expert_dtype="fp4",
        hidden_act="silu",
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        swiglu_limit=10.0,
        _dsv4_flash_exact_mode=exact,
    )


def _adapter_config() -> dict:
    return {
        "peft_type": "LORA",
        "_sglang_lora_format": DSV4_FLASH_LORA_FORMAT,
        "r": 1,
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_dora": False,
        "use_rslora": False,
        "moe_hybrid_shared_lora": False,
        "target_modules": sorted(DSV4_FLASH_REQUIRED_TARGET_MODULES),
    }


def test_inventory_has_exact_factor_count_roles_and_shapes() -> None:
    specs = build_dsv4_flash_exact_inventory(_official_config(), _adapter_config())

    assert len(specs) == DSV4_FLASH_LOGICAL_FACTOR_COUNT == 948
    assert DSV4_FLASH_NON_ROUTED_LOGICAL_PROJECTION_COUNT == 345
    assert DSV4_FLASH_ROUTED_BANK_COUNT == 43
    assert {spec.export_dtype for spec in specs} == {torch.bfloat16}
    assert len({spec.export_key for spec in specs}) == len(specs)

    roles = summarize_dsv4_flash_factor_roles(specs)
    assert roles == {
        "attention.wkv": 86,
        "attention.wo_a": 86,
        "attention.wo_b": 86,
        "attention.wq_a": 86,
        "attention.wq_b": 86,
        "output.lm_head": 2,
        "routed_expert.down_proj": 86,
        "routed_expert.gate_proj": 86,
        "routed_expert.up_proj": 86,
        "shared_expert.down_proj": 86,
        "shared_expert.gate_proj": 86,
        "shared_expert.up_proj": 86,
    }

    by_key = {spec.export_key: spec for spec in specs}
    prefix = "base_model.model.model.layers.0"
    assert by_key[f"{prefix}.self_attn.wq_a.lora_A.weight"].export_shape == (1, 4096)
    assert by_key[f"{prefix}.self_attn.wq_b.lora_B.weight"].export_shape == (32768, 1)
    assert by_key[f"{prefix}.self_attn.wo_a.lora_B.weight"].export_shape == (8192, 1)
    assert by_key[f"{prefix}.mlp.experts.w1.lora_A.weight"].export_shape == (
        256,
        1,
        4096,
    )
    assert by_key[f"{prefix}.mlp.experts.w2.lora_B.weight"].export_shape == (
        256,
        4096,
        1,
    )
    assert by_key["base_model.model.lm_head.lora_embedding_B"].export_shape == (
        129280,
        1,
    )


def test_exact_mode_rejects_missing_format_partial_targets_and_wrong_rank() -> None:
    config = _official_config()
    adapter = _adapter_config()

    for lora_format in (None, "shared_outer", "ordinary"):
        candidate = dict(adapter)
        candidate["_sglang_lora_format"] = lora_format
        with pytest.raises(ValueError, match="dsv4_expert_banks"):
            maybe_create_dsv4_flash_validator(config, candidate)

    candidate = dict(adapter)
    candidate["target_modules"] = candidate["target_modules"][:-1]
    with pytest.raises(ValueError, match="target_modules mismatch"):
        build_dsv4_flash_exact_inventory(config, candidate)

    candidate = dict(adapter, r=2)
    with pytest.raises(ValueError, match="rank-1/alpha-1"):
        build_dsv4_flash_exact_inventory(config, candidate)


def test_streaming_validator_rejects_shape_dtype_extra_duplicate_and_missing() -> None:
    config = _official_config()
    adapter = _adapter_config()
    specs = build_dsv4_flash_exact_inventory(config, adapter)

    validator = Dsv4FlashExactValidator(config, adapter)
    first = specs[0]
    with pytest.raises(ValueError, match="shape"):
        validator.observe(first.export_key, torch.empty((1, 1), dtype=torch.bfloat16))

    validator = Dsv4FlashExactValidator(config, adapter)
    with pytest.raises(ValueError, match="dtype"):
        validator.observe(
            first.export_key, torch.empty(first.export_shape, dtype=torch.float32)
        )

    validator = Dsv4FlashExactValidator(config, adapter)
    with pytest.raises(ValueError, match="Unexpected tensor"):
        validator.observe("extra", torch.empty(1, dtype=torch.bfloat16))

    validator = Dsv4FlashExactValidator(config, adapter)
    tensor = torch.empty(first.export_shape, dtype=torch.bfloat16)
    validator.observe(first.export_key, tensor)
    with pytest.raises(ValueError, match="Duplicate tensor"):
        validator.observe(first.export_key, tensor)
    with pytest.raises(ValueError, match="missing 947 of 948"):
        validator.finalize()


def test_streaming_validator_distinguishes_zero_from_nonzero_adapter() -> None:
    config = _official_config()
    adapter = _adapter_config()
    specs = build_dsv4_flash_exact_inventory(config, adapter)
    validator = Dsv4FlashExactValidator(config, adapter)
    first, second = specs[:2]
    validator.observe(
        first.export_key, torch.zeros(first.export_shape, dtype=torch.bfloat16)
    )
    assert validator.all_zero is True
    value = torch.zeros(second.export_shape, dtype=torch.bfloat16)
    value.flatten()[0] = 1
    validator.observe(second.export_key, value)
    assert validator.all_zero is False


def test_non_exact_non_dsv4_adapter_is_ignored() -> None:
    config = _official_config(exact=False)
    config.architectures = ["OtherForCausalLM"]
    assert maybe_create_dsv4_flash_validator(config, _adapter_config()) is None


def test_exact_request_routing_is_explicitly_rank_zero_only() -> None:
    validate_dsv4_flash_exact_request_routing(0)
    for candidate in (None, 1, 7):
        with pytest.raises(ValueError, match="routed_dp_rank=0"):
            validate_dsv4_flash_exact_request_routing(candidate)


def test_dsv4_attention_targets_do_not_alias_dsa_indexer_and_have_exact_dims() -> None:
    config = _official_config()
    normalized = _LORA_UTILS.get_normalized_target_modules(
        _adapter_config()["target_modules"]
    )
    assert "self_attn.wq_b" in normalized
    assert "indexer.wq_b" not in normalized
    assert normalized == {
        "down_proj",
        "gate_up_proj",
        "lm_head",
        "self_attn.wq_b",
        "wkv",
        "wo_a",
        "wo_b",
        "wq_a",
    }

    expected = {
        "wq_a": (4096, 1024),
        "self_attn.wq_b": (1024, 32768),
        "wkv": (4096, 512),
        "wo_a": (4096, 8192),
        "wo_b": (8192, 4096),
        "gate_up_proj": (4096, 4096),
        "down_proj": (2048, 4096),
        "lm_head": (4096, 129280),
    }
    for module_name, dims in expected.items():
        assert _LORA_UTILS.get_default_hidden_dim(module_name, config, 0) == dims


def test_only_exact_dsv4_attention_is_admitted_for_deterministic_inference() -> None:
    from sglang.srt.arg_groups.overrides import (
        ResolvedView,
        _deterministic_attention_backend,
    )

    exact = ResolvedView(
        SimpleNamespace(
            enable_deterministic_inference=True,
            attention_backend="dsv4",
            dsv4_flash_exact_mode=True,
        )
    )
    assert _deterministic_attention_backend(exact) == {}

    generic = ResolvedView(
        SimpleNamespace(
            enable_deterministic_inference=True,
            attention_backend="dsv4",
            dsv4_flash_exact_mode=False,
        )
    )
    with pytest.raises(ValueError, match="explicitly specified 'dsv4'"):
        _deterministic_attention_backend(generic)


def test_grouped_wo_a_lora_selects_only_the_matching_b_rows() -> None:
    from sglang.srt.lora.layers import ColumnParallelLinearWithLoRA

    class _Backend:
        @staticmethod
        def run_lora_a_sgemm(x, weights):
            assert x.is_contiguous()
            return x @ weights[0].transpose(0, 1)

        @staticmethod
        def run_lora_b_sgemm(
            *, x, weights, output_offset, output_offset_cpu, base_output
        ):
            del output_offset, output_offset_cpu
            assert base_output.is_contiguous()
            return base_output + x @ weights[0].transpose(0, 1)

    wrapper = ColumnParallelLinearWithLoRA.__new__(ColumnParallelLinearWithLoRA)
    torch.nn.Module.__init__(wrapper)
    wrapper.lora_backend = _Backend()
    wrapper.A_buffer = torch.tensor([[[1.0, 2.0]]])
    wrapper.B_buffer = torch.tensor(
        [[[[1.0]], [[2.0]], [[10.0]], [[20.0]], [[100.0]], [[200.0]]]]
    ).reshape(1, 6, 1)
    wrapper._grouped_output_offsets = (
        (3, 2),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
    )
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    base = torch.zeros(1, 3, 2)

    actual = wrapper.apply_grouped_lora(base, x)
    expected = torch.tensor([[[1.0, 2.0], [20.0, 40.0], [300.0, 600.0]]])
    assert torch.equal(actual, expected)


def test_marlin_lora_refreshes_lazy_ep_metadata_after_dispatch() -> None:
    from sglang.srt.lora.layers import FusedMoEWithLoRA

    expert_map = torch.arange(32, dtype=torch.int32)
    refreshed = SimpleNamespace(expert_map=expert_map, global_num_experts=256)
    calls = []

    class _QuantMethod:
        @staticmethod
        def get_marlin_quant_info(base_layer):
            calls.append(base_layer)
            return refreshed

    wrapper = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
    wrapper._lora_runner_backend = SimpleNamespace(is_marlin=lambda: True)
    wrapper._quant_info = SimpleNamespace(expert_map=None, global_num_experts=-1)
    base_layer = SimpleNamespace(
        quant_method=_QuantMethod(),
        dispatcher=SimpleNamespace(moe_ep_size=8),
    )

    assert wrapper._quant_info_after_dispatch(base_layer) is refreshed
    assert calls == [base_layer]


def test_marlin_lora_fails_closed_when_lazy_ep_map_is_missing() -> None:
    from sglang.srt.lora.layers import FusedMoEWithLoRA

    class _QuantMethod:
        @staticmethod
        def get_marlin_quant_info(_):
            return SimpleNamespace(expert_map=None, global_num_experts=-1)

    wrapper = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
    wrapper._lora_runner_backend = SimpleNamespace(is_marlin=lambda: True)
    base_layer = SimpleNamespace(
        quant_method=_QuantMethod(),
        dispatcher=SimpleNamespace(moe_ep_size=8),
    )

    with pytest.raises(RuntimeError, match="localized EP ids"):
        wrapper._quant_info_after_dispatch(base_layer)


def test_small_batch_moe_lora_alignment_skips_ep_sentinels() -> None:
    from sglang.srt.lora.lora_moe_runners import (
        _naive_moe_lora_align_block_size,
    )

    sorted_ids, expert_ids, padded = _naive_moe_lora_align_block_size(
        topk_ids=torch.tensor([[0, -1, 2]], dtype=torch.int32),
        seg_indptr=torch.tensor([0, 1], dtype=torch.int32),
        req_to_lora=torch.tensor([1], dtype=torch.int32),
        num_experts=3,
        block_size_m=2,
        max_loras=2,
        max_num_tokens_padded=8,
        max_num_m_blocks=4,
        adapter_enabled=torch.tensor([False, True]),
        device=torch.device("cpu"),
    )

    assert padded.tolist() == [0, 4]
    assert sorted_ids[8:12].tolist() == [0, 3, 2, 3]
    assert expert_ids[4:6].tolist() == [0, 2]
    assert expert_ids[4] != -1


def test_dsv4_exact_marlin_geometry_is_narrow_and_fail_closed() -> None:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        DSV4_EXACT_MARLIN_MAX_CHUNK_TOKENS,
        is_dsv4_exact_pinned_marlin_geometry,
    )

    admitted = {
        "is_mxfp4_marlin": True,
        "global_experts": 256,
        "local_experts": 32,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "topk": 6,
        "clamp_limit": 10.0,
    }
    assert DSV4_EXACT_MARLIN_MAX_CHUNK_TOKENS == 10
    assert is_dsv4_exact_pinned_marlin_geometry(**admitted)

    for field, value in {
        "is_mxfp4_marlin": False,
        "global_experts": 255,
        "local_experts": 31,
        "hidden_size": 4095,
        "intermediate_size": 2047,
        "topk": 5,
        "clamp_limit": 9.0,
    }.items():
        candidate = dict(admitted)
        candidate[field] = value
        assert not is_dsv4_exact_pinned_marlin_geometry(**candidate)


def test_chunked_moe_lora_info_rebases_segments_and_slices_tokens() -> None:
    from sglang.srt.lora.lora_moe_runners import LoRAInfo, slice_moe_lora_info

    weights = torch.zeros(1, 1, 1, 1)
    info = LoRAInfo(
        gate_up_lora_a_weights=weights,
        gate_up_lora_b_weights=weights,
        down_lora_a_weights=weights,
        down_lora_b_weights=weights,
        seg_indptr=torch.tensor([0, 3, 8, 12], dtype=torch.int32),
        req_to_lora=torch.tensor([0, 1, 0], dtype=torch.int32),
        lora_ranks=torch.tensor([1, 1], dtype=torch.int32),
        adapter_enabled=torch.tensor([True, True]),
        token_lora_mapping=torch.tensor([0] * 3 + [1] * 5 + [0] * 4),
        max_lora_rank=1,
        num_experts=1,
    )

    sliced = slice_moe_lora_info(info, 4, 10)

    assert sliced is not None
    assert sliced.seg_indptr.tolist() == [0, 0, 4, 6]
    assert sliced.req_to_lora is info.req_to_lora
    assert sliced.token_lora_mapping.tolist() == [1, 1, 1, 1, 0, 0]
    assert sliced.gate_up_lora_a_weights is weights
    assert slice_moe_lora_info(None, 0, 1) is None


def test_exact_batch_admits_eager_decode_but_rejects_real_decode_graph() -> None:
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    manager = LoRAManager.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(_dsv4_flash_exact_mode=True)
    manager.lora_backend = SimpleNamespace(_dsv4_flash_exact_batch_certified=False)
    batch = SimpleNamespace(
        batch_size=1,
        forward_mode=ForwardMode.DECODE,
        lora_ids=[None],
    )

    manager._validate_dsv4_flash_exact_batch(batch)
    assert manager.lora_backend._dsv4_flash_exact_batch_certified is True

    manager.max_bs_in_cuda_graph = 1
    with pytest.raises(RuntimeError, match="eager execution only"):
        manager._validate_dsv4_flash_exact_batch(batch)


def test_certified_all_zero_adapter_prepares_as_literal_base_noop() -> None:
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    prepared = {}

    class _Backend(SimpleNamespace):
        def prepare_lora_batch(self, **kwargs):
            prepared.update(kwargs)
            self.batch_info = SimpleNamespace(has_active_lora=True)

    backend = _Backend(
        _dsv4_flash_exact_batch_certified=False,
        prefill_cuda_graph_max_bs=None,
        prefill_cuda_graph_max_tokens=None,
    )
    adapter = SimpleNamespace(
        config=SimpleNamespace(r=1),
        scaling=1,
        _dsv4_flash_exact_adapter_certified=True,
        _dsv4_flash_exact_all_zero=True,
    )
    manager = LoRAManager.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(
        _dsv4_flash_exact_mode=True,
        _glm52_exact_mode=False,
    )
    manager.max_loras_per_batch = 3
    manager.memory_pool = SimpleNamespace(
        uid_to_buffer_id={None: 0, "zero": 2},
        get_buffer_id=lambda uid: {None: 0, "zero": 2}[uid],
    )
    manager.loras = {"zero": adapter}
    manager.lora_backend = backend
    batch = SimpleNamespace(
        batch_size=1,
        forward_mode=ForwardMode.DECODE,
        lora_ids=["zero"],
    )

    manager.prepare_lora_batch(batch)

    assert prepared["weight_indices"] == [0]
    assert prepared["lora_ranks"] == [0, 0, 0]
    assert prepared["scalings"] == [0, 0, 0]
    assert backend.batch_info.has_active_lora is False
    assert backend._dsv4_flash_exact_batch_certified is True

    manager.memory_pool.uid_to_buffer_id.pop(None)
    with pytest.raises(RuntimeError, match="resident base-model LoRA slot"):
        manager.prepare_lora_batch(batch)
