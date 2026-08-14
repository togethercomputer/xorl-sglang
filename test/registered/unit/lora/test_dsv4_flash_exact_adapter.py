import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.batch_invariant_ops import exact_temperature_scale_bf16_logits
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
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
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


def _exact_ingress_manager() -> TokenizerManager:
    manager = object.__new__(TokenizerManager)
    manager.context_len = 8192
    manager.num_reserved_tokens = 0
    manager.validate_total_tokens = True
    manager.allow_auto_truncate = False
    manager.preferred_sampling_params = {}
    manager.is_generation = True
    manager.server_args = SimpleNamespace(
        dsv4_flash_exact_mode=True,
        glm52_exact_mode=False,
        qwen35_gdn_exact_mode=False,
        qwen3_dense_exact_mode=False,
    )
    return manager


def _dsv4_parallel_context(
    *, dp_size: int, cp_size: int, dp_rank: int = 0, cp_rank: int = 0, gather
):
    source_ordinal = dp_rank * cp_size + cp_rank

    class _Group:
        world_size = 8
        rank_in_group = source_ordinal
        ranks = list(range(8))

        @staticmethod
        def all_gather_object(value):
            return gather(value)

    cp_start = dp_rank * cp_size
    return SimpleNamespace(
        attn_dp_size=dp_size,
        attn_cp_size=cp_size,
        attn_dp_rank=dp_rank,
        attn_cp_rank=cp_rank,
        attn_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        tp_group=_Group(),
        moe_ep_group=_Group(),
        attn_cp_group=SimpleNamespace(
            ranks=list(range(cp_start, cp_start + cp_size))
        ),
    )


def test_exact_lora_segments_collapse_cp_replicas_and_keep_multiple_requests() -> None:
    from sglang.srt.layers.logical_row_ownership import LogicalRowOwnership
    from sglang.srt.lora.lora_manager import _flatten_dp_request_segments

    ownership = LogicalRowOwnership(2, 4, 0, 0, 8)
    dp0 = (("adapter-a", 2), ("adapter-b", 1))
    dp1 = (("adapter-c", 1),)
    physical = [dp0] * 4 + [dp1] * 4

    assert _flatten_dp_request_segments(
        ownership,
        physical,
        [4, 2],
        family="DSV4-Flash",
    ) == [
        ("adapter-a", 2),
        ("adapter-b", 1),
        (None, 1),
        ("adapter-c", 1),
        (None, 1),
    ]


def test_exact_dp_lora_metadata_follows_every_request_owner(monkeypatch) -> None:
    import sglang.srt.lora.lora_manager as lora_manager_module
    from sglang.srt.lora.backend.base_backend import BaseLoRABackend
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    class _MemoryPool:
        uid_to_buffer_id = {}

        @staticmethod
        def get_buffer_id(uid):
            return _MemoryPool.uid_to_buffer_id[uid]

    adapter = SimpleNamespace(
        config=SimpleNamespace(r=1),
        scaling=1,
        _dsv4_flash_exact_adapter_certified=True,
        _dsv4_flash_exact_all_zero=False,
    )
    backend = object.__new__(BaseLoRABackend)
    backend._is_moe_lora = True
    backend.batch_info = None
    backend.context_parallel_mlp_batch_info = None
    backend.sgemm_batch_info = None

    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(_dsv4_flash_exact_mode=True)
    manager.max_loras_per_batch = 2
    manager.device = torch.device("cpu")
    manager.lora_backend = backend
    manager.memory_pool = _MemoryPool()
    manager.loras = {"adapter": adapter}
    manager.lora_refs = {"adapter": object()}
    fetched = []

    def _fetch(uids):
        fetched.append(set(uids))
        for uid in sorted(uids, key=lambda value: (value is not None, value or "")):
            _MemoryPool.uid_to_buffer_id.setdefault(
                uid, len(_MemoryPool.uid_to_buffer_id)
            )

    manager.fetch_new_loras = _fetch

    token_count_cases = (([1] * 8, False), (list(range(1, 9)), True))
    for token_counts, is_extend in token_count_cases:
        for local_rank in range(8):
            for owner_rank in range(8):
                global_uids = [None] * 8
                global_uids[owner_rank] = "adapter"
                # The owner may have admitted the adapter locally before the
                # gathered preparation, while idle ranks can still be empty.
                # The global mapping must use each rank's physical slot IDs.
                _MemoryPool.uid_to_buffer_id.clear()
                if local_rank == owner_rank:
                    _MemoryPool.uid_to_buffer_id["adapter"] = 0

                physical_segments = [
                    ((uid, token_counts[rank]),)
                    for rank, uid in enumerate(global_uids)
                ]

                def gather(local_segments):
                    assert local_segments == physical_segments[local_rank]
                    return physical_segments

                monkeypatch.setattr(
                    lora_manager_module,
                    "get_parallel",
                    lambda: _dsv4_parallel_context(
                        dp_size=8,
                        cp_size=1,
                        dp_rank=local_rank,
                        gather=gather,
                    ),
                )
                forward_batch = SimpleNamespace(
                    lora_ids=[global_uids[local_rank]],
                    global_num_tokens_cpu=token_counts,
                    is_extend_in_batch=is_extend,
                    extend_seq_lens_cpu=([token_counts[local_rank]] if is_extend else None),
                    forward_mode=ForwardMode.IDLE,
                )

                manager.prepare_dsv4_flash_exact_dp_lora_batch(forward_batch)

                info = backend.context_parallel_mlp_batch_info
                assert info.expected_tokens == sum(token_counts)
                assert info.weight_indices.tolist() == [
                    _MemoryPool.uid_to_buffer_id[
                        "adapter" if rank == owner_rank else None
                    ]
                    for rank in range(8)
                ]
                assert info.moe_lora_info.token_lora_mapping.tolist() == [
                    _MemoryPool.uid_to_buffer_id[
                        "adapter" if rank == owner_rank else None
                    ]
                    for rank, count in enumerate(token_counts)
                    for _ in range(count)
                ]

    assert fetched == [{None, "adapter"}] * 128


@pytest.mark.parametrize("dp_size", [1, 2, 4, 8])
def test_exact_decode_graph_reuses_fixed_dp_row_metadata(
    monkeypatch, dp_size: int
) -> None:
    import sglang.srt.lora.lora_manager as lora_manager_module
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    class _MemoryPool:
        uid_to_buffer_id = {None: 0, "adapter": 1}

        @staticmethod
        def get_buffer_id(uid):
            return _MemoryPool.uid_to_buffer_id[uid]

    adapter = SimpleNamespace(
        config=SimpleNamespace(r=1),
        scaling=1,
        _dsv4_flash_exact_adapter_certified=True,
        _dsv4_flash_exact_all_zero=False,
    )
    backend = object.__new__(TritonLoRABackend)
    backend.device = torch.device("cpu")
    backend.max_loras_per_batch = 2
    backend._is_moe_lora = True
    backend.context_parallel_mlp_batch_info = None
    backend.init_context_parallel_cuda_graph_batch_info(num_rows=dp_size)
    backend.moe_cg_buffers = {
        "adapter_enabled": torch.zeros(2, dtype=torch.int32),
        "token_lora_mapping": torch.full((dp_size,), -1, dtype=torch.int32),
    }

    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(_dsv4_flash_exact_mode=True)
    manager.max_loras_per_batch = 2
    manager.max_bs_in_cuda_graph = 1
    manager.dp_size = dp_size
    manager.device = torch.device("cpu")
    manager.lora_backend = backend
    manager.memory_pool = _MemoryPool()
    manager.loras = {"adapter": adapter}
    manager.lora_refs = {"adapter": object()}
    manager.fetch_new_loras = lambda _: None

    first_owner = dp_size // 2
    second_owner = dp_size - 1
    owners = [None] * dp_size
    owners[first_owner] = "adapter"

    cp_size = 8 // dp_size

    def gather(local_segments):
        expected_local = ((owners[0], 1),)
        assert local_segments == expected_local
        return [
            ((owners[physical_rank // cp_size], 1),)
            for physical_rank in range(8)
        ]

    monkeypatch.setattr(
        lora_manager_module,
        "get_parallel",
        lambda: _dsv4_parallel_context(
            dp_size=dp_size,
            cp_size=cp_size,
            gather=gather,
        ),
    )
    batch = SimpleNamespace(
        batch_size=1,
        lora_ids=[owners[0]],
        global_num_tokens_cpu=[1] * dp_size,
        is_extend_in_batch=False,
        forward_mode=ForwardMode.DECODE,
    )

    fixed = backend.context_parallel_cuda_graph_batch_info
    fixed_tensor_ids = {
        name: id(getattr(fixed, name))
        for name in (
            "seg_lens",
            "seg_indptr",
            "weight_indices",
            "lora_ranks",
            "scalings",
        )
    }
    manager.prepare_dsv4_flash_exact_dp_lora_batch(batch)

    assert backend.context_parallel_mlp_batch_info is fixed
    assert fixed.use_cuda_graph is True
    expected = [0] * dp_size
    expected[first_owner] = 1
    assert fixed.expected_tokens == fixed.bs == dp_size
    assert fixed.seg_lens.tolist() == [1] * dp_size
    assert fixed.seg_indptr.tolist() == list(range(dp_size + 1))
    assert fixed.weight_indices.tolist() == expected
    assert fixed.moe_lora_info.token_lora_mapping.tolist() == expected

    owners[first_owner] = None
    owners[second_owner] = "adapter"
    batch.lora_ids = [owners[0]]
    manager.prepare_dsv4_flash_exact_dp_lora_batch(batch)
    expected = [0] * dp_size
    expected[second_owner] = 1
    assert backend.context_parallel_mlp_batch_info is fixed
    assert fixed.weight_indices.tolist() == expected
    assert {
        name: id(getattr(fixed, name)) for name in fixed_tensor_ids
    } == fixed_tensor_ids


@pytest.mark.parametrize("dp_size", [1, 2, 4, 8])
def test_exact_decode_graph_moe_buffers_cover_physical_dp_rows(
    monkeypatch, dp_size: int
) -> None:
    import sglang.srt.lora.layers as lora_layers
    from sglang.srt.lora.lora_manager import init_lora_cuda_graph_moe_buffers

    class _FakeMoELoRA:
        pass

    fake_layer = _FakeMoELoRA()
    model = SimpleNamespace(modules=lambda: [object(), fake_layer])
    calls = []
    manager = SimpleNamespace(
        init_cuda_graph_moe_buffers=lambda *args: calls.append(args)
    )
    server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=1)),
        dsv4_flash_exact_mode=True,
        dp_size=dp_size,
        max_loras_per_batch=2,
    )
    monkeypatch.setattr(lora_layers, "FusedMoEWithLoRA", _FakeMoELoRA)

    init_lora_cuda_graph_moe_buffers(
        server_args=server_args,
        model=model,
        lora_manager=manager,
        dtype=torch.bfloat16,
    )

    assert calls == [(dp_size, 2, torch.bfloat16, fake_layer)]


def test_exact_radix_page_boundary_preserves_c4_overlap_addresses() -> None:
    from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

    pool = object.__new__(CompressStatePool)
    pool.swa_page_size = 256
    pool.ring_size = 8
    swa_locs = torch.tensor([255, 256, 257, 511, 512, -1])

    assert pool.translate_from_swa_loc_to_state_loc(swa_locs).tolist() == [
        7,
        8,
        9,
        15,
        16,
        -1,
    ]
    # A page-256 radix match is simultaneously a C4 and offline-C128 boundary,
    # so no request-scoped partial C128 row crosses into continuation decode.
    assert 256 % 4 == 0
    assert 256 % 128 == 0


def test_exact_radix_namespace_isolates_dp_owner_and_lora_generation() -> None:
    from sglang.srt.managers.scheduler import Scheduler

    def cache_key(owner: int, lora_generation: str) -> str:
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(dsv4_flash_exact_mode=True)
        scheduler.disable_radix_cache = False
        scheduler.tree_cache = SimpleNamespace(is_tree_cache=lambda: True)
        scheduler.ps = SimpleNamespace(attn_dp_rank=owner)
        req = SimpleNamespace(extra_key=lora_generation)
        scheduler._maybe_namespace_dsv4_exact_radix_cache(req)
        return req.extra_key

    generation_a = cache_key(0, "adapter-generation-a")
    generation_b = cache_key(0, "adapter-generation-b")
    owner_b = cache_key(1, "adapter-generation-a")

    assert generation_a == "adapter-generation-a|dsv4_exact_dp_owner=0"
    assert len({generation_a, generation_b, owner_b}) == 3


@pytest.mark.parametrize(
    ("speculative", "lora_id", "rejected"),
    [(True, "policy", True), (True, None, True), (False, None, False)],
)
def test_exact_speculative_request_boundary(
    monkeypatch, speculative, lora_id, rejected
) -> None:
    import sglang.srt.managers.scheduler as scheduler_module
    from sglang.srt.managers.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(dsv4_flash_exact_mode=True)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: not speculative)
    streamed = []
    scheduler.output_streamer = SimpleNamespace(
        stream_output=lambda reqs, return_logprob: streamed.append(
            (reqs, return_logprob)
        )
    )
    aborted = []
    monkeypatch.setattr(
        scheduler_module,
        "prepare_abort",
        lambda req, message, status_code: aborted.append(
            (req, message, status_code)
        ),
    )
    request = SimpleNamespace(
        rid="exact-speculative",
        lora_id=lora_id,
        return_logprob=True,
    )

    assert scheduler._reject_dsv4_exact_speculative_request(request) is rejected
    assert bool(aborted) is rejected
    assert bool(streamed) is rejected
    if rejected:
        assert "generic speculative scorer" in aborted[0][1]


def test_dynamic_lora_reload_mints_a_new_radix_generation() -> None:
    from sglang.srt.managers.io_struct import (
        LoadLoRAAdapterReqInput,
        LoRAUpdateOutput,
    )

    async def run_two_loads():
        class _Registry:
            num_registered_loras = 0

            async def register(self, _):
                self.num_registered_loras += 1

        manager = object.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(
            enable_lora=True,
            dp_size=1,
            enable_dp_attention=False,
            max_loaded_loras=None,
        )
        manager.auto_create_handle_loop = lambda: None
        manager.lora_update_lock = asyncio.Lock()
        manager.lora_registry = _Registry()
        manager.lora_ref_cache = {}
        observed_ids = []

        async def update(req):
            observed_ids.append(req.lora_id)
            return [
                LoRAUpdateOutput(
                    success=True,
                    loaded_adapters={req.lora_name: req.lora_path},
                )
            ]

        manager.update_lora_adapter_communicator = update
        for _ in range(2):
            result = await manager.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name="policy",
                    lora_path="policy-adapter",
                )
            )
            assert result.success
        return observed_ids, manager.lora_ref_cache["policy"].lora_id

    observed_ids, latest_id = asyncio.run(run_two_loads())
    assert observed_ids[0] != observed_ids[1]
    assert latest_id == observed_ids[1]


def test_exact_resolution_preserves_topology_capacity_lora_and_cache_options() -> None:
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        Phase,
    )
    from sglang.srt.server_args import ServerArgs

    config = _official_config()
    config.hidden_size = 8192
    config.n_routed_experts = 128
    args = ServerArgs(model_path="dummy", enable_lora=True)
    args.rl_on_policy_target = "xorl"
    args.nnodes = 2
    args.tp_size = 4
    args.dp_size = 2
    args.ep_size = 4
    args.pp_size = 2
    args.moe_dp_size = 2
    args.attn_cp_size = 2
    args.dcp_size = 2
    args.enable_dp_attention = True
    args.enable_prefill_cp = True
    args.enable_prefill_context_parallel = True
    args.enable_dsa_prefill_context_parallel = True
    args.chunked_prefill_size = 4096
    args.max_prefill_tokens = 12288
    args.prefill_max_requests = 3
    args.max_total_tokens = 24576
    args.max_running_requests = 37
    args.mem_fraction_static = 0.7
    args.max_lora_rank = 8
    args.max_loras_per_batch = 7
    args.max_loaded_loras = 7
    args.lora_backend = "csgmv"
    args.lora_target_modules = {"q_proj"}
    args.enable_two_batch_overlap = True
    args.disable_overlap_schedule = False
    args.disable_radix_cache = False
    args.disable_cuda_graph = False
    args.disable_cuda_graph_padding = False
    args.disable_piecewise_cuda_graph = False
    args.cuda_graph_bs_decode = [1]
    args.cuda_graph_max_bs_decode = 1
    args.cuda_graph_config = CudaGraphConfig()
    args.cuda_graph_config.decode.backend = Backend.FULL
    args.cuda_graph_config.decode.bs = [1]
    args.cuda_graph_config.decode.max_bs = 1
    args._cuda_graph_config_locked = {
        (Phase.DECODE, "backend"),
        (Phase.DECODE, "bs"),
        (Phase.DECODE, "max_bs"),
    }

    args._resolve_dsv4_flash_exact_contract(
        config, model_arch="DeepseekV4ForCausalLM"
    )

    assert (
        args.nnodes,
        args.tp_size,
        args.dp_size,
        args.ep_size,
        args.pp_size,
        args.moe_dp_size,
        args.attn_cp_size,
        args.dcp_size,
    ) == (2, 4, 2, 4, 2, 2, 2, 2)
    assert args.enable_dp_attention is True
    assert args.enable_prefill_cp is True
    assert args.enable_prefill_context_parallel is True
    assert args.enable_dsa_prefill_context_parallel is True
    assert args.chunked_prefill_size == 4096
    assert args.max_prefill_tokens == 12288
    assert args.prefill_max_requests == 3
    assert args.max_total_tokens == 24576
    assert args.max_running_requests == 37
    assert args.mem_fraction_static == 0.7
    assert args.max_lora_rank == 8
    assert args.max_loras_per_batch == 7
    assert args.max_loaded_loras == 7
    assert args.lora_backend == "csgmv"
    assert args.lora_target_modules == {"q_proj"}
    assert args.enable_two_batch_overlap is True
    assert args.disable_overlap_schedule is False
    assert args.disable_radix_cache is False
    assert args.disable_cuda_graph is False
    assert args.disable_cuda_graph_padding is False
    assert args.disable_piecewise_cuda_graph is False
    assert args.cuda_graph_bs_decode == [1]
    assert args.cuda_graph_max_bs_decode == 1
    assert args.cuda_graph_config.decode.backend == Backend.FULL


def test_gathered_mlp_lora_metadata_is_scoped_and_restored() -> None:
    from sglang.srt.lora.backend.base_backend import BaseLoRABackend

    backend = object.__new__(BaseLoRABackend)
    local = SimpleNamespace(expected_tokens=1)
    gathered = SimpleNamespace(expected_tokens=8)
    local_sgemm = object()
    backend.batch_info = local
    backend.context_parallel_mlp_batch_info = gathered
    backend.sgemm_batch_info = local_sgemm

    with backend.use_gathered_mlp_batch_info(8):
        assert backend.batch_info is gathered
        assert backend.sgemm_batch_info is None

    assert backend.batch_info is local
    assert backend.sgemm_batch_info is local_sgemm
    with pytest.raises(RuntimeError, match="metadata_rows=8, activation_rows=7"):
        with backend.use_gathered_mlp_batch_info(7):
            pass


@pytest.mark.parametrize(
    ("name", "value"),
    (("top_k", 8), ("top_p", 0.9), ("min_p", 0.1)),
)
def test_exact_ingress_rejects_filtered_sampling(name, value) -> None:
    request = GenerateReqInput(
        input_ids=[1],
        sampling_params={name: value, "max_new_tokens": 1},
        return_logprob=True,
    )
    request.normalize_batch_and_arguments()

    with pytest.raises(ValueError, match=rf"{name}={value}"):
        _exact_ingress_manager()._validate_one_request(request, [1])


def test_exact_ingress_allows_temperature_aware_sampling() -> None:
    request = GenerateReqInput(
        input_ids=[1],
        sampling_params={"temperature": 0.7, "max_new_tokens": 1},
        return_logprob=True,
    )
    request.normalize_batch_and_arguments()

    _exact_ingress_manager()._validate_one_request(request, [1])


def test_exact_ingress_rejects_nonpositive_temperature() -> None:
    request = GenerateReqInput(
        input_ids=[1],
        sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        return_logprob=True,
    )
    request.normalize_batch_and_arguments()

    with pytest.raises(ValueError, match="temperature=0.0"):
        _exact_ingress_manager()._validate_one_request(request, [1])


def test_exact_temperature_preserves_dsv4_bf16_boundary() -> None:
    logits = torch.tensor(
        [[1.25, -0.5, 0.75], [-1.0, 2.0, 0.25]],
        dtype=torch.bfloat16,
    )
    ones = torch.ones(2, dtype=torch.float32)
    mixed = torch.tensor([0.7, 1.3], dtype=torch.float32)

    assert exact_temperature_scale_bf16_logits(logits, None) is logits
    assert torch.equal(
        exact_temperature_scale_bf16_logits(logits, ones).view(torch.uint8),
        logits.view(torch.uint8),
    )
    assert torch.equal(
        exact_temperature_scale_bf16_logits(logits, mixed).view(torch.uint8),
        logits.bfloat16().div(mixed.unsqueeze(1)).bfloat16().view(torch.uint8),
    )


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
        "dsv4_exact_mode": True,
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
    assert not is_dsv4_exact_pinned_marlin_geometry(
        **{**admitted, "dsv4_exact_mode": False}
    )

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


@pytest.mark.parametrize(
    "setting_name", ("SGLANG_DP_USE_GATHERV", "SGLANG_DP_USE_REDUCE_SCATTER")
)
def test_exact_resolved_contract_rejects_alternate_dp_combine(
    setting_name: str,
) -> None:
    from sglang.srt.environ import envs
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(model_path="dummy")
    args.dsv4_flash_exact_mode = True
    setting = getattr(envs, setting_name)
    with setting.override(True), pytest.raises(ValueError, match=setting_name):
        args._validate_dsv4_flash_exact_resolved_contract()


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


def test_exact_batch_checks_decode_graph_metadata_shape_not_topology() -> None:
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
    manager.dp_size = 4
    manager._validate_dsv4_flash_exact_batch(batch)
    assert manager.lora_backend._dsv4_flash_exact_batch_certified is True

    invalid_batch = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.DECODE,
        lora_ids=[None, None],
    )
    manager.max_bs_in_cuda_graph = 2
    with pytest.raises(RuntimeError, match="one local one-token request"):
        manager._validate_dsv4_flash_exact_batch(invalid_batch)


@pytest.mark.parametrize("lora_ids", [["policy"], [None]])
def test_exact_target_verify_rejects_before_model_forward(lora_ids) -> None:
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    manager = LoRAManager.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(_dsv4_flash_exact_mode=True)
    manager.lora_backend = SimpleNamespace(_dsv4_flash_exact_batch_certified=False)

    batch = SimpleNamespace(
        batch_size=1,
        forward_mode=ForwardMode.TARGET_VERIFY,
        lora_ids=lora_ids,
    )
    with pytest.raises(RuntimeError, match="generic speculative scorer"):
        manager._validate_dsv4_flash_exact_batch(batch)


def test_exact_fp8_format_accepts_mechanism_and_rejects_alternates() -> None:
    from sglang.srt.server_args import (
        _validate_dsv4_flash_exact_quantization_config,
    )

    expected = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
    _validate_dsv4_flash_exact_quantization_config(
        SimpleNamespace(quantization_config=dict(expected))
    )

    alternates = {
        "quant_method": "compressed-tensors",
        "activation_scheme": "static",
        "fmt": "e5m2",
        "scale_fmt": "fp32",
        "weight_block_size": [64, 128],
    }
    for field, value in alternates.items():
        config = dict(expected)
        config[field] = value
        with pytest.raises(ValueError, match=rf"quantization_config\.{field}"):
            _validate_dsv4_flash_exact_quantization_config(
                SimpleNamespace(quantization_config=config)
            )


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
