import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.deepep_native_exact import (
    DEEPEP_DETERMINISTIC_PROTOCOL,
    DEEPEP_LOW_LATENCY_DETERMINISTIC_PROTOCOL,
    DeepEPNativeExactError,
    adapt_native_lora_context,
    adapt_native_runner_metadata,
    canonicalize_native_routing_metadata,
    combine_deterministic_bf16,
    native_exact_router_topk,
    native_zero_row_runner_routes,
    pack_native_low_latency_preweighted_routes,
    reduce_native_runner_routes_to_bf16,
    update_native_lora_graph_control,
    validate_native_receive,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig, PermuteMethodPool
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerOutput,
    post_permute_triton_to_deepep_ll,
    post_permute_triton_to_deepep_normal,
    pre_permute_deepep_ll_to_triton,
    pre_permute_deepep_normal_to_triton,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLExactDispatchOutput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend


def test_low_latency_protocol_identifier_is_versioned_and_unambiguous() -> None:
    assert DEEPEP_LOW_LATENCY_DETERMINISTIC_PROTOCOL == (
        "deepep_deterministic_hierarchical_bf16_v2"
    )
    assert DEEPEP_LOW_LATENCY_DETERMINISTIC_PROTOCOL == DEEPEP_DETERMINISTIC_PROTOCOL


def test_normal_dispatcher_receives_exact_flag_without_public_mode(monkeypatch):
    import sglang.srt.layers.moe.token_dispatcher.deepep as dispatcher_module

    class _CaptureImpl:
        def __init__(self, *, deepep_native_exact, **_kwargs):
            self.deepep_native_exact = deepep_native_exact

    monkeypatch.setattr(dispatcher_module, "_DeepEPDispatcherImplNormal", _CaptureImpl)
    monkeypatch.setattr(dispatcher_module, "_use_aiter", False)

    common = dict(
        group=object(),
        router_topk=2,
        num_experts=4,
        num_local_experts=2,
        hidden_size=8,
        params_dtype=torch.bfloat16,
        deepep_mode=dispatcher_module.DeepEPMode.NORMAL,
        deepep_native_exact=True,
    )
    dispatcher = dispatcher_module.DeepEPDispatcher(**common)

    assert dispatcher._normal_dispatcher.deepep_native_exact is True


def test_exact_auto_dispatcher_owns_deterministic_normal_and_low_latency(monkeypatch):
    import sglang.srt.layers.moe.token_dispatcher.deepep as dispatcher_module

    class _CaptureImpl:
        def __init__(self, *, deepep_native_exact, **_kwargs):
            self.deepep_native_exact = deepep_native_exact

    monkeypatch.setattr(dispatcher_module, "_DeepEPDispatcherImplNormal", _CaptureImpl)
    monkeypatch.setattr(
        dispatcher_module, "_DeepEPDispatcherImplLowLatency", _CaptureImpl
    )
    monkeypatch.setattr(dispatcher_module, "_use_aiter", False)

    dispatcher = dispatcher_module.DeepEPDispatcher(
        group=object(),
        router_topk=2,
        num_experts=4,
        num_local_experts=2,
        hidden_size=8,
        params_dtype=torch.bfloat16,
        deepep_mode=dispatcher_module.DeepEPMode.AUTO,
        deepep_native_exact=True,
    )

    assert dispatcher._normal_dispatcher.deepep_native_exact is True
    assert dispatcher._low_latency_dispatcher.deepep_native_exact is True


def test_exact_default_normal_combine_selects_one_call_deterministic(monkeypatch):
    import sglang.srt.layers.moe.deepep_native_exact as exact_module
    import sglang.srt.layers.moe.token_dispatcher.deepep as dispatcher_module

    calls = []

    def deterministic(local_leaf, **kwargs):
        calls.append((local_leaf, kwargs))
        return local_leaf, "deterministic-event"

    monkeypatch.setattr(exact_module, "combine_deterministic_bf16", deterministic)
    monkeypatch.setattr(
        dispatcher_module, "_deepep_precompile_tp_barrier", lambda: None
    )
    monkeypatch.setattr(
        dispatcher_module.DeepEPConfig,
        "get_instance",
        classmethod(lambda _cls: SimpleNamespace(normal_combine_config=None)),
    )

    impl = object.__new__(dispatcher_module._DeepEPDispatcherImplNormal)
    impl.deepep_native_exact = True
    impl.num_local_experts = 2
    impl.group = object()
    impl.handle = object()
    impl.async_finish = False
    impl._get_buffer = lambda: object()
    leaf = torch.ones((2, 8), dtype=torch.bfloat16)
    ids = torch.zeros((2, 2), dtype=torch.int64)
    weights = torch.ones((2, 2), dtype=torch.float32)

    output, event = impl._combine_core((leaf, ids, weights), None)

    assert output is leaf
    assert event == "deterministic-event"
    assert len(calls) == 1


def test_stock_normal_combine_does_not_receive_deterministic_kwargs(monkeypatch):
    import sglang.srt.layers.moe.token_dispatcher.deepep as dispatcher_module

    monkeypatch.setattr(
        dispatcher_module, "_deepep_precompile_tp_barrier", lambda: None
    )
    monkeypatch.setattr(
        dispatcher_module.DeepEPConfig,
        "get_instance",
        classmethod(lambda _cls: SimpleNamespace(normal_combine_config=None)),
    )
    calls = []

    class _Buffer:
        def combine(self, *args, **kwargs):
            calls.append((args, kwargs))
            return args[0], None, "stock-event"

    impl = object.__new__(dispatcher_module._DeepEPDispatcherImplNormal)
    impl.deepep_native_exact = False
    impl.handle = object()
    impl.async_finish = False
    impl._get_buffer = _Buffer
    hidden = torch.ones((2, 8), dtype=torch.bfloat16)

    output, event = impl._combine_core(hidden, None)

    assert output is hidden
    assert event == "stock-event"
    assert len(calls) == 1
    assert "reduction_mode" not in calls[0][1]


def test_native_routing_metadata_preserves_fp32_and_widens_bf16() -> None:
    weights = torch.tensor([[0.33333334, 0.66666669]], dtype=torch.float32)
    expected_bf16 = weights.to(torch.bfloat16).to(torch.float32)

    actual_fp32 = canonicalize_native_routing_metadata(weights)
    actual_bf16 = canonicalize_native_routing_metadata(weights.to(torch.bfloat16))

    assert actual_fp32.dtype is torch.float32
    assert actual_fp32.is_contiguous()
    torch.testing.assert_close(actual_fp32, weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_bf16, expected_bf16, rtol=0, atol=0)


@pytest.mark.parametrize("ep_size", [8, 16])
def test_native_deterministic_uses_one_versioned_combine(monkeypatch, ep_size):
    reduction_mode = object()
    monkeypatch.setitem(
        sys.modules,
        "deep_ep",
        SimpleNamespace(ReductionMode=SimpleNamespace(DETERMINISTIC=reduction_mode)),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: ep_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group=None: 3)
    calls = []

    class _Buffer:
        def combine(self, x, handle, **kwargs):
            calls.append((x, handle, kwargs))
            return x.clone(), None, "event"

    leaf = torch.ones((2, 16), dtype=torch.bfloat16).contiguous()
    ids = torch.zeros((2, 1), dtype=torch.int32)
    weights = torch.ones((2, 1), dtype=torch.float32)
    combined, event = combine_deterministic_bf16(
        leaf,
        recv_topk_ids=ids,
        recv_topk_weights=weights,
        num_local_experts=1,
        group=object(),
        buffer=_Buffer(),
        handle="dispatch-handle",
        config="combine-config",
        previous_event="previous-event",
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    assert torch.equal(combined, leaf)
    assert event == "event"
    assert len(calls) == 1
    assert calls[0][1] == "dispatch-handle"
    assert calls[0][2]["reduction_mode"] is reduction_mode


def test_native_router_metadata_remains_fp32_with_fixed_order_renorm():
    logits = torch.tensor([[1.0, 3.0, 2.0, -1.0]], dtype=torch.float32)

    weights, ids = native_exact_router_topk(logits, top_k=2, renormalize=True)

    assert weights.dtype is torch.float32
    assert ids.dtype is torch.int32
    assert ids.tolist() == [[1, 2]]
    scores = torch.softmax(logits, dim=1).gather(1, ids.to(torch.int64))
    expected = (
        (scores / (scores[:, 0] + scores[:, 1]).unsqueeze(-1))
        .to(torch.bfloat16)
        .to(torch.float32)
    )
    assert torch.equal(weights, expected)


@dataclass(frozen=True)
class _LoRAInfo:
    has_active_lora: bool
    single_adapter_id: int | None
    seg_indptr: torch.Tensor
    req_to_lora: torch.Tensor
    token_lora_mapping: torch.Tensor
    cg_buffers: object


def test_native_lora_context_is_remapped_by_shared_layer():
    original = _LoRAInfo(
        has_active_lora=True,
        single_adapter_id=7,
        seg_indptr=torch.tensor([0, 1, 3], dtype=torch.int32),
        req_to_lora=torch.tensor([7, 7], dtype=torch.int64),
        token_lora_mapping=torch.tensor([7, 7, 7], dtype=torch.int64),
        cg_buffers=object(),
    )

    adapted = adapt_native_lora_context(
        torch.empty((5, 8), dtype=torch.bfloat16),
        original,
        dispatch_mode="normal",
    )

    assert adapted.seg_indptr.tolist() == [0, 5]
    assert adapted.req_to_lora.tolist() == [7]
    assert adapted.token_lora_mapping.tolist() == [7] * 5
    assert adapted.cg_buffers is None


def test_native_low_latency_lora_context_covers_fixed_packed_rows():
    original = _LoRAInfo(
        has_active_lora=True,
        single_adapter_id=4,
        seg_indptr=torch.tensor([0, 2], dtype=torch.int32),
        req_to_lora=torch.tensor([4], dtype=torch.int64),
        token_lora_mapping=torch.tensor([4, 4], dtype=torch.int64),
        cg_buffers=object(),
    )

    adapted = adapt_native_lora_context(
        torch.empty((12, 8), dtype=torch.bfloat16),
        original,
        dispatch_mode="low_latency",
    )

    assert adapted.seg_indptr.tolist() == [0, 12]
    assert adapted.req_to_lora.tolist() == [4]
    assert adapted.token_lora_mapping.tolist() == [4] * 12
    assert adapted.cg_buffers is None


@dataclass(frozen=True)
class _GraphLoRAInfo:
    has_active_lora: bool
    single_adapter_id: int | None
    seg_indptr: torch.Tensor
    req_to_lora: torch.Tensor
    token_lora_mapping: torch.Tensor
    cg_buffers: dict
    lora_ranks: torch.Tensor
    adapter_enabled: torch.Tensor
    num_experts: int


def test_native_low_latency_lora_capture_records_full_static_receive_program(
    monkeypatch,
):
    import sglang.srt.model_executor.runner_utils.capture_mode as capture_mode

    monkeypatch.setattr(capture_mode, "get_is_capture_mode", lambda: True)
    replay_adapter_slot = torch.tensor([0], dtype=torch.int32)
    original = _GraphLoRAInfo(
        has_active_lora=False,
        single_adapter_id=None,
        seg_indptr=torch.tensor([0, 1], dtype=torch.int32),
        req_to_lora=replay_adapter_slot,
        token_lora_mapping=torch.tensor([0], dtype=torch.int32),
        cg_buffers={},
        lora_ranks=torch.tensor([0, 16], dtype=torch.int64),
        adapter_enabled=torch.zeros(2, dtype=torch.int32),
        num_experts=4,
    )
    hidden = torch.empty((12, 8), dtype=torch.bfloat16)
    topk_ids = torch.zeros((12, 1), dtype=torch.int32)

    first = adapt_native_lora_context(
        hidden,
        original,
        dispatch_mode="low_latency",
        topk_ids=topk_ids,
    )
    update_native_lora_graph_control(adapter_id=1, adapter_rank=16)
    second = adapt_native_lora_context(
        hidden,
        original,
        dispatch_mode="low_latency",
        topk_ids=topk_ids,
    )

    assert first.seg_indptr.tolist() == [0, 12]
    assert first.req_to_lora.item() == 1
    assert first.lora_ranks.tolist() == [0, 16]
    assert first.adapter_enabled.tolist() == [0, 1]
    assert second.cg_buffers is first.cg_buffers
    assert first.cg_buffers["sorted_token_ids_lora"].numel() >= 12 + 4 * 63
    assert first.cg_buffers["token_mask"].numel() == 2 * 12
    update_native_lora_graph_control(adapter_id=None, adapter_rank=None)
    assert first.lora_ranks.tolist() == [0, 0]
    assert first.adapter_enabled.tolist() == [0, 0]


def test_native_lora_context_rejects_mixed_adapter_ownership():
    original = _LoRAInfo(
        has_active_lora=True,
        single_adapter_id=None,
        seg_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        req_to_lora=torch.tensor([1, 2], dtype=torch.int64),
        token_lora_mapping=torch.tensor([1, 2], dtype=torch.int64),
        cg_buffers=None,
    )

    with pytest.raises(RuntimeError, match="one active adapter"):
        adapt_native_lora_context(
            torch.empty((2, 8), dtype=torch.bfloat16),
            original,
            dispatch_mode="normal",
        )


@pytest.mark.parametrize("owner_rank", range(8))
def test_native_deepep_lora_control_follows_every_request_owner(
    monkeypatch, owner_rank
):
    from sglang.srt.lora.lora_manager import LoRAManager

    physical_uids = [tuple() for _ in range(8)]
    physical_uids[owner_rank] = ("adapter",)
    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(_deepep_native_exact=True)
    manager.lora_backend = SimpleNamespace(deepep_native_moe_batch_info="stale")
    manager.max_loras_per_batch = 2
    manager.ep_size = 8
    manager.device = torch.device("cpu")
    manager.memory_pool = SimpleNamespace(get_buffer_id=lambda uid: 1)
    manager.loras = {
        "adapter": SimpleNamespace(
            config=SimpleNamespace(r=16),
            scaling=2.0,
        )
    }
    manager.lora_refs = {"adapter": object()}
    fetched = []
    manager.fetch_new_loras = lambda uids: fetched.append(uids)
    parallel = SimpleNamespace(
        moe_ep_size=8,
        moe_ep_rank=0,
        moe_ep_group=SimpleNamespace(
            world_size=8, all_gather_object=lambda local_uids: physical_uids
        ),
    )
    monkeypatch.setattr("sglang.srt.lora.lora_manager.get_parallel", lambda: parallel)

    manager.prepare_deepep_native_exact_dp_lora_batch(SimpleNamespace(lora_ids=[]))

    info = manager.lora_backend.deepep_native_moe_batch_info
    assert fetched == [{"adapter"}]
    assert info.has_active_lora
    assert info.single_adapter_id == 1
    assert info.expected_tokens is None
    assert info.moe_lora_info.adapter_enabled.tolist() == [0, 1]
    assert info.moe_lora_info.token_lora_mapping.numel() == 0


def test_native_deepep_lora_control_rejects_active_plus_base(monkeypatch):
    from sglang.srt.lora.lora_manager import LoRAManager

    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(_deepep_native_exact=True)
    manager.lora_backend = SimpleNamespace(deepep_native_moe_batch_info=None)
    manager.ep_size = 2
    parallel = SimpleNamespace(
        moe_ep_size=2,
        moe_ep_group=SimpleNamespace(
            world_size=2, all_gather_object=lambda local_uids: [("adapter",), (None,)]
        ),
    )
    monkeypatch.setattr("sglang.srt.lora.lora_manager.get_parallel", lambda: parallel)

    with pytest.raises(RuntimeError, match="active-plus-base"):
        manager.prepare_deepep_native_exact_dp_lora_batch(
            SimpleNamespace(lora_ids=["adapter"])
        )


def test_native_deepep_lora_control_ignores_dp_padding_uids(monkeypatch):
    from sglang.srt.lora.lora_manager import LoRAManager

    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(_deepep_native_exact=True)
    manager.lora_backend = SimpleNamespace(deepep_native_moe_batch_info=None)
    manager.max_loras_per_batch = 2
    manager.ep_size = 2
    manager.device = torch.device("cpu")
    manager.memory_pool = SimpleNamespace(get_buffer_id=lambda uid: 1)
    manager.loras = {
        "adapter": SimpleNamespace(
            config=SimpleNamespace(r=16),
            scaling=2.0,
        )
    }
    manager.lora_refs = {"adapter": object()}
    manager.fetch_new_loras = lambda uids: None

    def gather(local_uids):
        assert local_uids == ("adapter",)
        return [local_uids, tuple()]

    parallel = SimpleNamespace(
        moe_ep_size=2,
        moe_ep_rank=0,
        moe_ep_group=SimpleNamespace(world_size=2, all_gather_object=gather),
    )
    monkeypatch.setattr("sglang.srt.lora.lora_manager.get_parallel", lambda: parallel)

    manager.prepare_deepep_native_exact_dp_lora_batch(
        SimpleNamespace(
            lora_ids=["adapter", None, None],
            _original_batch_size=1,
        )
    )

    assert manager.lora_backend.deepep_native_moe_batch_info.single_adapter_id == 1


def test_native_deepep_dsv4_lora_control_collapses_cp_replicas(monkeypatch):
    from sglang.srt.lora.lora_manager import LoRAManager

    manager = object.__new__(LoRAManager)
    # Composite configs can reconstruct their text config without private
    # ModelConfig markers.  The resolved immutable ServerArgs-derived field on
    # the manager must still select the DSV4 logical owner plane.
    manager.base_hf_config = SimpleNamespace(_deepep_native_exact=True)
    manager.dsv4_flash_exact_mode = True
    manager.lora_backend = SimpleNamespace(deepep_native_moe_batch_info=None)
    manager.max_loras_per_batch = 2
    manager.ep_size = 8
    manager.device = torch.device("cpu")
    manager.memory_pool = SimpleNamespace(get_buffer_id=lambda uid: 1)
    manager.loras = {
        "adapter": SimpleNamespace(
            config=SimpleNamespace(r=16),
            scaling=2.0,
        )
    }
    manager.lora_refs = {"adapter": object()}
    manager.fetch_new_loras = lambda uids: None

    # DP0 owns one active request.  Its non-owning CP ranks and every synthetic
    # DP1 rank carry None-filled synchronization replicas.  The real runtime
    # can expose DP1 as a positive, non-IDLE row, so the empty request-ID plane
    # is the authoritative fabrication witness.
    physical_owners = [
        (("adapter",), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
        ((None,), False, 0),
        ((None,), False, 0),
        ((None,), False, 0),
        ((None,), False, 0),
    ]
    ranks = tuple(range(8))
    ep_group = SimpleNamespace(
        world_size=8,
        rank_in_group=0,
        ranks=ranks,
        all_gather_object=lambda local_owner: physical_owners,
    )
    parallel = SimpleNamespace(
        moe_ep_size=8,
        moe_ep_rank=0,
        moe_ep_group=ep_group,
        tp_group=ep_group,
        attn_dp_size=2,
        attn_dp_rank=0,
        attn_cp_size=4,
        attn_cp_rank=0,
        attn_cp_group=SimpleNamespace(ranks=(0, 1, 2, 3)),
        attn_tp_size=1,
        moe_tp_size=1,
    )
    monkeypatch.setattr("sglang.srt.lora.lora_manager.get_parallel", lambda: parallel)

    manager.prepare_deepep_native_exact_dp_lora_batch(
        SimpleNamespace(
            lora_ids=["adapter"],
            rids=["active-request"],
            _original_batch_size=1,
            # DP1's request-free synchronization batch became one positive,
            # non-IDLE row.  Request ownership, not the row count, excludes it.
            original_global_num_tokens_cpu=[10, 1],
        )
    )

    assert manager.lora_backend.deepep_native_moe_batch_info.single_adapter_id == 1


def test_native_deepep_dsv4_lora_control_rejects_real_base_owner(monkeypatch):
    from sglang.srt.lora.lora_manager import LoRAManager

    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(
        _deepep_native_exact=True,
        _dsv4_flash_exact_mode=True,
    )
    manager.lora_backend = SimpleNamespace(deepep_native_moe_batch_info=None)
    manager.ep_size = 8
    physical_owners = [
        (("adapter",), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
    ]
    ranks = tuple(range(8))
    ep_group = SimpleNamespace(
        world_size=8,
        rank_in_group=0,
        ranks=ranks,
        all_gather_object=lambda local_owner: physical_owners,
    )
    parallel = SimpleNamespace(
        moe_ep_size=8,
        moe_ep_group=ep_group,
        tp_group=ep_group,
        attn_dp_size=2,
        attn_dp_rank=0,
        attn_cp_size=4,
        attn_cp_rank=0,
        attn_cp_group=SimpleNamespace(ranks=(0, 1, 2, 3)),
        attn_tp_size=1,
        moe_tp_size=1,
    )
    monkeypatch.setattr("sglang.srt.lora.lora_manager.get_parallel", lambda: parallel)

    with pytest.raises(RuntimeError, match="active-plus-base"):
        manager.prepare_deepep_native_exact_dp_lora_batch(
            SimpleNamespace(
                lora_ids=["adapter"],
                rids=["active-request"],
                _original_batch_size=1,
                # DP1 has a real base request, rather than idle padding.
                original_global_num_tokens_cpu=[4, 1],
            )
        )


def test_native_deepep_glm52_lora_control_collapses_cp_replicas(monkeypatch):
    from sglang.srt.lora.lora_manager import LoRAManager

    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(
        _deepep_native_exact=True,
        _glm52_exact_mode=True,
    )
    manager.lora_backend = SimpleNamespace(deepep_native_moe_batch_info=None)
    manager.max_loras_per_batch = 2
    manager.ep_size = 4
    manager.device = torch.device("cpu")
    manager.memory_pool = SimpleNamespace(get_buffer_id=lambda uid: 1)
    manager.loras = {
        "adapter": SimpleNamespace(
            config=SimpleNamespace(r=16),
            scaling=2.0,
        )
    }
    manager.lora_refs = {"adapter": object()}
    manager.fetch_new_loras = lambda uids: None
    physical_owners = [
        (("adapter",), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
        ((None,), False, 1),
    ]
    ep_group = SimpleNamespace(
        world_size=4,
        rank_in_group=0,
        all_gather_object=lambda local_owner: physical_owners,
    )
    parallel = SimpleNamespace(
        moe_ep_size=4,
        moe_ep_rank=0,
        moe_ep_group=ep_group,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=4,
        attn_cp_rank=0,
        attn_tp_size=1,
    )
    monkeypatch.setattr("sglang.srt.lora.lora_manager.get_parallel", lambda: parallel)

    manager.prepare_deepep_native_exact_dp_lora_batch(
        SimpleNamespace(
            lora_ids=["adapter"],
            rids=["active-request"],
            _original_batch_size=1,
            original_global_num_tokens_cpu=[4],
        )
    )

    assert manager.lora_backend.deepep_native_moe_batch_info.single_adapter_id == 1


def test_native_deepep_lora_control_rejects_partial_ep_group(monkeypatch):
    from sglang.srt.lora.lora_manager import LoRAManager

    manager = object.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace(_deepep_native_exact=True)
    manager.lora_backend = SimpleNamespace(deepep_native_moe_batch_info=None)
    manager.ep_size = 8
    parallel = SimpleNamespace(
        moe_ep_size=8,
        moe_ep_group=SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr("sglang.srt.lora.lora_manager.get_parallel", lambda: parallel)

    with pytest.raises(RuntimeError, match="full expert-parallel group"):
        manager.prepare_deepep_native_exact_dp_lora_batch(
            SimpleNamespace(lora_ids=["adapter"])
        )


def test_idle_owner_selects_native_deepep_lora_control_before_dispatch():
    from sglang.srt.lora.layers import FusedMoEWithLoRA

    control = SimpleNamespace(expected_tokens=None)
    base_calls = []
    seen = []
    wrapper = SimpleNamespace(
        lora_backend=SimpleNamespace(
            get_batch_info_for_rows=lambda rows: None,
            deepep_native_moe_batch_info=control,
        ),
        moe_runner_config=SimpleNamespace(deepep_native_exact=True),
        base_layer=SimpleNamespace(
            forward=lambda *args, **kwargs: base_calls.append((args, kwargs))
        ),
        _get_lora_info=lambda batch_info: seen.append(batch_info) or "lora-info",
        _forward_with_lora=lambda hidden, topk, lora_info, **kwargs: lora_info,
    )

    result = FusedMoEWithLoRA.forward(
        wrapper,
        torch.empty((0, 8), dtype=torch.bfloat16),
        object(),
    )

    assert result == "lora-info"
    assert seen == [control]
    assert base_calls == []


@pytest.mark.parametrize("native_exact", [False, True], ids=["ordinary", "native"])
def test_moe_runner_gates_native_deepep_lora_adaptation(monkeypatch, native_exact):
    """Only native exact rewrites ownership after DeepEP transport."""

    received_rows = torch.empty((5, 8), dtype=torch.bfloat16)
    runner_input = SimpleNamespace(
        hidden_states=received_rows,
        topk_ids=torch.zeros((5, 2), dtype=torch.int32),
    )
    dispatch_output = DeepEPNormalDispatchOutput(
        hidden_states=torch.empty((5, 8), dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_ids=torch.zeros((5, 2), dtype=torch.int64),
        topk_weights=torch.ones((5, 2), dtype=torch.float32),
        num_recv_tokens_per_expert=[5],
    )
    lora_info = _LoRAInfo(
        has_active_lora=True,
        single_adapter_id=7,
        seg_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        req_to_lora=torch.tensor([7, -1], dtype=torch.int64),
        token_lora_mapping=torch.tensor([7, -1], dtype=torch.int64),
        cg_buffers=object(),
    )
    captured = {}

    def fake_build_lora_hooks(hidden_states, resolved_lora_info, topk_ids, **kwargs):
        captured["hidden_states"] = hidden_states
        captured["lora_info"] = resolved_lora_info
        captured["topk_ids"] = topk_ids
        captured["mul_routed_weight"] = kwargs["mul_routed_weight"]
        return "hooks"

    class FakeRunnerCore:
        runner_backend = MoeRunnerBackend.TRITON

        def run(self, passed_input, _quant_info, _running_state, *, hooks):
            assert passed_input is runner_input
            assert hooks == "hooks"
            return SimpleNamespace(hidden_states=received_rows)

    monkeypatch.setattr(
        PermuteMethodPool,
        "get_pre_permute",
        classmethod(lambda _cls, *_args: lambda *_unused: runner_input),
    )
    monkeypatch.setattr(
        PermuteMethodPool,
        "get_post_permute",
        classmethod(lambda _cls, *_args: lambda output, *_unused: output),
    )
    monkeypatch.setattr(
        "sglang.srt.lora.lora_moe_runners.build_lora_hooks",
        fake_build_lora_hooks,
    )

    runner = object.__new__(MoeRunner)
    runner.fused_func = None
    runner.lora_enabled = True
    runner.runner_core = FakeRunnerCore()
    runner.config = MoeRunnerConfig(
        no_combine=False,
        deepep_native_exact=native_exact,
    )
    runner.down_gemm_overlap_args = None
    runner.meta_overlap_args = None

    runner.run(dispatch_output, object(), lora_info=lora_info)

    assert captured["hidden_states"] is received_rows
    assert captured["topk_ids"] is runner_input.topk_ids
    if native_exact:
        assert captured["lora_info"].seg_indptr.tolist() == [0, 5]
        assert captured["lora_info"].token_lora_mapping.tolist() == [7] * 5
    else:
        assert captured["lora_info"] is lora_info
        assert captured["lora_info"].seg_indptr.tolist() == [0, 1, 2]
        assert captured["lora_info"].token_lora_mapping.tolist() == [7, -1]
    assert captured["mul_routed_weight"] is True


def test_native_receive_contract_accepts_zero_rows():
    validate_native_receive(
        torch.empty((0, 8), dtype=torch.bfloat16).contiguous(),
        torch.empty((0, 4), dtype=torch.int64),
        torch.empty((0, 4), dtype=torch.float32),
        num_local_experts=2,
    )


def test_native_runner_adapter_converts_deepep_ids_to_int32_without_narrowing_weights():
    ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.5, 0.0], [0.75, 0.25]], dtype=torch.float32)

    runner_ids, runner_weights = adapt_native_runner_metadata(ids, weights)

    assert runner_ids.dtype is torch.int32
    assert runner_ids.is_contiguous()
    assert torch.equal(runner_ids.to(torch.int64), ids)
    assert runner_weights.dtype is torch.float32
    assert torch.equal(runner_weights, weights)


def test_native_runner_routes_reduce_in_fp32_then_store_bf16_leaf():
    routes = torch.tensor(
        [[[256.0, 1.0], [1.0, 2.0], [-256.0, 4.0]]],
        dtype=torch.bfloat16,
    )
    ids = torch.tensor([[0, 1, 0]], dtype=torch.int32)
    weights = torch.ones((1, 3), dtype=torch.float32)

    leaf = reduce_native_runner_routes_to_bf16(routes, ids, weights)

    assert leaf.dtype is torch.bfloat16
    assert torch.equal(leaf, torch.tensor([[1.0, 7.0]], dtype=torch.bfloat16))


def test_glm_native_runner_scales_fp32_reduction_before_bf16_wire_leaf():
    routes = torch.tensor(
        [[[1.25, -2.0], [0.5, 3.0]]],
        dtype=torch.bfloat16,
    )
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32)

    leaf = reduce_native_runner_routes_to_bf16(
        routes,
        ids,
        weights,
        routed_scaling_factor=2.5,
    )

    expected = ((routes.float() * weights.unsqueeze(-1)).sum(dim=1) * 2.5).to(
        torch.bfloat16
    )
    assert leaf.dtype is torch.bfloat16
    assert torch.equal(leaf, expected)


def test_dsv4_low_latency_preserves_runner_preweighted_bf16_routes():
    routes = torch.tensor(
        [[[1.25, -2.0]], [[3.0, 4.0]], [[99.0, 99.0]]],
        dtype=torch.bfloat16,
    )
    ids = torch.tensor([[0], [1], [-1]], dtype=torch.int32)

    packed = pack_native_low_latency_preweighted_routes(routes, ids)

    assert packed.dtype is torch.bfloat16
    assert torch.equal(packed[:2], routes[:2, 0])
    assert torch.count_nonzero(packed[2]).item() == 0


def test_native_runner_zero_receive_rows_bypass_kernels():
    hidden = torch.empty((0, 8), dtype=torch.bfloat16)
    ids = torch.empty((0, 4), dtype=torch.int32)
    routes = native_zero_row_runner_routes(hidden, ids)
    leaf = reduce_native_runner_routes_to_bf16(
        routes,
        ids,
        torch.empty((0, 4), dtype=torch.float32),
    )

    assert routes.shape == (0, 4, 8)
    assert leaf.shape == (0, 8)
    assert leaf.dtype is torch.bfloat16


@pytest.mark.parametrize(
    ("leaf_dtype", "ids", "weights", "match"),
    [
        (torch.float32, [[0, -1]], [[1.0, 0.0]], "BF16"),
        (torch.bfloat16, [[-1, -1]], [[1.0, 0.0]], "no local route"),
        (torch.bfloat16, [[2, -1]], [[1.0, 0.0]], "non-local"),
        (torch.bfloat16, [[0, -1]], [[float("nan"), 0.0]], "not finite"),
    ],
)
def test_native_receive_contract_fails_closed(leaf_dtype, ids, weights, match):
    with pytest.raises(DeepEPNativeExactError, match=match):
        validate_native_receive(
            torch.zeros((1, 8), dtype=leaf_dtype).contiguous(),
            torch.tensor(ids, dtype=torch.int64),
            torch.tensor(weights, dtype=torch.float32),
            num_local_experts=2,
        )


def test_deepep_normal_triton_adapters_preserve_real_receive_metadata(monkeypatch):
    try:
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPNormalDispatchOutput,
        )
    except ImportError as error:
        pytest.skip(
            f"ambient sgl_kernel ABI cannot import the dispatcher stack: {error}"
        )
    hidden = torch.zeros((2, 8), dtype=torch.bfloat16).contiguous()
    ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.5, 0.0], [0.75, 0.25]], dtype=torch.float32)
    dispatch = DeepEPNormalDispatchOutput(hidden, None, ids, weights, [2, 1])
    quant = TritonMoeQuantInfo(
        w13_weight=torch.zeros((2, 4, 8), dtype=torch.bfloat16),
        w2_weight=torch.zeros((2, 8, 2), dtype=torch.bfloat16),
    )
    sorted_ids = torch.tensor([0, 1], dtype=torch.int32)
    expert_ids = torch.tensor([0, 1], dtype=torch.int32)
    padded = torch.tensor([2], dtype=torch.int32)

    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe

    monkeypatch.setattr(
        fused_moe,
        "_prepare_fused_moe_run",
        lambda *args, **kwargs: (
            {"tile": "test"},
            None,
            True,
            sorted_ids,
            expert_ids,
            padded,
        ),
    )
    state = {}
    runner_config = MoeRunnerConfig(
        num_experts=4,
        num_local_experts=2,
        glm52_exact_mode=True,
        routed_scaling_factor=1.0,
    )
    runner_input = pre_permute_deepep_normal_to_triton(
        dispatch,
        quant,
        runner_config,
        state,
    )
    local_leaf = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16)
    combine_input = post_permute_triton_to_deepep_normal(
        TritonRunnerOutput(hidden_states=local_leaf),
        quant,
        runner_config,
        state,
    )

    assert runner_input.hidden_states is hidden
    assert runner_input.topk_ids.dtype is torch.int32
    assert torch.equal(runner_input.topk_ids.to(torch.int64), ids)
    assert runner_input.topk_weights is weights
    assert state["down_moe_use_tma"] is True
    assert combine_input.topk_ids.dtype is torch.int32
    assert torch.equal(combine_input.topk_ids.to(torch.int64), ids)
    assert combine_input.topk_weights is weights
    assert combine_input.hidden_states.dtype is torch.bfloat16
    assert torch.equal(combine_input.hidden_states, local_leaf)


def test_deepep_ll_triton_adapter_builds_preweighted_expert_routes(monkeypatch):
    hidden = (
        torch.arange(4 * 3 * 8, dtype=torch.float32).reshape(4, 3, 8).to(torch.bfloat16)
    )
    packed_weights = torch.tensor(
        [
            [0.25, 0.5, 99.0],
            [0.75, 99.0, 99.0],
            [0.125, 0.625, 0.875],
            [99.0, 99.0, 99.0],
        ],
        dtype=torch.float32,
    )
    counts = torch.tensor([2, 1, 3, 0], dtype=torch.int32)
    original_ids = torch.tensor([[0, 6], [2, -1]], dtype=torch.int64)
    original_weights = torch.tensor([[0.25, 0.75], [1.0, 0.0]], dtype=torch.float32)
    dispatch = DeepEPLLExactDispatchOutput(
        hidden,
        None,
        original_ids,
        original_weights,
        counts,
        2,
        packed_weights,
    )
    quant = TritonMoeQuantInfo(
        w13_weight=torch.zeros((4, 4, 8), dtype=torch.bfloat16),
        w2_weight=torch.zeros((4, 8, 2), dtype=torch.bfloat16),
    )
    sorted_ids = torch.tensor([0, 1], dtype=torch.int32)
    expert_ids = torch.tensor([0, 1], dtype=torch.int32)
    padded = torch.tensor([2], dtype=torch.int32)

    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe

    monkeypatch.setattr(
        fused_moe,
        "_prepare_fused_moe_run",
        lambda *args, **kwargs: (
            {"tile": "ll-test"},
            None,
            False,
            sorted_ids,
            expert_ids,
            padded,
        ),
    )
    state = {}
    runner_input = pre_permute_deepep_ll_to_triton(
        dispatch,
        quant,
        MoeRunnerConfig(deepep_native_exact=True),
        state,
    )

    assert runner_input.hidden_states.shape == (12, 8)
    assert runner_input.topk_ids.tolist() == [
        [0],
        [0],
        [-1],
        [1],
        [-1],
        [-1],
        [2],
        [2],
        [2],
        [-1],
        [-1],
        [-1],
    ]
    assert runner_input.topk_weights.flatten().tolist() == [
        0.25,
        0.5,
        0.0,
        0.75,
        0.0,
        0.0,
        0.125,
        0.625,
        0.875,
        0.0,
        0.0,
        0.0,
    ]
    assert state["deepep_ll_deterministic_routes"] is True

    # Model the weighted BF16 route buffer after a nonzero active-LoRA down
    # delta has been added. The post-permute boundary must communicate these
    # exact bytes and replace the combine coefficients with ones.
    route_output = hidden.reshape(12, 1, 8).clone().add_(3)
    combine = post_permute_triton_to_deepep_ll(
        TritonRunnerOutput(route_output),
        quant,
        MoeRunnerConfig(deepep_native_exact=True),
        state,
    )
    assert combine.hidden_states.shape == hidden.shape
    expected = route_output.squeeze(1).reshape_as(hidden).clone()
    expected.reshape(12, 8)[runner_input.topk_ids.squeeze(1) < 0] = 0
    assert torch.equal(combine.hidden_states, expected)
    # Invalid capacity rows are zero even though their route buffer deliberately
    # contains a nonzero weighted base + LoRA value.
    assert (
        torch.count_nonzero(
            combine.hidden_states.reshape(12, 8)[runner_input.topk_ids.squeeze(1) < 0]
        ).item()
        == 0
    )
    assert combine.topk_ids is original_ids
    assert torch.equal(combine.topk_weights, torch.ones_like(original_weights))


def test_deepep_ll_combine_passes_rank_leaf_routed_scaling(monkeypatch):
    import sglang.srt.layers.moe.token_dispatcher.deepep as dispatcher_module

    reduction_mode = object()
    monkeypatch.setitem(
        sys.modules,
        "deep_ep",
        SimpleNamespace(
            LowLatencyReductionMode=SimpleNamespace(DETERMINISTIC=reduction_mode)
        ),
    )
    monkeypatch.setattr(
        dispatcher_module, "_deepep_precompile_tp_barrier", lambda: None
    )
    calls = []

    class _Buffer:
        def low_latency_combine(self, **kwargs):
            calls.append(kwargs)
            return kwargs["x"], None, lambda: None

    impl = object.__new__(dispatcher_module._DeepEPDispatcherImplLowLatency)
    impl.overlap_args = None
    impl.meta_overlap_args = None
    impl.return_recv_hook = True
    impl.deepep_native_exact = True
    impl.handle = "dispatch-handle"
    impl.routed_scaling_factor = 2.5
    impl._get_buffer = lambda: _Buffer()

    routes = torch.ones((2, 3, 8), dtype=torch.bfloat16)
    ids = torch.zeros((1, 2), dtype=torch.int64)
    weights = torch.ones((1, 2), dtype=torch.float32)
    output, _, _ = impl._combine_core(routes, ids, weights)

    assert output is routes
    assert len(calls) == 1
    assert calls[0]["reduction_mode"] is reduction_mode
    assert calls[0]["input_is_preweighted"] is False
    assert calls[0]["routed_scaling_factor"] == 2.5


def test_deepep_ll_draft_runner_uses_stock_combine_abi(monkeypatch):
    import sglang.srt.layers.moe.token_dispatcher.deepep as dispatcher_module

    monkeypatch.setattr(
        dispatcher_module, "_deepep_precompile_tp_barrier", lambda: None
    )
    calls = []

    class _Buffer:
        def low_latency_combine(self, **kwargs):
            calls.append(kwargs)
            return kwargs["x"], None, lambda: None

    impl = object.__new__(dispatcher_module._DeepEPDispatcherImplLowLatency)
    impl.overlap_args = None
    impl.meta_overlap_args = None
    impl.return_recv_hook = True
    impl.handle = "dispatch-handle"
    impl.routed_scaling_factor = 2.5
    impl.deepep_native_exact = False
    impl._get_buffer = lambda: _Buffer()

    routes = torch.ones((2, 3, 8), dtype=torch.bfloat16)
    ids = torch.zeros((1, 2), dtype=torch.int64)
    weights = torch.ones((1, 2), dtype=torch.float32)
    output, _, _ = impl._combine_core(routes, ids, weights)

    assert output is routes
    assert len(calls) == 1
    assert "reduction_mode" not in calls[0]
    assert "input_is_preweighted" not in calls[0]
    assert "routed_scaling_factor" not in calls[0]


def test_stock_low_latency_reduction_is_benchmark_only(monkeypatch):
    import sglang.srt.layers.moe.token_dispatcher.deepep as dispatcher_module

    monkeypatch.delenv("SGLANG_DEEPEP_BENCHMARK_STOCK_LL_REDUCTION", raising=False)
    assert dispatcher_module._benchmark_stock_low_latency_reduction() is False
    monkeypatch.setenv("SGLANG_DEEPEP_BENCHMARK_STOCK_LL_REDUCTION", "1")
    assert dispatcher_module._benchmark_stock_low_latency_reduction() is True
    monkeypatch.setenv("SGLANG_DEEPEP_BENCHMARK_STOCK_LL_REDUCTION", "true")
    assert dispatcher_module._benchmark_stock_low_latency_reduction() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
