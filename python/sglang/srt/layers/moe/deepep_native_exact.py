"""Serving-side native DeepEP BF16 transport and canonical FP64 fold.

The implementation is intentionally independent of XoRL's trainer module.
Both sides agree only on the versioned numerical contract: the real top-k
dispatch handle transports BF16 local leaves through one deterministic
hierarchical combine. Rank-serial folds live only in tests and benchmarks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

DEEPEP_DETERMINISTIC_PROTOCOL = "deepep_deterministic_hierarchical_bf16_v2"
DEEPEP_LOW_LATENCY_DETERMINISTIC_PROTOCOL = DEEPEP_DETERMINISTIC_PROTOCOL
logger = logging.getLogger(__name__)
_engagement_logged = False
_low_latency_engagement_logged = False
_native_lora_graph_buffers: dict[tuple, dict] = {}


class DeepEPNativeExactError(RuntimeError):
    pass


def log_low_latency_deterministic_engagement(*, group, hidden_size: int) -> None:
    """Publish the decode protocol only after its real dispatch API engages."""

    global _low_latency_engagement_logged
    if _low_latency_engagement_logged:
        return
    ep_size = dist.get_world_size(group)
    if ep_size < 2:
        raise DeepEPNativeExactError(
            f"DeepEP low-latency deterministic combine requires EP>=2, got EP{ep_size}"
        )
    logger.info(
        "Native DeepEP exact low-latency ENGAGED: protocol=%s ep_size=%d "
        "dispatch_wire_dtype=bf16 route_metadata_dtype=fp32 "
        "expert_route_store_dtype=bf16 "
        "fold=receiver_fp64_tree8_bf16_node_leaf_fp64_node_fold "
        "wire_width=%d combine_calls=1",
        DEEPEP_LOW_LATENCY_DETERMINISTIC_PROTOCOL,
        ep_size,
        int(hidden_size),
    )
    _low_latency_engagement_logged = True


@dataclass(frozen=True)
class NativeDeepEPGeometry:
    ep_size: int
    ep_rank: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise DeepEPNativeExactError(
                f"invalid native DeepEP geometry: rank={self.ep_rank}, size={self.ep_size}"
            )
        if self.hidden_size <= 0:
            raise DeepEPNativeExactError(
                "native DeepEP requires a positive hidden size"
            )

    @property
    def wire_width(self) -> int:
        return self.hidden_size


def native_exact_router_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the versioned FP32 routing metadata for native DeepEP.

    Router logits are produced by the batch-invariant GEMM in the thin model
    adapter.  This shared boundary then owns FP32 softmax/top-k and the same
    fixed left-to-right FP32 denominator used by the trainer.  These tensors
    are DeepEP metadata; expert values and every combine payload remain BF16.
    """

    if router_logits.ndim != 2 or router_logits.dtype is not torch.float32:
        raise DeepEPNativeExactError(
            "native DeepEP router logits must be FP32 [tokens, experts] metadata"
        )
    if top_k <= 0 or top_k > router_logits.shape[1]:
        raise DeepEPNativeExactError(
            "native DeepEP top-k is outside the expert geometry"
        )

    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (  # noqa: PLC0415
        bi_router_topk_weights,
    )

    scores = torch.softmax(router_logits, dim=1, dtype=torch.float32)
    weights, expert_ids = torch.topk(scores, top_k, dim=-1)
    weights = bi_router_topk_weights(weights, renormalize, torch.bfloat16)
    # DeepEP normal mode requires FP32 top-k metadata.  The values themselves
    # have already crossed the BF16 boundary; widening here adds no precision.
    return (
        weights.to(torch.float32).contiguous(),
        expert_ids.to(torch.int32).contiguous(),
    )


def adapt_native_lora_context(
    hidden_states: torch.Tensor,
    lora_info,
    *,
    dispatch_mode: str,
    topk_ids: torch.Tensor | None = None,
):
    """Remap physical-batch LoRA ownership after native DeepEP dispatch."""

    if lora_info is None:
        return None
    if dispatch_mode not in ("normal", "low_latency"):
        return lora_info
    from sglang.srt.model_executor.runner_utils.capture_mode import (  # noqa: PLC0415
        get_is_capture_mode,
    )

    capture_mode = get_is_capture_mode()
    if not lora_info.has_active_lora and not capture_mode:
        return lora_info

    adapter_id = lora_info.single_adapter_id
    if adapter_id is None and not capture_mode:
        raise RuntimeError(
            "Native DeepEP MoE-LoRA currently requires one active adapter per "
            "physical batch; mixed adapter ownership metadata is not transported yet"
        )
    rows = hidden_states.shape[0]
    if capture_mode:
        if topk_ids is None or topk_ids.ndim != 2 or topk_ids.shape[0] != rows:
            raise RuntimeError(
                "Native DeepEP CUDA-graph LoRA capture requires fixed receive-row "
                "top-k metadata"
            )
        if lora_info.cg_buffers is None:
            raise RuntimeError(
                "Native DeepEP CUDA-graph LoRA capture requires graph-lifetime "
                "MoE workspaces"
            )
        if lora_info.req_to_lora.numel() == 0:
            raise RuntimeError(
                "Native DeepEP CUDA-graph LoRA capture requires one replay-updated "
                "adapter slot"
            )

        # Capture intentionally uses an inactive adapter, but must record the
        # same expert-major row program that active replay executes. Ordinary
        # MoE graph workspaces cover the physical batch, not DeepEP's fixed
        # receive buffer. Keep one stable workspace per receive geometry so
        # captured pointers stay live across every replay.
        max_loras = int(lora_info.lora_ranks.numel())
        num_experts = int(lora_info.num_experts)
        top_k = int(topk_ids.shape[1])
        block_size_m = 64
        max_num_tokens_padded = rows * top_k + num_experts * (block_size_m - 1)
        max_num_tokens_padded = (
            (max_num_tokens_padded + block_size_m - 1) // block_size_m
        ) * block_size_m
        max_num_m_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m
        key = (
            id(lora_info.cg_buffers),
            hidden_states.device.type,
            hidden_states.device.index,
            rows,
            top_k,
            num_experts,
            max_loras,
        )
        graph_buffers = _native_lora_graph_buffers.get(key)
        if graph_buffers is None:
            device = hidden_states.device
            graph_buffers = dict(lora_info.cg_buffers)
            graph_buffers.update(
                sorted_token_ids_lora=torch.empty(
                    max_loras * max_num_tokens_padded,
                    dtype=torch.int32,
                    device=device,
                ),
                expert_ids_lora=torch.empty(
                    max_loras * max_num_m_blocks,
                    dtype=torch.int32,
                    device=device,
                ),
                num_tokens_post_padded_lora=torch.empty(
                    max_loras,
                    dtype=torch.int32,
                    device=device,
                ),
                lora_ids=torch.arange(max_loras, dtype=torch.int32, device=device),
                cumsum_buffer=torch.zeros(
                    max_loras * (num_experts + 1),
                    dtype=torch.int32,
                    device=device,
                ),
                token_mask=torch.empty(
                    max_loras * rows * top_k,
                    dtype=torch.int32,
                    device=device,
                ),
                native_seg_indptr=torch.tensor(
                    [0, rows], dtype=lora_info.seg_indptr.dtype, device=device
                ),
                native_token_lora_mapping=torch.empty(
                    rows,
                    dtype=lora_info.token_lora_mapping.dtype,
                    device=device,
                ),
                native_req_to_lora=torch.zeros(
                    1,
                    dtype=lora_info.req_to_lora.dtype,
                    device=device,
                ),
                native_lora_ranks=torch.zeros_like(lora_info.lora_ranks),
                native_adapter_enabled=torch.zeros_like(lora_info.adapter_enabled),
            )
            _native_lora_graph_buffers[key] = graph_buffers

        return replace(
            lora_info,
            seg_indptr=graph_buffers["native_seg_indptr"],
            # Native DeepEP broadcasts one adapter identity across the EP
            # group. These controls are independent of the physical-DP graph
            # inputs so every expert rank executes the same active adapter.
            req_to_lora=graph_buffers["native_req_to_lora"],
            lora_ranks=graph_buffers["native_lora_ranks"],
            adapter_enabled=graph_buffers["native_adapter_enabled"],
            token_lora_mapping=graph_buffers["native_token_lora_mapping"],
            cg_buffers=graph_buffers,
        )

    return replace(
        lora_info,
        seg_indptr=torch.tensor(
            [0, rows],
            dtype=lora_info.seg_indptr.dtype,
            device=hidden_states.device,
        ),
        req_to_lora=torch.tensor(
            [adapter_id],
            dtype=lora_info.req_to_lora.dtype,
            device=hidden_states.device,
        ),
        token_lora_mapping=torch.full(
            (rows,),
            adapter_id,
            dtype=lora_info.token_lora_mapping.dtype,
            device=hidden_states.device,
        ),
        cg_buffers=None,
    )


def update_native_lora_graph_control(
    *, adapter_id: int | None, adapter_rank: int | None
) -> None:
    """Update graph-static native-DeepEP LoRA control on every EP rank."""

    if (adapter_id is None) != (adapter_rank is None):
        raise RuntimeError(
            "Native DeepEP CUDA-graph LoRA control requires adapter id and rank together"
        )
    for graph_buffers in _native_lora_graph_buffers.values():
        req_to_lora = graph_buffers["native_req_to_lora"]
        lora_ranks = graph_buffers["native_lora_ranks"]
        adapter_enabled = graph_buffers["native_adapter_enabled"]
        lora_ranks.zero_()
        adapter_enabled.zero_()
        req_to_lora.zero_()
        if adapter_id is None:
            continue
        if not 0 <= adapter_id < lora_ranks.numel():
            raise RuntimeError(
                "Native DeepEP CUDA-graph adapter slot is outside the captured bank: "
                f"slot={adapter_id}, slots={lora_ranks.numel()}"
            )
        if adapter_rank <= 0:
            raise RuntimeError(
                f"Native DeepEP CUDA-graph active adapter rank must be positive, got {adapter_rank}"
            )
        req_to_lora.fill_(adapter_id)
        lora_ranks[adapter_id] = adapter_rank
        adapter_enabled[adapter_id] = 1


def validate_native_receive(
    local_leaf: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
) -> None:
    if local_leaf.ndim != 2 or local_leaf.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(
            "native DeepEP communication requires a BF16 [recv_rows, hidden] leaf"
        )
    if not local_leaf.is_contiguous():
        raise DeepEPNativeExactError("native DeepEP rank leaf must be contiguous")
    if recv_topk_ids.ndim != 2 or recv_topk_weights.shape != recv_topk_ids.shape:
        raise DeepEPNativeExactError(
            "native DeepEP ids and weights must share [recv_rows, topk] shape"
        )
    if recv_topk_ids.shape[0] != local_leaf.shape[0]:
        raise DeepEPNativeExactError(
            "native DeepEP metadata does not cover every receive row"
        )
    if recv_topk_ids.dtype not in (torch.int32, torch.int64):
        raise DeepEPNativeExactError(
            "native DeepEP receive expert ids must be integral"
        )
    if recv_topk_weights.dtype is not torch.float32:
        raise DeepEPNativeExactError(
            "native DeepEP receive routing weights must be FP32 metadata"
        )
    if num_local_experts <= 0:
        raise DeepEPNativeExactError(
            "native DeepEP requires a positive local expert count"
        )

    valid = recv_topk_ids >= 0
    if recv_topk_ids.numel() and bool(torch.any(recv_topk_ids < -1)):
        raise DeepEPNativeExactError(
            "native DeepEP receive ids contain a marker below -1"
        )
    if bool(torch.any(recv_topk_ids[valid] >= num_local_experts)):
        raise DeepEPNativeExactError("native DeepEP delivered a non-local expert id")
    if recv_topk_ids.shape[0] and bool(torch.any(~valid.any(dim=1))):
        raise DeepEPNativeExactError(
            "native DeepEP delivered a receive row with no local route"
        )
    if recv_topk_weights.numel() and not bool(torch.isfinite(recv_topk_weights).all()):
        raise DeepEPNativeExactError(
            "native DeepEP receive routing weights are not finite"
        )


def adapt_native_runner_metadata(
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt DeepEP receive metadata to the serving-runner ABI.

    DeepEP normal mode exposes expert ids as int64, while the CUDA Triton/JIT
    activation path requires int32 expert ids.  The conversion belongs to the
    shared native program, not to a model adapter.  Routing weights are
    required to remain FP32 and are never silently narrowed.
    """

    if recv_topk_ids.ndim != 2 or recv_topk_weights.shape != recv_topk_ids.shape:
        raise DeepEPNativeExactError(
            "native DeepEP runner metadata must share [recv_rows, topk] shape"
        )
    if recv_topk_ids.dtype not in (torch.int32, torch.int64):
        raise DeepEPNativeExactError(
            f"native DeepEP runner expert ids must be integral, got {recv_topk_ids.dtype}"
        )
    if recv_topk_weights.dtype is not torch.float32:
        raise DeepEPNativeExactError(
            f"native DeepEP runner weights must remain FP32, got {recv_topk_weights.dtype}"
        )
    return (
        recv_topk_ids.to(torch.int32).contiguous(),
        recv_topk_weights.contiguous(),
    )


def canonicalize_native_routing_metadata(
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Preserve routing coefficients in DeepEP's required FP32 metadata ABI.

    Routing coefficients are kernel metadata, not expert-value wire payloads.
    Rounding an FP32 coefficient through BF16 here changes the fused
    ``no_combine=False`` rank leaf before its declared BF16 storage boundary.
    Expert outputs and rank leaves remain BF16.
    """

    if routing_weights.dtype not in (torch.bfloat16, torch.float32):
        raise DeepEPNativeExactError(
            "native DeepEP routing coefficients must be BF16 or FP32, "
            f"got {routing_weights.dtype}"
        )
    if routing_weights.requires_grad:
        raise DeepEPNativeExactError(
            "native DeepEP v1 requires frozen routing coefficients"
        )
    return routing_weights.to(torch.float32).contiguous()


def reduce_native_runner_routes_to_bf16(
    route_output: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Reduce rank-local expert routes in FP32, then store one BF16 leaf.

    This compatibility helper is for a runner that explicitly returns one
    unweighted BF16 row per receive top-k slot. The selected native exact
    program instead transports the fused ``no_combine=False`` local leaf.
    """

    if route_output.ndim != 3:
        raise DeepEPNativeExactError(
            "native DeepEP no-combine runner output must be [recv_rows, topk, hidden]"
        )
    if route_output.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(
            f"native DeepEP runner routes must be BF16, got {route_output.dtype}"
        )
    if (
        recv_topk_ids.shape != route_output.shape[:2]
        or recv_topk_weights.shape != recv_topk_ids.shape
    ):
        raise DeepEPNativeExactError(
            "native DeepEP runner routes and receive metadata have different row/top-k geometry"
        )
    if (
        recv_topk_ids.dtype is not torch.int32
        or recv_topk_weights.dtype is not torch.float32
    ):
        raise DeepEPNativeExactError(
            "native DeepEP runner reduction requires int32 ids and FP32 routing weights"
        )
    routed_scaling_factor = float(routed_scaling_factor)
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise DeepEPNativeExactError(
            "native DeepEP routed scaling factor must be finite and positive"
        )
    valid = recv_topk_ids >= 0
    weighted_fp32 = torch.where(
        valid.unsqueeze(-1),
        route_output.to(torch.float32) * recv_topk_weights.unsqueeze(-1),
        torch.zeros((), dtype=torch.float32, device=route_output.device),
    )
    return (
        weighted_fp32.sum(dim=1)
        .mul(routed_scaling_factor)
        .to(torch.bfloat16)
        .contiguous()
    )


def pack_native_low_latency_bf16_routes(
    route_output: torch.Tensor,
    runner_topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Pack BF16 expert rows and zero invalid low-latency capacity rows.

    The helper deliberately performs no arithmetic. Exact serving callers
    supply the already weighted BF16 route buffer produced by the same fused
    epilogue and active-LoRA hook program as normal mode, then pair these bytes
    with unit combine weights. This keeps communication at BF16 while DeepEP
    owns only the FP32 owner-rank sum, model scale, and hierarchical fold.
    """

    if route_output.ndim != 3 or route_output.shape[1] != 1:
        raise DeepEPNativeExactError(
            "native low-latency BF16 routes must be [packed_rows, 1, hidden]"
        )
    if route_output.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(
            f"native low-latency routes must be BF16, got {route_output.dtype}"
        )
    if runner_topk_ids.shape != route_output.shape[:2]:
        raise DeepEPNativeExactError(
            "native low-latency route ids do not cover every packed row"
        )
    if runner_topk_ids.dtype is not torch.int32:
        raise DeepEPNativeExactError("native low-latency expert ids must be int32")
    return (
        torch.where(
            (runner_topk_ids >= 0).unsqueeze(-1),
            route_output,
            torch.zeros((), dtype=torch.bfloat16, device=route_output.device),
        )
        .squeeze(1)
        .contiguous()
    )


def pack_native_low_latency_preweighted_routes(
    route_output: torch.Tensor,
    runner_topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Compatibility alias for already-weighted Marlin route bytes."""

    return pack_native_low_latency_bf16_routes(route_output, runner_topk_ids)


def native_zero_row_runner_routes(
    recv_hidden: torch.Tensor,
    recv_topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Construct the no-combine runner result for a valid empty receive batch."""

    if (
        recv_hidden.ndim != 2
        or recv_hidden.shape[0] != 0
        or recv_hidden.dtype is not torch.bfloat16
    ):
        raise DeepEPNativeExactError(
            "native DeepEP zero-row bypass requires empty BF16 receive rows"
        )
    if recv_topk_ids.ndim != 2 or recv_topk_ids.shape[0] != 0:
        raise DeepEPNativeExactError(
            "native DeepEP zero-row metadata must be empty [0, topk]"
        )
    return recv_hidden.new_empty((0, recv_topk_ids.shape[1], recv_hidden.shape[1]))


def combine_deterministic_bf16(
    local_leaf: torch.Tensor,
    *,
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    num_local_experts: int,
    group,
    buffer,
    handle,
    config,
    previous_event,
    async_finish: bool,
    allocate_on_comm_stream: bool,
):
    """Run one normal-mode DeepEP deterministic hierarchical combine."""

    geometry = NativeDeepEPGeometry(
        ep_size=dist.get_world_size(group),
        ep_rank=dist.get_rank(group),
        hidden_size=local_leaf.shape[1],
    )
    validate_native_receive(
        local_leaf,
        recv_topk_ids,
        recv_topk_weights,
        num_local_experts=num_local_experts,
    )
    if geometry.ep_size < 2:
        raise DeepEPNativeExactError(
            f"DeepEP deterministic combine requires EP>=2, got EP{geometry.ep_size}"
        )
    try:
        from deep_ep import ReductionMode  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise DeepEPNativeExactError(
            "installed DeepEP lacks ReductionMode.DETERMINISTIC"
        ) from exc
    reduction_mode = getattr(
        ReductionMode,
        "DETERMINISTIC",
        None,
    )
    if reduction_mode is None:
        raise DeepEPNativeExactError(
            "installed DeepEP lacks ReductionMode.DETERMINISTIC"
        )

    combined, combined_weights, event = buffer.combine(
        local_leaf,
        handle,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream,
        config=config,
        reduction_mode=reduction_mode,
    )
    if combined_weights is not None:
        raise DeepEPNativeExactError(
            "DeepEP deterministic value combine unexpectedly returned routing metadata"
        )
    if combined.dtype is not torch.bfloat16:
        raise DeepEPNativeExactError(
            f"DeepEP deterministic combine widened BF16 to {combined.dtype}"
        )

    global _engagement_logged
    if not _engagement_logged:
        logger.info(
            "Native DeepEP exact combine ENGAGED: protocol=%s ep_size=%d "
            "wire_dtype=bf16 "
            "fold=receiver_fp64_tree8_bf16_node_leaf_fp64_node_fold "
            "wire_width=%d combine_calls=1",
            DEEPEP_DETERMINISTIC_PROTOCOL,
            geometry.ep_size,
            geometry.wire_width,
        )
        _engagement_logged = True
    return combined, event


__all__ = [
    "canonicalize_native_routing_metadata",
    "combine_deterministic_bf16",
    "DEEPEP_DETERMINISTIC_PROTOCOL",
    "DEEPEP_LOW_LATENCY_DETERMINISTIC_PROTOCOL",
    "DeepEPNativeExactError",
    "NativeDeepEPGeometry",
    "adapt_native_lora_context",
    "adapt_native_runner_metadata",
    "native_zero_row_runner_routes",
    "native_exact_router_topk",
    "log_low_latency_deterministic_engagement",
    "pack_native_low_latency_bf16_routes",
    "pack_native_low_latency_preweighted_routes",
    "reduce_native_runner_routes_to_bf16",
    "update_native_lora_graph_control",
    "validate_native_receive",
]
