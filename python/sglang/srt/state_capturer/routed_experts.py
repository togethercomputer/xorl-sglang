from typing import Any, Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_dp_local_slice_cpu,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
)
from sglang.srt.state_capturer.base import (
    BaseDeviceCache,
    BaseHostCache,
    BaseTopkCapturer,
    TopkCaptureOutput,
)


def _routed_experts_device_cache_rows(
    *,
    chunked_prefill_size: int,
    max_prefill_tokens: int,
    max_running_requests: int,
    dp_size: int,
) -> int:
    """Size the shared DP capture buffer for the largest forward token batch.

    Both prefill limits bound one DP scheduler.  With DP attention and no MoE
    all-to-all backend, routed-expert capture sees the DP-concatenated token
    tensor, so reconstruct the global row bound in both chunked and unchunked
    modes.  Request count is not a prefill token bound: one long prompt can
    contain many more tokens than running requests.
    """
    if chunked_prefill_size > 0:
        prefill_rows = chunked_prefill_size * dp_size
    else:
        prefill_rows = max_prefill_tokens * dp_size
    decode_rows = max_running_requests * dp_size
    return max(prefill_rows, decode_rows)


def routed_experts_capture_enabled() -> bool:
    """Whether the server was launched with routed-expert capture capability.

    Single source of truth for the capability gate: ``RoutedExpertsCapturer.create``
    returns ``None`` exactly when this is False, so request-level validation and
    capturer construction cannot drift apart and start silently returning
    missing or stale rows.
    """
    return (
        get_exec().features.enable_return_routed_experts
        or get_exec().features.enable_return_expert_logits
    )


class RoutedExpertsCaptureOutput(TopkCaptureOutput):
    """Routed expert indices plus optional float32 selected-router weights."""

    def __init__(
        self,
        *,
        out_cache_loc: torch.Tensor,
        topk: torch.Tensor,
        host_cache: BaseHostCache,
        expert_logits: Optional[torch.Tensor],
        expert_logits_host_cache: Optional[BaseHostCache],
    ):
        super().__init__(out_cache_loc, topk, host_cache)
        self.expert_logits = expert_logits
        self.expert_logits_host_cache = expert_logits_host_cache

    def map_device_tensors(self, fn):
        super().map_device_tensors(fn)
        if self.expert_logits is not None:
            self.expert_logits = fn(self.expert_logits)

    def finalize(self):
        super().finalize()
        if self.expert_logits is not None:
            self.expert_logits_host_cache.buffer[self.out_cache_loc] = (
                self.expert_logits
            )


class RoutedExpertsCapturer(BaseTopkCapturer):
    """Capturer for routed experts with host buffer.

    Routed experts share a global device buffer across DP ranks (indexed by
    dp_rank), so `_get_local_slice` overrides the default to apply DP-rank-aware
    slicing. The device cache also holds extra columns for any fused shared
    experts; the host cache and user-facing return drop them via the
    [:topk_size] truncation.
    """

    @staticmethod
    def create(
        *,
        model: torch.nn.Module,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["RoutedExpertsCapturer"]:
        if not routed_experts_capture_enabled():
            return None
        if not get_exec().moe.disable_shared_experts_fusion and hasattr(
            model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0
        return RoutedExpertsCapturer(
            model_config,
            num_tokens=num_tokens,
            max_running_requests=max_running_requests,
            num_fused_shared_experts=num_fused_shared_experts,
            device=device,
        )

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        topk_size = model_config.hf_text_config.num_experts_per_tok
        num_layers = model_config.hf_text_config.num_hidden_layers

        # Scale by dp_size so the buffer covers the full DP-concatenated batch.
        # _get_local_slice indexes into [attention_dp_rank * cuda_graph_batch, ...)
        # and otherwise overflows on dp_rank > 0 when max_running_requests >
        # chunked_prefill_size.
        # FIXME: spec decoding's num_verify_tokens is still not accounted for.
        max_batch_size = _routed_experts_device_cache_rows(
            chunked_prefill_size=get_schedule().chunked_prefill_size,
            max_prefill_tokens=get_schedule().max_prefill_tokens,
            max_running_requests=max_running_requests,
            dp_size=get_parallel().dp_size,
        )

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=topk_size,
            device=device,
            name="routed_experts",
            device_topk_size=topk_size + num_fused_shared_experts,
        )
        self.capture_topk_weights = get_exec().features.enable_return_expert_logits
        self.expert_logits_host_cache = None
        self.expert_logits_device_cache = None
        if self.capture_topk_weights:
            self.expert_logits_host_cache = BaseHostCache(
                num_tokens,
                num_layers,
                topk_size,
                name="expert_logits",
                dtype=torch.float32,
            )
            self.expert_logits_device_cache = BaseDeviceCache(
                max_batch_size,
                num_layers,
                topk_size + num_fused_shared_experts,
                device,
                name="expert_logits",
                dtype=torch.float32,
            )

        # DeepEP a2a path: each attn-TP rank only sees its scattered slice of
        # topk_ids. All-gather across attn-TP at capture time so device_cache
        # holds the full batch and the existing _get_local_slice / D2H sync
        # paths work unchanged. Pre-allocate the gather target.
        if get_moe_a2a_backend().is_deepep():
            attn_tp_size = (
                get_parallel().attn_tp_size if is_dp_attention_enabled() else 1
            )
            self.gather_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0] * attn_tp_size,
                    self.device_cache.buffer.shape[2],
                ),
                dtype=torch.int32,
                device=device,
            )
            self.expert_logits_gather_buffer = (
                torch.empty_like(self.gather_buffer, dtype=torch.float32)
                if self.capture_topk_weights
                else None
            )

    def capture(
        self,
        layer_id: int,
        topk_indices: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        if get_moe_a2a_backend().is_deepep():
            local_topk = topk_indices
            topk_indices = self.gather_buffer[
                : local_topk.size(0) * get_parallel().attn_tp_size
            ]
            attn_tp_all_gather_into_tensor(topk_indices, local_topk)
            if self.capture_topk_weights:
                local_weights = topk_weights
                topk_weights = self.expert_logits_gather_buffer[
                    : local_weights.size(0) * get_parallel().attn_tp_size
                ]
                attn_tp_all_gather_into_tensor(topk_weights, local_weights)
        super().capture(layer_id, topk_indices)
        if self.capture_topk_weights:
            if topk_weights is None:
                raise RuntimeError("expert-logit capture requires topk_weights")
            self.expert_logits_device_cache.capture(
                layer_id, topk_weights.to(torch.float32)
            )

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        # Under DeepEP, capture() already attn_tp_all_gathered into the head of
        # the per-rank buffer, so the local DP rank's data lives at [0:N_local]
        # rather than at the global [start_pos:end_pos] offset.
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            # GPU->CPU sync would break overlap; operate on CPU directly.
            local_start_pos, local_num_tokens = get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )
            local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos, local_end_pos = 0, forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]

    def _get_local_expert_logits_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> Optional[torch.Tensor]:
        if not self.capture_topk_weights:
            return None
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            local_start_pos, local_num_tokens = get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )
            local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos, local_end_pos = 0, forward_batch.out_cache_loc.shape[0]
        return self.expert_logits_device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]

    def get_expert_logits(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool,
        start_len: int = 0,
    ) -> Optional[torch.Tensor]:
        if not self.capture_topk_weights:
            return None
        if start_len < 0:
            raise ValueError(f"{start_len=} must be non-negative")
        start_len = min(start_len, seqlen - 1)
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][start_len : seqlen - 1]
            .cpu()
            .clone()
        )
        return self.expert_logits_host_cache.buffer[cache_pool_idx]

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[RoutedExpertsCaptureOutput]:
        indices = self._get_local_slice(forward_batch, can_run_graph, cuda_graph_batch)
        expert_logits = self._get_local_expert_logits_slice(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        if no_copy_to_cpu:
            return RoutedExpertsCaptureOutput(
                out_cache_loc=forward_batch.out_cache_loc,
                topk=indices,
                host_cache=self.host_cache,
                expert_logits=expert_logits,
                expert_logits_host_cache=self.expert_logits_host_cache,
            )
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = indices.cpu()
        if expert_logits is not None:
            self.expert_logits_host_cache.buffer[out_cache_loc_cpu] = (
                expert_logits.cpu()
            )
        return None


def get_global_experts_capturer() -> Optional[RoutedExpertsCapturer]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().experts_capturer


def set_global_experts_capturer(capturer: Optional[RoutedExpertsCapturer]):
    from sglang.srt.runtime_context import get_resources

    get_resources().experts_capturer = capturer


def extract_routed_experts_from_meta_info(data):
    # To solve the performance issue, we return the experts_ids in base64
    # We left this function for user to change it back to normal int32
    # See detokenizer_manager::_extract_routed_experts
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32
    )
    return routed_experts


def extract_expert_logits_from_meta_info(data):
    expert_logits_base64 = data["meta_info"].get("expert_logits", None)
    return np.frombuffer(
        pybase64.b64decode(expert_logits_base64.encode("utf-8")), dtype=np.float32
    )


def extract_expert_ids_from_meta_info(data, field: str):
    """Decode one partitioned expert-ID payload from a ``/generate`` response.

    ``field`` is ``"input_expert_ids"`` or ``"output_expert_ids"``. Returns a
    flat int32 array; reshape with ``meta_info["expert_ids_schema"]`` as
    ``(<field>_num_rows, num_layers, top_k)``.
    """
    payload = data["meta_info"].get(field, None)
    if payload is None:
        return None
    return np.frombuffer(pybase64.b64decode(payload.encode("utf-8")), dtype=np.int32)


def disable_routed_experts_capture_for_draft(model: Any) -> None:
    """Opt every draft MoE ``TopK`` out of routed-experts (R3) capture.

    Capture is target-only; a draft ``TopK`` must never write the target's
    process-global buffer. ``HashTopK`` has no ``topk_config`` and never
    captures, so it is left untouched.
    """
    # Lazy import: ``layers.moe.topk`` imports ``get_global_experts_capturer``
    # from this module, so a top-level import here would be circular.
    from sglang.srt.layers.moe.topk import TopK

    for module in model.modules():
        if isinstance(module, TopK):
            module.topk_config.allow_routed_experts_capture = False
