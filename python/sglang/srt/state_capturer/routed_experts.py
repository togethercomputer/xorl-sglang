import logging
import os
from typing import Any, Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
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

logger = logging.getLogger(__name__)

_PP_ROUTED_EXPERT_IDS = "__sglang_pp_routed_expert_ids__"
_PP_ROUTED_EXPERT_WEIGHTS = "__sglang_pp_routed_expert_weights__"
_PP_ROUTED_EXPERT_OUT_CACHE_LOC = "__sglang_pp_routed_expert_out_cache_loc__"
_PP_ROUTED_EXPERT_REQ_POOL_INDICES = "__sglang_pp_routed_expert_req_pool_indices__"


def add_routed_experts_to_pp_output(
    tensor_dict: dict[str, torch.Tensor],
    capture_output: Optional["RoutedExpertsCaptureOutput"],
) -> None:
    """Attach the terminal PP stage's complete route capture to its reply."""
    if capture_output is None:
        return
    tensor_dict[_PP_ROUTED_EXPERT_IDS] = capture_output.topk
    tensor_dict[_PP_ROUTED_EXPERT_OUT_CACHE_LOC] = capture_output.out_cache_loc
    tensor_dict[_PP_ROUTED_EXPERT_REQ_POOL_INDICES] = capture_output.req_pool_indices
    if capture_output.expert_logits is not None:
        tensor_dict[_PP_ROUTED_EXPERT_WEIGHTS] = capture_output.expert_logits


def publish_routed_experts_from_pp_output(
    tensor_dict: dict[str, torch.Tensor],
) -> None:
    """Publish a completed PP capture in the response-owning stage's cache."""
    if _PP_ROUTED_EXPERT_IDS not in tensor_dict:
        orphaned_keys = {
            _PP_ROUTED_EXPERT_WEIGHTS,
            _PP_ROUTED_EXPERT_OUT_CACHE_LOC,
            _PP_ROUTED_EXPERT_REQ_POOL_INDICES,
        }.intersection(tensor_dict)
        if orphaned_keys:
            raise RuntimeError(
                "DeepEP PP output contains routed metadata without expert ids: "
                f"keys={sorted(orphaned_keys)}"
            )
        return
    if _PP_ROUTED_EXPERT_OUT_CACHE_LOC not in tensor_dict:
        raise RuntimeError(
            "DeepEP PP output contains routed experts without owner cache locations"
        )
    if _PP_ROUTED_EXPERT_REQ_POOL_INDICES not in tensor_dict:
        raise RuntimeError(
            "DeepEP PP output contains routed experts without owner request-pool "
            "indices"
        )
    capturer = get_global_experts_capturer()
    if capturer is None:
        raise RuntimeError(
            "DeepEP PP output contains routed experts but capture is not enabled"
        )
    capturer.publish_pp_capture(
        out_cache_loc=tensor_dict[_PP_ROUTED_EXPERT_OUT_CACHE_LOC],
        req_pool_indices=tensor_dict[_PP_ROUTED_EXPERT_REQ_POOL_INDICES],
        indices=tensor_dict[_PP_ROUTED_EXPERT_IDS],
        expert_logits=tensor_dict.get(_PP_ROUTED_EXPERT_WEIGHTS),
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


class RoutedExpertsCaptureOutput(TopkCaptureOutput):
    """Routed expert indices plus optional float32 selected-router weights."""

    def __init__(
        self,
        *,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        topk: torch.Tensor,
        host_cache: BaseHostCache,
        expert_logits: Optional[torch.Tensor],
        expert_logits_host_cache: Optional[BaseHostCache],
    ):
        super().__init__(out_cache_loc, topk, host_cache)
        self.req_pool_indices = req_pool_indices
        self.expert_logits = expert_logits
        self.expert_logits_host_cache = expert_logits_host_cache

    def map_device_tensors(self, fn):
        super().map_device_tensors(fn)
        self.req_pool_indices = fn(self.req_pool_indices)
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
        if not (
            get_exec().features.enable_return_routed_experts
            or get_exec().features.enable_return_expert_logits
        ):
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
        from sglang.srt.distributed.utils import get_pp_indices

        self.pp_start_layer, self.pp_end_layer = get_pp_indices(
            num_layers,
            get_parallel().pp_rank,
            get_parallel().pp_size,
        )
        self.first_routed_layer = int(
            getattr(model_config.hf_text_config, "first_k_dense_replace", 0)
        )

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

        # DeepEP a2a path: prefill CP and attention TP can each leave a rank
        # with only a sequence-local slice of top-k ids.  Pre-allocate for the
        # larger group; capture() selects the group used by the current phase.
        if get_moe_a2a_backend().is_deepep():
            capture_group_size = (
                max(get_parallel().attn_tp_size, get_parallel().attn_cp_size)
                if is_dp_attention_enabled()
                else 1
            )
            self.gather_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0] * capture_group_size,
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
            # CP ranks can own unequal numbers of real interleaved rows.  Use
            # a separate fixed-width staging buffer so the
            # collective sees the same shape on every rank while padding is
            # deterministic zero, never a live DP/MLP-sync router row.  It is
            # consumed by the device-cache copy before the next routed layer.
            self.capture_local_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0],
                    self.device_cache.buffer.shape[2],
                ),
                dtype=torch.int32,
                device=device,
            )
            self.expert_logits_capture_local_buffer = (
                torch.empty_like(self.capture_local_buffer, dtype=torch.float32)
                if self.capture_topk_weights
                else None
            )
        # Set once per eager forward after CP metadata and MLP-sync padding
        # have both been prepared.  CP metadata describes the real physical
        # rows owned by one CP rank, while the router tensor may contain a
        # larger DP/MLP-sync suffix.  The suffix must never enter the capture
        # all-gather: doing so changes the rank-block stride and maps padding
        # routes onto real logical tokens during restoration.
        self._deepep_prefill_cp_capture_rows = None
        self._deepep_prefill_cp_local_rows = None
        self._deepep_prefill_cp_active = False

    def prepare_forward(
        self,
        forward_batch: ForwardBatch,
        owner_forward_batch: Optional[ForwardBatch] = None,
    ) -> None:
        """Pin the CP plan and bind it to the batch used at forward end.

        The eager runner executes a fixed-buffer copy of the scheduler's live
        ``ForwardBatch``.  CP metadata is built on that execution copy, while
        ``ModelRunner.on_forward_end`` receives the original owner batch.  The
        route sidechannel must therefore publish the exact restoration plan to
        the owner instead of silently treating rank-major CP rows as logical
        rows at forward end.
        """
        self._deepep_prefill_cp_capture_rows = None
        self._deepep_prefill_cp_local_rows = None
        self._deepep_prefill_cp_active = False
        if not get_moe_a2a_backend().is_deepep():
            return
        parallel = get_parallel()
        metadata = getattr(forward_batch, "attn_cp_metadata", None)
        if parallel.attn_cp_size <= 1 or metadata is None:
            return
        if not forward_batch.forward_mode.is_context_parallel_extend():
            return

        per_rank_rows = getattr(metadata, "per_rank_actual_token", None)
        if not per_rank_rows or len(per_rank_rows) != parallel.attn_cp_size:
            raise RuntimeError(
                "DeepEP CP routed-expert capture requires one physical capacity "
                f"count per CP rank: cp_size={parallel.attn_cp_size}, "
                f"per_rank_rows={per_rank_rows}"
            )
        physical_rank_rows = int(per_rank_rows[0])
        if physical_rank_rows < 0 or any(
            int(rows) != physical_rank_rows for rows in per_rank_rows
        ):
            raise RuntimeError(
                "DeepEP CP routed-expert capture requires equal physical capacity "
                f"counts: per_rank_rows={per_rank_rows}"
            )

        # `per_rank_actual_token` is physical capacity, not a real-token
        # count: prepare_mlp_sync_batch has already appended DP padding and
        # pad_logical_token_to_physical may add another CP-alignment suffix.
        # `_original_num_tokens` is recorded immediately before DP padding and
        # therefore defines the sidechannel's real logical prefix.
        real_total_rows = getattr(forward_batch, "_original_num_tokens", None)
        if real_total_rows is None:
            extend_rows = getattr(forward_batch, "extend_seq_lens_cpu", None)
            real_total_rows = (
                sum(int(rows) for rows in extend_rows)
                if extend_rows is not None
                else int(metadata.total_seq_lens)
            )
        real_total_rows = int(real_total_rows)
        if real_total_rows < 0 or real_total_rows > int(metadata.total_seq_lens):
            raise RuntimeError(
                "DeepEP CP routed-expert capture has an invalid real token count: "
                f"real={real_total_rows}, physical_logical={metadata.total_seq_lens}"
            )

        base_rows, extra_rows = divmod(real_total_rows, parallel.attn_cp_size)
        real_rows_by_rank = [
            base_rows + int(rank < extra_rows) for rank in range(parallel.attn_cp_size)
        ]
        capture_rank_rows = max(real_rows_by_rank, default=0)
        if capture_rank_rows > physical_rank_rows:
            raise RuntimeError(
                "DeepEP CP routed-expert real rows exceed physical capacity: "
                f"real_rows_by_rank={real_rows_by_rank}, "
                f"physical_rank_rows={physical_rank_rows}"
            )

        self._deepep_prefill_cp_capture_rows = capture_rank_rows
        self._deepep_prefill_cp_local_rows = real_rows_by_rank[parallel.attn_cp_rank]
        self._deepep_prefill_cp_active = True
        forward_batch._deepep_route_capture_rank_rows = capture_rank_rows
        forward_batch._deepep_route_capture_total_rows = real_total_rows
        if owner_forward_batch is not None and owner_forward_batch is not forward_batch:
            owner_forward_batch.attn_cp_metadata = metadata
            owner_forward_batch._deepep_route_capture_rank_rows = capture_rank_rows
            owner_forward_batch._deepep_route_capture_total_rows = real_total_rows
            owner_forward_batch._original_num_tokens = real_total_rows
        if os.environ.get("SGLANG_DEEPEP_ROUTE_CAPTURE_DEBUG") == "1":
            logger.warning(
                "DeepEP routed-expert CP preparation: pp_rank=%s cp_rank=%s "
                "original_num_tokens=%s input_rows=%s out_cache_rows=%s "
                "metadata_total_rows=%s metadata_logical_rows=%s "
                "metadata_physical_rows=%s real_rows_by_rank=%s "
                "capture_rank_rows=%s local_real_rows=%s",
                parallel.pp_rank,
                parallel.attn_cp_rank,
                real_total_rows,
                forward_batch.input_ids.shape[0],
                forward_batch.out_cache_loc.shape[0],
                metadata.total_seq_lens,
                getattr(metadata, "per_rank_logical_token", None),
                per_rank_rows,
                real_rows_by_rank,
                capture_rank_rows,
                self._deepep_prefill_cp_local_rows,
            )

    def capture(
        self,
        layer_id: int,
        topk_indices: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        if get_moe_a2a_backend().is_deepep():
            local_topk = topk_indices
            capture_rank_rows = None
            local_real_rows = None
            # This is deliberately local-forward state.  The DP-synchronized
            # extend flag may be true on a decode owner when another owner is
            # extending, but only prepare_forward() creates a CP restoration
            # plan for a local context-parallel extend.
            use_prefill_cp = bool(
                getattr(self, "_deepep_prefill_cp_active", False)
                and get_parallel().attn_cp_size > 1
            )
            capture_group_size = (
                get_parallel().attn_cp_size
                if use_prefill_cp
                else get_parallel().attn_tp_size
            )
            if use_prefill_cp:
                capture_rank_rows = getattr(
                    self, "_deepep_prefill_cp_capture_rows", None
                )
                local_real_rows = getattr(self, "_deepep_prefill_cp_local_rows", None)
                debug_first_owned_routed_layer = max(
                    getattr(self, "pp_start_layer", 0),
                    getattr(self, "first_routed_layer", 0),
                )
                debug_this_layer = (
                    os.environ.get("SGLANG_DEEPEP_ROUTE_CAPTURE_DEBUG") == "1"
                    and layer_id == debug_first_owned_routed_layer
                )
                if debug_this_layer:
                    local_zero_rows = (
                        (
                            torch.count_nonzero(local_topk[:, : self.topk_size], dim=1)
                            == 0
                        )
                        .nonzero(as_tuple=False)
                        .flatten()
                    )
                    logger.warning(
                        "DeepEP routed-expert CP local capture: pp_rank=%s "
                        "cp_rank=%s layer=%s router_rows=%s "
                        "capture_rank_rows=%s local_real_rows=%s "
                        "zero_router_rows=%s first_router_rows=%s",
                        get_parallel().pp_rank,
                        get_parallel().attn_cp_rank,
                        layer_id,
                        local_topk.shape[0],
                        capture_rank_rows,
                        local_real_rows,
                        local_zero_rows.tolist(),
                        local_topk[: min(20, local_topk.shape[0]), : self.topk_size]
                        .detach()
                        .cpu()
                        .tolist(),
                    )
                if capture_rank_rows is not None:
                    if local_real_rows is None:
                        raise RuntimeError(
                            "DeepEP CP routed-expert capture is missing its local "
                            "real-row count"
                        )
                    if local_topk.size(0) < local_real_rows:
                        raise RuntimeError(
                            "DeepEP CP routed-expert router tensor is shorter than "
                            "the prepared real row prefix: "
                            f"router_rows={local_topk.size(0)}, "
                            f"local_real_rows={local_real_rows}"
                        )
                    if capture_rank_rows == 0:
                        local_topk = local_topk[:0]
                    elif local_real_rows == capture_rank_rows:
                        local_topk = local_topk[:local_real_rows]
                    else:
                        staged_topk = self.capture_local_buffer[:capture_rank_rows]
                        staged_topk.zero_()
                        staged_topk[:local_real_rows].copy_(
                            local_topk[:local_real_rows]
                        )
                        local_topk = staged_topk
                if debug_this_layer:
                    staged_zero_rows = (
                        (
                            torch.count_nonzero(local_topk[:, : self.topk_size], dim=1)
                            == 0
                        )
                        .nonzero(as_tuple=False)
                        .flatten()
                    )
                    logger.warning(
                        "DeepEP routed-expert CP staged capture: pp_rank=%s "
                        "cp_rank=%s layer=%s staged_rows=%s zero_staged_rows=%s "
                        "staged_values=%s",
                        get_parallel().pp_rank,
                        get_parallel().attn_cp_rank,
                        layer_id,
                        local_topk.shape[0],
                        staged_zero_rows.tolist(),
                        local_topk[:, : self.topk_size].detach().cpu().tolist(),
                    )
            topk_indices = self.gather_buffer[: local_topk.size(0) * capture_group_size]
            if use_prefill_cp:
                attn_cp_all_gather_into_tensor(topk_indices, local_topk)
                if debug_this_layer:
                    gathered_zero_rows = (
                        (
                            torch.count_nonzero(
                                topk_indices[:, : self.topk_size], dim=1
                            )
                            == 0
                        )
                        .nonzero(as_tuple=False)
                        .flatten()
                    )
                    logger.warning(
                        "DeepEP routed-expert CP gathered capture: pp_rank=%s "
                        "cp_rank=%s layer=%s gathered_rows=%s "
                        "zero_gathered_rows=%s gathered_values=%s",
                        get_parallel().pp_rank,
                        get_parallel().attn_cp_rank,
                        layer_id,
                        topk_indices.shape[0],
                        gathered_zero_rows.tolist(),
                        topk_indices[:, : self.topk_size].detach().cpu().tolist(),
                    )
            else:
                attn_tp_all_gather_into_tensor(topk_indices, local_topk)
            if self.capture_topk_weights:
                if topk_weights is None:
                    raise RuntimeError("expert-logit capture requires topk_weights")
                local_weights = topk_weights
                if use_prefill_cp and capture_rank_rows is not None:
                    if local_weights.size(0) < local_real_rows:
                        raise RuntimeError(
                            "DeepEP CP routed-expert weight tensor is shorter than "
                            "the prepared real row prefix: "
                            f"weight_rows={local_weights.size(0)}, "
                            f"local_real_rows={local_real_rows}"
                        )
                    if capture_rank_rows == 0:
                        local_weights = local_weights[:0]
                    elif local_real_rows == capture_rank_rows:
                        local_weights = local_weights[:local_real_rows]
                    else:
                        staged_weights = self.expert_logits_capture_local_buffer[
                            :capture_rank_rows
                        ]
                        staged_weights.zero_()
                        staged_weights[:local_real_rows].copy_(
                            local_weights[:local_real_rows]
                        )
                        local_weights = staged_weights
                topk_weights = self.expert_logits_gather_buffer[
                    : local_weights.size(0) * capture_group_size
                ]
                if use_prefill_cp:
                    attn_cp_all_gather_into_tensor(topk_weights, local_weights)
                else:
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
        # Under DeepEP, capture() already gathered the active sequence-sharding
        # group into the head of the per-rank buffer, so the local DP rank's
        # data lives at [0:N_local] rather than at a global DP offset.
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            # GPU->CPU sync would break overlap; operate on CPU directly.
            local_start_pos, local_num_tokens = get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )
            local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos, local_end_pos = self._get_deepep_local_row_bounds(
                forward_batch
            )
        rows = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]
        if (
            os.environ.get("SGLANG_DEEPEP_ROUTE_CAPTURE_DEBUG") == "1"
            and getattr(forward_batch, "_deepep_route_capture_rank_rows", None)
            is not None
        ):
            owned_start = max(self.pp_start_layer, self.first_routed_layer)
            raw_zero_rows = (
                (
                    torch.count_nonzero(
                        rows[:, owned_start : self.pp_end_layer], dim=(1, 2)
                    )
                    == 0
                )
                .nonzero(as_tuple=False)
                .flatten()
            )
            logger.warning(
                "DeepEP routed-expert CP raw forward-end capture: pp_rank=%s "
                "cp_rank=%s raw_rows=%s owned_layers=[%s,%s) "
                "zero_raw_rows=%s",
                get_parallel().pp_rank,
                get_parallel().attn_cp_rank,
                rows.shape[0],
                owned_start,
                self.pp_end_layer,
                raw_zero_rows.tolist(),
            )
        rows = self._restore_deepep_cp_logical_rows(
            rows, forward_batch, payload_name="routed-expert ids"
        )
        if (
            os.environ.get("SGLANG_DEEPEP_ROUTE_CAPTURE_DEBUG") == "1"
            and getattr(forward_batch, "_deepep_route_capture_rank_rows", None)
            is not None
        ):
            logical_zero_rows = (
                (
                    torch.count_nonzero(
                        rows[:, owned_start : self.pp_end_layer], dim=(1, 2)
                    )
                    == 0
                )
                .nonzero(as_tuple=False)
                .flatten()
            )
            logger.warning(
                "DeepEP routed-expert CP logical forward-end capture: pp_rank=%s "
                "cp_rank=%s logical_rows=%s zero_logical_rows=%s",
                get_parallel().pp_rank,
                get_parallel().attn_cp_rank,
                rows.shape[0],
                logical_zero_rows.tolist(),
            )
        return rows

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
            local_start_pos, local_end_pos = self._get_deepep_local_row_bounds(
                forward_batch
            )
        rows = self.expert_logits_device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]
        rows = self._restore_deepep_cp_logical_rows(
            rows, forward_batch, payload_name="routed-expert weights"
        )
        return rows

    @staticmethod
    def _get_deepep_local_row_bounds(
        forward_batch: ForwardBatch,
    ) -> tuple[int, int]:
        """Return the physical rows already gathered by DeepEP capture.

        ``capture()`` all-gathers each CP-local top-k tensor across attn-CP
        before writing the device cache.  A CP prefill therefore occupies the
        complete rank-major *physical* buffer, including per-rank padding.  It
        must not be sliced by either one rank's logical length or the global
        logical request length.
        """
        if not get_moe_a2a_backend().is_deepep():
            return 0, forward_batch.out_cache_loc.shape[0]

        metadata = getattr(forward_batch, "attn_cp_metadata", None)
        if get_parallel().attn_cp_size <= 1 or metadata is None:
            return 0, forward_batch.out_cache_loc.shape[0]

        capture_rank_rows = getattr(
            forward_batch, "_deepep_route_capture_rank_rows", None
        )
        if capture_rank_rows is not None:
            return 0, int(capture_rank_rows) * get_parallel().attn_cp_size

        per_rank_rows = getattr(metadata, "per_rank_actual_token", None)
        if not per_rank_rows:
            raise RuntimeError(
                "DeepEP CP routed-expert capture requires per-rank token metadata"
            )
        cp_size = get_parallel().attn_cp_size
        if len(per_rank_rows) != cp_size:
            raise RuntimeError(
                "DeepEP CP routed-expert capture has incomplete CP metadata: "
                f"cp_size={cp_size}, per_rank_rows={per_rank_rows}"
            )
        physical_rank_rows = int(per_rank_rows[0])
        if physical_rank_rows < 0 or any(
            int(rows) != physical_rank_rows for rows in per_rank_rows
        ):
            raise RuntimeError(
                "DeepEP CP routed-expert capture requires equal physical row counts: "
                f"per_rank_rows={per_rank_rows}"
            )
        return 0, physical_rank_rows * cp_size

    @staticmethod
    def _restore_deepep_cp_logical_rows(
        rows: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        payload_name: str,
    ) -> torch.Tensor:
        """Restore an already-gathered CP buffer to logical token order."""
        if not get_moe_a2a_backend().is_deepep():
            return rows
        metadata = getattr(forward_batch, "attn_cp_metadata", None)
        if get_parallel().attn_cp_size <= 1 or metadata is None:
            return rows

        from sglang.srt.layers.cp.base import get_cp_strategy

        strategy = get_cp_strategy()
        if strategy is None:
            raise RuntimeError(
                "DeepEP CP routed-expert capture requires an active CP strategy"
            )
        if strategy.name != "interleave":
            raise RuntimeError(
                "DeepEP CP routed-expert capture currently admits only the "
                f"interleave strategy, got {strategy.name!r}"
            )

        cp_size = get_parallel().attn_cp_size
        capture_rank_rows = getattr(
            forward_batch, "_deepep_route_capture_rank_rows", None
        )
        capture_total_rows = getattr(
            forward_batch, "_deepep_route_capture_total_rows", None
        )
        if capture_rank_rows is not None:
            physical_rank_rows = int(capture_rank_rows)
            total_rows = int(capture_total_rows)
        else:
            per_rank_rows = getattr(metadata, "per_rank_actual_token", None)
            if not per_rank_rows:
                raise RuntimeError(
                    "DeepEP CP routed-expert capture requires physical row metadata"
                )
            physical_rank_rows = int(per_rank_rows[0])
            total_rows = int(metadata.total_seq_lens)
        expected_physical_rows = cp_size * physical_rank_rows
        if rows.shape[0] != expected_physical_rows:
            raise RuntimeError(
                f"DeepEP CP {payload_name} has the wrong physical row count: "
                f"captured={rows.shape[0]}, expected={expected_physical_rows}"
            )

        flat_indices = torch.arange(total_rows, device=rows.device)
        gather_indices = (
            flat_indices % cp_size
        ) * physical_rank_rows + flat_indices // cp_size
        logical_rows = rows.index_select(0, gather_indices)
        expected_rows = (
            int(capture_total_rows)
            if capture_total_rows is not None
            else forward_batch.out_cache_loc.shape[0]
        )
        if logical_rows.shape[0] != expected_rows:
            raise RuntimeError(
                f"DeepEP CP {payload_name} gather returned the wrong row count: "
                f"gathered={logical_rows.shape[0]}, expected={expected_rows}"
            )
        return logical_rows

    @staticmethod
    def _trim_deepep_mlp_sync_padding(
        rows: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        payload_name: str,
    ) -> torch.Tensor:
        """Drop local DP padding before routes are published by cache location.

        MLP synchronization pads ``input_ids`` and ``out_cache_loc`` to the
        local DP collective width.  The appended cache locations are zero, so
        publishing their route rows would overwrite the real token already
        stored at KV slot zero.  DeepEP capture must retain those rows through
        MoE execution and CP restoration, then remove them at this boundary.
        """
        if not get_moe_a2a_backend().is_deepep():
            return rows
        original_num_tokens = getattr(forward_batch, "_original_num_tokens", None)
        if original_num_tokens is None:
            return rows
        original_num_tokens = int(original_num_tokens)
        if original_num_tokens < 0 or original_num_tokens > rows.shape[0]:
            raise RuntimeError(
                f"DeepEP {payload_name} has invalid unpadded token count: "
                f"original={original_num_tokens}, captured={rows.shape[0]}"
            )
        return rows[:original_num_tokens]

    def _mask_to_owned_pp_layers(
        self,
        rows: torch.Tensor,
        *,
        payload_name: str,
    ) -> torch.Tensor:
        """Zero stale cache columns that are not owned by this PP stage."""
        if rows.ndim != 3 or rows.shape[1] != self.num_layers:
            raise RuntimeError(
                f"DeepEP PP {payload_name} has invalid shape {tuple(rows.shape)}; "
                f"expected [tokens, {self.num_layers}, topk]"
            )
        merged = rows.clone()
        merged[:, : self.pp_start_layer] = 0
        merged[:, self.pp_end_layer :] = 0
        return merged

    @staticmethod
    def _validate_pp_payload(
        payload: torch.Tensor,
        local: torch.Tensor,
        *,
        payload_name: str,
    ) -> None:
        if payload.shape != local.shape:
            raise RuntimeError(
                f"DeepEP PP {payload_name} shape mismatch: "
                f"incoming={tuple(payload.shape)}, local={tuple(local.shape)}"
            )
        if payload.dtype != local.dtype:
            raise RuntimeError(
                f"DeepEP PP {payload_name} dtype mismatch: "
                f"incoming={payload.dtype}, local={local.dtype}"
            )

    def _transport_deepep_pp_layers(
        self,
        indices: torch.Tensor,
        expert_logits: Optional[torch.Tensor],
        *,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        incoming_pp_proxy_tensors,
        outgoing_pp_proxy_tensors,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Carry disjoint route columns downstream with the PP activation.

        A PP all-reduce at forward end deadlocks: PP0 cannot enter the
        collective and also send the activation that lets PP1 run.  The normal
        PP proxy already flows in the required order, so attach the masked
        stage-local captures to it.  Intermediate stages add disjoint columns;
        the terminal stage returns the complete tensor for host publication.
        """
        parallel = get_parallel()
        if not get_moe_a2a_backend().is_deepep() or parallel.pp_size <= 1:
            return indices, expert_logits, out_cache_loc, req_pool_indices

        indices = self._mask_to_owned_pp_layers(
            indices, payload_name="routed-expert ids"
        )
        if expert_logits is not None:
            expert_logits = self._mask_to_owned_pp_layers(
                expert_logits, payload_name="routed-expert weights"
            )

        if parallel.pp_group.is_first_rank:
            # These tensors identify the PP0 request/KV pools that own the
            # route rows.  Clone them on the forward stream before the live
            # ScheduleBatch or its ring slot can be reused.
            owner_out_cache_loc = out_cache_loc.clone()
            owner_req_pool_indices = req_pool_indices.clone()
        else:
            owner_out_cache_loc = None
            owner_req_pool_indices = None

        if not parallel.pp_group.is_first_rank:
            if incoming_pp_proxy_tensors is None:
                raise RuntimeError(
                    "DeepEP PP routed-expert capture is missing the upstream PP proxy"
                )
            incoming = incoming_pp_proxy_tensors.tensors
            if _PP_ROUTED_EXPERT_IDS not in incoming:
                raise RuntimeError(
                    "DeepEP PP routed-expert capture is missing upstream expert ids"
                )
            upstream_indices = incoming[_PP_ROUTED_EXPERT_IDS]
            self._validate_pp_payload(
                upstream_indices, indices, payload_name="routed-expert ids"
            )
            indices = indices + upstream_indices

            if _PP_ROUTED_EXPERT_OUT_CACHE_LOC not in incoming:
                raise RuntimeError(
                    "DeepEP PP routed-expert capture is missing upstream owner "
                    "cache locations"
                )
            if _PP_ROUTED_EXPERT_REQ_POOL_INDICES not in incoming:
                raise RuntimeError(
                    "DeepEP PP routed-expert capture is missing upstream owner "
                    "request-pool indices"
                )
            owner_out_cache_loc = incoming[_PP_ROUTED_EXPERT_OUT_CACHE_LOC]
            owner_req_pool_indices = incoming[_PP_ROUTED_EXPERT_REQ_POOL_INDICES]

            if expert_logits is not None:
                if _PP_ROUTED_EXPERT_WEIGHTS not in incoming:
                    raise RuntimeError(
                        "DeepEP PP routed-expert capture is missing upstream weights"
                    )
                upstream_logits = incoming[_PP_ROUTED_EXPERT_WEIGHTS]
                self._validate_pp_payload(
                    upstream_logits,
                    expert_logits,
                    payload_name="routed-expert weights",
                )
                expert_logits = expert_logits + upstream_logits

        if (
            owner_out_cache_loc.ndim != 1
            or owner_out_cache_loc.numel() != indices.shape[0]
        ):
            raise RuntimeError(
                "DeepEP PP routed-expert owner cache locations have invalid shape: "
                f"shape={tuple(owner_out_cache_loc.shape)}, "
                f"route_rows={indices.shape[0]}"
            )
        if owner_req_pool_indices.ndim != 1:
            raise RuntimeError(
                "DeepEP PP routed-expert owner request-pool indices have invalid "
                f"shape: shape={tuple(owner_req_pool_indices.shape)}"
            )

        if parallel.pp_group.is_last_rank:
            return (
                indices,
                expert_logits,
                owner_out_cache_loc,
                owner_req_pool_indices,
            )

        if outgoing_pp_proxy_tensors is None:
            raise RuntimeError(
                "DeepEP PP routed-expert capture is missing the downstream PP proxy"
            )
        outgoing_pp_proxy_tensors[_PP_ROUTED_EXPERT_IDS] = indices
        outgoing_pp_proxy_tensors[_PP_ROUTED_EXPERT_OUT_CACHE_LOC] = owner_out_cache_loc
        outgoing_pp_proxy_tensors[_PP_ROUTED_EXPERT_REQ_POOL_INDICES] = (
            owner_req_pool_indices
        )
        if expert_logits is not None:
            outgoing_pp_proxy_tensors[_PP_ROUTED_EXPERT_WEIGHTS] = expert_logits
        return None, None, None, None

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

    def get_topk(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool,
        start_len: int = 0,
    ) -> torch.Tensor:
        """Read routes and expose any real-token lookup through dummy KV slot 0.

        Slot 0 is reserved for graph/DP padding.  A real request position that
        resolves there cannot be made correct by publishing padding routes to
        the dummy slot; log the exact positions so the owning mapping can be
        repaired.  The warning is intentionally unconditional because such a
        lookup is a routed-expert sidechannel correctness failure.
        """
        if start_len < 0:
            raise ValueError(f"{start_len=} must be non-negative")
        start_len = min(start_len, seqlen - 1)
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][start_len : seqlen - 1]
            .cpu()
            .clone()
        )
        zero_positions = (cache_pool_idx == 0).nonzero(as_tuple=False).flatten()
        if (
            zero_positions.numel()
            or os.environ.get("SGLANG_DEEPEP_ROUTE_CAPTURE_DEBUG") == "1"
        ):
            logger.warning(
                "DeepEP routed-expert lookup mapping: req_pool_idx=%s "
                "start_len=%s seqlen=%s cache_rows=%s zero_relative_positions=%s "
                "zero_absolute_positions=%s cache_pool_indices=%s",
                req_pool_idx,
                start_len,
                seqlen,
                cache_pool_idx.numel(),
                zero_positions.tolist(),
                (zero_positions + start_len).tolist(),
                cache_pool_idx.tolist(),
            )
        return self.host_cache.buffer[cache_pool_idx]

    def publish_pp_capture(
        self,
        *,
        out_cache_loc: torch.Tensor,
        req_pool_indices: Optional[torch.Tensor] = None,
        indices: torch.Tensor,
        expert_logits: Optional[torch.Tensor],
    ) -> None:
        """Commit the terminal PP capture under the owner's KV-cache indices.

        The terminal stage cannot publish directly: HTTP output extraction runs
        on PP0 and indexes PP0's request/KV pools.  The normal PP response ring
        carries the merged tensors back, and PP0 commits them here.
        """
        expected_shape = (out_cache_loc.numel(), self.num_layers, self.topk_size)
        if tuple(indices.shape) != expected_shape or indices.dtype != torch.int32:
            raise RuntimeError(
                "DeepEP PP routed-expert output has invalid ids: "
                f"shape={tuple(indices.shape)}, dtype={indices.dtype}, "
                f"expected_shape={expected_shape}, expected_dtype=torch.int32"
            )
        if self.capture_topk_weights:
            if expert_logits is None:
                raise RuntimeError(
                    "DeepEP PP routed-expert output is missing selected-router weights"
                )
            if (
                tuple(expert_logits.shape) != expected_shape
                or expert_logits.dtype != torch.float32
            ):
                raise RuntimeError(
                    "DeepEP PP routed-expert output has invalid weights: "
                    f"shape={tuple(expert_logits.shape)}, "
                    f"dtype={expert_logits.dtype}, expected_shape={expected_shape}, "
                    "expected_dtype=torch.float32"
                )
        elif expert_logits is not None:
            raise RuntimeError(
                "DeepEP PP output returned selected-router weights when weight "
                "capture is disabled"
            )

        out_cache_loc_cpu = out_cache_loc.cpu()
        indices_cpu = indices.cpu()
        if os.environ.get("SGLANG_DEEPEP_ROUTE_CAPTURE_DEBUG") == "1":
            zero_rows = (
                (torch.count_nonzero(indices_cpu, dim=(1, 2)) == 0)
                .nonzero(as_tuple=False)
                .flatten()
            )
            unique_locs, loc_counts = torch.unique(
                out_cache_loc_cpu, return_counts=True
            )
            duplicate_locs = unique_locs[loc_counts > 1]
            logger.warning(
                "DeepEP routed-expert PP publication: route_rows=%s "
                "zero_route_rows=%s zero_out_cache_rows=%s duplicate_out_cache_locs=%s "
                "zero_route_out_cache_locs=%s req_pool_indices=%s out_cache_locs=%s",
                indices_cpu.shape[0],
                zero_rows.tolist(),
                (out_cache_loc_cpu == 0).nonzero(as_tuple=False).flatten().tolist(),
                duplicate_locs.tolist(),
                out_cache_loc_cpu[zero_rows].tolist(),
                (
                    req_pool_indices.cpu().tolist()
                    if req_pool_indices is not None
                    else None
                ),
                out_cache_loc_cpu.tolist(),
            )
        self.host_cache.buffer[out_cache_loc_cpu] = indices_cpu
        if expert_logits is not None:
            self.expert_logits_host_cache.buffer[out_cache_loc_cpu] = (
                expert_logits.cpu()
            )

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
        incoming_pp_proxy_tensors=None,
        outgoing_pp_proxy_tensors=None,
    ) -> Optional[RoutedExpertsCaptureOutput]:
        indices = self._get_local_slice(forward_batch, can_run_graph, cuda_graph_batch)
        expert_logits = self._get_local_expert_logits_slice(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        indices = self._trim_deepep_mlp_sync_padding(
            indices, forward_batch, payload_name="routed-expert ids"
        )
        if expert_logits is not None:
            expert_logits = self._trim_deepep_mlp_sync_padding(
                expert_logits,
                forward_batch,
                payload_name="routed-expert weights",
            )
        owner_out_cache_loc = forward_batch.out_cache_loc
        if get_moe_a2a_backend().is_deepep():
            original_num_tokens = getattr(forward_batch, "_original_num_tokens", None)
            if original_num_tokens is not None:
                original_num_tokens = int(original_num_tokens)
                if (
                    original_num_tokens < 0
                    or original_num_tokens > owner_out_cache_loc.numel()
                ):
                    raise RuntimeError(
                        "DeepEP routed-expert owner cache locations have invalid "
                        "unpadded token count: "
                        f"original={original_num_tokens}, "
                        f"captured={owner_out_cache_loc.numel()}"
                    )
                owner_out_cache_loc = owner_out_cache_loc[:original_num_tokens]
        (
            indices,
            expert_logits,
            owner_out_cache_loc,
            owner_req_pool_indices,
        ) = self._transport_deepep_pp_layers(
            indices,
            expert_logits,
            out_cache_loc=owner_out_cache_loc,
            req_pool_indices=forward_batch.req_pool_indices,
            incoming_pp_proxy_tensors=incoming_pp_proxy_tensors,
            outgoing_pp_proxy_tensors=outgoing_pp_proxy_tensors,
        )
        if indices is None:
            return None
        parallel = get_parallel()
        if (
            get_moe_a2a_backend().is_deepep()
            and parallel.pp_size > 1
            and parallel.pp_group.is_last_rank
        ):
            # The terminal stage must return the merged GPU tensors to PP0 via
            # the normal output ring.  Publishing here would fill PP1's host
            # cache, while HTTP extraction reads PP0's disjoint cache.
            return RoutedExpertsCaptureOutput(
                out_cache_loc=owner_out_cache_loc,
                req_pool_indices=owner_req_pool_indices,
                topk=indices,
                host_cache=self.host_cache,
                expert_logits=expert_logits,
                expert_logits_host_cache=self.expert_logits_host_cache,
            )
        if no_copy_to_cpu:
            return RoutedExpertsCaptureOutput(
                out_cache_loc=owner_out_cache_loc,
                req_pool_indices=owner_req_pool_indices,
                topk=indices,
                host_cache=self.host_cache,
                expert_logits=expert_logits,
                expert_logits_host_cache=self.expert_logits_host_cache,
            )
        out_cache_loc_cpu = owner_out_cache_loc.cpu()
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


def disable_routed_experts_capture_for_draft(model: Any) -> None:
    """Opt every draft MoE router out of routed-experts (R3) capture.

    Capture is target-only; a draft router must never write the target's
    process-global buffer.
    """
    # Lazy import: ``layers.moe.topk`` imports ``get_global_experts_capturer``
    # from this module, so a top-level import here would be circular.
    from sglang.srt.layers.moe.hash_topk import HashTopK
    from sglang.srt.layers.moe.topk import TopK

    for module in model.modules():
        if isinstance(module, (TopK, HashTopK)):
            module.topk_config.allow_routed_experts_capture = False
