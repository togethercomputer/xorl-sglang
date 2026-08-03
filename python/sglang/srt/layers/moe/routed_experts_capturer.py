import logging
from abc import ABC
from typing import Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_attention_dp_rank,
    get_attention_tp_size,
    get_dp_local_info,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize


class _RoutedExpertsDeviceCache:
    def __init__(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
        capture_topk_weights: bool = False,
    ) -> None:
        self.buffer = torch.zeros(
            (
                max(
                    get_global_server_args().chunked_prefill_size
                    * get_global_server_args().dp_size,
                    max_running_requests,
                ),
                num_hidden_layers,
                num_experts_per_tok + num_fused_shared_experts,
            ),
            dtype=torch.int32,
            device=device,
        )
        self.topk_weights_buffer = None
        if capture_topk_weights:
            self.topk_weights_buffer = torch.zeros(
                (
                    max(
                        get_global_server_args().chunked_prefill_size
                        * get_global_server_args().dp_size,
                        max_running_requests,
                    ),
                    num_hidden_layers,
                    num_experts_per_tok + num_fused_shared_experts,
                ),
                dtype=torch.float32,
                device=device,
            )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        size = get_tensor_size_bytes(self.buffer)
        if self.topk_weights_buffer is not None:
            size += get_tensor_size_bytes(self.topk_weights_buffer)
        return size

    def capture_fwd_routed_experts(self, layer_id: int, topk_ids: torch.Tensor):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids

    def capture_fwd_topk_weights(self, layer_id: int, topk_weights: torch.Tensor):
        if self.topk_weights_buffer is not None:
            batch, _ = topk_weights.shape
            self.topk_weights_buffer[:batch, layer_id, :] = topk_weights.float()

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_MB = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )


class _RoutedExpertsHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        capture_topk_weights: bool = False,
    ) -> None:
        self.num_tokens = num_tokens
        self.buffer = torch.zeros(
            (
                num_tokens,
                num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.topk_weights_buffer = None
        if capture_topk_weights:
            self.topk_weights_buffer = torch.zeros(
                (
                    num_tokens,
                    num_hidden_layers,
                    num_experts_per_tok,
                ),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        size = get_tensor_size_bytes(self.buffer)
        if self.topk_weights_buffer is not None:
            size += get_tensor_size_bytes(self.topk_weights_buffer)
        return size

    def set_experts_buffer(self, layer_id: int, loc: torch.Tensor, top_k: torch.Tensor):
        self.buffer[layer_id, loc, :] = top_k.to(device="cpu", non_blocking=True)

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers."""
        buffer_size_GB = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.num_tokens}, size: {buffer_size_GB:.2f} GB"
        )


class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool,
        model_config: ModelConfig,
        num_fused_shared_experts: int,
        num_tokens: int,
        max_running_requests: int,
        device: str,
        capture_topk_weights: bool = False,
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=num_fused_shared_experts,
                device=device,
                capture_topk_weights=capture_topk_weights,
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        raise NotImplementedError

    def capture(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def get_expert_logits(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        raise NotImplementedError

    def get_host_cache(self):
        raise NotImplementedError

    def get_device_cache(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
        capture_topk_weights: bool = False,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        self.num_hidden_layers = model_config.hf_text_config.num_hidden_layers
        self.num_experts_per_tok = model_config.hf_text_config.num_experts_per_tok
        self.capture_topk_weights = capture_topk_weights

        self.host_cache = _RoutedExpertsHostCache(
            num_tokens=num_tokens,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            capture_topk_weights=capture_topk_weights,
        )

        self.device_cache = _RoutedExpertsDeviceCache(
            max_running_requests=max_running_requests,
            num_hidden_layers=self.num_hidden_layers,
            num_experts_per_tok=self.num_experts_per_tok,
            num_fused_shared_experts=self.num_fused_shared_experts,
            device=device,
            capture_topk_weights=capture_topk_weights,
        )

        if get_moe_a2a_backend().is_deepep():
            attn_tp_size = get_attention_tp_size() if is_dp_attention_enabled() else 1
            self.gather_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0] * attn_tp_size,
                    self.device_cache.buffer.shape[2],
                ),
                dtype=torch.int32,
                device=device,
            )
            if capture_topk_weights:
                self.gather_weights_buffer = torch.empty(
                    (
                        self.device_cache.buffer.shape[0] * attn_tp_size,
                        self.device_cache.buffer.shape[2],
                    ),
                    dtype=torch.float32,
                    device=device,
                )

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
            # handle with cuda graph padding
            if can_run_graph:
                local_start_pos = get_attention_dp_rank() * cuda_graph_batch
                local_end_pos = local_start_pos + local_num_tokens
            else:
                local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos = 0
            local_end_pos = forward_batch.out_cache_loc.shape[0]

        # FIXME: sync explicitly here, overlap scheduler breaks here.
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.num_experts_per_tok
        ].cpu()
        if (
            self.capture_topk_weights
            and self.host_cache.topk_weights_buffer is not None
        ):
            self.host_cache.topk_weights_buffer[out_cache_loc_cpu] = (
                self.device_cache.topk_weights_buffer[
                    local_start_pos:local_end_pos, :, : self.num_experts_per_tok
                ].cpu()
            )

    def capture(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        if get_moe_a2a_backend().is_deepep():
            local_topk_ids = topk_ids
            topk_ids = self.gather_buffer[
                : local_topk_ids.size(0) * get_attention_tp_size()
            ]
            attn_tp_all_gather_into_tensor(topk_ids, local_topk_ids)
            if topk_weights is not None and self.capture_topk_weights:
                local_topk_weights = topk_weights
                topk_weights = self.gather_weights_buffer[
                    : local_topk_weights.size(0) * get_attention_tp_size()
                ]
                attn_tp_all_gather_into_tensor(topk_weights, local_topk_weights)
        self.device_cache.capture_fwd_routed_experts(layer_id, topk_ids)
        if topk_weights is not None:
            self.device_cache.capture_fwd_topk_weights(layer_id, topk_weights)

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1].cpu().clone()
        )
        return self.get_host_cache().buffer[cache_pool_idx]

    def get_expert_logits(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        if not self.capture_topk_weights or self.host_cache.topk_weights_buffer is None:
            return None
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][: seqlen - 1].cpu().clone()
        )
        return self.host_cache.topk_weights_buffer[cache_pool_idx]

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        self._sync_fwd_experts_buffer_DtoH(
            forward_batch=forward_batch,
            can_run_graph=can_run_graph,
            cuda_graph_batch=cuda_graph_batch,
        )

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def _sync_fwd_experts_buffer_DtoH(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: int,
    ):
        pass

    def capture(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ):
        pass

    def get_routed_experts(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        pass

    def get_expert_logits(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
    ):
        pass

    def on_forward_end(self, forward_batch, can_run_graph, cuda_graph_batch):
        pass

    def get_host_cache(self):
        pass

    def get_device_cache(self):
        pass


_global_expert_capturer: Optional[RoutedExpertsCapturer] = _RoutedExpertsCapturerNoop()


def get_global_experts_capturer():
    return _global_expert_capturer


def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer


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
    # Decode expert routing weights (topk_weights) from base64-encoded float32
    # See detokenizer_manager::_extract_expert_logits
    expert_logits_base64 = data["meta_info"].get("expert_logits", None)
    if expert_logits_base64 is None:
        return None
    expert_logits = np.frombuffer(
        pybase64.b64decode(expert_logits_base64.encode("utf-8")), dtype=np.float32
    )
    return expert_logits
