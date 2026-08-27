import dataclasses
import logging
from typing import List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_GB = 1024 * 1024 * 1024
_MB = 1024 * 1024


def get_tensor_size_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class BaseDeviceCache:
    def __init__(
        self,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
        name: str,
        dtype: torch.dtype = torch.int32,
    ):
        self.buffer = torch.zeros(
            (max_batch_size, num_layers, topk_size),
            dtype=dtype,
            device=device,
        )
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.name = name
        self._log_allocation()

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        batch = topk_indices.shape[0]
        self.buffer[:batch, layer_id, :] = topk_indices

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def _log_allocation(self):
        size_mb = self.get_buffer_size_bytes() / _MB
        logger.info(
            f"DeviceCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "
            f"size={size_mb:.2f} MB"
        )


class BaseHostCache:
    def __init__(
        self,
        num_tokens: int,
        num_layers: int,
        topk_size: int,
        name: str,
        dtype: torch.dtype = torch.int32,
    ):
        self.buffer = torch.zeros(
            (num_tokens, num_layers, topk_size),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.name = name
        self._log_allocation()

    def get_buffer_size_bytes(self):
        return get_tensor_size_bytes(self.buffer)

    def _log_allocation(self):
        size_gb = self.get_buffer_size_bytes() / _GB
        logger.info(
            f"HostCache[{self.name}] allocated: shape={tuple(self.buffer.shape)}, "
            f"size={size_gb:.2f} GB"
        )


@dataclasses.dataclass
class TopkCaptureOutput:
    """Holds GPU tensors captured during forward for overlap scheduling.
    map_device_tensors() D2H-copies them before copy_done.record() (may run on
    the dedicated result-copy stream); finalize() runs after copy_done.synchronize().
    """

    out_cache_loc: torch.Tensor
    topk: torch.Tensor
    host_cache: BaseHostCache

    def map_device_tensors(self, fn):
        # Device-tensor fields only; caller injects the copy+safety primitive
        # (see GenerationBatchResult.copy_to_cpu).
        self.out_cache_loc = fn(self.out_cache_loc)
        self.topk = fn(self.topk)

    def finalize(self):
        self.host_cache.buffer[self.out_cache_loc] = self.topk


class BaseTopkCapturer:
    def __init__(
        self,
        num_tokens: int,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        device: str,
        name: str,
        device_topk_size: Optional[int] = None,
    ):
        """device_topk_size defaults to topk_size; pass a different value when
        the device buffer needs extra columns (e.g. fused shared experts) that
        are dropped before writing to host_cache via [:topk_size] truncation.
        """
        self.num_layers = num_layers
        self.topk_size = topk_size
        # Which layer ids have actually written route rows. Dense (non-MoE)
        # layers never call capture(), so their planes in the host buffer stay
        # zero -- indistinguishable on the wire from a real expert id 0.
        # Recording the captured set lets a response state which plane indices
        # carry real routes instead of leaving the consumer to guess.
        self._layer_captured = [False] * num_layers

        self.host_cache = BaseHostCache(num_tokens, num_layers, topk_size, name=name)
        self.device_cache = BaseDeviceCache(
            max_batch_size,
            num_layers,
            device_topk_size if device_topk_size is not None else topk_size,
            device,
            name=name,
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        self._layer_captured[layer_id] = True
        self.device_cache.capture(layer_id, topk_indices)

    @property
    def captured_layer_ids(self) -> List[int]:
        """Layer ids that have written route rows, ascending (the MoE layers)."""
        return [i for i, seen in enumerate(self._layer_captured) if seen]

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        """Return the device_cache slice for this forward batch, GPU-resident.

        Default assumes per-rank-local capture: each rank writes [:local_num_tokens)
        to its own device_cache. Subclasses with global-tensor capture semantics
        (e.g. shared cuda graph buffer indexed by dp_rank) should override and
        consume can_run_graph / cuda_graph_batch.
        """
        del can_run_graph, cuda_graph_batch  # reserved for subclass override
        num_tokens = forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[:num_tokens, :, : self.topk_size]

    def get_rows(
        self,
        *,
        req_pool_idx: int,
        start: int,
        end: int,
        req_to_token_pool: ReqToTokenPool,
    ) -> torch.Tensor:
        """Gather host rows for this request's forward positions ``[start, end)``.

        Positions index the request's own token slots via ``req_to_token``, so a
        partial gather copies only its own slice of the mapping and indexes only
        its own host rows -- an input-only or output-only request never
        materializes the full history just to slice it.
        """
        if start < 0:
            raise ValueError(f"{start=} must be non-negative")
        if end < start:
            raise ValueError(f"{end=} must be >= {start=}")
        if end == start:
            # Basic slicing would return a *view* aliasing the whole pinned host
            # buffer, which the IPC pickler would then serialize in full. Build a
            # standalone empty tensor with the right trailing shape instead.
            return torch.empty(
                (0, self.num_layers, self.topk_size),
                dtype=self.host_cache.buffer.dtype,
            )
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][start:end].cpu().clone()
        )
        # Advanced (tensor) indexing copies, so the result never aliases the
        # shared host buffer.
        return self.host_cache.buffer[cache_pool_idx]

    def get_topk(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
    ) -> torch.Tensor:
        """Legacy full-history gather: rows ``[start_len, seqlen - 1)``.

        The exclusive ``seqlen - 1`` bound is the causal convention: a row
        belongs to the forward position that predicts the *next* token, and the
        forward at ``seqlen - 1`` never ran. See
        :mod:`sglang.srt.state_capturer.expert_route_selection`.
        """
        if start_len < 0:
            raise ValueError(f"{start_len=} must be non-negative")
        end = max(0, seqlen - 1)
        return self.get_rows(
            req_pool_idx=req_pool_idx,
            start=min(start_len, end),
            end=end,
            req_to_token_pool=req_to_token_pool,
        )

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        """If no_copy_to_cpu is True, return a TopkCaptureOutput holding GPU tensors so
        the overlap thread can do non-blocking D2H + finalize itself. Otherwise sync
        D2H inline and return None (legacy non-overlap path).
        """
        slice_gpu = self._get_local_slice(
            forward_batch, can_run_graph, cuda_graph_batch
        )
        if no_copy_to_cpu:
            return TopkCaptureOutput(
                out_cache_loc=forward_batch.out_cache_loc,
                topk=slice_gpu,
                host_cache=self.host_cache,
            )
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = slice_gpu.cpu()
        return None
