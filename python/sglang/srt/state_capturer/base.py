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
        decode_graph_stride: Optional[int],
    ) -> torch.Tensor:
        """Return the device_cache slice for this forward batch, GPU-resident.

        Default assumes per-rank-local capture: each rank writes [:local_num_tokens)
        to its own device_cache. Subclasses with global-tensor capture semantics
        (e.g. shared cuda graph buffer indexed by dp_rank) should override and
        consume decode_graph_stride.
        """
        del decode_graph_stride  # reserved for subclass override
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

    @staticmethod
    def _own_rows(rows_view: torch.Tensor) -> torch.Tensor:
        """Detach a capture slice from the shared device cache.

        The overlap path hands these rows to ``GenerationBatchResult.copy_to_cpu``,
        whose D2H runs on ``copy_stream`` *deliberately overlapping the next
        forward* (scheduler.py: ``copy_stream.wait_stream(forward_stream)`` then
        the copy, with no reverse dependency). ``_async_d2h`` calls
        ``record_stream`` on the source, which stops the caching allocator from
        recycling a block early -- and that is enough for every other tensor it
        copies, because those are freshly allocated per forward.

        It is not enough here. ``BaseDeviceCache.buffer`` is allocated once and
        ``capture()`` writes it **in place** every forward, so it is never
        recycled by the allocator and ``record_stream`` has nothing to hold.
        Handing out a view lets the next forward overwrite those rows while the
        D2H is still reading them, and the request receives a valid-shaped
        tensor of another forward's expert ids.

        Copying on the forward stream, before returning, restores the invariant
        the copy path already assumes: everything it is given is privately owned
        for this forward. The copy is rows x layers x top_k int32 -- kilobytes.

        The non-overlap path needs none of this: it ``.cpu()``s synchronously on
        the forward stream before returning, so no window exists.
        """
        return rows_view.clone()

    @staticmethod
    def _num_real_rows(forward_batch: ForwardBatch) -> int:
        """How many leading rows of ``out_cache_loc`` belong to real tokens.

        MLP-sync padding appends dummy rows *after* the real ones and pads
        ``out_cache_loc`` with **zeros** (``_pad_tensor_to_size`` defaults to
        ``value=0``), so every padding row points at KV slot 0. Writing the
        whole tensor would stamp padding-derived routes over whichever request
        currently owns that slot -- silent corruption at a valid shape, not a
        crash.

        ``ForwardBatch._original_num_tokens`` is recorded by
        ``_pad_inputs_to_size`` immediately before it pads, and is ``None`` on
        every path that never padded (no DP/MLP-sync, and the decode cuda graph,
        which returns before ``_prepare_eager_forward_batch``). ``None``
        therefore means "no padding was applied", not "unknown".
        ``post_forward_mlp_sync_batch`` trims positions/seq_lens with the same
        field but leaves ``out_cache_loc`` padded, which is why the trim has to
        happen here.
        """
        padded = forward_batch.out_cache_loc.shape[0]
        original = forward_batch._original_num_tokens
        if original is None:
            return padded
        return min(original, padded)

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        decode_graph_stride: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        """If no_copy_to_cpu is True, return a TopkCaptureOutput holding GPU tensors so
        the overlap thread can do non-blocking D2H + finalize itself. Otherwise sync
        D2H inline and return None (legacy non-overlap path).
        """
        # Trim once, here, so the overlap and non-overlap paths below both
        # inherit it -- finalize() must not need its own copy of the guard.
        rows = self._num_real_rows(forward_batch)
        out_cache_loc = forward_batch.out_cache_loc[:rows]
        slice_gpu = self._get_local_slice(forward_batch, decode_graph_stride)[:rows]
        if no_copy_to_cpu:
            return TopkCaptureOutput(
                out_cache_loc=out_cache_loc,
                topk=self._own_rows(slice_gpu),
                host_cache=self.host_cache,
            )
        self.host_cache.buffer[out_cache_loc.cpu()] = slice_gpu.cpu()
        return None
