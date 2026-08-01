"""Exact GLM-5.2 sparse selection and selected-KV contracts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import logging
from typing import Iterator, Mapping, Sequence

import torch

GLM52_SELECTOR_CONTRACT_VERSION = "glm52_fp8_sampler_selector_v1"
_SELECTOR_CONTRACT = GLM52_SELECTOR_CONTRACT_VERSION
_PACKER_CONTRACT = "glm52_dynamic_selected_kv_v1"
_RECEIPT_IDENTITIES_LOGGED: set[tuple[int, int, str]] = set()


def _require_host_true(condition: torch.Tensor, message: str) -> None:
    """Fail closed at an eager host validation boundary."""

    if not bool(condition.all().item()):
        raise RuntimeError(message)


def _normalize_row_metadata(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 2:
        raise ValueError(
            f"exact sparse scores must be rank two, got {tuple(scores.shape)}"
        )
    if scores.dtype != torch.float32:
        raise TypeError(f"exact sparse scores must be float32, got {scores.dtype}")
    if lengths.ndim != 1 or lengths.shape[0] != scores.shape[0]:
        raise ValueError(
            "exact sparse lengths must contain exactly one value per score row"
        )
    lengths = lengths.to(device=scores.device, dtype=torch.int64)
    if row_starts is None:
        starts = torch.zeros_like(lengths)
    else:
        if row_starts.ndim != 1 or row_starts.shape != lengths.shape:
            raise ValueError(
                "exact sparse row_starts must contain exactly one value per score row"
            )
        starts = row_starts.to(device=scores.device, dtype=torch.int64)

    width = scores.shape[1]
    _require_host_true(starts >= 0, "exact sparse row starts must be non-negative")
    _require_host_true(lengths >= 0, "exact sparse row lengths must be non-negative")
    _require_host_true(
        starts + lengths <= width,
        "exact sparse legal score interval exceeds the score width",
    )
    return lengths, starts


def select_canonical_logical_topk(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    *,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select request-local logical keys with the stable sparse ordering contract.

    Legal scores for row ``r`` occupy ``[row_starts[r], row_starts[r] +
    lengths[r])`` in ``scores``. Winners are chosen by descending score, with
    lower request-local logical index winning an exact tie. Returned winners
    are then sorted into ascending request-local logical order and padded with
    ``-1``.
    """

    if topk <= 0:
        raise ValueError(f"exact sparse topk must be positive, got {topk}")
    lengths, starts = _normalize_row_metadata(scores, lengths, row_starts)
    rows, width = scores.shape

    columns = torch.arange(width, device=scores.device, dtype=torch.int64)
    columns = columns.unsqueeze(0).expand(rows, -1)
    legal = (columns >= starts.unsqueeze(1)) & (
        columns < (starts + lengths).unsqueeze(1)
    )
    _require_host_true(
        ~(torch.isnan(scores) & legal),
        "exact sparse selector encountered NaN in a legal score",
    )

    take = min(topk, width)
    if take == 0:
        return torch.full((rows, topk), -1, device=scores.device, dtype=torch.int32)

    # Top-k supplies only the boundary score. Its arbitrary choice among equal
    # values is discarded: values above the boundary always win, and a prefix
    # count over exact boundary ties chooses the lower logical keys needed to
    # fill the row. This preserves the stable lexicographic contract without a
    # full O(width log width) sort of every production row.
    legal_scores = scores.masked_fill(~legal, float("-inf"))
    boundary_score = torch.topk(
        legal_scores, take, dim=1, largest=True, sorted=False
    ).values.amin(dim=1, keepdim=True)
    above_boundary = legal & (scores > boundary_score)
    at_boundary = legal & (scores == boundary_score)

    selected_counts = lengths.clamp(max=take)
    boundary_needed = (
        selected_counts - above_boundary.sum(dim=1, dtype=torch.int64)
    ).clamp(min=0)
    boundary_rank = torch.cumsum(at_boundary, dim=1, dtype=torch.int64)
    selected_at_boundary = at_boundary & (boundary_rank <= boundary_needed.unsqueeze(1))
    full_rows = lengths >= take
    selected = torch.where(
        full_rows.unsqueeze(1), above_boundary | selected_at_boundary, legal
    )

    # Selected logical columns are unique. Top-k over these integer columns is
    # therefore only a canonical ascending pack; repeated sentinel values are
    # semantically identical and become suffix padding below.
    logical_columns = columns - starts.unsqueeze(1)
    sentinel = width + 1
    selected_columns = torch.where(selected, logical_columns, sentinel)
    canonical = torch.topk(
        selected_columns, take, dim=1, largest=False, sorted=True
    ).values
    selected_slots = torch.arange(take, device=scores.device).unsqueeze(0)
    canonical = torch.where(
        selected_slots < selected_counts.unsqueeze(1), canonical, -1
    )

    if take < topk:
        canonical = torch.nn.functional.pad(canonical, (0, topk - take), value=-1)
    return canonical.to(torch.int32)


def validate_canonical_logical_indices(
    selected_logical: torch.Tensor,
    *,
    max_logical_width: int,
) -> torch.Tensor:
    """Validate canonical ascending logical rows and return selected counts."""

    if selected_logical.ndim != 2:
        raise ValueError(
            "exact sparse selected logical indices must have query and top-k dimensions"
        )
    if selected_logical.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "exact sparse selected logical indices must use an integer dtype"
        )
    if max_logical_width < 0:
        raise ValueError("exact sparse logical width must be non-negative")

    _require_host_true(
        selected_logical >= -1,
        "exact sparse selected logical padding must use exactly -1",
    )
    valid = selected_logical >= 0
    _require_host_true(
        (~valid).cummax(dim=1).values.logical_and(valid).logical_not(),
        "exact sparse selected logical indices must have suffix-only padding",
    )
    _require_host_true(
        (~valid) | (selected_logical < max_logical_width),
        "exact sparse selected logical index exceeds the request page table",
    )
    if selected_logical.shape[1] > 1:
        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        strictly_ascending = selected_logical[:, 1:] > selected_logical[:, :-1]
        _require_host_true(
            (~adjacent_valid) | strictly_ascending,
            "exact sparse selected logical indices must be unique and ascending",
        )
    return valid.sum(dim=1, dtype=torch.int32)


@dataclass(frozen=True)
class PackedSelectedKV:
    kv: torch.Tensor
    compact_indices: torch.Tensor
    physical_indices: torch.Tensor
    selected_counts: torch.Tensor


def pack_selected_kv_dynamic(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    selected_logical: torch.Tensor,
) -> PackedSelectedKV:
    """Copy selected physical KV rows into canonical request-local order."""

    if kv_cache.ndim < 2:
        raise ValueError(
            "exact sparse KV cache must have a row dimension and payload dimensions"
        )
    if page_table.ndim != 2:
        raise ValueError("exact sparse page table must be rank two")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("exact sparse page table must use an integer dtype")
    if (
        page_table.device != kv_cache.device
        or selected_logical.device != kv_cache.device
    ):
        raise ValueError(
            "exact sparse KV cache, page table, and selection must share a device"
        )
    if page_table.shape[0] != selected_logical.shape[0]:
        raise ValueError(
            "exact sparse decode requires one page-table row per selected-logical row"
        )
    if not kv_cache.is_contiguous():
        raise RuntimeError(
            "exact sparse selected-KV packing requires contiguous cache rows; refusing "
            "an implicit full-cache copy"
        )
    if kv_cache.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "exact sparse dynamic selected-KV packing is eager-only; CUDA graph capture is unsupported"
        )

    selected_counts = validate_canonical_logical_indices(
        selected_logical, max_logical_width=page_table.shape[1]
    )
    valid = selected_logical >= 0
    physical = torch.gather(
        page_table.to(torch.int64), 1, selected_logical.clamp(min=0).to(torch.int64)
    )
    _require_host_true(
        (~valid) | ((physical >= 0) & (physical < kv_cache.shape[0])),
        "exact sparse selected logical key maps to an invalid physical KV row",
    )

    flat_physical = physical.masked_select(valid)
    # Index the storage bytes, not arithmetic values. This keeps packing a
    # layout operation for BF16 and native FP8 cache formats alike.
    kv_bytes = kv_cache.view(torch.uint8)
    packed_bytes = torch.index_select(kv_bytes, 0, flat_physical)
    packed_kv = packed_bytes.view(kv_cache.dtype).view(
        flat_physical.shape[0], *kv_cache.shape[1:]
    )

    row_offsets = torch.cumsum(selected_counts.to(torch.int64), dim=0) - selected_counts
    compact_positions = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    compact_indices = row_offsets.unsqueeze(1) + compact_positions
    compact_indices = torch.where(valid, compact_indices, -1).to(torch.int32)

    return PackedSelectedKV(
        kv=packed_kv,
        compact_indices=compact_indices,
        physical_indices=flat_physical,
        selected_counts=selected_counts,
    )


@dataclass(frozen=True)
class Glm52SparseLayerReceipt:
    layer_id: int
    producer_layer: int
    query_rows: int
    topk: int
    selected_rows: int
    selector_contract: str = _SELECTOR_CONTRACT
    packer_contract: str = _PACKER_CONTRACT
    attention_backend: str = "flashmla_sparse"


class Glm52SparseReceiptBook:
    """Fail-closed model-forward engagement receipts for exact sparse decode."""

    def __init__(self) -> None:
        self._expected_producers: dict[int, int] | None = None
        self._expected_full_layers: frozenset[int] = frozenset()
        self._selector: dict[int, tuple[int, int, int]] = {}
        self._packer: dict[int, tuple[int, int, int]] = {}
        self.last_receipts: tuple[Glm52SparseLayerReceipt, ...] = ()

    @property
    def active(self) -> bool:
        return self._expected_producers is not None

    @contextmanager
    def invocation(
        self,
        *,
        producer_by_layer: Mapping[int, int],
        full_layers: Sequence[int],
    ) -> Iterator[None]:
        if self.active:
            raise RuntimeError(
                "exact sparse receipt scope cannot overlap another forward"
            )
        self._expected_producers = dict(producer_by_layer)
        self._expected_full_layers = frozenset(full_layers)
        self._selector.clear()
        self._packer.clear()
        self.last_receipts = ()
        try:
            yield
            self._finish()
        finally:
            self._expected_producers = None
            self._expected_full_layers = frozenset()
            self._selector.clear()
            self._packer.clear()

    def record_selector(self, layer_id: int, selected: torch.Tensor) -> None:
        self._require_active()
        if layer_id in self._selector:
            raise RuntimeError(
                f"exact sparse selector engaged twice at layer {layer_id}"
            )
        if layer_id not in self._expected_full_layers:
            raise RuntimeError(
                f"exact sparse selector engaged on shared layer {layer_id}"
            )
        counts = (selected >= 0).sum(dim=1)
        self._selector[layer_id] = (
            selected.shape[0],
            selected.shape[1],
            int(counts.sum().item()),
        )

    def record_packer(self, layer_id: int, packed: PackedSelectedKV) -> None:
        self._require_active()
        if layer_id in self._packer:
            raise RuntimeError(
                f"exact sparse selected-KV packer engaged twice at layer {layer_id}"
            )
        if layer_id not in self._expected_producers:
            raise RuntimeError(
                f"exact sparse selected-KV packer engaged on unexpected layer {layer_id}"
            )
        self._packer[layer_id] = (
            packed.compact_indices.shape[0],
            packed.compact_indices.shape[1],
            int(packed.selected_counts.sum().item()),
        )

    def _finish(self) -> None:
        assert self._expected_producers is not None
        selector_layers = frozenset(self._selector)
        packer_layers = frozenset(self._packer)
        expected_layers = frozenset(self._expected_producers)
        if selector_layers != self._expected_full_layers:
            raise RuntimeError(
                "exact sparse selector engagement mismatch: "
                f"expected {sorted(self._expected_full_layers)}, got {sorted(selector_layers)}"
            )
        if packer_layers != expected_layers:
            raise RuntimeError(
                "exact sparse selected-KV engagement mismatch: "
                f"expected {sorted(expected_layers)}, got {sorted(packer_layers)}"
            )

        receipts = []
        for layer_id in sorted(expected_layers):
            producer = self._expected_producers[layer_id]
            selected_shape = self._selector[producer]
            packed_shape = self._packer[layer_id]
            if packed_shape != selected_shape:
                raise RuntimeError(
                    f"exact sparse layer {layer_id} packed state disagrees with producer {producer}"
                )
            receipts.append(
                Glm52SparseLayerReceipt(
                    layer_id=layer_id,
                    producer_layer=producer,
                    query_rows=packed_shape[0],
                    topk=packed_shape[1],
                    selected_rows=packed_shape[2],
                )
            )
        self.last_receipts = tuple(receipts)

    def _require_active(self) -> None:
        if not self.active:
            raise RuntimeError(
                "exact sparse engagement occurred outside a model-forward receipt scope"
            )


def log_glm52_sparse_receipt_once(
    receipts: Sequence[Glm52SparseLayerReceipt],
    *,
    start_layer: int,
    end_layer: int,
    request_ids: Sequence[str] | None,
    request_count: int,
    receipt_logger: logging.Logger,
) -> None:
    """Emit one receipt for each distinct co-resident request set."""

    expected_layers = list(range(start_layer, end_layer))
    if [receipt.layer_id for receipt in receipts] != expected_layers:
        raise RuntimeError(
            "exact sparse receipt layer range does not match the pipeline stage"
        )
    if not receipts:
        raise RuntimeError("exact sparse receipt cannot be empty")
    if any(
        receipt.selector_contract != _SELECTOR_CONTRACT
        or receipt.packer_contract != _PACKER_CONTRACT
        or receipt.attention_backend != "flashmla_sparse"
        for receipt in receipts
    ):
        raise RuntimeError(
            "exact sparse receipt contains an unexpected numerical contract"
        )
    query_rows = {receipt.query_rows for receipt in receipts}
    topks = {receipt.topk for receipt in receipts}
    if len(query_rows) != 1 or len(topks) != 1:
        raise RuntimeError("exact sparse receipt shapes differ across attention layers")
    query_rows = query_rows.pop()
    topk = topks.pop()
    if (
        request_count <= 0
        or request_ids is None
        or len(request_ids) != request_count
        or not request_ids
        or any(
            not isinstance(request_id, str) or not request_id
            for request_id in request_ids
        )
        or len(set(request_ids)) != len(request_ids)
    ):
        raise RuntimeError(
            "exact sparse receipt requires a non-empty unique request set"
        )
    request_set_sha256 = hashlib.sha256(
        b"\x00".join(request_id.encode() for request_id in sorted(request_ids))
    ).hexdigest()
    identity = (start_layer, end_layer, request_set_sha256)
    if identity in _RECEIPT_IDENTITIES_LOGGED:
        return
    receipt_logger.info(
        "GLM-5.2 exact sparse decode receipt: layer_range=%s:%s "
        "attention_layers=%s full_producers=%s requests=%s query_rows=%s topk=%s request_set_sha256=%s "
        "selected_rows_min=%s selected_rows_max=%s selector=%s packer=%s "
        "attention_backend=flashmla_sparse cuda_graph=false",
        start_layer,
        end_layer,
        len(receipts),
        len({receipt.producer_layer for receipt in receipts}),
        len(request_ids),
        query_rows,
        topk,
        request_set_sha256,
        min(receipt.selected_rows for receipt in receipts),
        max(receipt.selected_rows for receipt in receipts),
        _SELECTOR_CONTRACT,
        _PACKER_CONTRACT,
    )
    _RECEIPT_IDENTITIES_LOGGED.add(identity)


__all__ = [
    "GLM52_SELECTOR_CONTRACT_VERSION",
    "Glm52SparseLayerReceipt",
    "Glm52SparseReceiptBook",
    "PackedSelectedKV",
    "pack_selected_kv_dynamic",
    "log_glm52_sparse_receipt_once",
    "select_canonical_logical_topk",
    "validate_canonical_logical_indices",
]
