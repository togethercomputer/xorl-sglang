"""Exact GLM-5.2 sparse selection and selected-KV contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
    validate: bool = True,
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
    if validate:
        lengths, starts = _normalize_row_metadata(scores, lengths, row_starts)
    else:
        if scores.ndim != 2 or lengths.ndim != 1 or lengths.shape[0] != scores.shape[0]:
            raise ValueError(
                "exact sparse trusted selector received invalid tensor ranks"
            )
        lengths = lengths.to(device=scores.device, dtype=torch.int64)
        starts = (
            torch.zeros_like(lengths)
            if row_starts is None
            else row_starts.to(device=scores.device, dtype=torch.int64)
        )
    rows, width = scores.shape

    columns = torch.arange(width, device=scores.device, dtype=torch.int64)
    columns = columns.unsqueeze(0).expand(rows, -1)
    legal = (columns >= starts.unsqueeze(1)) & (
        columns < (starts + lengths).unsqueeze(1)
    )
    if validate:
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
    contract_ok: torch.Tensor | None = None


def pack_selected_kv_dynamic(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    selected_logical: torch.Tensor,
    *,
    validate: bool = True,
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

    selected_counts = (
        validate_canonical_logical_indices(
            selected_logical, max_logical_width=page_table.shape[1]
        )
        if validate
        else (selected_logical >= 0).sum(dim=1, dtype=torch.int32)
    )
    valid = selected_logical >= 0
    physical = torch.gather(
        page_table.to(torch.int64), 1, selected_logical.clamp(min=0).to(torch.int64)
    )
    if validate:
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


def pack_selected_kv_static(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    selected_logical: torch.Tensor,
    *,
    max_selected_tokens: int,
) -> PackedSelectedKV:
    """Graph-safe fixed-shape form of :func:`pack_selected_kv_dynamic`.

    Live request rows are compacted into the leading logical interval consumed
    by FlashMLA. Every inactive or invalid slot receives a private trash row,
    so ``index_copy_`` has unique destinations and remains deterministic. The
    device scalar ``contract_ok`` records canonical-index, physical-page, and
    capacity validity without a host synchronization inside capture.
    """

    if kv_cache.ndim < 2 or not kv_cache.is_contiguous():
        raise RuntimeError(
            "exact sparse static selected-KV packing requires contiguous cache rows"
        )
    if page_table.ndim != 2 or selected_logical.ndim != 2:
        raise ValueError(
            "exact sparse static packing requires rank-two page and selection tables"
        )
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("exact sparse page table must use an integer dtype")
    if selected_logical.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "exact sparse selected logical indices must use an integer dtype"
        )
    if page_table.shape[0] != selected_logical.shape[0]:
        raise ValueError(
            "exact sparse static packing requires one page-table row per request"
        )
    if not (page_table.device == selected_logical.device == kv_cache.device):
        raise ValueError("exact sparse static packing tensors must share one device")
    if page_table.shape[1] < 1 or kv_cache.shape[0] < 1 or max_selected_tokens < 1:
        raise ValueError(
            "exact sparse static packing requires positive storage capacities"
        )

    valid = selected_logical >= 0
    selected_counts = valid.sum(dim=1, dtype=torch.int32)
    row_offsets = torch.cumsum(selected_counts.to(torch.int64), dim=0) - selected_counts
    compact_positions = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    destinations = row_offsets.unsqueeze(1) + compact_positions
    within_capacity = destinations < max_selected_tokens

    suffix_only = (~valid).cummax(dim=1).values.logical_and(valid).logical_not().all()
    logical_in_range = (
        (selected_logical >= -1) & ((~valid) | (selected_logical < page_table.shape[1]))
    ).all()
    if selected_logical.shape[1] > 1:
        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        ascending = (
            (~adjacent_valid) | (selected_logical[:, 1:] > selected_logical[:, :-1])
        ).all()
    else:
        ascending = torch.ones((), dtype=torch.bool, device=selected_logical.device)
    capacity_ok = selected_counts.sum(dtype=torch.int64) <= max_selected_tokens

    safe_logical = selected_logical.clamp(min=0, max=page_table.shape[1] - 1).to(
        torch.int64
    )
    physical = torch.gather(page_table.to(torch.int64), 1, safe_logical)
    physical_in_range = (
        (~valid) | ((physical >= 0) & (physical < kv_cache.shape[0]))
    ).all()
    contract_ok = (
        suffix_only & logical_in_range & ascending & capacity_ok & physical_in_range
    )

    slots = selected_logical.numel()
    slot_ids = torch.arange(
        slots, device=selected_logical.device, dtype=torch.int64
    ).view_as(selected_logical)
    trash = max_selected_tokens + slot_ids
    copy_destinations = torch.where(
        valid & within_capacity, destinations, trash
    ).reshape(-1)
    source_rows = physical.clamp(min=0, max=kv_cache.shape[0] - 1).reshape(-1)
    gathered = torch.index_select(kv_cache.view(torch.uint8), 0, source_rows)
    packed_storage = torch.zeros(
        (max_selected_tokens + slots, *kv_cache.view(torch.uint8).shape[1:]),
        dtype=torch.uint8,
        device=kv_cache.device,
    )
    packed_storage.index_copy_(0, copy_destinations, gathered)
    packed_kv = (
        packed_storage[:max_selected_tokens]
        .view(kv_cache.dtype)
        .view(max_selected_tokens, *kv_cache.shape[1:])
    )
    compact_indices = torch.where(valid & within_capacity, destinations, -1).to(
        torch.int32
    )
    return PackedSelectedKV(
        kv=packed_kv,
        compact_indices=compact_indices,
        physical_indices=physical.reshape(-1),
        selected_counts=selected_counts,
        contract_ok=contract_ok,
    )


__all__ = [
    "PackedSelectedKV",
    "pack_selected_kv_dynamic",
    "pack_selected_kv_static",
    "select_canonical_logical_topk",
    "validate_canonical_logical_indices",
]
