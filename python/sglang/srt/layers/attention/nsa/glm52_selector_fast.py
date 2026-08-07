"""Fused fast path for the exact GLM-5.2 sparse-selection contract.

Implements the same selection semantics as
:func:`sglang.srt.layers.attention.nsa.glm52_selector.select_canonical_logical_topk`
(the reference implementation, which stays the in-tree oracle): winners by
descending float32 score, exact ties choose the smaller request-local logical
index, output ascending, ``-1`` suffix padding, int32. One Triton program per
query row replaces the eager post-score chain (masked_fill, float top-k, amin,
compares, width-length int64 cumsum, int64 top-k, pad, cast).

Every output is an integer index; the only floating-point operations are
comparisons on the literal score bytes, so launch configuration and tiling
cannot change output bytes. Signed zeros are canonicalized (``-0.0 -> +0.0``)
before the monotonic key transform so the IEEE ``==`` tie class at zero stays a
single key class. A NaN inside a row's legal interval is out-of-contract: the
kernel raises a device-side per-row flag instead of reproducing the oracle's
unspecified full-row behavior; NaN-free inputs match the oracle bitwise.

All passes are bounded to each row's legal interval ``[start, end)``: the
production decode allocation is far wider than the legal interval (max-context
logits behind a ~4k live prefix), and sweeping the allocation instead of the
interval costs two orders of magnitude while changing no bytes.

These kernels are the exact serving contract's default execution: the
architecture resolver selects them whenever the exact sparse path serves.
The eager reference implementations in ``glm52_selector`` remain in-tree
solely as the certification baselines the test suite compares against;
no production flag dispatches them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention.nsa.glm52_selector import PackedSelectedKV


@triton.jit
def _glm52_keys(x, legal):
    """Monotonic integer key: larger float => larger key; excluded => -1."""

    xz = tl.where(x == 0.0, 0.0, x)
    is_nan = x != x
    bits = xz.to(tl.int32, bitcast=True)
    flipped = tl.where(bits < 0, ~bits, bits | (-2147483648))
    key = flipped.to(tl.int64) & 0xFFFFFFFF
    return tl.where(legal & (~is_nan), key, -1), is_nan


@triton.jit
def _glm52_stable_topk_kernel(
    scores_ptr,
    lengths_ptr,
    starts_ptr,
    out_ptr,
    flag_ptr,
    width,
    take,
    stride_sr,
    stride_or,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts_ptr + row).to(tl.int64)
    length = tl.load(lengths_ptr + row).to(tl.int64)
    end = start + length
    c_lo = start // CHUNK
    c_hi = tl.minimum(tl.cdiv(end, CHUNK), tl.cdiv(width, CHUNK))
    full = length >= take

    # Pass 1: NaN-in-legal contract flag.
    nan_seen = tl.zeros((), dtype=tl.int64)
    for c in tl.range(c_lo, c_hi):
        cols = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
        m = cols < width
        x = tl.load(scores_ptr + row * stride_sr + cols, mask=m, other=0.0)
        legal = (cols >= start) & (cols < end) & m
        key_p1, is_nan = _glm52_keys(x, legal)
        nan_seen += tl.sum((legal & is_nan).to(tl.int64))
    # Unconditional status write: a reused (capture-owned) buffer must never
    # carry a stale flag from a previous invocation.
    tl.store(flag_ptr + row, (nan_seen > 0).to(tl.int32))

    # Pass 2: boundary key V = take-th largest legal key (full rows only).
    # V is the smallest v with count(key > v) < take; pure counting, no
    # arithmetic on scores, so V is bit-identical to topk().values.amin().
    lo = tl.zeros((), dtype=tl.int64)
    hi = tl.full((), 4294967295, dtype=tl.int64)
    if full:
        for it in tl.range(0, 32):
            mid = (lo + hi) // 2
            cnt = tl.zeros((), dtype=tl.int64)
            for c in tl.range(c_lo, c_hi):
                cols = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
                m = cols < width
                x = tl.load(scores_ptr + row * stride_sr + cols, mask=m, other=0.0)
                legal = (cols >= start) & (cols < end) & m
                key, nan_p2 = _glm52_keys(x, legal)
                cnt += tl.sum((key > mid).to(tl.int64))
            found = cnt < take
            hi = tl.where(found, mid, hi)
            lo = tl.where(found, lo, mid + 1)
    boundary = hi

    # Pass 3: strict-above count A and tie budget B = take - A.
    above_total = tl.zeros((), dtype=tl.int64)
    if full:
        for c in tl.range(c_lo, c_hi):
            cols = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
            m = cols < width
            x = tl.load(scores_ptr + row * stride_sr + cols, mask=m, other=0.0)
            legal = (cols >= start) & (cols < end) & m
            key, nan_p3 = _glm52_keys(x, legal)
            above_total += tl.sum((key > boundary).to(tl.int64))
    tie_budget = take - above_total

    # Pass 4: ordered emit. Ascending column sweep with carried prefix counts:
    # winners land at their final ascending positions in one pass.
    tie_carry = tl.zeros((), dtype=tl.int64)
    win_carry = tl.zeros((), dtype=tl.int64)
    for c in tl.range(c_lo, c_hi):
        cols = c * CHUNK + tl.arange(0, CHUNK).to(tl.int64)
        m = cols < width
        x = tl.load(scores_ptr + row * stride_sr + cols, mask=m, other=0.0)
        legal = (cols >= start) & (cols < end) & m
        key, nan_p4 = _glm52_keys(x, legal)
        tie = legal & (key == boundary)
        tie_rank = tie_carry + tl.cumsum(tie.to(tl.int64), axis=0)
        winner_full = (key > boundary) | (tie & (tie_rank <= tie_budget))
        winner = tl.where(full, winner_full, legal)
        pos = win_carry + tl.cumsum(winner.to(tl.int64), axis=0) - 1
        logical = (cols - start).to(tl.int32)
        emit = winner & (pos < take)
        tl.store(out_ptr + row * stride_or + pos, logical, mask=emit)
        tie_carry += tl.sum(tie.to(tl.int64))
        win_carry += tl.sum(winner.to(tl.int64))


def select_canonical_logical_topk_fused(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    *,
    row_starts: torch.Tensor | None = None,
    nan_flags_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused equivalent of the reference selector (trusted-input form).

    Returns ``(canonical, nan_flags)``. ``canonical`` matches the oracle
    bitwise for every input whose legal intervals are NaN-free. ``nan_flags``
    is int32 ``[rows]``, nonzero where a legal NaN was seen (out-of-contract;
    the caller defers the check to a step boundary, never inside capture).
    """

    if topk <= 0:
        raise ValueError(f"exact sparse fused topk must be positive, got {topk}")
    if scores.ndim != 2:
        raise ValueError(
            f"exact sparse fused scores must be rank two, got {tuple(scores.shape)}"
        )
    if scores.dtype != torch.float32:
        raise TypeError(
            f"exact sparse fused scores must be float32, got {scores.dtype}"
        )
    rows, width = scores.shape
    lengths = lengths.to(device=scores.device, dtype=torch.int64)
    starts = (
        torch.zeros_like(lengths)
        if row_starts is None
        else row_starts.to(device=scores.device, dtype=torch.int64)
    )
    if nan_flags_out is not None:
        flags = nan_flags_out
    else:
        flags = torch.zeros(max(rows, 1), device=scores.device, dtype=torch.int32)
    out = torch.full((rows, topk), -1, device=scores.device, dtype=torch.int32)
    take = min(topk, width)
    if take == 0 or rows == 0:
        return out, flags

    scores = scores.contiguous()
    _glm52_stable_topk_kernel[(rows,)](
        scores,
        lengths,
        starts,
        out,
        flags,
        width,
        take,
        scores.stride(0),
        out.stride(0),
        CHUNK=2048,
        num_warps=8,
    )
    return out, flags


@dataclass(frozen=True)
class Glm52SelectionPlan:
    """Producer-scoped selected-KV plan shared by all consumer layers."""

    physical_padded: torch.Tensor  # int32 [M, topk]; -1 where invalid
    selected_counts: torch.Tensor  # int32 [M]
    contract_ok: torch.Tensor  # bool scalar, device-side


def build_selection_plan(
    page_table: torch.Tensor,
    selected_logical: torch.Tensor,
    *,
    kv_cache_rows: int,
) -> Glm52SelectionPlan:
    """Translate one producer's canonical selection to physical KV rows.

    Byte-identical to the per-layer derivations inside
    ``pack_selected_kv_static`` for the tensors it produces; computed once per
    producer instead of once per consumer layer. Graph-safe: fixed shapes, no
    host synchronization; ``contract_ok`` is a deferred device scalar.
    """

    if page_table.ndim != 2 or selected_logical.ndim != 2:
        raise ValueError(
            "exact sparse selection plan requires rank-two page/selection tables"
        )
    if page_table.shape[0] != selected_logical.shape[0]:
        raise ValueError(
            "exact sparse selection plan requires one page-table row per request"
        )

    valid = selected_logical >= 0
    selected_counts = valid.sum(dim=1, dtype=torch.int32)

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

    safe_logical = selected_logical.clamp(min=0, max=page_table.shape[1] - 1).to(
        torch.int64
    )
    physical = torch.gather(page_table.to(torch.int64), 1, safe_logical)
    physical_in_range = (
        (~valid) | ((physical >= 0) & (physical < kv_cache_rows))
    ).all()
    contract_ok = suffix_only & logical_in_range & ascending & physical_in_range
    physical_padded = torch.where(valid, physical, -1).to(torch.int32)

    return Glm52SelectionPlan(
        physical_padded=physical_padded,
        selected_counts=selected_counts,
        contract_ok=contract_ok,
    )


@dataclass(frozen=True)
class Glm52ProducerSelectionPlan:
    """Producer-published plan: built once per producer, read by its consumers.

    Carried on the per-forward ``CanonicalLogicalIndices`` payload, so its
    lifetime is the IndexShare context's. The build executes as ordinary
    kernels inside the forward — capture records the builders and every replay
    recomputes plan values from the live page-table/selection buffers; nothing
    is memoized Python-side.
    """

    plan: Glm52SelectionPlan
    kv_cache_rows: int
    rows: int


@triton.jit
def _glm52_pack_selected_kv_kernel(
    sel_ptr,
    phys_ptr,
    counts_ptr,
    plan_ok_ptr,
    kv_ptr,
    out_kv_ptr,
    out_compact_ptr,
    out_ok_ptr,
    n_rows,
    topk,
    n_tiles,
    max_sel,
    cache_rows,
    stride_sel,
    stride_phys,
    stride_compact,
    SLOTS: tl.constexpr,
    LANES: tl.constexpr,
    ROW_U64: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row = pid // n_tiles
    tile = pid % n_tiles

    # Destination base: exclusive prefix sum of the producer-invariant counts.
    m_idx = tl.arange(0, BLOCK_M).to(tl.int64)
    cvec = tl.load(counts_ptr + m_idx, mask=m_idx < n_rows, other=0).to(tl.int64)
    row_off = tl.sum(tl.where(m_idx < row, cvec, 0))
    if pid == 0:
        total = tl.sum(cvec)
        ok = (tl.load(plan_ok_ptr) != 0) & (total <= max_sel)
        tl.store(out_ok_ptr, ok.to(tl.uint8))

    cols = tile * SLOTS + tl.arange(0, SLOTS).to(tl.int64)
    cmask = cols < topk
    sel = tl.load(sel_ptr + row * stride_sel + cols, mask=cmask, other=-1)
    valid = cmask & (sel >= 0)
    # Canonical suffix padding makes the column ordinal the compact position.
    dest = row_off + cols
    live = valid & (dest < max_sel)
    tl.store(
        out_compact_ptr + row * stride_compact + cols,
        tl.where(live, dest, -1).to(tl.int32),
        mask=cmask,
    )
    if tl.sum(live.to(tl.int32)) > 0:
        phys = tl.load(phys_ptr + row * stride_phys + cols, mask=cmask, other=0).to(
            tl.int64
        )
        src = tl.minimum(tl.maximum(phys, 0), cache_rows - 1)
        lanes = tl.arange(0, LANES).to(tl.int64)
        copy = live[:, None] & (lanes < ROW_U64)[None, :]
        vals = tl.load(kv_ptr + src[:, None] * ROW_U64 + lanes[None, :], mask=copy)
        tl.store(out_kv_ptr + dest[:, None] * ROW_U64 + lanes[None, :], vals, mask=copy)


def pack_selected_kv_fused(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    selected_logical: torch.Tensor,
    *,
    max_selected_tokens: int,
    plan: Glm52SelectionPlan | None = None,
    out_kv: torch.Tensor | None = None,
    out_compact: torch.Tensor | None = None,
    out_contract: torch.Tensor | None = None,
) -> PackedSelectedKV:
    """One fused copy kernel replacing the eager ``pack_selected_kv_static``.

    Byte contract, gated against the reference oracle: ``kv`` rows
    ``[0, sum(selected_counts))`` byte-identical; ``compact_indices``,
    ``selected_counts``, and ``contract_ok`` semantics identical (plan checks
    plus the capacity check). Sources are clamped to the cache interval exactly
    as the oracle clamps, so out-of-contract physical values reproduce oracle
    bytes while ``contract_ok`` reads false.

    DEAD REGION POLICY: rows in ``[sum(counts), max_selected_tokens)`` are
    never written — no zero fill, no trash rows. FlashMLA addresses only the
    live prefix through ``compact_indices``/cu_seqlens, so those bytes are
    unspecified; a reused output workspace may hold stale rows there by
    design (captured-workspace doctrine). Consumers must never read past the
    live prefix.

    Determinism: a pure position-wise byte copy — every destination element is
    written by exactly one program with one loaded value, and there is no
    accumulation, so launch configuration cannot change output bytes.

    Graph-safe: fixed shapes keyed by ``(M, topk, row_bytes)``, no host sync;
    grid and constants derive from static shapes only. Pass a precomputed
    ``plan`` (one per producer) to hoist the page-table translation, counts,
    and canonical checks out of the consumer layers; pass ``out_*``
    workspaces for capture-owned reuse (``compact_indices`` and the contract
    flag are fully rewritten each call; only the ``kv`` dead region is stale).

    ``physical_indices`` is populated from ``plan.physical_padded`` (``-1`` at
    invalid slots); no consumer reads it in packed mode.
    """

    if kv_cache.ndim < 2 or not kv_cache.is_contiguous():
        raise RuntimeError(
            "exact sparse fused selected-KV packing requires contiguous cache rows"
        )
    if page_table.ndim != 2 or selected_logical.ndim != 2:
        raise ValueError(
            "exact sparse fused packing requires rank-two page and selection tables"
        )
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("exact sparse page table must use an integer dtype")
    if selected_logical.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "exact sparse selected logical indices must use an integer dtype"
        )
    if page_table.shape[0] != selected_logical.shape[0]:
        raise ValueError(
            "exact sparse fused packing requires one page-table row per request"
        )
    if not (page_table.device == selected_logical.device == kv_cache.device):
        raise ValueError("exact sparse fused packing tensors must share one device")
    if page_table.shape[1] < 1 or kv_cache.shape[0] < 1 or max_selected_tokens < 1:
        raise ValueError(
            "exact sparse fused packing requires positive storage capacities"
        )
    if selected_logical.stride(1) != 1:
        raise RuntimeError(
            "exact sparse fused packing requires unit-stride selection columns; "
            "an implicit contiguous() copy must not hide in integration"
        )

    rows, topk = selected_logical.shape
    row_bytes = kv_cache[0].numel() * kv_cache.element_size()
    if row_bytes % 8 != 0:
        raise RuntimeError(
            f"exact sparse fused packing requires 8-byte-divisible KV rows, "
            f"got {row_bytes}"
        )
    row_u64 = row_bytes // 8

    if plan is None:
        plan = build_selection_plan(
            page_table, selected_logical, kv_cache_rows=kv_cache.shape[0]
        )
    if plan.physical_padded.shape != selected_logical.shape:
        raise ValueError(
            "exact sparse fused packing plan does not match the selection shape"
        )
    if plan.physical_padded.stride(1) != 1:
        raise RuntimeError(
            "exact sparse fused packing requires unit-stride plan columns"
        )

    if out_kv is None:
        out_kv = torch.empty(
            (max_selected_tokens, *kv_cache.shape[1:]),
            dtype=kv_cache.dtype,
            device=kv_cache.device,
        )
    elif (
        out_kv.shape != (max_selected_tokens, *kv_cache.shape[1:])
        or out_kv.dtype != kv_cache.dtype
        or not out_kv.is_contiguous()
    ):
        raise ValueError(
            "exact sparse fused packing out_kv workspace has the wrong layout"
        )
    if out_compact is None:
        out_compact = torch.empty(
            (rows, topk), dtype=torch.int32, device=kv_cache.device
        )
    elif (
        out_compact.shape != (rows, topk)
        or out_compact.dtype != torch.int32
        or out_compact.stride(1) != 1
    ):
        raise ValueError(
            "exact sparse fused packing out_compact workspace has the wrong layout"
        )
    if out_contract is None:
        out_contract = torch.empty((), dtype=torch.bool, device=kv_cache.device)
    elif out_contract.shape != () or out_contract.dtype != torch.bool:
        raise ValueError(
            "exact sparse fused packing out_contract workspace has the wrong layout"
        )

    SLOTS = 16
    n_tiles = triton.cdiv(topk, SLOTS)
    _glm52_pack_selected_kv_kernel[(rows * n_tiles,)](
        selected_logical,
        plan.physical_padded,
        plan.selected_counts,
        plan.contract_ok.view(torch.uint8),
        kv_cache.view(torch.int64),
        out_kv.view(torch.int64),
        out_compact,
        out_contract.view(torch.uint8),
        rows,
        topk,
        n_tiles,
        max_selected_tokens,
        kv_cache.shape[0],
        selected_logical.stride(0),
        plan.physical_padded.stride(0),
        out_compact.stride(0),
        SLOTS=SLOTS,
        LANES=triton.next_power_of_2(row_u64),
        ROW_U64=row_u64,
        BLOCK_M=triton.next_power_of_2(rows),
        num_warps=8,
    )
    return PackedSelectedKV(
        kv=out_kv,
        compact_indices=out_compact,
        physical_indices=plan.physical_padded.reshape(-1),
        selected_counts=plan.selected_counts,
        contract_ok=out_contract,
    )


logger = logging.getLogger(__name__)

_applied = False


def _apply_glm52_exact_fastpath() -> None:
    """Install the exact GLM-5.2 implementation tuple once per worker.

    ``ModelRunner`` calls this private resolver after ``ServerArgs`` has
    selected GLM-5.2 exact mode and before constructing any attention backend.
    No model layer, user flag, or diagnostic API may partially select it.
    """
    global _applied
    if _applied:
        return
    from sglang.srt.batch_invariant_ops import bi_gemm_configs
    from sglang.srt.layers.xorl_batch_invariant import force_xorl_bi_family

    force_xorl_bi_family("v2")
    bi_gemm_configs._force_bi_gemm_config_table(True)
    bi_gemm_configs._set_glm52_tier_a_enabled(True)
    logger.info(
        "Exact GLM-5.2 zero-K3 serving resolved: fused selector+packer, "
        "producer-dataflow plans, canonical_v3b MoE transport "
        "(CP prefill: v3), families=v2, Tier-A GEMM tables, "
        "model-tail status check, prebuilt RoPE cache only"
    )
    _applied = True


__all__ = [
    "Glm52ProducerSelectionPlan",
    "Glm52SelectionPlan",
    "build_selection_plan",
    "pack_selected_kv_fused",
    "select_canonical_logical_topk_fused",
]
