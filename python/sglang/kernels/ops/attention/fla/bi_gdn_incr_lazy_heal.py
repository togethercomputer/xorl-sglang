"""Batched extend-time cache heal for the incremental hybrid.

``the batched heal`` (default OFF; requires
``the INCR mode=1``) replaces the per-(request, layer)
host-serial ``BIGDNIncrDecodeRunner.warm_slot`` loop on the extend path with
ONE batched launch set per (layer, extend batch): the same stock stage
drivers/binaries warm_slot runs, driven over a gap-encoded varlen batch (the
certified the FAST path encoding, exactly the slab structure ``_step_incr`` and
``flush_slots`` already use), then one masked transport-commit kernel per
persistent cache tensor.

BYTE-NEUTRAL HOST WORK: outputs must be bit-identical to the per-request
path -- committed cache rows [:fill] byte-equal, rows >= fill untouched, no
pool tensor written (the heal writes ONLY the incremental row caches, never
``ssm_states``/``boundary``/``scratch``/``rows_*`` -- no new pool-copy call
site, the R4 poison discipline is not re-triggered).  The win is
host-sync/serialization removal:

  per (request, layer) flag-off: 2 pageable-H2D ``torch.tensor`` calls (cu,
  ci) + a HIDDEN D2H sync -- ``chunk_local_cumsum``'s ``**kwargs`` swallows
  the ``chunk_indices`` warm_slot passes (cumsum.py), so
  ``prepare_chunk_indices`` re-derives them via ``.tolist()`` on a device
  tensor (+ its H2D upload); ~24 host-serial launches.

  per (layer, extend batch) flag-on: ONE pageable-H2D upload (slots+fills
  packed) + ~50 launches, R-invariant (census: 3,969 -> 63 sync-forcing
  calls and 7,688 -> 1,519 launches per 8-request x 31-layer window, 63x /
  5.1x, bytes equal).  cu/ci/chunk_offsets are built on device;
  ``chunk_indices`` is passed EXPLICITLY to every driver and the cumsum
  stage launches its kernel directly (the flush_slots precedent), because
  the stock ``prepare_chunk_indices`` must never see a gap-encoded cu (it
  renumbers nonempty segments).

LIFETIME: the batched heal runs eagerly and synchronously inside the same
``_bi_gdn_decode_seed`` call that runs warm_slot today -- no heal state
survives the call, so there is NO new host-state seam and no new
capture-lifetime exposure (cache validity remains device-tensor content
refreshed at fixed addresses between replays, the seed_from_extend class).
The fully-lazy variant (defer past the seed call) is NOT built here.

The batched and per-request forms are byte-compared across batch composition,
defer mode, and solve dispatch, including untouched cache tails and pools.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.fla.bi_gdn_prefill import (
    BI_GDN_SOLVE_TRIL_DECODE,
    chunk_gated_delta_rule_fwd_h,
    solve_tril,
)
from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE
from sglang.kernels.ops.attention.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum_scalar_kernel
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.fla.solve_tril_decode import solve_tril_decode
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd

# Internal choice for the exact Qwen3.5-family GDN serving contract.
# Installed by the architecture resolver (see
# sglang.kernels.ops.attention.fla.qwen35_gdn_exact) -- there are no
# per-feature environment variables on this surface. The False defaults
# keep every non-contract server on the stock path, bit-for-bit
# unaffected; tests may set these module attributes directly on a fresh
# runner (never after a capture).
# The heal batches INCR's warm_slot and is meaningless without INCR -- the
# architecture resolver selects them together by construction.
BI_GDN_LAZY_HEAL_ENABLED = False


@triton.jit
def bi_gdn_lazy_heal_commit_kernel(
    src,  # [n*CHUNK, ELEM] slab rows (request-major, gap layout)
    dst,  # [S*CHUNK, ELEM] persistent cache rows (slot-major)
    slot_indices,  # [n] int64
    fills,  # [n] int32
    ELEM,  # runtime: elements per row (dtype from the pointers)
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Masked row transport: cache[slot, r] <- slab[i_b, r] for r < fill.

    Pure byte movement (integer-surface class): rows >= fill are never
    stored, so cache tails keep their bytes exactly like the per-request
    ``state.x[slot, :fill] = slab`` slice-assign it replaces.
    """
    i_b, i_r = tl.program_id(0), tl.program_id(1)
    fill = tl.load(fills + i_b)
    if i_r < fill:
        slot = tl.load(slot_indices + i_b)
        src_row = src + (i_b.to(tl.int64) * CHUNK + i_r) * ELEM
        dst_row = dst + (slot.to(tl.int64) * CHUNK + i_r) * ELEM
        for i in range(0, tl.cdiv(ELEM, BLOCK)):
            o = i * BLOCK + tl.arange(0, BLOCK)
            m = o < ELEM
            tl.store(dst_row + o, tl.load(src_row + o, mask=m, other=0), mask=m)


def _commit_rows(
    slab: torch.Tensor,
    cache_rows: torch.Tensor,
    idx: torch.Tensor,
    fills: torch.Tensor,
    n: int,
) -> None:
    elem = slab.numel() // (n * CHUNK_SIZE)
    assert cache_rows.is_contiguous() and slab.is_contiguous()
    bi_gdn_lazy_heal_commit_kernel[(n, CHUNK_SIZE)](
        slab.view(n * CHUNK_SIZE, elem),
        cache_rows.view(-1, elem),
        idx,
        fills,
        elem,
        CHUNK=CHUNK_SIZE,
        BLOCK=1024,
        num_warps=4,
    )


@torch.no_grad()
def warm_slots_batched(runner, cache, pairs) -> None:
    """Batched equivalent of ``runner.warm_slot(cache, slot, fill)`` over all
    ``(slot, fill)`` pairs of one layer's extend batch.

    Same stock stage drivers/binaries as warm_slot, over ONE gap-encoded
    varlen batch (request i owns physical rows [64i, 64i+64) with logical
    extent fill_i; odd cu segments are the certified zero-length gaps).
    Committed bytes are covered by direct comparison with the per-request
    path. Eager only -- never under capture.
    """
    todo = [(int(s), int(f)) for s, f in pairs if int(f) > 0 and int(s) >= 0]
    if not todo:
        return
    if len(todo) == 1:
        # Nothing to batch across: dispatch to the FROZEN per-request path
        # (identical bytes by definition; avoids the batched path's fixed
        # 64-row slab cost -- the census fixture measured batched-at-n=1
        # marginally slower in wall exposure while flag-off is certified).
        runner.warm_slot(cache, todo[0][0], todo[0][1])
        return
    _warm_slots_batched_impl(runner, cache, todo)


@torch.no_grad()
def _warm_slots_batched_impl(runner, cache, todo) -> None:
    """The batched heal body (n >= 1 sanitized pairs).  Exposed separately so
    component tests can exercise the batched machinery at n=1 even though production
    dispatches single-request batches to warm_slot."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "the batched heal: the batched heal is a host-side eager "
            "event and must never run under graph capture."
        )
    state = runner._layer_caches(cache)
    device = cache.boundary.device
    hv, k_dim, v_dim = cache.hv, cache.k, cache.v
    hg = (cache.qkv_dim - hv * v_dim) // (2 * k_dim)
    n = len(todo)
    rows = n * CHUNK_SIZE

    # ONE pageable H2D for all host-known integers; everything else derives
    # on device.
    sf = torch.tensor(
        [[s for s, _ in todo], [f for _, f in todo]],
        dtype=torch.int64,
        device=device,
    )
    idx = sf[0].contiguous()
    fl32 = sf[1].to(torch.int32)
    # gap-encoded cu_seqlens (even entries 64*i, odd entries 64*i + fill_i)
    # and explicit chunk_indices [(2i, 0)] -- the flush_slots/_step_incr
    # structure; int32 like warm_slot's cu/ci.
    base = torch.arange(n + 1, dtype=torch.int32, device=device) * CHUNK_SIZE
    cu = torch.zeros(2 * n + 1, dtype=torch.int32, device=device)
    cu[0::2] = base
    cu[1::2] = base[:-1] + fl32
    ci = torch.stack(
        (
            torch.arange(n, dtype=torch.int32, device=device) * 2,
            torch.zeros(n, dtype=torch.int32, device=device),
        ),
        dim=1,
    ).contiguous()

    # pool-row gathers (transport; identical bytes at slab addresses)
    rows_qkv = cache.rows_qkv.index_select(0, idx)  # [n, 64, QKV]
    g_rows = cache.rows_g.index_select(0, idx).view(1, rows, hv)
    beta_rows = cache.rows_beta.index_select(0, idx).view(1, rows, hv)

    # stage 1: head expansion + STOCK l2norm (row-local; tails are
    # stale-but-finite pool bytes, processed then never committed)
    q, k, v = cache._split(rows_qkv.reshape(rows, cache.qkv_dim))
    rep = hv // hg
    qi = q.repeat_interleave(rep, dim=2).contiguous()
    ki = k.repeat_interleave(rep, dim=2).contiguous()
    l2q = l2norm_fwd(qi)
    l2k = l2norm_fwd(ki)

    # stage 2: STOCK varlen cumsum -- kernel-direct with the driver's exact
    # launch (num_warps=8/num_stages=3; flush_slots precedent), because the
    # chunk_local_cumsum wrapper drops chunk_indices and would re-derive
    # them from the gap cu via a .tolist() D2H (and renumber the segments).
    gcum = torch.empty_like(g_rows)
    chunk_local_cumsum_scalar_kernel[(n, hv)](
        s=g_rows,
        o=gcum,
        scale=None,
        cu_seqlens=cu,
        chunk_indices=ci,
        T=rows,
        B=1,
        H=hv,
        BT=CHUNK_SIZE,
        HEAD_FIRST=False,
        REVERSE=False,
        HAS_SCALE=False,
        IS_VARLEN=True,
        num_warps=8,
        num_stages=3,
    )

    # stages 3-5: STOCK kkt / solve / w_u drivers, chunk_indices explicit
    A = chunk_scaled_dot_kkt_fwd(
        l2k,
        beta_rows,
        g_cumsum=gcum,
        cu_seqlens=cu,
        output_dtype=torch.float32,
        chunk_indices=ci,
    )
    if BI_GDN_SOLVE_TRIL_DECODE:
        # mirror warm_slot's consolidated call-site dispatch (single flag
        # authority; eager per extend batch -- per-call Di scratch is fine)
        ai32 = solve_tril_decode(
            A=A, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32
        )
    else:
        ai32 = solve_tril(
            A=A, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32
        )
    ai16 = ai32.to(torch.bfloat16)  # rtne; same elementwise op as warm_slot
    w, u = recompute_w_u_fwd(
        l2k,
        v.contiguous(),
        beta_rows,
        g_cumsum=gcum,
        A=ai16,
        cu_seqlens=cu,
        chunk_indices=ci,
    )

    # commits: one masked transport launch per persistent cache tensor
    _commit_rows(l2q, state.l2q, idx, fl32, n)
    _commit_rows(l2k, state.l2k, idx, fl32, n)
    _commit_rows(A, state.A, idx, fl32, n)
    _commit_rows(ai32, state.Ai32, idx, fl32, n)
    _commit_rows(ai16, state.Ai16, idx, fl32, n)
    _commit_rows(w, state.w, idx, fl32, n)
    _commit_rows(u, state.u, idx, fl32, n)

    if runner.defer_writeback:
        # the DEFER v_new warm through the STOCK fwd_h driver (P1: rows == the
        # per-step committed values; h0 = the just-seeded chunk boundary).
        #
        # ENCODING NOTE (OOB hazard found in bring-up): fwd_h's grid is
        # per-SEQUENCE, not per-chunk-index -- under the GAP encoding the odd
        # tail segments are NON-empty (T = 64 - fill), so the kernel would
        # run them and write ``h`` at chunk_offsets slots beyond the driver's
        # NT=len(chunk_indices) allocation (silent corruption or IMA
        # depending on allocator layout).  The fwd_h stage therefore uses a
        # DENSE per-request encoding: each request is one full 64-row
        # sequence.  v_new[j] = u[j] - w[j] @ bf16(h0) is stored BEFORE the
        # kernel's g-scaling of the recurrence state, is per-row in u/w/h0,
        # and h0 is per-sequence -- so rows j < fill are bit-identical to
        # warm_slot's T=fill call and rows >= fill are dead garbage the
        # masked commit never moves (covered by the digest comparison).
        hs = cache.boundary.index_select(0, idx).transpose(-1, -2).contiguous()
        ci_dense = torch.stack(
            (
                torch.arange(n, dtype=torch.int32, device=device),
                torch.zeros(n, dtype=torch.int32, device=device),
            ),
            dim=1,
        ).contiguous()
        co_dense = torch.arange(n + 1, dtype=torch.int32, device=device)
        _, vnew_rows, _ = chunk_gated_delta_rule_fwd_h(
            k=l2k,
            w=w,
            u=u,
            g=gcum,
            initial_state=hs,
            output_final_state=False,
            cu_seqlens=base,  # dense [0, 64, 128, ...]
            chunk_indices=ci_dense,
            chunk_offsets=co_dense,
        )
        _commit_rows(vnew_rows, state.v_new, idx, fl32, n)
