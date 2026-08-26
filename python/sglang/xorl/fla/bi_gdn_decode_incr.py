"""incremental-exact BI-GDN decode.

Row-p incremental evaluation of the partial-chunk rescan, behind
``the INCR mode`` (default OFF; requires
``the FAST transport=1``).  The frozen oracle program is
``bi_gdn_decode.py`` + the stage kernels of
``bi_chunk_gated_delta_rule_prefill``; this module never modifies them, and
the FAST path's ``bi_gdn_decode_fast.py`` is composed with (its append / state
scatter / output gather kernels and its runner dispatch surface), never
edited.

**v1.1 flag semantics (2026-08-04, coordinator-approved): the INCR flag is a
HYBRID.**  Under graph capture (``get_is_capture_mode()`` -- which covers the
capture-mode warmups -- or an actively capturing stream) the runner executes
the full incremental program below, so graph REPLAY replays the incremental
kernels; on the true eager path (ramp, drain, sub-bucket batches) it executes
the FAST path's FAST program -- the two programs are byte-certified identical, so
dispatching between them is behavior-free, and FAST's eager step has fewer
launches (v1 measured the all-INCR eager ramp/drain costing the endpoint).
THE DISPATCH SEAM: the incremental caches must be consistent with ``rows_*``
up to the current fill whenever a replay runs, so the eager branch appends a
CACHE-MAINTENANCE tail (the same certified prep / token-expand / l2norm /
l2-commit / cumsum / kkt_rowband / solve_row / solve-commit / wu_rowband
kernels, WITHOUT fwd_h / o / epilogue -- FAST already produced the state and
output).  Caches are therefore fresh on EVERY path by construction; there is
no host-side staleness tracking to get wrong; eager-to-graph transitions
therefore preserve the same cache contents by construction.

**composed stack: hybrid x writeback deferral.**  When
``the writeback deferral`` is armed (DEFER section below), the
persistent ``v_new`` row cache joins the dispatch-seam contract: the deferred
program's o stage reads it mid-chunk instead of recomputing v_new inside
fwd_h, so the eager maintenance tail ALSO runs the DEFER program's UNMODIFIED
``vnew_rowband`` kernel -- with chunk-completed slots masked to PAD (-1)
because (a) FAST's epilogue has already advanced ``boundary`` (vnew's h0)
for exactly those slots this step, and (b) a completed chunk's band rows are
provably dead (rewritten by the next chunk's own band writes before any
read).  The v_new cache stays fresh by construction on every path; the
hybrid-DEFER transition cells (plain + poison-armed) byte-gate the composed
trajectory end to end (eager FAST -> graph INCR+DEFER -> completion ->
finish-flush -> radix-warm readback).

Design: incremental row-band recomputation with the full-grid output kernel
retained where narrower tiles change arithmetic bytes.
The chunked composition is causal inside the chunk in BYTES (201,600 row-pair
fill-invariance checks, 0 violations), so per-row stage intermediates cached
at first computation are legal, and row p of each stage can be recomputed
bit-identically with narrow row-band kernels -- except ``chunk_fwd_o``, whose
narrow tiles measurably flip 1 ulp at rare value edges and which therefore
runs SAME-GRID-MASKED at BM=64 (probe-measured: 420/420 at BM=64; 34 byte failures at
BM in {16,32} -- do not narrow the o stage).

Persistent per-slot per-layer caches (device tensors at fixed addresses,
written ONLY by captured kernels; fill/slot validity is derived in-kernel
from ``seq_lens``/``slot_indices`` CONTENT -- never host-side warmth logic,
which a CUDA-graph capture would bake in; GLM SelectionPlan lesson):

  l2q_rows/l2k_rows [S, 64, HV, K] bf16   l2-normalized interleaved q/k rows
  A_rows            [S, 64, HV, 64] fp32  kkt rows (strictly-lower masked)
  Ai32_rows         [S, 64, HV, 64] fp32  solve fp32 rows (upper cols zero)
  Ai16_rows         [S, 64, HV, 64] bf16  rtne(Ai32) rows (w_u operand)
  w_rows/u_rows     [S, 64, HV, K/V] bf16 pseudo key/value rows

Raw v/g/beta rows are read strided from the existing ``rows_qkv``/``rows_g``/
``rows_beta`` pools (identical bytes, pool addresses -- transport per the
the FAST-path doctrine).  Tails are stale-but-finite by construction (zero-init,
only ever written with finite stage outputs); the stale-mode probe cells
certify exactly this operating condition, including the band overwrite
semantics (recomputed band rows j < p are bitwise the cached rows because
each stage's row j consumes only rows <= j -- the P1 property).

Per decode step, per GDN layer (all launches captured; p is content):

  1. prep         : g slab from rows_g (+ token substitution) + cu_gap refresh
  2. append       : the FAST path's kernel persists the token row at rows_*[slot, p]
  3. token expand : packed token row -> dense HV-expanded q/k rows (transport)
  4. l2norm       : STOCK kernel over the bs*HV fresh rows only
  5. l2 commit    : fresh l2 rows -> l2q_rows/l2k_rows[slot, p] (masked)
  6. cumsum       : STOCK varlen scalar kernel over the g slab (the FAST path's
                    certified launch); the probe program measured the serial increment
                    NOT bit-safe -- the kernel is always rerun
  7. kkt_rowband  : BM=16 band of A from l2k cache + beta pool + gcum slab
  8. solve_row x4 : constexpr-D single-row substitution + ieee block merges,
                    content early-return dispatch (3 of 4 exit immediately)
  9. solve commit : assemble row p -> Ai32_rows (fp32) + Ai16_rows (rtne);
                    upper columns written as exact zeros (w_u contraction
                    weights beyond col p must be exactly 0)
 10. wu_rowband   : BM=16 band of w/u from Ai16 row band + v/beta pools
 11. fwd_h        : the FAST path's pool-direct fwd_h variant re-addressed to the
                    slot-major l2k/w/u caches; boundary pool -> scratch pool,
                    v_new slab (full extent), h[0] slab (bf16(boundary))
 12. o stage      : transport export of cached l2q/l2k rows to gap-encoded
                    slabs, then the ORACLE BINARY chunk_fwd_kernel_o with
                    the FAST path's certified launch (see the S7 section note: every
                    copy of this kernel so far has flipped a bit somewhere)
 13. scatter      : the FAST path's epilogue (scratch->ssm every step; boundary
                    advance on completed chunks)  [v1 KEEPS the per-step
                    pool writeback -- stock semantics; the deferral is a
                    later sub-flag per design risk #3]
 14. out gather   : the FAST path's kernel reads row p per request

Implementation details constrained by byte-comparison tests:
  - The per-step pool writeback (kept in v1) requires the fwd_h recurrence at
    T=fill every step; that launch is the P3-certified
    completion-from-cached-w/u variant and ALSO the exact launch the oracle
    itself performs at fill T over bitwise-identical inputs.  Its SAVE_NEW_VALUE
    output supplies v_new rows 0..p and its h[0] store supplies bf16(boundary),
    so the design's separate ``vnew_rowband`` kernel and ``h0_bf16`` cache are
    subsumed (fewer new binaries, no extra cache).
  - ``gcum_rows`` is a per-step slab, not a persistent cache: the stock varlen
    cumsum at T=fill (the FAST path's certified stage) reproduces rows <= p bitwise
    (P1 fill-invariance), so persisting them buys nothing.
  - The o stage runs the oracle's own compiled kernel over gap-encoded slabs
    instead of a same-grid BM=64 slot-addressed copy: component tests
    falsified the copy at 1 bf16 ulp (1 element, H=4 rank shape, ties cell)
    -- the same SSG0 copied-kernel codegen class the FAST path hit on this exact
    kernel.  The design's intent (identical tile shapes/config to the oracle
    o kernel) is satisfied exactly by the oracle binary itself; the price is
    one transport export of the cached l2 rows (rows <= fill only).

Extend seeding: ``warm_slot`` exports the suffix rows' intermediates through
the STOCK stage drivers (P1 proves those rows equal the per-step values; the
fp32 solve export is the same kernel, store dtype only) and writes them into
the caches.  It runs eagerly per request at extend end -- a content refresh
at fixed addresses between graph replays, the same class as
``seed_from_extend`` itself -- never during decode capture or replay.

PAD semantics mirror the FAST path exactly: reads clamp slot to 0, ALL cache/pool
stores are masked on ``slot >= 0`` (eager -1 rows are inert); graph pads map
to reserved slot 0 and are last-writer racy on slot-0 bytes only (excluded
from byte gates in multi-pad cells; pad OUTPUT rows additionally race through
the shared slot-0 caches under the incremental path and are likewise excluded
-- they are DP-sync dummies discarded upstream, and live-slot bytes are
provably untouched by pad rows).

The incremental path is verified against the full-rescan oracle at stage and
multi-step boundaries, including cache crossings, graph replay, and
adversarial values.

--- DEFER: writeback deferral sub-flag ------------------------------

``the writeback deferral`` (default OFF; raises unless INCR is
armed) defers the per-step pool writeback -- the fwd_h recurrence at T=fill
(93.5 us/layer/step) and the scratch->ssm scatter (45.2 us/layer/step), 54%
of the INCR layer step -- to FLUSH POINTS.  ``ssm_states[slot]`` loses its
stock every-step meaning between flush points; ``boundary`` semantics are
unchanged (advanced at completions exactly as before).  Mid-chunk, the o
stage's inputs come from:
  - an h0 transport export (h slab <- bf16(boundary[slot]), the same rtne
    downcast fwd_h's h[0] store performs -- gate G2), and
  - a ``v_new`` per-slot row cache fed by the the probe program P2/P3-certified
    ``vnew_rowband`` (BM=16, BV=32, k-tile order preserved; v_new[j] =
    u[j] - w[j] @ bf16(h0) is per-row given h0, which is constant within a
    chunk), exported rows <= p to the request-major slab.

Flush points (state materialization writes ssm_states[slot]):
  (a) chunk completion (fill hits 64), in-capture: flush-point-gated fwd_h
      (verbatim body + content early-return) + flush-point-gated scatter --
      the write is enqueued INSIDE step(), i.e. strictly before the same
      step's mamba-track copy launch (gdn_backend.py:1026-1028,:1109-1111),
      so a same-step track read always sees the completion bytes;
  (b) mamba-track points, in-capture: the same gate also fires on
      ``seq_len % track_interval == 0`` (a strict subset of (a) when the
      server-args ``mamba_track_interval % 64 == 0`` guard holds -- asserted
      fail-closed at backend init -- and a LIVE mid-chunk defense that
      materializes state-at-fill if that config guard ever drifts);
  (c) request finish -> radix ``no_buffer`` insert, host-side eager:
      ``flush_slots`` (mamba_radix_cache.cache_finished_req hook) re-runs the
      UNGATED v1 binaries -- the certified cumsum launch over a g slab
      rebuilt from ``rows_g`` (byte-identical to the step's substituted slab
      because the append kernel persisted the token row), then v1's fwd_h at
      T=fill and v1's stock scatter (fill<64 => no boundary advance).
      Host rule: SKIP when consumed_len % 64 == 0 -- the completion already
      materialized the state, and re-running fwd_h at T=64 after the
      boundary advance would re-apply the chunk (double-application hazard);
      also skip when no decode token was consumed (prefill writeback fresh).
Flush triggers are host-side scheduler events OUTSIDE captured regions or
tensor-content gates INSIDE them -- never Python state consulted at capture
(the GLM SelectionPlan bake hazard).  Flushes are idempotent (same inputs,
same binaries, same bytes).

Test-side deferred-row poisoning (the DEFER byte suites): between steps the
tests overwrite live slots' non-flush-point ssm rows with the qNaN pattern
0x7FC0DEAD, so ANY unflushed reader produces loud corruption in gates instead
of silent staleness (formerly an env-armed in-step branch; the corruption
instrument now lives entirely in the tests).  Boundary is never
poisoned (load-bearing).  Reserved graph slot 0 may carry poison bytes (pads
map there; MambaPool.free_slots start at 1, so no live request reads them).

Pool-reader enumeration (this tree, verified 2026-08-04; ssm = MambaPool
temporal, bound at gdn_backend.py:889-891):
  R1 mamba-track decode copy: hybrid_linear_attn_backend.py:88-97 (kernel),
     :758-766 (driver), fired gdn_backend.py:1026-1028,:1109-1111 right
     after step(); mask content = seq_lens % mamba_track_interval == 0
     (schedule_batch.py:3203-3207, extra_buffer strategy only per :3179);
     interval % 64 == 0 asserted (server_args.py:2336).  Covered by flush
     points (a)+(b); ordering by same-stream enqueue inside step().
  R2 extend-boundary track copy: hybrid_linear_attn_backend.py:794-797 +
     gdn_backend.py:1320-1322 -- reads extend-end bytes (prefill writeback,
     which this flag does NOT defer).  Fresh.
  R3 extend h0 read + pre_states clone: bi_gdn_prefill.py:1248-1251 (driver
     reads ssm_states[cache_indices] as initial state) + gdn_backend.py:
     1294-1298.  Slot bytes at extend start come from alloc-zero
     (memory_pool.py:366-377), tree COW copy (mamba_radix_cache.py:
     1049-1066), or a prior prefill writeback -- never mid-chunk decode
     bytes (64-aligned admission: schedule_policy.py:85-90,:662-665,
     :745-748,:830-833,:871-874; seed_from_extend assert
     bi_gdn_decode.py:416-419).  Fresh.
  R4 radix fork/COW pool copies: memory_pool.py:403-418 (copy_from /
     fork_from).  Sources are tree-owned slots (written at their own flush
     points) or the LIVE slot at cache_unfinished_req's extend-end fork
     (mamba_radix_cache.py:660-671) -- extend-end bytes are prefill-fresh
     (the only writer sequence reaching that call is an extend forward).
     Not guarded in code (no cheap context check exists there); poison mode
     surfaces any violation.  RESIDUAL-RISK entry, see final report.
  R5 no_buffer finish insert: mamba_radix_cache.py cache_finished_req
     (:573-586) hands the LIVE slot to the tree at len(token_ids) tokens ->
     flush point (c), the hook installed by gdn_backend when this flag is
     on.  extra_buffer finish keeps the ping-pong TRACK slot instead
     (:563-572), written at track points = completions.  Fresh either way.
  R6 HiCache D2H write-through: hi_mamba_radix_cache.py (subclass, inherits
     cache_finished_req -> hook applies) -- reads tree-forked slots, fresh
     at fork time per R4/R5.
  R7 retraction/abort: schedule_batch.py release_req (:2640-2657) ->
     release_kv_cache(is_insert=False) -- state DISCARDED; slot freed;
     MambaPool.alloc zeroes conv+temporal at re-alloc (memory_pool.py:
     366-377) and prefix reuse overwrites via R4 copy, so a discarded
     slot's stale/poison bytes are provably unreachable before reuse-reinit.
  R8 debug/dump observers: gdn_backend.py:843 (_debug_exact_state ssm
     digest), :569/:596 (dump-hook ssm clones) -- env-gated debug only;
     their mid-chunk ssm bytes CHANGE under deferral (baselines differ).
  R9 MTP/spec verify + mixed-chunk + session resume: unreachable or loudly
     rejected under the BI GDN decode contract (single-token assert
     gdn_backend.py:976-979; 64-aligned admission R3) -- unchanged v1-level
     exposure; if any is ever enabled, the per-step writeback becomes
     load-bearing again and this flag must not be armed.

Conv/ssm symmetry: conv_states update EVERY step (causal_conv1d_update,
gdn_backend.py:909-916) and are never deferred; every pair-copying consumer
(R1 track kernel copies conv then ssm; R4/R5 hand both to the tree) reads at
a flush point, so a fresh-conv/stale-ssm pair is impossible at any
enumerated read point.

Certification: test_bi_gdn_decode_incr_defer_a0.py (deferred vs non-deferred
INCR byte equality of pool state AT EVERY FLUSH POINT; poison suite; capture
lifecycle with deferral armed; direct comparison with deferral disabled) ->
in-server radix-warm continuation gate (warm logprob bytes == cold reference,
both arms, poison arm clean) -> 80-row + graph-32 ruler -> A/B.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE
from sglang.kernels.ops.attention.fla.cumsum import (
    chunk_local_cumsum,
    chunk_local_cumsum_scalar_kernel,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd, l2norm_fwd_kernel
from sglang.kernels.ops.attention.fla.op import safe_exp
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd
from sglang.xorl.fla.bi_gdn_decode_fast import (
    BIGDNFastDecodeRunner,
    bi_gdn_fast_append_kernel,
    bi_gdn_fast_out_gather_kernel,
    bi_gdn_fast_state_scatter_kernel,
)
from sglang.xorl.fla.bi_gdn_prefill import (
    BI_GDN_SOLVE_TRIL_DECODE,
    chunk_fwd_kernel_o,
    chunk_gated_delta_rule_fwd_h,
    exp,
    solve_tril,
)
from sglang.xorl.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from sglang.xorl.fla.solve_tril_decode import (
    _sum_rows_16_fla_tree,
    solve_tril_decode,
)

# Internal choices for the exact Qwen3.5-family GDN serving contract.
# Installed as a unit by the architecture resolver (see
# sglang.xorl.fla.qwen35_gdn_exact) -- there are no
# per-feature environment variables on this surface. The False defaults
# keep every non-contract server on the stock path, bit-for-bit
# unaffected; tests may set these module attributes directly on a fresh
# runner (never after a capture).
# Invariants the resolver preserves by construction (and the backend
# re-asserts at init): INCR is a mode of the fast runner's dispatch surface
# and cannot run without it; the writeback deferral is a sub-mode of the
# incremental runner; the slim o-stage reads the deferral's v_new cache and
# never arms without it.
BI_GDN_DECODE_INCR_ENABLED = False

BI_GDN_INCR_DEFER_ENABLED = False

# Slim o-stage -- a sub-mode of the deferral program (consulted
# only in the defer branch): the per-step vnew/l2 slab EXPORTS are deleted
# and the ORACLE chunk_fwd_kernel_o reads the slot-major caches DIRECTLY
# through a second cu tensor whose even entries hold 64*slot (content
# refreshed in-kernel per step -- the certified dynamism-as-content pattern;
# the oracle binary, grid, tile shapes, and launch config are untouched; `h`
# stays request-major because the varlen o kernel indexes it positionally).
# The g operand comes from a per-slot `gcum` row cache (row p committed at
# stage 9; fill-invariance makes committed rows bitwise-stable within a
# chunk); the o output lands in a slot-major buffer read by a slot-addressed
# out gather.
BI_GDN_VNEW_SLIM_ENABLED = False

TBUF = CHUNK_SIZE  # physical cache buffer rows (== CHUNK_SIZE == 64)
TBUF_C = tl.constexpr(TBUF)


# --- transport: g slab build (+ cu_gap refresh), token row head expansion ----
# Verbatim subsets of the FAST path's certified ingest kernel (read-time token
# substitution, clamp-for-reads).


@triton.jit
def bi_gdn_incr_prep_kernel(
    rows_g,  # [S, CHUNK, H] fp32 pool
    g_tok,  # [bs, H] fp32 current-token rows
    slot_indices,  # [bs] int32, may contain -1 (eager PAD)
    seq_lens,  # [bs] int
    g_slab,  # [bs*CHUNK, H] fp32 slab
    cu_gap,  # [2*bs+1] int32 gap-encoded cu_seqlens; odd entries written here
    H: tl.constexpr,
    BH: tl.constexpr,
    CHUNK: tl.constexpr,
):
    i_b, i_r = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK  # torch.remainder semantics

    if i_r == 0:
        # refresh the request's logical extent for the varlen stage kernels
        tl.store(cu_gap + 2 * i_b + 1, i_b * CHUNK + fb + 1)

    src_g = rows_g + (slot_c * CHUNK + i_r) * H
    if i_r == fb:
        src_g = g_tok + i_b.to(tl.int64) * H
    o_h = tl.arange(0, BH)
    m_h = o_h < H
    b_g = tl.load(src_g + o_h, mask=m_h, other=0.0)
    tl.store(g_slab + (i_b.to(tl.int64) * CHUNK + i_r) * H + o_h, b_g, mask=m_h)


@triton.jit
def bi_gdn_incr_token_expand_kernel(
    qkv_tok,  # [bs, QKV] bf16 packed [q(HG*K) | k(HG*K) | v(H*V)] token rows
    q_tok,  # [bs, H, K] bf16 head-expanded
    k_tok,  # [bs, H, K] bf16 head-expanded
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    QKV: tl.constexpr,
    BHK: tl.constexpr,
):
    i_b = tl.program_id(0).to(tl.int64)
    o_hk = tl.arange(0, BHK)
    m_hk = o_hk < H * K
    hh = o_hk // K
    kk = o_hk % K
    src_off = (hh // (H // HG)) * K + kk
    src = qkv_tok + i_b * QKV
    b_q = tl.load(src + src_off, mask=m_hk, other=0.0)
    tl.store(q_tok + i_b * H * K + o_hk, b_q, mask=m_hk)
    b_k = tl.load(src + HG * K + src_off, mask=m_hk, other=0.0)
    tl.store(k_tok + i_b * H * K + o_hk, b_k, mask=m_hk)


@triton.jit
def bi_gdn_incr_l2_commit_kernel(
    l2q_tok,  # [bs, H, K] bf16 (stock l2norm output on the fresh rows)
    l2k_tok,  # [bs, H, K] bf16
    l2q_rows,  # [S, CHUNK, H, K] bf16 cache
    l2k_rows,  # [S, CHUNK, H, K] bf16 cache
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    K: tl.constexpr,
    BHK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    i_b = tl.program_id(0)
    slot = tl.load(slot_indices + i_b)
    if slot >= 0:
        sl = tl.load(seq_lens + i_b).to(tl.int32)
        fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
        dst = (slot.to(tl.int64) * CHUNK + fb) * H * K
        o = tl.arange(0, BHK)
        m = o < H * K
        b_q = tl.load(l2q_tok + i_b.to(tl.int64) * H * K + o, mask=m, other=0.0)
        tl.store(l2q_rows + dst + o, b_q, mask=m)
        b_k = tl.load(l2k_tok + i_b.to(tl.int64) * H * K + o, mask=m, other=0.0)
        tl.store(l2k_rows + dst + o, b_k, mask=m)


# --- S3: chunk_scaled_dot_kkt row band (the probe program kkt_rowband, slot-addressed) --
# Arithmetic body verbatim from the certified probe kernel; only base-address
# computation (slot/request from tensor content) and the masked store differ.


@triton.jit
def bi_gdn_incr_kkt_rowband_kernel(
    k,  # [S, CHUNK, H, K] bf16 l2k cache
    beta,  # [S, CHUNK, H] fp32 pool
    g_cumsum,  # [bs*CHUNK, H] fp32 slab
    A,  # [S, CHUNK, H, CHUNK] fp32 cache
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    CHUNK: tl.constexpr,
):
    i_b, i_h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    T = ((sl - 1) % CHUNK + CHUNK) % CHUNK + 1
    r0 = ((T - 1) // BM) * BM
    ext = TBUF_C  # production condition: full physical extent, stale tails

    k += slot_c * CHUNK * H * K
    beta += slot_c * CHUNK * H
    g_cumsum += i_b.to(tl.int64) * CHUNK * H
    A += slot_c * CHUNK * H * BT

    o_t = tl.arange(0, BT)
    o_m = r0 + tl.arange(0, BM)

    p_beta = tl.make_block_ptr(beta + i_h, (ext,), (H,), (r0,), (BM,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BM, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_kr = tl.make_block_ptr(
            k + i_h * K, (ext, K), (H * K, 1), (r0, i_k * BK), (BM, BK), (1, 0)
        )
        p_kc = tl.make_block_ptr(
            k + i_h * K, (ext, K), (H * K, 1), (0, i_k * BK), (BT, BK), (1, 0)
        )
        b_kr = tl.load(p_kr, boundary_check=(0, 1))
        b_kc = tl.load(p_kc, boundary_check=(0, 1))
        b_A += tl.dot(b_kr, tl.trans(b_kc))

    p_gr = tl.make_block_ptr(g_cumsum + i_h, (ext,), (H,), (r0,), (BM,), (0,))
    p_gc = tl.make_block_ptr(g_cumsum + i_h, (ext,), (H,), (0,), (BT,), (0,))
    b_gr = tl.load(p_gr, boundary_check=(0,))
    b_gc = tl.load(p_gc, boundary_check=(0,))
    b_A = b_A * safe_exp(b_gr[:, None] - b_gc[None, :])

    b_A *= b_beta[:, None]
    b_A = tl.where(o_m[:, None] > o_t[None, :], b_A, 0)
    if slot >= 0:
        p_A = tl.make_block_ptr(
            A + i_h * BT, (TBUF_C, BT), (BT * H, 1), (r0, 0), (BM, BT), (1, 0)
        )
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


# --- S4: solve_tril single-row substitution + ieee block-merge chain ---------
# the probe program solve_row, slot-addressed; four constexpr-D variants are always all
# launched, with per-request content early-return dispatch.  Row-p segments
# are staged per REQUEST (private), then committed by the commit kernel.


@triton.jit
def bi_gdn_incr_solve_row_kernel(
    A,  # [S, CHUNK, H, CHUNK] fp32 cache
    Ai32,  # [S, CHUNK, H, CHUNK] fp32 cache
    OutDiag,  # [bs, H, 16] fp32 staging
    OutSeg,  # [bs, H, 3, 16] fp32 staging
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_b, i_h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    T = ((sl - 1) % CHUNK + CHUNK) % CHUNK + 1
    p = T - 1
    if p // 16 != D:
        return
    r = p - D * 16
    r0 = D * 16
    ext = TBUF_C

    o_i = tl.arange(0, 16)
    m_I = (o_i[:, None] == o_i[None, :]).to(tl.float32)

    A += slot_c * CHUNK * H * BT + i_h * BT
    Ai32 += slot_c * CHUNK * H * BT + i_h * BT

    # ---- diagonal-block forward-substitution step for local row r ----
    # oracle in-loop operand rows j<i are the computed rows WITHOUT the m_I
    # diagonal; the cached final rows minus identity reproduce them exactly
    # (rows 0 and 1 degenerate to the init values; rows >= r are zero-masked).
    b_a = -tl.load(A + p * H * BT + o_i + r0)
    b_a = tl.where(o_i < r, b_a, 0.0)
    p_diag = tl.make_block_ptr(
        Ai32, (TBUF_C, BT), (H * BT, 1), (r0, r0), (16, 16), (1, 0)
    )
    b_cached = tl.load(p_diag, boundary_check=(0, 1))
    b_op = b_cached - m_I
    # The oracle's EXPLICIT num_warps=2 tl.sum association tree (imported
    # from the solve port), so this kernel's diagonal bytes no longer
    # depend on its own launch config. The previous inline
    # ``tl.sum(b_a[:, None] * b_op, 0)`` was byte-equal only at
    # num_warps<=2 -- and Triton 3.6.0 miscompiles this kernel's D=1 merge
    # dot chain at num_warps<=2 (silent zeros), so the launch moved to 4
    # warps and the diag tree must be pinned explicitly.
    b_new = b_a + tl.reshape(
        _sum_rows_16_fla_tree(tl.reshape(b_a, [1, 16]), tl.reshape(b_op, [1, 16, 16])),
        [16],
    )
    b_diag = b_new + tl.where(o_i == r, 1.0, 0.0)
    tl.store(OutDiag + (i_b * H + i_h) * 16 + o_i, b_diag)

    if D >= 1:
        # assembled Ai_dd operand: cached rows (<r final), fresh row r
        b_Aidd = tl.where((o_i == r)[:, None], b_diag[None, :], b_cached)

        # nested form: Ai_{d,d-1} = -(Ai_dd @ A_{d,d-1}) @ Ai_{d-1,d-1}
        p_A_prev = tl.make_block_ptr(
            A, (ext, BT), (H * BT, 1), (r0, r0 - 16), (16, 16), (1, 0)
        )
        b_A_prev = tl.load(p_A_prev, boundary_check=(0, 1))
        p_Ai_prev = tl.make_block_ptr(
            Ai32, (TBUF_C, BT), (H * BT, 1), (r0 - 16, r0 - 16), (16, 16), (1, 0)
        )
        b_Ai_prev = tl.load(p_Ai_prev, boundary_check=(0, 1))
        b_seg = -tl.dot(
            tl.dot(b_Aidd, b_A_prev, input_precision=DOT_PRECISION),
            b_Ai_prev,
            input_precision=DOT_PRECISION,
        )
        b_row = tl.sum(tl.where((o_i == r)[:, None], b_seg, 0.0), 0)
        tl.store(OutSeg + ((i_b * H + i_h) * 3 + (D - 1)) * 16 + o_i, b_row)

        # sum form: Ai_{d,e} = -Ai_dd @ sum_{m=e..d-1} A_{d,m} @ Ai_{m,e}
        # (the oracle writes the m-terms left to right in ascending m; the
        #  accumulation below is left-associated starting at m = e to match)
        for e in tl.static_range(D - 1):
            p_A_de = tl.make_block_ptr(
                A, (ext, BT), (H * BT, 1), (r0, e * 16), (16, 16), (1, 0)
            )
            b_A_de = tl.load(p_A_de, boundary_check=(0, 1))
            p_Ai_ee = tl.make_block_ptr(
                Ai32, (TBUF_C, BT), (H * BT, 1), (e * 16, e * 16), (16, 16), (1, 0)
            )
            b_Ai_ee = tl.load(p_Ai_ee, boundary_check=(0, 1))
            b_M = tl.dot(b_A_de, b_Ai_ee, input_precision=DOT_PRECISION)
            for m in tl.static_range(D - 1 - e):
                mm = e + 1 + m
                p_A_dm = tl.make_block_ptr(
                    A, (ext, BT), (H * BT, 1), (r0, mm * 16), (16, 16), (1, 0)
                )
                b_A_dm = tl.load(p_A_dm, boundary_check=(0, 1))
                p_Ai_me = tl.make_block_ptr(
                    Ai32,
                    (TBUF_C, BT),
                    (H * BT, 1),
                    (mm * 16, e * 16),
                    (16, 16),
                    (1, 0),
                )
                b_Ai_me = tl.load(p_Ai_me, boundary_check=(0, 1))
                b_M += tl.dot(b_A_dm, b_Ai_me, input_precision=DOT_PRECISION)
            b_seg2 = -tl.dot(b_Aidd, b_M, input_precision=DOT_PRECISION)
            b_row2 = tl.sum(tl.where((o_i == r)[:, None], b_seg2, 0.0), 0)
            tl.store(OutSeg + ((i_b * H + i_h) * 3 + e) * 16 + o_i, b_row2)


@triton.jit
def bi_gdn_incr_solve_commit_kernel(
    OutDiag,  # [bs, H, 16] fp32 staging
    OutSeg,  # [bs, H, 3, 16] fp32 staging
    Ai32,  # [S, CHUNK, H, CHUNK] fp32 cache
    Ai16,  # [S, CHUNK, H, CHUNK] bf16 cache
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    BT: tl.constexpr,
    CHUNK: tl.constexpr,
):
    i_b, i_h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    if slot >= 0:
        sl = tl.load(seq_lens + i_b).to(tl.int32)
        p = ((sl - 1) % CHUNK + CHUNK) % CHUNK
        d = p // 16
        o = tl.arange(0, 64)
        # cols < d*16 come from the merge segments (seg e covers cols
        # [16e, 16e+16)); cols in the diagonal block come from OutDiag; all
        # higher cols are EXACT zeros (the solve export's upper triangle --
        # w_u's contraction weights beyond col p must be exactly 0).
        b_seg = tl.load(OutSeg + (i_b * H + i_h) * 48 + o, mask=o < d * 16, other=0.0)
        m_diag = (o >= d * 16) & (o < (d + 1) * 16)
        idx_diag = tl.where(m_diag, o - d * 16, 0)
        b_diag = tl.load(
            OutDiag + (i_b * H + i_h) * 16 + idx_diag, mask=m_diag, other=0.0
        )
        val = tl.where(o < d * 16, b_seg, tl.where(m_diag, b_diag, 0.0))
        base = (slot.to(tl.int64) * CHUNK + p) * H * BT + i_h * BT
        tl.store(Ai32 + base + o, val)
        tl.store(Ai16 + base + o, val.to(tl.bfloat16))


# --- S5: recompute_w_u row band (the probe program wu_rowband, slot-addressed; v/beta
# read strided from the packed rows_qkv / rows_beta pools) -------------------


@triton.jit
def bi_gdn_incr_wu_rowband_kernel(
    k,  # [S, CHUNK, H, K] bf16 l2k cache
    qkv,  # [S, CHUNK, QKV] bf16 pool (v region read strided)
    beta,  # [S, CHUNK, H] fp32 pool
    w,  # [S, CHUNK, H, K] bf16 cache
    u,  # [S, CHUNK, H, V] bf16 cache
    Ai,  # [S, CHUNK, H, CHUNK] bf16 cache (Ai16)
    g,  # [bs*CHUNK, H] fp32 slab
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    QKV: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    CHUNK: tl.constexpr,
):
    i_b, i_h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    T = ((sl - 1) % CHUNK + CHUNK) % CHUNK + 1
    r0 = ((T - 1) // BM) * BM
    ext = TBUF_C

    k += slot_c * CHUNK * H * K
    beta += slot_c * CHUNK * H
    Ai += slot_c * CHUNK * H * BT
    g += i_b.to(tl.int64) * CHUNK * H
    w += slot_c * CHUNK * H * K
    u += slot_c * CHUNK * H * V
    # v rows live in the packed qkv pool at column offset 2*HG*K + i_h*V with
    # row stride QKV -- identical bytes to a dense [CHUNK, H, V] cache, pool
    # addresses (transport).
    v = qkv + slot_c * CHUNK * QKV + 2 * HG * K + i_h * V

    p_beta = tl.make_block_ptr(beta + i_h, (ext,), (H,), (0,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + i_h, (ext,), (H,), (0,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        Ai + i_h * BT, (ext, BT), (H * BT, 1), (r0, 0), (BM, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (ext, V), (QKV, 1), (0, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(
            u + i_h * V, (TBUF_C, V), (H * V, 1), (r0, i_v * BV), (BM, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        if slot >= 0:
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + i_h * K, (ext, K), (H * K, 1), (0, i_k * BK), (BT, BK), (1, 0)
        )
        p_w = tl.make_block_ptr(
            w + i_h * K, (TBUF_C, K), (H * K, 1), (r0, i_k * BK), (BM, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        if slot >= 0:
            tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# --- S6: fwd_h at T=fill from the slot-major caches --------------------------
# Body verbatim from the FAST path's certified pool-direct variant
# (bi_gdn_fast_fwd_h_kernel); the ONLY change is the k/v/w base addressing
# (slot-major caches instead of request-major slabs).  v_new and h remain
# request-major slabs; h0 reads the fp32 boundary pool and ht writes the fp32
# scratch pool slot-direct with transposed strides, masked for PAD rows.
# In v1 this runs EVERY step (per-step pool writeback kept): it is the exact
# launch the oracle performs at fill T over bitwise-identical inputs, and the
# P3-certified completion variant at T=64.


@triton.jit
def bi_gdn_incr_fwd_h_kernel(
    k,  # [S, CHUNK, H, K] bf16 l2k cache
    v,  # [S, CHUNK, H, V] bf16 u cache
    w,  # [S, CHUNK, H, K] bf16 w cache
    v_new,  # [bs*CHUNK, H, V] bf16 slab
    g,  # [bs*CHUNK, H] fp32 slab
    h,  # [bs, H, K, V] bf16 slab (the h[0] block per request)
    boundary,  # h0 source pool [S, H, V, K] fp32
    scratch,  # ht destination pool [S, H, V, K] fp32
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * BT
    sl = tl.load(seq_lens + i_n).to(tl.int32)
    T = ((sl - 1) % BT + BT) % BT + 1
    NT = tl.cdiv(T, BT)
    boh = i_n
    slot = tl.load(slot_indices + i_n)
    slot_c = tl.maximum(slot, 0)
    bos_kv = slot_c.to(tl.int64) * BT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos_kv * H + i_h) * V
    k += (bos_kv * H + i_h) * K
    w += (bos_kv * H + i_h) * K
    v_new += (bos * H + i_h).to(tl.int64) * V

    h0 = boundary + (slot_c.to(tl.int64) * H + i_h) * V * K
    ht = scratch + (slot_c.to(tl.int64) * H + i_h) * V * K

    # load initial state (pool layout [V, K]; trainer view (K, V) via strides)
    p_h0_1 = tl.make_block_ptr(h0, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1))
    b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    if K > 64:
        p_h0_2 = tl.make_block_ptr(h0, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1))
        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
    if K > 128:
        p_h0_3 = tl.make_block_ptr(
            h0, (K, V), (1, K), (128, i_v * BV), (64, BV), (0, 1)
        )
        b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
    if K > 192:
        p_h0_4 = tl.make_block_ptr(
            h0, (K, V), (1, K), (192, i_v * BV), (64, BV), (0, 1)
        )
        b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        p_h1 = tl.make_block_ptr(
            h + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (128, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (192, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        p_v = tl.make_block_ptr(
            v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        m_t = (i_t * BT + tl.arange(0, BT)) < T
        b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(
            tl.float32
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
        b_g_last = exp(b_g_last)
        b_h1 *= b_g_last
        if K > 64:
            b_h2 *= b_g_last
        if K > 128:
            b_h3 *= b_g_last
        if K > 192:
            b_h4 *= b_g_last

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    # store final state slot-direct into the scratch pool; masked for eager
    # PAD (-1) rows exactly like the oracle's masked pool writeback.
    if slot >= 0:
        p_ht = tl.make_block_ptr(ht, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (1, K), (128, i_v * BV), (64, BV), (0, 1)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (1, K), (192, i_v * BV), (64, BV), (0, 1)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# --- DEFER: writeback-deferral kernels ------------------------------
# All gates are tensor-content (seq_lens / slot_indices) so the launches are
# capture-safe; TRACK_IVL is a server-config constant (constexpr) -- baking it
# into a capture is correct because server args are immutable for the process
# lifetime, unlike per-step Python state (the GLM bake hazard class).


@triton.jit
def bi_gdn_incr_defer_h0_export_kernel(
    h,  # [bs, H, K, V] bf16 slab (the h[0] block per request)
    boundary,  # [S, H, V, K] fp32 pool
    slot_indices,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """Transport: h slab <- bf16(boundary[slot]).  The load/store block pairs
    are verbatim from bi_gdn_incr_fwd_h_kernel's h0 load + h[0] store (fp32
    pool read through the transposed view, rtne bf16 downcast at store) --
    gate G2: equal bytes to fwd_h's h[0]."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    slot = tl.load(slot_indices + i_n)
    slot_c = tl.maximum(slot, 0)
    h += (i_n * H + i_h).to(tl.int64) * K * V
    h0 = boundary + (slot_c.to(tl.int64) * H + i_h) * V * K

    p_h0_1 = tl.make_block_ptr(h0, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1))
    b_h1 = tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    p_h1 = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
    tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
    if K > 64:
        p_h0_2 = tl.make_block_ptr(h0, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1))
        b_h2 = tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        p_h2 = tl.make_block_ptr(h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
    if K > 128:
        p_h0_3 = tl.make_block_ptr(
            h0, (K, V), (1, K), (128, i_v * BV), (64, BV), (0, 1)
        )
        b_h3 = tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        p_h3 = tl.make_block_ptr(h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
    if K > 192:
        p_h0_4 = tl.make_block_ptr(
            h0, (K, V), (1, K), (192, i_v * BV), (64, BV), (0, 1)
        )
        b_h4 = tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
        p_h4 = tl.make_block_ptr(h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def bi_gdn_incr_defer_vnew_rowband_kernel(
    w,  # [S, CHUNK, H, K] bf16 w cache
    u,  # [S, CHUNK, H, V] bf16 u cache
    boundary,  # [S, H, V, K] fp32 pool (h0; constant within a chunk)
    v_new,  # [S, CHUNK, H, V] bf16 cache (the DEFER program)
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """the probe program vnew_rowband (P2/P3-certified: BV=32 as the oracle's
    check_shared_mem()->False pin, k-tile order k1..k4, dot downcast of the
    fp32 h0 load), slot-addressed; the h0 read uses the boundary pool's
    transposed view exactly like the certified fwd_h variant.  Band rows
    j < p recompute the cached bytes (P1; h0 constant within the chunk);
    rows > p in the band are stale-but-finite tails, never read."""
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    T = ((sl - 1) % CHUNK + CHUNK) % CHUNK + 1
    r0 = ((T - 1) // BM) * BM
    ext = TBUF_C  # production condition: full physical extent, stale tails

    w += slot_c * CHUNK * H * K
    u += slot_c * CHUNK * H * V
    v_new += slot_c * CHUNK * H * V
    h0 = boundary + (slot_c * H + i_h) * V * K

    p_w = tl.make_block_ptr(
        w + i_h * K, (ext, K), (H * K, 1), (r0, 0), (BM, 64), (1, 0)
    )
    b_w = tl.load(p_w, boundary_check=(0, 1))
    p_h1 = tl.make_block_ptr(h0, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1))
    b_h1 = tl.load(p_h1, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
    if K > 64:
        p_w2 = tl.make_block_ptr(
            w + i_h * K, (ext, K), (H * K, 1), (r0, 64), (BM, 64), (1, 0)
        )
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        p_h2 = tl.make_block_ptr(h0, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1))
        b_h2 = tl.load(p_h2, boundary_check=(0, 1)).to(tl.float32)
        b_v += tl.dot(b_w2, b_h2.to(b_w2.dtype))
    if K > 128:
        p_w3 = tl.make_block_ptr(
            w + i_h * K, (ext, K), (H * K, 1), (r0, 128), (BM, 64), (1, 0)
        )
        b_w3 = tl.load(p_w3, boundary_check=(0, 1))
        p_h3 = tl.make_block_ptr(h0, (K, V), (1, K), (128, i_v * BV), (64, BV), (0, 1))
        b_h3 = tl.load(p_h3, boundary_check=(0, 1)).to(tl.float32)
        b_v += tl.dot(b_w3, b_h3.to(b_w3.dtype))
    if K > 192:
        p_w4 = tl.make_block_ptr(
            w + i_h * K, (ext, K), (H * K, 1), (r0, 192), (BM, 64), (1, 0)
        )
        b_w4 = tl.load(p_w4, boundary_check=(0, 1))
        p_h4 = tl.make_block_ptr(h0, (K, V), (1, K), (192, i_v * BV), (64, BV), (0, 1))
        b_h4 = tl.load(p_h4, boundary_check=(0, 1)).to(tl.float32)
        b_v += tl.dot(b_w4, b_h4.to(b_w4.dtype))

    p_u = tl.make_block_ptr(
        u + i_h * V, (ext, V), (H * V, 1), (r0, i_v * BV), (BM, BV), (1, 0)
    )
    b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v
    if slot >= 0:
        p_vn = tl.make_block_ptr(
            v_new + i_h * V, (TBUF_C, V), (H * V, 1), (r0, i_v * BV), (BM, BV), (1, 0)
        )
        tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def bi_gdn_incr_defer_vnew_export_kernel(
    v_new_rows,  # [S, CHUNK, H, V] bf16 cache
    v_new_slab,  # [bs*CHUNK, H, V] bf16 slab (o-stage input)
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    V: tl.constexpr,
    BHV: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Transport: cached v_new rows <= fb -> the gap-encoded slab (rows > fb
    never read; the varlen o kernel masks by cu content).  PAD rows read the
    clamped slot-0 cache -- their slab rows feed discarded DP-sync outputs
    (the the INCR path slot-0 convention, excluded from byte gates)."""
    i_b, i_r = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
    if i_r <= fb:
        src = v_new_rows + (slot_c * CHUNK + i_r) * H * V
        dst = (i_b.to(tl.int64) * CHUNK + i_r) * H * V
        o = tl.arange(0, BHV)
        m = o < H * V
        tl.store(v_new_slab + dst + o, tl.load(src + o, mask=m, other=0.0), mask=m)


@triton.jit
def bi_gdn_incr_defer_fwd_h_flushpt_kernel(
    k,  # [S, CHUNK, H, K] bf16 l2k cache
    v,  # [S, CHUNK, H, V] bf16 u cache
    w,  # [S, CHUNK, H, K] bf16 w cache
    v_new,  # [bs*CHUNK, H, V] bf16 slab
    g,  # [bs*CHUNK, H] fp32 slab
    h,  # [bs, H, K, V] bf16 slab
    boundary,  # h0 source pool [S, H, V, K] fp32
    scratch,  # ht destination pool [S, H, V, K] fp32
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    TRACK_IVL: tl.constexpr,
):
    """bi_gdn_incr_fwd_h_kernel's body VERBATIM behind a flush-point
    content gate: programs whose request is not at a chunk completion
    (T == BT) or a mamba-track point (seq_len % TRACK_IVL == 0, live
    defense -- redundant when the interval%64 guard holds) return before
    any load.  Its per-request slab writes (v_new rows, h[0]) duplicate the
    bytes the deferral stages already wrote (P1/P3 + gate G2)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * BT
    sl = tl.load(seq_lens + i_n).to(tl.int32)
    T = ((sl - 1) % BT + BT) % BT + 1
    completed = T == BT
    if TRACK_IVL > 0:
        tracked = (sl % TRACK_IVL) == 0
    else:
        tracked = False
    if not (completed | tracked):
        return
    NT = tl.cdiv(T, BT)
    boh = i_n
    slot = tl.load(slot_indices + i_n)
    slot_c = tl.maximum(slot, 0)
    bos_kv = slot_c.to(tl.int64) * BT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos_kv * H + i_h) * V
    k += (bos_kv * H + i_h) * K
    w += (bos_kv * H + i_h) * K
    v_new += (bos * H + i_h).to(tl.int64) * V

    h0 = boundary + (slot_c.to(tl.int64) * H + i_h) * V * K
    ht = scratch + (slot_c.to(tl.int64) * H + i_h) * V * K

    # load initial state (pool layout [V, K]; trainer view (K, V) via strides)
    p_h0_1 = tl.make_block_ptr(h0, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1))
    b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    if K > 64:
        p_h0_2 = tl.make_block_ptr(h0, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1))
        b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
    if K > 128:
        p_h0_3 = tl.make_block_ptr(
            h0, (K, V), (1, K), (128, i_v * BV), (64, BV), (0, 1)
        )
        b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
    if K > 192:
        p_h0_4 = tl.make_block_ptr(
            h0, (K, V), (1, K), (192, i_v * BV), (64, BV), (0, 1)
        )
        b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        p_h1 = tl.make_block_ptr(
            h + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (128, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t_int64 * H * K * V,
                (K, V),
                (V, 1),
                (192, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        p_v = tl.make_block_ptr(
            v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        m_t = (i_t * BT + tl.arange(0, BT)) < T
        b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(
            tl.float32
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
        b_g_last = exp(b_g_last)
        b_h1 *= b_g_last
        if K > 64:
            b_h2 *= b_g_last
        if K > 128:
            b_h3 *= b_g_last
        if K > 192:
            b_h4 *= b_g_last

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    # store final state slot-direct into the scratch pool; masked for eager
    # PAD (-1) rows exactly like the oracle's masked pool writeback.
    if slot >= 0:
        p_ht = tl.make_block_ptr(ht, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (1, K), (128, i_v * BV), (64, BV), (0, 1)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (1, K), (192, i_v * BV), (64, BV), (0, 1)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def bi_gdn_incr_defer_state_scatter_flushpt_kernel(
    scratch,  # [S, H, V, K] fp32 pool (final states from the gated fwd_h)
    ssm,  # [S, H, V, K] fp32 pool
    boundary,  # [S, H, V, K] fp32 pool
    slot_indices,
    seq_lens,
    num_slots,
    elem_per_entry: tl.constexpr,  # H*V*K
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TRACK_IVL: tl.constexpr,
):
    """bi_gdn_fast_state_scatter_kernel's body verbatim behind the SAME
    flush-point gate as the gated fwd_h (the pair must agree: scratch is
    only valid for gated rows).  Boundary advance stays completion-only."""
    pid_req = tl.program_id(0)
    pid_block = tl.program_id(1).to(tl.int64)

    slot = tl.load(slot_indices + pid_req).to(tl.int64)
    # Early exit for eager PAD (-1) rows: no state writes (oracle parity).
    if slot < 0:
        return
    # Bounds check to avoid illegal memory access
    if slot >= num_slots:
        return

    sl = tl.load(seq_lens + pid_req).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
    completed = fb + 1 == CHUNK
    if TRACK_IVL > 0:
        tracked = (sl % TRACK_IVL) == 0
    else:
        tracked = False
    if not (completed | tracked):
        return

    base = slot * elem_per_entry
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elem_per_entry

    data = tl.load(scratch + base + offsets, mask=mask)
    tl.store(ssm + base + offsets, data, mask=mask)
    if completed:
        # completed chunks advance the boundary; skipping the write when not
        # completed is byte-equivalent to the oracle's where-rewrite of the
        # unchanged value.
        tl.store(boundary + base + offsets, data, mask=mask)


# --- S7: chunk_fwd_o -- the ORACLE BINARY over the gap-encoded slabs ---------
# The o stage is the codegen-fragile kernel of this composition (probe-measured: its
# narrow tiles flip 1 ulp; the FAST path: even a verbatim COPY of the kernel with a
# rewritten preamble flipped 1 ulp at the wide-head shape -- the SSG0 class).
# Byte comparison falsified a slot-addressed 3D-grid copy the same way (1 bf16
# ulp, 1 element, H=4 ties cell).  So the o stage runs the oracle's own
# compiled binary (chunk_fwd_kernel_o) over the FAST path's certified gap-encoded
# varlen slab representation; a transport-only export kernel feeds it the
# cached l2q/l2k rows (with read-time token substitution, which also keeps
# PAD rows private + deterministic exactly like the FAST path's ingest).


@triton.jit
def bi_gdn_incr_l2_export_kernel(
    l2q_rows,  # [S, CHUNK, H, K] bf16 cache
    l2k_rows,  # [S, CHUNK, H, K] bf16 cache
    l2q_tok,  # [bs, H, K] bf16 (fresh token rows, post l2norm)
    l2k_tok,  # [bs, H, K] bf16
    q_slab,  # [bs*CHUNK, H, K] bf16
    k_slab,  # [bs*CHUNK, H, K] bf16
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    K: tl.constexpr,
    BHK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    i_b, i_r = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
    if i_r <= fb:  # rows > fb are never read (the varlen kernels mask by T)
        src_q = l2q_rows + (slot_c * CHUNK + i_r) * H * K
        src_k = l2k_rows + (slot_c * CHUNK + i_r) * H * K
        if i_r == fb:
            src_q = l2q_tok + i_b.to(tl.int64) * H * K
            src_k = l2k_tok + i_b.to(tl.int64) * H * K
        dst = (i_b.to(tl.int64) * CHUNK + i_r) * H * K
        o = tl.arange(0, BHK)
        m = o < H * K
        tl.store(q_slab + dst + o, tl.load(src_q + o, mask=m, other=0.0), mask=m)
        tl.store(k_slab + dst + o, tl.load(src_k + o, mask=m, other=0.0), mask=m)


# --- slim-output and transport-fusion kernels -------------------------------
# TRANSPORT ONLY (integer-surface freedom): byte-movement merges/re-addressing
# of the transport kernels above; every part's body is verbatim from its
# source kernel.  The arithmetic stage binaries, operands, and order are
# untouched. The race audit follows directly from disjoint destination spans:
# The append part's pool writes overlap other programs' pool reads only on the
# shared pad slot 0, which the existing conventions already exclude; live
# slots are unique and their row-fb reads are token-substituted).


@triton.jit
def bi_gdn_w3_prep_pack_kernel(
    rows_qkv,  # [S, CHUNK, QKV] bf16 pool
    rows_g,  # [S, CHUNK, H] fp32 pool
    rows_beta,  # [S, CHUNK, H] fp32 pool
    qkv_tok,  # [bs, QKV] bf16 current-token rows
    g_tok,  # [bs, H] fp32
    beta_tok,  # [bs, H] fp32
    slot_indices,
    seq_lens,
    g_slab,  # [bs*CHUNK, H] fp32 slab
    cu_gap,  # [2*bs+1] int32 request-major gap cu (odd entries refreshed)
    cu_slot,  # [2*bs+1] int32 slot-major gap cu (W3.2; both entries refreshed)
    q_tok,  # [bs, H, K] bf16 head-expanded token rows
    k_tok,  # [bs, H, K] bf16
    h_slab,  # [bs, H, K, V] bf16 (the o-stage h operand)
    boundary,  # [S, H, V, K] fp32 pool
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    QKV: tl.constexpr,
    BH: tl.constexpr,
    BHK: tl.constexpr,
    BQKV: tl.constexpr,
    BVH0: tl.constexpr,  # h0 tile BV (transport axis, sweepable)
    NT_TILES: tl.constexpr,  # cdiv(V, BVH0) * H
    TILES_PER_PROG: tl.constexpr,  # cdiv(NT_TILES, CHUNK)
    CHUNK: tl.constexpr,
    DO_APPEND: tl.constexpr,
    WITH_EXPAND: tl.constexpr,
    WITH_H0: tl.constexpr,
    WITH_CU_SLOT: tl.constexpr,
):
    """W3.3 F1: prep (+ cu refresh) + token expand + append + h0 export in
    one launch.  Parts: bi_gdn_incr_prep_kernel (always),
    bi_gdn_incr_token_expand_kernel (WITH_EXPAND, i_r == fb program),
    bi_gdn_fast_append_kernel (DO_APPEND, i_r == fb program, masked),
    bi_gdn_incr_defer_h0_export_kernel tiles (WITH_H0, block-ptr pairs
    verbatim, distributed over the CHUNK grid axis).  The early h0 read is
    byte-equal to 11a's late read: `boundary` is constant within a step
    until the gated scatter, which runs after the o stage."""
    i_b, i_r = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK  # torch.remainder semantics

    if i_r == 0:
        # refresh the request's logical extent for the varlen stage kernels
        tl.store(cu_gap + 2 * i_b + 1, i_b * CHUNK + fb + 1)
        if WITH_CU_SLOT:
            base32 = (slot_c * CHUNK).to(tl.int32)
            tl.store(cu_slot + 2 * i_b, base32)
            tl.store(cu_slot + 2 * i_b + 1, base32 + fb + 1)

    # --- prep part (bi_gdn_incr_prep_kernel body) ---
    src_g = rows_g + (slot_c * CHUNK + i_r) * H
    if i_r == fb:
        src_g = g_tok + i_b.to(tl.int64) * H
    o_h = tl.arange(0, BH)
    m_h = o_h < H
    b_g = tl.load(src_g + o_h, mask=m_h, other=0.0)
    tl.store(g_slab + (i_b.to(tl.int64) * CHUNK + i_r) * H + o_h, b_g, mask=m_h)

    # --- token expand part (bi_gdn_incr_token_expand_kernel body) ---
    if WITH_EXPAND:
        if i_r == fb:
            o_hk = tl.arange(0, BHK)
            m_hk = o_hk < H * K
            hh = o_hk // K
            kk = o_hk % K
            src_off = (hh // (H // HG)) * K + kk
            src = qkv_tok + i_b.to(tl.int64) * QKV
            b_q = tl.load(src + src_off, mask=m_hk, other=0.0)
            tl.store(q_tok + i_b.to(tl.int64) * H * K + o_hk, b_q, mask=m_hk)
            b_k = tl.load(src + HG * K + src_off, mask=m_hk, other=0.0)
            tl.store(k_tok + i_b.to(tl.int64) * H * K + o_hk, b_k, mask=m_hk)

    # --- append part (bi_gdn_fast_append_kernel body, masked) ---
    if DO_APPEND:
        if i_r == fb and slot >= 0:
            dst = (slot.to(tl.int64) * CHUNK + fb) * QKV
            o_q = tl.arange(0, BQKV)
            m_q = o_q < QKV
            b_qa = tl.load(qkv_tok + i_b.to(tl.int64) * QKV + o_q, mask=m_q, other=0.0)
            tl.store(rows_qkv + dst + o_q, b_qa, mask=m_q)
            b_ga = tl.load(g_tok + i_b.to(tl.int64) * H + o_h, mask=m_h, other=0.0)
            tl.store(
                rows_g + (slot.to(tl.int64) * CHUNK + fb) * H + o_h, b_ga, mask=m_h
            )
            b_ba = tl.load(beta_tok + i_b.to(tl.int64) * H + o_h, mask=m_h, other=0.0)
            tl.store(
                rows_beta + (slot.to(tl.int64) * CHUNK + fb) * H + o_h,
                b_ba,
                mask=m_h,
            )

    # --- h0 export part (bi_gdn_incr_defer_h0_export_kernel tile pairs) ---
    if WITH_H0:
        h_base = h_slab + i_b.to(tl.int64) * H * K * V
        h0_base = boundary + slot_c * H * V * K
        NV: tl.constexpr = NT_TILES // H
        for t in tl.static_range(TILES_PER_PROG):
            tile = i_r * TILES_PER_PROG + t
            if tile < NT_TILES:
                i_v = tile % NV
                i_h = tile // NV
                h0 = h0_base + i_h.to(tl.int64) * V * K
                hd = h_base + i_h.to(tl.int64) * K * V
                p_h0_1 = tl.make_block_ptr(
                    h0, (K, V), (1, K), (0, i_v * BVH0), (64, BVH0), (0, 1)
                )
                b_h1 = tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
                p_h1 = tl.make_block_ptr(
                    hd, (K, V), (V, 1), (0, i_v * BVH0), (64, BVH0), (1, 0)
                )
                tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    p_h0_2 = tl.make_block_ptr(
                        h0, (K, V), (1, K), (64, i_v * BVH0), (64, BVH0), (0, 1)
                    )
                    b_h2 = tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
                    p_h2 = tl.make_block_ptr(
                        hd, (K, V), (V, 1), (64, i_v * BVH0), (64, BVH0), (1, 0)
                    )
                    tl.store(
                        p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1)
                    )
                if K > 128:
                    p_h0_3 = tl.make_block_ptr(
                        h0, (K, V), (1, K), (128, i_v * BVH0), (64, BVH0), (0, 1)
                    )
                    b_h3 = tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
                    p_h3 = tl.make_block_ptr(
                        hd, (K, V), (V, 1), (128, i_v * BVH0), (64, BVH0), (1, 0)
                    )
                    tl.store(
                        p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1)
                    )
                if K > 192:
                    p_h0_4 = tl.make_block_ptr(
                        h0, (K, V), (1, K), (192, i_v * BVH0), (64, BVH0), (0, 1)
                    )
                    b_h4 = tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
                    p_h4 = tl.make_block_ptr(
                        hd, (K, V), (V, 1), (192, i_v * BVH0), (64, BVH0), (1, 0)
                    )
                    tl.store(
                        p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1)
                    )


@triton.jit
def bi_gdn_w3_export_pack_kernel(
    v_new_rows,  # [S, CHUNK, H, V] bf16 cache
    v_new_slab,  # [bs*CHUNK, H, V] bf16 slab
    l2q_rows,  # [S, CHUNK, H, K] bf16 cache
    l2k_rows,  # [S, CHUNK, H, K] bf16 cache
    l2q_tok,  # [bs, H, K] bf16
    l2k_tok,  # [bs, H, K] bf16
    q_slab,  # [bs*CHUNK, H, K] bf16
    k_slab,  # [bs*CHUNK, H, K] bf16
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BHK: tl.constexpr,
    BHV: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """W3.3 F2 (FUSE on, SLIM off): bi_gdn_incr_defer_vnew_export_kernel +
    bi_gdn_incr_l2_export_kernel in one launch (both bodies verbatim; both
    run after the vnew row band, which l2 export never depended on)."""
    i_b, i_r = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
    if i_r <= fb:  # rows > fb are never read (the varlen kernels mask by T)
        # vnew export part
        src_v = v_new_rows + (slot_c * CHUNK + i_r) * H * V
        dst_v = (i_b.to(tl.int64) * CHUNK + i_r) * H * V
        o_v = tl.arange(0, BHV)
        m_v = o_v < H * V
        tl.store(
            v_new_slab + dst_v + o_v,
            tl.load(src_v + o_v, mask=m_v, other=0.0),
            mask=m_v,
        )
        # l2 export part (token substitution at fb)
        src_q = l2q_rows + (slot_c * CHUNK + i_r) * H * K
        src_k = l2k_rows + (slot_c * CHUNK + i_r) * H * K
        if i_r == fb:
            src_q = l2q_tok + i_b.to(tl.int64) * H * K
            src_k = l2k_tok + i_b.to(tl.int64) * H * K
        dst = (i_b.to(tl.int64) * CHUNK + i_r) * H * K
        o = tl.arange(0, BHK)
        m = o < H * K
        tl.store(q_slab + dst + o, tl.load(src_q + o, mask=m, other=0.0), mask=m)
        tl.store(k_slab + dst + o, tl.load(src_k + o, mask=m, other=0.0), mask=m)


@triton.jit
def bi_gdn_w3_solve_commit_gcum_kernel(
    OutDiag,  # [bs, H, 16] fp32 staging
    OutSeg,  # [bs, H, 3, 16] fp32 staging
    Ai32,  # [S, CHUNK, H, CHUNK] fp32 cache
    Ai16,  # [S, CHUNK, H, CHUNK] bf16 cache
    gcum_slab,  # [bs*CHUNK, H] fp32 slab (this step's cumsum output)
    gcum_rows,  # [S, CHUNK, H] fp32 cache (W3.2)
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    BT: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """W3.2: bi_gdn_incr_solve_commit_kernel body verbatim + a one-float
    commit of this step's gcum row p into the slot-major gcum cache (the
    o stage's g operand under SLIM).  P1 fill-invariance (the probe program, 201,600
    checks) proves the committed row is bitwise what the per-step slab
    recompute produces at every later fill in the chunk."""
    i_b, i_h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    if slot >= 0:
        sl = tl.load(seq_lens + i_b).to(tl.int32)
        p = ((sl - 1) % CHUNK + CHUNK) % CHUNK
        d = p // 16
        o = tl.arange(0, 64)
        b_seg = tl.load(OutSeg + (i_b * H + i_h) * 48 + o, mask=o < d * 16, other=0.0)
        m_diag = (o >= d * 16) & (o < (d + 1) * 16)
        idx_diag = tl.where(m_diag, o - d * 16, 0)
        b_diag = tl.load(
            OutDiag + (i_b * H + i_h) * 16 + idx_diag, mask=m_diag, other=0.0
        )
        val = tl.where(o < d * 16, b_seg, tl.where(m_diag, b_diag, 0.0))
        base = (slot.to(tl.int64) * CHUNK + p) * H * BT + i_h * BT
        tl.store(Ai32 + base + o, val)
        tl.store(Ai16 + base + o, val.to(tl.bfloat16))
        # W3.2 gcum row-p commit (pure transport: slab value -> cache)
        gv = tl.load(gcum_slab + (i_b.to(tl.int64) * CHUNK + p) * H + i_h)
        tl.store(gcum_rows + (slot.to(tl.int64) * CHUNK + p) * H + i_h, gv)


@triton.jit
def bi_gdn_w3_scatter_gather_kernel(
    scratch,  # [S, H, V, K] fp32 pool
    ssm,  # [S, H, V, K] fp32 pool
    boundary,  # [S, H, V, K] fp32 pool
    o_src,  # o slab (request-major) or o_slot buffer (slot-major, SLIM)
    out,  # [bs, H, V] bf16
    slot_indices,
    seq_lens,
    num_slots,
    elem_per_entry: tl.constexpr,  # H*V*K
    row_elems: tl.constexpr,  # H*V
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TRACK_IVL: tl.constexpr,
    GATED: tl.constexpr,
    SLOT_MAJOR: tl.constexpr,
):
    """W3.3 F3: state scatter (flush-gated under DEFER via GATED, ungated
    under v1/FAST semantics) + out gather in one launch.  The gather part is
    unconditional exactly like bi_gdn_fast_out_gather_kernel; the scatter
    part keeps bi_gdn_fast_state_scatter_kernel /
    bi_gdn_incr_defer_state_scatter_flushpt_kernel's guards and gates
    verbatim.  elem_per_entry >= row_elems always, so the gather blocks are
    a prefix of the grid."""
    pid_req = tl.program_id(0)
    pid_block = tl.program_id(1).to(tl.int64)
    sl = tl.load(seq_lens + pid_req).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
    slot = tl.load(slot_indices + pid_req).to(tl.int64)

    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # --- out gather part ---
    if start < row_elems:
        m_g = offsets < row_elems
        if SLOT_MAJOR:
            src = (tl.maximum(slot, 0) * CHUNK + fb) * row_elems
        else:
            src = (pid_req.to(tl.int64) * CHUNK + fb) * row_elems
        d_g = tl.load(o_src + src + offsets, mask=m_g)
        tl.store(out + pid_req.to(tl.int64) * row_elems + offsets, d_g, mask=m_g)

    # --- state scatter part ---
    if slot >= 0:
        if slot < num_slots:
            completed = fb + 1 == CHUNK
            if GATED:
                if TRACK_IVL > 0:
                    tracked = (sl % TRACK_IVL) == 0
                else:
                    tracked = False
                do_write = completed | tracked
            else:
                do_write = True
            if do_write:
                base = slot * elem_per_entry
                mask = offsets < elem_per_entry
                data = tl.load(scratch + base + offsets, mask=mask)
                tl.store(ssm + base + offsets, data, mask=mask)
                if completed:
                    tl.store(boundary + base + offsets, data, mask=mask)


@triton.jit
def bi_gdn_w3_slim_out_gather_kernel(
    o_slot,  # [S*CHUNK, H, V] bf16 slot-major o buffer
    out,  # [bs, H, V] bf16
    slot_indices,
    seq_lens,
    row_elems: tl.constexpr,  # H*V
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """W3.2 (SLIM on, FUSE off): bi_gdn_fast_out_gather_kernel re-addressed
    to the slot-major o buffer (clamped reads for eager PAD rows -- their
    outputs are DP-sync dummies, the existing convention)."""
    pid_req = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1).to(tl.int64)
    slot = tl.load(slot_indices + pid_req)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + pid_req).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
    src = (slot_c * CHUNK + fb) * row_elems
    dst = pid_req * row_elems
    offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_elems
    data = tl.load(o_slot + src + offsets, mask=mask)
    tl.store(out + dst + offsets, data, mask=mask)


# --- per-layer slot caches ----------------------------------------------------


@dataclass
class BIGDNIncrLayerCaches:
    """Per-slot per-layer intermediate row caches (fixed-address device
    tensors; content is the only state -- see module docstring)."""

    l2q: torch.Tensor
    l2k: torch.Tensor
    A: torch.Tensor
    Ai32: torch.Tensor
    Ai16: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    # DEFER: allocated only under the writeback deferral
    # (the deferral's mid-chunk o-stage input; v1 recomputed it in fwd_h).
    v_new: torch.Tensor | None = None
    # W3.2: per-slot gcum row cache (the slim o stage's g operand);
    # allocated only under the slim o-stage.
    gcum: torch.Tensor | None = None

    @staticmethod
    def allocate(
        num_slots: int,
        h: int,
        k: int,
        v: int,
        device: torch.device,
        with_v_new: bool = False,
        with_gcum: bool = False,
    ) -> BIGDNIncrLayerCaches:
        bf16, fp32 = torch.bfloat16, torch.float32
        # zero-init => tails are stale-but-FINITE from the first step on
        # (the certified operating condition); zeros also satisfy the two
        # structural invariants (A strictly-lower masked, Ai upper cols 0).
        z = lambda *sh, dt: torch.zeros(*sh, dtype=dt, device=device)
        return BIGDNIncrLayerCaches(
            l2q=z(num_slots, CHUNK_SIZE, h, k, dt=bf16),
            l2k=z(num_slots, CHUNK_SIZE, h, k, dt=bf16),
            A=z(num_slots, CHUNK_SIZE, h, CHUNK_SIZE, dt=fp32),
            Ai32=z(num_slots, CHUNK_SIZE, h, CHUNK_SIZE, dt=fp32),
            Ai16=z(num_slots, CHUNK_SIZE, h, CHUNK_SIZE, dt=bf16),
            w=z(num_slots, CHUNK_SIZE, h, k, dt=bf16),
            u=z(num_slots, CHUNK_SIZE, h, v, dt=bf16),
            v_new=z(num_slots, CHUNK_SIZE, h, v, dt=bf16) if with_v_new else None,
            gcum=z(num_slots, CHUNK_SIZE, h, dt=fp32) if with_gcum else None,
        )


# --- driver --------------------------------------------------------------------


class BIGDNIncrDecodeRunner(BIGDNFastDecodeRunner):
    """Incremental-exact decode runner (one instance per backend; per-step
    slabs shared across layers, per-slot row caches per layer)."""

    _INCR_ATTR = "_bi_incr_layer_caches"

    def __init__(self) -> None:
        super().__init__()
        # sweepable launch axes for the incremental kernels; defaults mirror
        # the probe-certified configs (the probe program P2/P4).
        self.kkt_num_warps = 8
        self.kkt_num_stages = 3
        # 4 warps, not the probe-certified 2: Triton 3.6.0 miscompiles the
        # D=1 specialization's 16x16 fp32 ieee dot chain at num_warps<=2 --
        # the OutSeg merge row silently stores zeros (TTIR/PTX correct and
        # unconditional; the dot pipeline produces zeros), so rows 16..31 of
        # every chunk go byte-wrong. At num_warps>=4 the kernel's row is
        # bitwise equal to the full-solve oracle (solve_tril and
        # solve_tril_decode agree bitwise). Covered by
        # test_bi_gdn_incr_step_oracle.py (step-vs-rescan byte A/B + solve
        # row vs full solve across launch configs).
        self.solve_num_warps = 4
        self.solve_num_stages = 3
        self.wu_num_warps = 4
        self.wu_num_stages = 3
        self.prep_num_warps = 4
        self.export_num_warps = 4
        # fwd_h axes inherited from the fast runner (fwd_h_num_warps/stages/bv)
        # v1.1 hybrid: the INCR slab store is SEPARATE from the inherited
        # the FAST path slab store (self._slabs/_bs/_dims), because the eager branch
        # runs the FAST program (which owns those) in the same process.
        self._incr_bs = 0
        self._incr_dims: tuple | None = None
        self._incr_slabs: dict[str, torch.Tensor] = {}
        self._capture_mode_fn = None
        # Writeback deferral is selected once by the architecture resolver
        # before runner construction.
        self.defer_writeback = BI_GDN_INCR_DEFER_ENABLED
        # mamba-track interval for the in-capture flush gate; 0 = no track
        # machinery (no_buffer strategy).  Set by the backend at init from
        # server args (a process constant -- safe to bake into captures).
        self.track_interval = 0
        # DEFER sweepable launch axes (probe-certified defaults: vnew BV=32
        # is the oracle's check_shared_mem()->False pin, w4s2 the P2 config).
        self.vnew_num_warps = 4
        self.vnew_num_stages = 2
        self.h0_export_num_warps = 4
        self.vnew_export_num_warps = 4
        # The slim v_new path is consulted only in the deferred branch;
        # fuse_small is captured from the resolver-owned fast-runner choice.
        self.slim_vnew = BI_GDN_VNEW_SLIM_ENABLED
        # W3 sweepable transport launch axes
        self.prep_pack_num_warps = 4
        self.export_pack_num_warps = 4
        self.h0_tile_bv = 32

    def _graph_path_active(self) -> bool:
        """True under graph capture (including capture-mode warmups) or an
        actively capturing stream -- the contexts whose work is replayed."""
        if torch.cuda.is_current_stream_capturing():
            return True
        if self._capture_mode_fn is None:
            # lazy import: the fla layer must not import model_executor at
            # module scope (import cycle).
            from sglang.srt.model_executor.runner_utils.capture_mode import (
                get_is_capture_mode,
            )

            self._capture_mode_fn = get_is_capture_mode
        return bool(self._capture_mode_fn())

    # -- allocation ------------------------------------------------------------

    def _layer_caches(self, cache) -> BIGDNIncrLayerCaches:
        state = getattr(cache, self._INCR_ATTR, None)
        if state is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "the INCR mode: per-layer caches must be "
                    "allocated before graph capture (warmup allocates them); "
                    "allocation during capture is a bug (the INCR path)."
                )
            hv, hg = (
                cache.hv,
                (cache.qkv_dim - cache.hv * cache.v) // (2 * cache.k),
            )
            del hg
            state = BIGDNIncrLayerCaches.allocate(
                num_slots=cache.boundary.shape[0],
                h=hv,
                k=cache.k,
                v=cache.v,
                device=cache.boundary.device,
                with_v_new=self.defer_writeback,
                with_gcum=self.defer_writeback and self.slim_vnew,
            )
            setattr(cache, self._INCR_ATTR, state)
        if self.defer_writeback and state.v_new is None:
            # a cache first touched by a non-defer runner (tests): attach the
            # the DEFER program cache lazily -- eager only, same class as allocate().
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "the writeback deferral: the v_new cache must "
                    "be allocated before graph capture (the DEFER program)."
                )
            state.v_new = torch.zeros_like(state.u)
        if self.defer_writeback and self.slim_vnew and state.gcum is None:
            # W3.2: same lazy-attach class as the v_new cache above.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "the slim o-stage: the gcum cache must be "
                    "allocated before graph capture (W3.2)."
                )
            state.gcum = torch.zeros(
                state.u.shape[0],
                CHUNK_SIZE,
                state.u.shape[2],
                dtype=torch.float32,
                device=state.u.device,
            )
        return state

    def _ensure_incr_slabs(self, bs: int, cache, device: torch.device) -> None:
        hv, hg = cache.hv, (cache.qkv_dim - cache.hv * cache.v) // (2 * cache.k)
        dims = (hv, hg, cache.k, cache.v, cache.qkv_dim)
        if self._incr_bs >= bs and self._incr_dims == dims:
            return
        if self._incr_dims is not None and self._incr_dims != dims:
            raise RuntimeError(
                "the INCR mode: GDN layer dims changed across "
                f"layers: {self._incr_dims} != {dims}."
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the INCR mode: slabs must be allocated before "
                "graph capture (the INCR path)."
            )
        bs = max(bs, self._incr_bs)
        h, k, v = hv, cache.k, cache.v
        rows = bs * CHUNK_SIZE
        bf16, fp32 = torch.bfloat16, torch.float32
        cu_gap = torch.zeros(2 * bs + 1, dtype=torch.int32, device=device)
        cu_gap[0::2] = (
            torch.arange(bs + 1, dtype=torch.int32, device=device) * CHUNK_SIZE
        )
        cu_gap[1::2] = cu_gap[0:-1:2]
        chunk_indices = torch.stack(
            (
                torch.arange(bs, dtype=torch.int32, device=device) * 2,
                torch.zeros(bs, dtype=torch.int32, device=device),
            ),
            dim=1,
        ).contiguous()
        o_slot_keep = self._incr_slabs.get("o_slot")
        self._incr_slabs = {
            "q_tok": torch.zeros(bs, h, k, dtype=bf16, device=device),
            "k_tok": torch.zeros(bs, h, k, dtype=bf16, device=device),
            "l2q_tok": torch.zeros(bs, h, k, dtype=bf16, device=device),
            "l2k_tok": torch.zeros(bs, h, k, dtype=bf16, device=device),
            "q_norm": torch.zeros(rows, h, k, dtype=bf16, device=device),
            "k_norm": torch.zeros(rows, h, k, dtype=bf16, device=device),
            "g": torch.zeros(rows, h, dtype=fp32, device=device),
            "g_cum": torch.zeros(rows, h, dtype=fp32, device=device),
            "v_new": torch.zeros(rows, h, v, dtype=bf16, device=device),
            "h": torch.zeros(bs, h, k, v, dtype=bf16, device=device),
            "o": torch.zeros(rows, h, v, dtype=bf16, device=device),
            "solve_diag": torch.zeros(bs, h, 16, dtype=fp32, device=device),
            "solve_seg": torch.zeros(bs, h, 3, 16, dtype=fp32, device=device),
            # W3.2: slot-major gap cu for the slim o stage; content refreshed
            # in-kernel every step (both entries dynamic), used only when
            # slim engages.  Cheap, allocated unconditionally.
            "cu_slot": torch.zeros(2 * bs + 1, dtype=torch.int32, device=device),
            "cu_gap": cu_gap,
            "chunk_indices": chunk_indices,
        }
        if o_slot_keep is not None:
            # W3.2: the slot-major o buffer is keyed by the POOL slot count,
            # not bs -- carry it across slab regrowth.
            self._incr_slabs["o_slot"] = o_slot_keep
        self._incr_bs = bs
        self._incr_dims = dims

    def _ensure_o_slot(self, cache, h: int, v: int) -> torch.Tensor:
        """W3.2: slot-major o output buffer [S*CHUNK, H, V] (per-step
        scratch shared across layers; fixed address for capture)."""
        num_rows = cache.boundary.shape[0] * CHUNK_SIZE
        buf = self._incr_slabs.get("o_slot")
        if buf is not None and buf.shape[0] == num_rows:
            return buf
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the slim o-stage: the o_slot buffer must be "
                "allocated before graph capture (W3.2)."
            )
        buf = torch.zeros(
            num_rows, h, v, dtype=torch.bfloat16, device=cache.boundary.device
        )
        self._incr_slabs["o_slot"] = buf
        return buf

    # -- extend-time cache warm-up ----------------------------------------------

    @torch.no_grad()
    def warm_slot(self, cache, slot: int, fill: int) -> None:
        """Export the partial suffix's per-row intermediates into the caches.

        Runs eagerly per request after ``seed_from_extend`` (never inside
        decode capture/replay).  Uses the STOCK stage drivers at the suffix
        fill; P1 fill-invariance proves the exported rows equal the values
        the incremental per-step kernels would have committed, and the fp32
        solve export is the same kernel with a different store dtype.
        """
        state = self._layer_caches(cache)
        if fill <= 0:
            return
        hv, k_dim, v_dim = cache.hv, cache.k, cache.v
        hg = (cache.qkv_dim - hv * v_dim) // (2 * k_dim)
        device = cache.boundary.device
        q, k, v = cache._split(cache.rows_qkv[slot, :fill])
        rep = hv // hg
        qi = q.repeat_interleave(rep, dim=2).contiguous()
        ki = k.repeat_interleave(rep, dim=2).contiguous()
        l2q = l2norm_fwd(qi)
        l2k = l2norm_fwd(ki)
        cu = torch.tensor([0, fill], dtype=torch.int32, device=device)
        ci = torch.tensor([[0, 0]], dtype=torch.int32, device=device)
        g_rows = cache.rows_g[slot, :fill].view(1, fill, hv)
        beta_rows = cache.rows_beta[slot, :fill].view(1, fill, hv)
        gcum = chunk_local_cumsum(
            g_rows, chunk_size=CHUNK_SIZE, cu_seqlens=cu, chunk_indices=ci
        )
        A = chunk_scaled_dot_kkt_fwd(
            l2k,
            beta_rows,
            g_cumsum=gcum,
            cu_seqlens=cu,
            output_dtype=torch.float32,
            chunk_indices=ci,
        )
        if BI_GDN_SOLVE_TRIL_DECODE:
            # the defaults consolidation composition: mirror the call-site dispatch of
            # bi_chunk_gated_delta_rule_prefill (the solve port) and the fast runner
            # (db97305f7) so this warm path launches the same solve kernels
            # as every other full-solve site (single flag authority,
            # bi_gdn_prefill.BI_GDN_SOLVE_TRIL_DECODE). Eager, per request,
            # never under capture -- solve_tril_decode's per-call Di scratch
            # is fine here; the capture-safety Di slab (db97305f7) is only
            # required on captured paths.
            ai32 = solve_tril_decode(
                A=A, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32
            )
        else:
            ai32 = solve_tril(
                A=A, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32
            )
        ai16 = ai32.to(torch.bfloat16)  # rtne; gate G3 == the bf16 export
        w, u = recompute_w_u_fwd(
            l2k,
            v.contiguous(),
            beta_rows,
            g_cumsum=gcum,
            A=ai16,
            cu_seqlens=cu,
            chunk_indices=ci,
        )
        state.l2q[slot, :fill] = l2q[0]
        state.l2k[slot, :fill] = l2k[0]
        state.A[slot, :fill] = A[0]
        state.Ai32[slot, :fill] = ai32[0]
        state.Ai16[slot, :fill] = ai16[0]
        state.w[slot, :fill] = w[0]
        state.u[slot, :fill] = u[0]
        if self.defer_writeback and self.slim_vnew:
            # W3.2: seed the gcum row cache (P1: suffix rows equal
            # the values the per-step commit would have written).
            state.gcum[slot, :fill] = gcum[0]
        if self.defer_writeback:
            # DEFER: warm the v_new rows through the STOCK fwd_h
            # driver at the suffix fill (P1: rows == the per-step committed
            # values; h0 = the just-seeded chunk boundary).
            co = torch.arange(2, dtype=torch.int32, device=device)
            h0w = cache.boundary[slot].transpose(-1, -2).unsqueeze(0).contiguous()
            _, vnew_rows, _ = chunk_gated_delta_rule_fwd_h(
                k=l2k,
                w=w,
                u=u,
                g=gcum,
                initial_state=h0w,
                output_final_state=False,
                cu_seqlens=cu,
                chunk_indices=ci,
                chunk_offsets=co,
            )
            state.v_new[slot, :fill] = vnew_rows[0]

    # -- DEFER: eager flush (host-triggered, outside capture) ----------

    @torch.no_grad()
    def flush_slots(
        self,
        cache,
        ssm_states: torch.Tensor,
        slots: list[int],
        seq_lens_list: list[int],
    ) -> None:
        """Materialize the deferred pool state for the given slots at their
        current fills: ssm_states[slot] <- state after seq_len consumed
        tokens.  Runs the UNGATED v1 binaries (certified cumsum launch over
        a g slab rebuilt from rows_g -- byte-identical to the step's
        substituted slab because the append kernel persisted the token row
        -- then v1's fwd_h at T=fill and v1's stock scatter).

        Slots at seq_len % 64 == 0 are SKIPPED: the completion step already
        materialized their state, and re-running fwd_h at T=64 after the
        boundary advance would re-apply the chunk (double-application
        hazard).  Idempotent for mid-chunk fills (same binaries, same
        inputs, same bytes).  Eager only -- never inside capture.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the writeback deferral: flush_slots is a "
                "host-side eager event and must never run under graph "
                "capture (the DEFER program)."
            )
        todo = [
            (s, L)
            for s, L in zip(slots, seq_lens_list)
            if L % CHUNK_SIZE != 0 and L > 0
        ]
        if not todo:
            return
        state = self._layer_caches(cache)
        device = cache.boundary.device
        h, k, v = cache.hv, cache.k, cache.v
        n = len(todo)
        rows = n * CHUNK_SIZE
        idx = torch.tensor([s for s, _ in todo], dtype=torch.int32, device=device)
        sl = torch.tensor([L for _, L in todo], dtype=torch.int64, device=device)
        # g slab: transport from the rows_g pool (identical bytes to the last
        # step's substituted slab -- append persisted the token row).
        g_slab = cache.rows_g.index_select(0, idx.long()).reshape(rows, h).contiguous()
        g_cum = torch.empty_like(g_slab)
        cu = torch.zeros(2 * n + 1, dtype=torch.int32, device=device)
        cu[0::2] = torch.arange(n + 1, dtype=torch.int32, device=device) * CHUNK_SIZE
        cu[1::2] = cu[0:-1:2] + (
            torch.remainder(sl - 1, CHUNK_SIZE).to(torch.int32) + 1
        )
        ci = torch.stack(
            (
                torch.arange(n, dtype=torch.int32, device=device) * 2,
                torch.zeros(n, dtype=torch.int32, device=device),
            ),
            dim=1,
        ).contiguous()
        chunk_local_cumsum_scalar_kernel[(n, h)](
            s=g_slab,
            o=g_cum,
            scale=None,
            cu_seqlens=cu,
            chunk_indices=ci,
            T=rows,
            B=1,
            H=h,
            BT=CHUNK_SIZE,
            HEAD_FIRST=False,
            REVERSE=False,
            HAS_SCALE=False,
            IS_VARLEN=True,
            num_warps=8,
            num_stages=3,
        )
        v_new_tmp = torch.empty(rows, h, v, dtype=torch.bfloat16, device=device)
        h_tmp = torch.empty(n, h, k, v, dtype=torch.bfloat16, device=device)
        bi_gdn_incr_fwd_h_kernel[(triton.cdiv(v, self.fwd_h_bv), n * h)](
            state.l2k,
            state.u,
            state.w,
            v_new_tmp,
            g_cum,
            h_tmp,
            cache.boundary,
            cache.scratch,
            idx,
            sl,
            H=h,
            K=k,
            V=v,
            BT=CHUNK_SIZE,
            BV=self.fwd_h_bv,
            num_warps=self.fwd_h_num_warps,
            num_stages=self.fwd_h_num_stages,
        )
        elem = h * v * k
        block = self.scatter_block
        bi_gdn_fast_state_scatter_kernel[(n, triton.cdiv(elem, block))](
            cache.scratch,
            ssm_states,
            cache.boundary,
            idx,
            sl,
            ssm_states.shape[0],
            elem_per_entry=elem,
            CHUNK=CHUNK_SIZE,
            BLOCK_SIZE=block,
        )

    # -- the per-step pipeline ----------------------------------------------------

    def step(
        self,
        cache,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        qkv_rows: torch.Tensor,
        g_rows: torch.Tensor,
        beta_rows: torch.Tensor,
        ssm_states: torch.Tensor,
    ) -> torch.Tensor:
        """One batched single-token decode step -- v1.1 HYBRID dispatch.

        Same contract as ``BIGDNFastDecodeRunner.step``: indices [bs] int32
        mamba slots (graph pads -> reserved slot 0; eager PAD rows -> -1,
        clamp+masked), seq_lens content-authoritative for the fill; returns
        core_attn_out [bs, HV, V] bf16 and preserves stock pool semantics
        (per-step ssm writeback; boundary advance at chunk completion).

        Graph capture (and capture-mode warmups) -> the full incremental
        program (so replay replays it).  True eager -> the FAST path's byte-identical
        FAST program plus the cache-maintenance head (module docstring: the
        caches stay fresh on every path; no staleness tracking exists).
        """
        if self._graph_path_active():
            return self._step_incr(
                cache, indices, seq_lens, qkv_rows, g_rows, beta_rows, ssm_states
            )
        out = BIGDNFastDecodeRunner.step(
            self,
            cache=cache,
            indices=indices,
            seq_lens=seq_lens,
            qkv_rows=qkv_rows,
            g_rows=g_rows,
            beta_rows=beta_rows,
            ssm_states=ssm_states,
        )
        self._step_incr(
            cache,
            indices,
            seq_lens,
            qkv_rows,
            g_rows,
            beta_rows,
            ssm_states,
            head_only=True,
        )
        return out

    def _step_incr(
        self,
        cache,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        qkv_rows: torch.Tensor,
        g_rows: torch.Tensor,
        beta_rows: torch.Tensor,
        ssm_states: torch.Tensor,
        head_only: bool = False,
    ):
        """The incremental program.  ``head_only=True`` is the eager-branch
        cache-maintenance mode: stages 1 + 3-10 only (no append -- the FAST
        program already persisted the token row; no fwd_h/o/epilogue -- FAST
        already produced the state and output).  The maintenance head never
        reads ``boundary``/``scratch``, so running it after FAST's epilogue
        (which may advance the boundary on completed chunks) is order-safe.
        """
        bs = qkv_rows.shape[0]
        device = qkv_rows.device
        self._ensure_incr_slabs(bs, cache, device)
        rows_c = self._layer_caches(cache)
        s = self._incr_slabs
        h, hg, k, v = cache.hv, self._incr_dims[1], cache.k, cache.v
        qkv_dim = cache.qkv_dim
        rows = bs * CHUNK_SIZE
        cu = s["cu_gap"][: 2 * bs + 1]
        ci = s["chunk_indices"][:bs]

        # The slim transport engages only in the deferred branch.
        use_fuse = self.fuse_small
        use_slim = self.slim_vnew and self.defer_writeback

        # 1-3. g slab + cu refresh, append, token expand (+ W3 folds)
        if use_fuse or use_slim:
            # W3.3 F1 / W3.2 cu_slot refresh: one prep-pack launch replaces
            # prep [+ append + expand when FUSE; + h0 export when FUSE on the
            # defer graph path].  Under SLIM-only the extras stay separate
            # launches (launch structure unchanged; only cu_slot is added).
            with_h0 = use_fuse and self.defer_writeback and not head_only
            nv_tiles = (triton.cdiv(v, self.h0_tile_bv) * h) if with_h0 else 1
            bi_gdn_w3_prep_pack_kernel[(bs, CHUNK_SIZE)](
                cache.rows_qkv,
                cache.rows_g,
                cache.rows_beta,
                qkv_rows,
                g_rows,
                beta_rows,
                indices,
                seq_lens,
                s["g"],
                cu,
                s["cu_slot"][: 2 * bs + 1],
                s["q_tok"],
                s["k_tok"],
                s["h"],
                cache.boundary,
                H=h,
                HG=hg,
                K=k,
                V=v,
                QKV=qkv_dim,
                BH=self._next_pow2(h),
                BHK=self._next_pow2(h * k),
                BQKV=self._next_pow2(qkv_dim),
                BVH0=self.h0_tile_bv,
                NT_TILES=nv_tiles,
                TILES_PER_PROG=triton.cdiv(nv_tiles, CHUNK_SIZE),
                CHUNK=CHUNK_SIZE,
                DO_APPEND=use_fuse and not head_only,
                WITH_EXPAND=use_fuse,
                WITH_H0=with_h0,
                WITH_CU_SLOT=use_slim and not head_only,
                num_warps=self.prep_pack_num_warps,
            )
        else:
            bi_gdn_incr_prep_kernel[(bs, CHUNK_SIZE)](
                cache.rows_g,
                g_rows,
                indices,
                seq_lens,
                s["g"],
                cu,
                H=h,
                BH=self._next_pow2(h),
                CHUNK=CHUNK_SIZE,
                num_warps=self.prep_num_warps,
            )
        # 2. persist the token row into the rows_* pools (the FAST path's kernel);
        #    skipped in maintenance mode (FAST already appended these bytes)
        #    and under FUSE (folded into the prep pack)
        if not head_only and not use_fuse:
            bi_gdn_fast_append_kernel[(bs,)](
                cache.rows_qkv,
                cache.rows_g,
                cache.rows_beta,
                qkv_rows,
                g_rows,
                beta_rows,
                indices,
                seq_lens,
                H=h,
                QKV=qkv_dim,
                BQKV=self._next_pow2(qkv_dim),
                BH=self._next_pow2(h),
                CHUNK=CHUNK_SIZE,
                num_warps=4,
            )
        # 3. token q/k head expansion (transport; folded under FUSE)
        if not use_fuse:
            bi_gdn_incr_token_expand_kernel[(bs,)](
                qkv_rows,
                s["q_tok"],
                s["k_tok"],
                H=h,
                HG=hg,
                K=k,
                QKV=qkv_dim,
                BHK=self._next_pow2(h * k),
                num_warps=4,
            )
        # 4. STOCK l2norm over the bs*h fresh rows only
        l2_rows = bs * h
        for src, dst in ((s["q_tok"], s["l2q_tok"]), (s["k_tok"], s["l2k_tok"])):
            l2norm_fwd_kernel[(triton.cdiv(l2_rows, 16),)](
                src,
                dst,
                1e-6,
                T=l2_rows,
                D=k,
                BD=self._next_pow2(k),
                BT=16,
                num_warps=8,
                num_stages=3,
            )
        # 5. commit the fresh l2 rows into the caches at row p
        bi_gdn_incr_l2_commit_kernel[(bs,)](
            s["l2q_tok"],
            s["l2k_tok"],
            rows_c.l2q,
            rows_c.l2k,
            indices,
            seq_lens,
            H=h,
            K=k,
            BHK=self._next_pow2(h * k),
            CHUNK=CHUNK_SIZE,
            num_warps=4,
        )
        # 6. STOCK varlen cumsum over the g slab (the FAST path's certified launch;
        #    the serial increment is NOT bit-safe -- always rerun the kernel)
        chunk_local_cumsum_scalar_kernel[(bs, h)](
            s=s["g"],
            o=s["g_cum"],
            scale=None,
            cu_seqlens=cu,
            chunk_indices=ci,
            T=rows,
            B=1,
            H=h,
            BT=CHUNK_SIZE,
            HEAD_FIRST=False,
            REVERSE=False,
            HAS_SCALE=False,
            IS_VARLEN=True,
            num_warps=8,
            num_stages=3,
        )
        # 7. kkt row band (BM=16)
        bi_gdn_incr_kkt_rowband_kernel[(bs, h)](
            rows_c.l2k,
            cache.rows_beta,
            s["g_cum"],
            rows_c.A,
            indices,
            seq_lens,
            H=h,
            K=k,
            BT=CHUNK_SIZE,
            BK=64,
            BM=16,
            CHUNK=CHUNK_SIZE,
            num_warps=self.kkt_num_warps,
            num_stages=self.kkt_num_stages,
        )
        # 8. solve row (4 constexpr-D launches; content early-return dispatch)
        for d in range(4):
            bi_gdn_incr_solve_row_kernel[(bs, h)](
                rows_c.A,
                rows_c.Ai32,
                s["solve_diag"],
                s["solve_seg"],
                indices,
                seq_lens,
                H=h,
                BT=CHUNK_SIZE,
                D=d,
                CHUNK=CHUNK_SIZE,
                DOT_PRECISION="ieee",
                num_warps=self.solve_num_warps,
                num_stages=self.solve_num_stages,
            )
        # 9. commit the assembled fp32 row + rtne bf16 row (upper cols zero)
        #    [+ W3.2: the gcum row-p commit for the slim o stage]
        if use_slim:
            bi_gdn_w3_solve_commit_gcum_kernel[(bs, h)](
                s["solve_diag"],
                s["solve_seg"],
                rows_c.Ai32,
                rows_c.Ai16,
                s["g_cum"],
                rows_c.gcum,
                indices,
                seq_lens,
                H=h,
                BT=CHUNK_SIZE,
                CHUNK=CHUNK_SIZE,
                num_warps=2,
            )
        else:
            bi_gdn_incr_solve_commit_kernel[(bs, h)](
                s["solve_diag"],
                s["solve_seg"],
                rows_c.Ai32,
                rows_c.Ai16,
                indices,
                seq_lens,
                H=h,
                BT=CHUNK_SIZE,
                CHUNK=CHUNK_SIZE,
                num_warps=2,
            )
        # 10. w/u row band (BM=16)
        bi_gdn_incr_wu_rowband_kernel[(bs, h)](
            rows_c.l2k,
            cache.rows_qkv,
            cache.rows_beta,
            rows_c.w,
            rows_c.u,
            rows_c.Ai16,
            s["g_cum"],
            indices,
            seq_lens,
            H=h,
            HG=hg,
            K=k,
            V=v,
            QKV=qkv_dim,
            BT=CHUNK_SIZE,
            BK=64,
            BV=64,
            BM=16,
            CHUNK=CHUNK_SIZE,
            num_warps=self.wu_num_warps,
            num_stages=self.wu_num_stages,
        )
        if head_only:
            # eager-branch maintenance ends here for the ROW caches: they now
            # hold rows <= p for every live slot, matching what the full
            # incremental step would have committed (same kernels, same input
            # bytes).  the composed head (composed head): under writeback
            # deferral the persistent v_new cache is ALSO maintenance state
            # (the deferred program's o stage reads it mid-chunk instead of
            # recomputing v_new inside fwd_h), so its row band is maintained
            # here too -- via the UNMODIFIED the DEFER program kernel, with
            # chunk-COMPLETED slots masked to PAD (-1) through the kernel's
            # certified slot-masking:
            #   (a) order safety: the maintenance head runs AFTER FAST's
            #       epilogue, which has already advanced ``boundary`` for
            #       exactly the completed slots this step; vnew_rowband reads
            #       ``boundary`` as h0, so unmasked completed slots would
            #       consume the NEXT chunk's h0 (the one boundary read the
            #       maintenance head would otherwise perform);
            #   (b) deadness: a completed chunk's band rows are never read
            #       again -- every row j <= p of the NEXT chunk is rewritten
            #       by that chunk's own band writes (graph 11b or this
            #       maintenance) before any o-stage/export read of row j, and
            #       flush_slots/warm_slot never read v_new.
            # Divergence vs the all-graph trajectory is therefore confined to
            # dead bytes (row fill-1 of chunks completed on an eager step);
            # the hybrid-DEFER transition test byte-gates the trajectory.
            # 11a/11c fill per-step o-stage slabs (not cache state): skipped.
            if self.defer_writeback:
                indices_maint = indices.masked_fill((seq_lens % CHUNK_SIZE) == 0, -1)
                bi_gdn_incr_defer_vnew_rowband_kernel[(triton.cdiv(v, 32), bs * h)](
                    rows_c.w,
                    rows_c.u,
                    cache.boundary,
                    rows_c.v_new,
                    indices_maint,
                    seq_lens,
                    H=h,
                    K=k,
                    V=v,
                    BV=32,
                    BM=16,
                    CHUNK=CHUNK_SIZE,
                    num_warps=self.vnew_num_warps,
                    num_stages=self.vnew_num_stages,
                )
            return None
        if not self.defer_writeback:
            # 11. fwd_h at T=fill from the caches: v_new slab + h[0] slab +
            #     per-step state writeback (boundary pool -> scratch pool)
            bi_gdn_incr_fwd_h_kernel[(triton.cdiv(v, self.fwd_h_bv), bs * h)](
                rows_c.l2k,
                rows_c.u,
                rows_c.w,
                s["v_new"],
                s["g_cum"],
                s["h"],
                cache.boundary,
                cache.scratch,
                indices,
                seq_lens,
                H=h,
                K=k,
                V=v,
                BT=CHUNK_SIZE,
                BV=self.fwd_h_bv,
                num_warps=self.fwd_h_num_warps,
                num_stages=self.fwd_h_num_stages,
            )
        else:
            # the DEFER program (11a-c): mid-chunk the o stage's fwd_h products
            # come from transport + the certified row band instead of the
            # full recurrence; the pool writeback is deferred to flush points
            # (stage 13, gated fwd_h + scatter).
            # 11a. h slab <- bf16(boundary[slot])  (transport; gate G2;
            #      folded into the W3.3 prep pack under FUSE)
            if not use_fuse:
                bi_gdn_incr_defer_h0_export_kernel[
                    (triton.cdiv(v, self.fwd_h_bv), bs * h)
                ](
                    s["h"],
                    cache.boundary,
                    indices,
                    H=h,
                    K=k,
                    V=v,
                    BV=self.fwd_h_bv,
                    num_warps=self.h0_export_num_warps,
                )
            # 11b. v_new row band -> v_new cache (P2/P3-certified band;
            #      rows j < p recompute the cached bytes, P1)
            bi_gdn_incr_defer_vnew_rowband_kernel[(triton.cdiv(v, 32), bs * h)](
                rows_c.w,
                rows_c.u,
                cache.boundary,
                rows_c.v_new,
                indices,
                seq_lens,
                H=h,
                K=k,
                V=v,
                BV=32,
                BM=16,
                CHUNK=CHUNK_SIZE,
                num_warps=self.vnew_num_warps,
                num_stages=self.vnew_num_stages,
            )
            # 11c. cached v_new rows <= p -> the gap-encoded slab (transport)
            #      W3.2 (SLIM): DELETED -- the o stage reads the v_new cache
            #      directly.  W3.3 (FUSE, slim off): merged with the l2
            #      export into one export-pack launch.
            if use_slim:
                pass
            elif use_fuse:
                bi_gdn_w3_export_pack_kernel[(bs, CHUNK_SIZE)](
                    rows_c.v_new,
                    s["v_new"],
                    rows_c.l2q,
                    rows_c.l2k,
                    s["l2q_tok"],
                    s["l2k_tok"],
                    s["q_norm"],
                    s["k_norm"],
                    indices,
                    seq_lens,
                    H=h,
                    K=k,
                    V=v,
                    BHK=self._next_pow2(h * k),
                    BHV=self._next_pow2(h * v),
                    CHUNK=CHUNK_SIZE,
                    num_warps=self.export_pack_num_warps,
                )
            else:
                bi_gdn_incr_defer_vnew_export_kernel[(bs, CHUNK_SIZE)](
                    rows_c.v_new,
                    s["v_new"],
                    indices,
                    seq_lens,
                    H=h,
                    V=v,
                    BHV=self._next_pow2(h * v),
                    CHUNK=CHUNK_SIZE,
                    num_warps=self.vnew_export_num_warps,
                )
        # 12. o stage: the ORACLE BINARY chunk_fwd_kernel_o with the FAST path's
        #     certified launch.  Copies of this kernel are falsified (SSG0
        #     codegen class) -- never copy it.
        #     - default: transport-export the cached l2 rows to the
        #       gap-encoded slabs first (skipped under FUSE+defer, where the
        #       export pack above already wrote them);
        #     - W3.2 (SLIM): NO exports -- the oracle binary reads the
        #       slot-major caches directly through the slot-based cu content
        #       (same binary, same grid, same tile shapes; only index-tensor
        #       CONTENT and pointer operands differ -- the certified the FAST path
        #       dynamism-as-content pattern), and writes the slot-major
        #       o buffer.  `h` stays request-major (positional indexing).
        if not use_slim and not (use_fuse and self.defer_writeback):
            bi_gdn_incr_l2_export_kernel[(bs, CHUNK_SIZE)](
                rows_c.l2q,
                rows_c.l2k,
                s["l2q_tok"],
                s["l2k_tok"],
                s["q_norm"],
                s["k_norm"],
                indices,
                seq_lens,
                H=h,
                K=k,
                BHK=self._next_pow2(h * k),
                CHUNK=CHUNK_SIZE,
                num_warps=self.export_num_warps,
            )

        def _o_grid(meta):
            return (triton.cdiv(v, meta["BV"]), bs, h)

        if use_slim:
            o_slot = self._ensure_o_slot(cache, h, v)
            chunk_fwd_kernel_o[_o_grid](
                q=rows_c.l2q,
                k=rows_c.l2k,
                v=rows_c.v_new,
                h=s["h"],
                g=rows_c.gcum,
                g_gamma=None,
                o=o_slot,
                cu_seqlens=s["cu_slot"][: 2 * bs + 1],
                chunk_indices=ci,
                scale=cache.k**-0.5,
                T=rows,
                H=h,
                K=k,
                V=v,
                BT=CHUNK_SIZE,
            )
        else:
            chunk_fwd_kernel_o[_o_grid](
                q=s["q_norm"],
                k=s["k_norm"],
                v=s["v_new"],
                h=s["h"],
                g=s["g_cum"],
                g_gamma=None,
                o=s["o"],
                cu_seqlens=cu,
                chunk_indices=ci,
                scale=cache.k**-0.5,
                T=rows,
                H=h,
                K=k,
                V=v,
                BT=CHUNK_SIZE,
            )
        elem = h * v * k
        block = self.scatter_block
        row_elems = h * v
        out = torch.empty(bs, h, v, dtype=torch.bfloat16, device=device)
        if not self.defer_writeback:
            # 13. state epilogue (the FAST path): scratch->ssm every step; boundary
            #     advance on completed chunks (v1 keeps stock pool semantics)
            if use_fuse:
                # W3.3 F3: scatter + gather in one launch (ungated, v1)
                bi_gdn_w3_scatter_gather_kernel[(bs, triton.cdiv(elem, block))](
                    cache.scratch,
                    ssm_states,
                    cache.boundary,
                    s["o"],
                    out,
                    indices,
                    seq_lens,
                    ssm_states.shape[0],
                    elem_per_entry=elem,
                    row_elems=row_elems,
                    CHUNK=CHUNK_SIZE,
                    BLOCK_SIZE=block,
                    TRACK_IVL=0,
                    GATED=False,
                    SLOT_MAJOR=False,
                )
                return out
            bi_gdn_fast_state_scatter_kernel[(bs, triton.cdiv(elem, block))](
                cache.scratch,
                ssm_states,
                cache.boundary,
                indices,
                seq_lens,
                ssm_states.shape[0],
                elem_per_entry=elem,
                CHUNK=CHUNK_SIZE,
                BLOCK_SIZE=block,
            )
        else:
            # the DEFER program (13a-c): flush-point materialization.  The
            # gated fwd_h runs the FULL recurrence only for requests at a
            # flush point (chunk completion / mamba-track point) -- content
            # early-return idles it for the other 63/64 -- and the gated
            # scatter writes ssm (+ boundary on completion) for exactly the
            # same rows.  Enqueued inside step(), i.e. before the same
            # step's mamba-track copy launch (ordering precondition).
            bi_gdn_incr_defer_fwd_h_flushpt_kernel[
                (triton.cdiv(v, self.fwd_h_bv), bs * h)
            ](
                rows_c.l2k,
                rows_c.u,
                rows_c.w,
                s["v_new"],
                s["g_cum"],
                s["h"],
                cache.boundary,
                cache.scratch,
                indices,
                seq_lens,
                H=h,
                K=k,
                V=v,
                BT=CHUNK_SIZE,
                BV=self.fwd_h_bv,
                TRACK_IVL=self.track_interval,
                num_warps=self.fwd_h_num_warps,
                num_stages=self.fwd_h_num_stages,
            )
            if use_fuse:
                # W3.3 F3: gated scatter + out gather in one launch (the
                # gather part reads the o buffer written above; the poison
                # kernel below touches disjoint ssm rows, order-free)
                bi_gdn_w3_scatter_gather_kernel[(bs, triton.cdiv(elem, block))](
                    cache.scratch,
                    ssm_states,
                    cache.boundary,
                    self._incr_slabs["o_slot"] if use_slim else s["o"],
                    out,
                    indices,
                    seq_lens,
                    ssm_states.shape[0],
                    elem_per_entry=elem,
                    row_elems=row_elems,
                    CHUNK=CHUNK_SIZE,
                    BLOCK_SIZE=block,
                    TRACK_IVL=self.track_interval,
                    GATED=True,
                    SLOT_MAJOR=use_slim,
                )
            else:
                bi_gdn_incr_defer_state_scatter_flushpt_kernel[
                    (bs, triton.cdiv(elem, block))
                ](
                    cache.scratch,
                    ssm_states,
                    cache.boundary,
                    indices,
                    seq_lens,
                    ssm_states.shape[0],
                    elem_per_entry=elem,
                    CHUNK=CHUNK_SIZE,
                    BLOCK_SIZE=block,
                    TRACK_IVL=self.track_interval,
                )
            if use_fuse:
                return out
        # 14. output row gather (the FAST path; slot-addressed variant under SLIM)
        if use_slim:
            bi_gdn_w3_slim_out_gather_kernel[(bs, triton.cdiv(row_elems, block))](
                self._incr_slabs["o_slot"],
                out,
                indices,
                seq_lens,
                row_elems=row_elems,
                CHUNK=CHUNK_SIZE,
                BLOCK_SIZE=block,
            )
        else:
            bi_gdn_fast_out_gather_kernel[(bs, triton.cdiv(row_elems, block))](
                s["o"],
                out,
                seq_lens,
                row_elems=row_elems,
                CHUNK=CHUNK_SIZE,
                BLOCK_SIZE=block,
            )
        return out
