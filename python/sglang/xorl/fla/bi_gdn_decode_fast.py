"""BI-GDN decode marshal fast path.

Transport-only rewrite of ``BIGDNDecodeCache.step()``'s data marshaling,
engaged by the exact Qwen3.5-family serving contract. The frozen oracle
program is
``bi_gdn_decode.py`` + the stage kernels it drives through
``bi_chunk_gated_delta_rule_prefill``; this module never modifies them.

Design (the design record D1-D3):

**Gap-sequence varlen encoding (D1).** The intermediate stage kernels run
UNMODIFIED (the same compiled binaries the oracle runs) over fixed per-request
64-row slabs. The slab is described to the varlen kernels by a synthetic
cu_seqlens with 2*bs sequences: sequence 2i is request i (``cu[2i] = 64*i``
constant, ``cu[2i+1] = 64*i + fill_i`` refreshed in-kernel each step) and the
odd "gap" sequence covers the slab tail (never referenced -- chunk_indices
lists only the even sequences). Every kernel thus reads ``bos = 64*i`` and
``T = fill_i`` through its ordinary varlen preamble: identical code, identical
tile shapes and masks, different index-tensor CONTENT at fixed addresses --
which is exactly what CUDA-graph replay refreshes. An earlier draft used
copied kernels with a rewritten preamble; byte comparison caught a 1-ulp codegen divergence
in the fwd_o copy at the wide-head shape (2/131072 outputs; all inputs
bit-identical), so copied arithmetic variants were
abandoned for the oracle binaries themselves wherever the kernel does
arithmetic.

**Capture against the pool (D2).** The fused ingest kernel reads the per-slot
row pools (``rows_qkv``/``rows_g``/``rows_beta``) directly at ``slot_indices``
content (clamped for PAD rows) and writes dense slabs (split + HK->HV head
expansion + read-time token substitution: pure byte transport). The one
arithmetic-stage variant is fwd_h, which reads the fp32 initial state directly
from the ``boundary`` pool and writes the final state directly into the
``scratch`` pool with transposed block-ptr strides (pool layout [slot, H, V,
K] vs trainer [N, H, K, V]; fp32 loads/stores of identical values); its
compute section is byte-verified against the oracle at both head shapes (ssm,
h-buffer, and v_new equality) and remains constrained to the verified configs.
No staging slots, no backup/restore copies, no ``torch.nonzero``, no
index_select/index_copy scatter-backs.

**Fused prologue/epilogue (D3).** One append kernel persists the new token's
rows at ``rows_*[slot, fill_before]`` (masked on ``slot >= 0``); the state
epilogue adapts the in-tree ``fused_mamba_state_scatter_with_mask`` pattern
(mamba_state_scatter_triton.py: per-request early-exit masking, bounds guard,
flat block copy) to scatter ``scratch[slot] -> ssm_states[slot]`` and advance
``boundary[slot]`` on completed chunks; a small gather kernel reads each
request's output row.

Token substitution: the ingest kernel substitutes the CURRENT token's
qkv/g/beta row at read time for row ``fill_before`` of each request (live and
PAD alike). For live rows this equals append-then-read; for PAD rows it
reproduces byte-for-byte what the oracle's staging produced (slot-0 content
with the pad row's own token overlaid), so core_attn_out matches the oracle on
PAD rows too.

PAD semantics mirrored from the oracle:
  - graph replay maps pads to reserved mamba slot 0 (``_replay_metadata``);
    both programs then write reserved-slot state for pad rows, and with >1 pad
    row both are last-writer racy on slot 0 only (oracle: duplicate-index
    ``index_copy_``); slot-0 bytes are excluded from byte gates in multi-pad
    cells, live slots are always gated.
  - eager PAD_SLOT_ID (-1) rows: reads clamp to slot 0, ALL state writes are
    skipped (``slot >= 0`` masks), exactly the oracle's clamp+masked-write
    pattern (bi_gdn_decode.py:512-518 / :202-205).

NOT contract state: the ``scratch`` pool is a per-step temporary in both
programs (the oracle's graph path clobbers ``scratch[0:graph_bs]`` with
workspace values every step); its bytes intentionally differ between the two
programs and are excluded from byte gates. Slab bytes at rows >= fill are
stale-but-deterministic and are never read (every consumer masks by T).

The fused transport is covered by component byte tests spanning synthetic,
value-edge, and captured inputs, plus graph-versus-prefill comparisons.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum_scalar_kernel
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd_kernel
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd_kernel
from sglang.xorl.fla.bi_gdn_prefill import (
    BI_GDN_SOLVE_TRIL_DECODE,
    IS_TMA_SUPPORTED,
    chunk_fwd_kernel_o,
    exp,
    merge_16x16_to_64x64_inverse_kernel,
)
from sglang.xorl.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd_kernel,
)
from sglang.xorl.fla.solve_tril_decode import (
    DIAG_HEAD_GROUP,
    DIAG_NUM_STAGES,
    DIAG_NUM_WARPS,
    MERGE_NUM_STAGES,
    MERGE_NUM_WARPS,
    solve_tril_64x64_diag_inv_grouped_kernel,
    solve_tril_64x64_merge_inv_kernel,
)

# Internal choices for the exact Qwen3.5-family GDN serving contract.
# Installed as a unit by the architecture resolver (see
# sglang.xorl.fla.qwen35_gdn_exact) -- there are no
# per-feature environment variables on this surface. The False defaults
# keep every non-contract server on the stock path, bit-for-bit
# unaffected; tests may set these module attributes directly on a fresh
# runner (never after a capture).
BI_GDN_DECODE_FAST_ENABLED = False

# Small-kernel and gated-launch fusion in the fast and incremental
# decode paths.  TRANSPORT-ONLY: every fused kernel is a byte-movement merge
# of existing transport kernels (integer-surface freedom); the arithmetic
# stage binaries, operands, and order are untouched.  Certified by
# torch.equal byte gates. This module attribute is the authority for BOTH the fast runner
# and the incremental runner (which imports it from here).
BI_GDN_FUSE_SMALL_ENABLED = False


# --- D2/D3: fused ingest (split + head-expand + read-time token substitution) -


@triton.jit
def bi_gdn_fast_ingest_kernel(
    rows_qkv,  # [S, CHUNK, QKV] bf16 pool
    rows_g,  # [S, CHUNK, H] fp32 pool
    rows_beta,  # [S, CHUNK, H] fp32 pool
    qkv_tok,  # [bs, QKV] bf16 current-token rows
    g_tok,  # [bs, H] fp32
    beta_tok,  # [bs, H] fp32
    slot_indices,  # [bs] int32, may contain -1 (eager PAD)
    seq_lens,  # [bs] int
    q_raw,  # [bs*CHUNK, H, K] bf16 slab (head-expanded)
    k_raw,  # [bs*CHUNK, H, K] bf16 slab (head-expanded)
    v_slab,  # [bs*CHUNK, H, V] bf16 slab
    g_slab,  # [bs*CHUNK, H] fp32 slab
    beta_slab,  # [bs*CHUNK, H] fp32 slab
    cu_gap,  # [2*bs+1] int32 gap-encoded cu_seqlens; odd entries written here
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    QKV: tl.constexpr,
    BHK: tl.constexpr,  # next_pow2(H*K)
    BHV: tl.constexpr,  # next_pow2(H*V)
    BH: tl.constexpr,  # next_pow2(H)
    CHUNK: tl.constexpr,
):
    i_b, i_r = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK  # torch.remainder semantics
    use_tok = i_r == fb

    if i_r == 0:
        # refresh the request's logical extent for the varlen stage kernels
        tl.store(cu_gap + 2 * i_b + 1, i_b * CHUNK + fb + 1)

    # source row pointer: pool row, or the current token's row (substitution)
    src_qkv = rows_qkv + (slot_c * CHUNK + i_r) * QKV
    if use_tok:
        src_qkv = qkv_tok + i_b.to(tl.int64) * QKV
    src_g = rows_g + (slot_c * CHUNK + i_r) * H
    src_beta = rows_beta + (slot_c * CHUNK + i_r) * H
    if use_tok:
        src_g = g_tok + i_b.to(tl.int64) * H
        src_beta = beta_tok + i_b.to(tl.int64) * H

    dst_row = i_b.to(tl.int64) * CHUNK + i_r

    # q / k: expanded heads (dst head hh <- src head hh // (H // HG)),
    # byte transport of the packed [q(HG*K) | k(HG*K) | v(H*V)] row.
    o_hk = tl.arange(0, BHK)
    m_hk = o_hk < H * K
    hh = o_hk // K
    kk = o_hk % K
    src_off = (hh // (H // HG)) * K + kk
    b_q = tl.load(src_qkv + src_off, mask=m_hk, other=0.0)
    tl.store(q_raw + dst_row * H * K + o_hk, b_q, mask=m_hk)
    b_k = tl.load(src_qkv + HG * K + src_off, mask=m_hk, other=0.0)
    tl.store(k_raw + dst_row * H * K + o_hk, b_k, mask=m_hk)

    # v: direct copy (packed [H, V] region is already the slab layout)
    o_hv = tl.arange(0, BHV)
    m_hv = o_hv < H * V
    b_v = tl.load(src_qkv + 2 * HG * K + o_hv, mask=m_hv, other=0.0)
    tl.store(v_slab + dst_row * H * V + o_hv, b_v, mask=m_hv)

    # g / beta
    o_h = tl.arange(0, BH)
    m_h = o_h < H
    b_g = tl.load(src_g + o_h, mask=m_h, other=0.0)
    tl.store(g_slab + dst_row * H + o_h, b_g, mask=m_h)
    b_b = tl.load(src_beta + o_h, mask=m_h, other=0.0)
    tl.store(beta_slab + dst_row * H + o_h, b_b, mask=m_h)


# --- D3: append prologue (persist the token row at rows_*[slot, fill_before]) -


@triton.jit
def bi_gdn_fast_append_kernel(
    rows_qkv,
    rows_g,
    rows_beta,
    qkv_tok,
    g_tok,
    beta_tok,
    slot_indices,
    seq_lens,
    H: tl.constexpr,
    QKV: tl.constexpr,
    BQKV: tl.constexpr,
    BH: tl.constexpr,
    CHUNK: tl.constexpr,
):
    i_b = tl.program_id(0)
    slot = tl.load(slot_indices + i_b)
    if slot >= 0:
        sl = tl.load(seq_lens + i_b).to(tl.int32)
        fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
        dst = (slot.to(tl.int64) * CHUNK + fb) * QKV
        o_q = tl.arange(0, BQKV)
        m_q = o_q < QKV
        b_q = tl.load(qkv_tok + i_b.to(tl.int64) * QKV + o_q, mask=m_q, other=0.0)
        tl.store(rows_qkv + dst + o_q, b_q, mask=m_q)
        o_h = tl.arange(0, BH)
        m_h = o_h < H
        b_g = tl.load(g_tok + i_b.to(tl.int64) * H + o_h, mask=m_h, other=0.0)
        tl.store(rows_g + (slot.to(tl.int64) * CHUNK + fb) * H + o_h, b_g, mask=m_h)
        b_b = tl.load(beta_tok + i_b.to(tl.int64) * H + o_h, mask=m_h, other=0.0)
        tl.store(rows_beta + (slot.to(tl.int64) * CHUNK + fb) * H + o_h, b_b, mask=m_h)


# --- fwd_h variant: pool-direct initial/final state (the one arithmetic-stage
# variant; body verbatim from bi_gdn_prefill's vendored
# chunk_gated_delta_rule_fwd_kernel_h_blockdim64 specialized to the oracle
# decode call: USE_G=True, USE_GK=False, USE_INITIAL_STATE=True,
# STORE_FINAL_STATE=True, SAVE_NEW_VALUE=True, USE_EXP2=False). h0 reads the
# fp32 boundary pool and ht writes the fp32 scratch pool slot-direct with
# transposed strides. Compute bit-verified vs the oracle at H=4 and H=32
# (ssm/h/v_new byte equality) and constrained to the verified launch configs.


@triton.jit
def bi_gdn_fast_fwd_h_kernel(
    k,
    v,
    w,
    v_new,
    g,
    h,
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
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
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


# --- D3: state epilogue, adapted from mamba_state_scatter_triton.py's
# fused_mamba_state_scatter_with_mask (per-request early-exit masking, bounds
# guard, flat block copy; extended with the completed-chunk boundary advance).


@triton.jit
def bi_gdn_fast_state_scatter_kernel(
    scratch,  # [S, H, V, K] fp32 pool (final states from fwd_h)
    ssm,  # [S, H, V, K] fp32 pool (stock pool semantics)
    boundary,  # [S, H, V, K] fp32 pool
    slot_indices,
    seq_lens,
    num_slots,
    elem_per_entry: tl.constexpr,  # H*V*K
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
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


@triton.jit
def bi_gdn_fast_out_gather_kernel(
    o_slab,  # [bs*CHUNK, H, V] bf16
    out,  # [bs, H, V] bf16
    seq_lens,
    row_elems: tl.constexpr,  # H*V
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_req = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1).to(tl.int64)
    sl = tl.load(seq_lens + pid_req).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK
    src = (pid_req * CHUNK + fb) * row_elems
    dst = pid_req * row_elems
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_elems
    data = tl.load(o_slab + src + offsets, mask=mask)
    tl.store(out + dst + offsets, data, mask=mask)


# --- W3.3 (the transport fusion): fused transport kernels -------
# Byte-movement merges of the kernels above; bodies are verbatim per part.
# Race audit: the append part writes only
# rows_*[slot, fb] of the program's OWN request; every same-request read of
# that row is token-substituted, live slots are unique per batch, and the
# only cross-program overlap is shared pad slot 0 (multi-pad slot-0 bytes and
# pad output rows are excluded from byte gates by the existing convention;
# a single pad has no cross-program writer and stays deterministic).


@triton.jit
def bi_gdn_w3_fast_ingest_append_kernel(
    rows_qkv,
    rows_g,
    rows_beta,
    qkv_tok,
    g_tok,
    beta_tok,
    slot_indices,
    seq_lens,
    q_raw,
    k_raw,
    v_slab,
    g_slab,
    beta_slab,
    cu_gap,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    QKV: tl.constexpr,
    BHK: tl.constexpr,
    BHV: tl.constexpr,
    BH: tl.constexpr,
    BQKV: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """bi_gdn_fast_ingest_kernel + bi_gdn_fast_append_kernel in one launch
    (the append lands in the i_r == fb program, masked on slot >= 0)."""
    i_b, i_r = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + i_b)
    slot_c = tl.maximum(slot, 0).to(tl.int64)
    sl = tl.load(seq_lens + i_b).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK  # torch.remainder semantics
    use_tok = i_r == fb

    if i_r == 0:
        tl.store(cu_gap + 2 * i_b + 1, i_b * CHUNK + fb + 1)

    src_qkv = rows_qkv + (slot_c * CHUNK + i_r) * QKV
    if use_tok:
        src_qkv = qkv_tok + i_b.to(tl.int64) * QKV
    src_g = rows_g + (slot_c * CHUNK + i_r) * H
    src_beta = rows_beta + (slot_c * CHUNK + i_r) * H
    if use_tok:
        src_g = g_tok + i_b.to(tl.int64) * H
        src_beta = beta_tok + i_b.to(tl.int64) * H

    dst_row = i_b.to(tl.int64) * CHUNK + i_r

    o_hk = tl.arange(0, BHK)
    m_hk = o_hk < H * K
    hh = o_hk // K
    kk = o_hk % K
    src_off = (hh // (H // HG)) * K + kk
    b_q = tl.load(src_qkv + src_off, mask=m_hk, other=0.0)
    tl.store(q_raw + dst_row * H * K + o_hk, b_q, mask=m_hk)
    b_k = tl.load(src_qkv + HG * K + src_off, mask=m_hk, other=0.0)
    tl.store(k_raw + dst_row * H * K + o_hk, b_k, mask=m_hk)

    o_hv = tl.arange(0, BHV)
    m_hv = o_hv < H * V
    b_v = tl.load(src_qkv + 2 * HG * K + o_hv, mask=m_hv, other=0.0)
    tl.store(v_slab + dst_row * H * V + o_hv, b_v, mask=m_hv)

    o_h = tl.arange(0, BH)
    m_h = o_h < H
    b_g = tl.load(src_g + o_h, mask=m_h, other=0.0)
    tl.store(g_slab + dst_row * H + o_h, b_g, mask=m_h)
    b_b = tl.load(src_beta + o_h, mask=m_h, other=0.0)
    tl.store(beta_slab + dst_row * H + o_h, b_b, mask=m_h)

    # --- fused append (bi_gdn_fast_append_kernel body, this row only) ---
    if use_tok and slot >= 0:
        dst = (slot.to(tl.int64) * CHUNK + fb) * QKV
        o_q = tl.arange(0, BQKV)
        m_q = o_q < QKV
        b_qa = tl.load(qkv_tok + i_b.to(tl.int64) * QKV + o_q, mask=m_q, other=0.0)
        tl.store(rows_qkv + dst + o_q, b_qa, mask=m_q)
        b_ga = tl.load(g_tok + i_b.to(tl.int64) * H + o_h, mask=m_h, other=0.0)
        tl.store(rows_g + (slot.to(tl.int64) * CHUNK + fb) * H + o_h, b_ga, mask=m_h)
        b_ba = tl.load(beta_tok + i_b.to(tl.int64) * H + o_h, mask=m_h, other=0.0)
        tl.store(rows_beta + (slot.to(tl.int64) * CHUNK + fb) * H + o_h, b_ba, mask=m_h)


@triton.jit
def bi_gdn_w3_fast_epilogue_kernel(
    scratch,
    ssm,
    boundary,
    o_slab,
    out,
    slot_indices,
    seq_lens,
    num_slots,
    elem_per_entry: tl.constexpr,  # H*V*K
    row_elems: tl.constexpr,  # H*V
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """bi_gdn_fast_state_scatter_kernel + bi_gdn_fast_out_gather_kernel in
    one launch (grid (bs, cdiv(elem_per_entry, BLOCK_SIZE));
    elem_per_entry >= row_elems always, so the gather blocks are a prefix).
    The gather part stays unconditional exactly like the original (no slot
    guard); the scatter part keeps the original guards/gates."""
    pid_req = tl.program_id(0)
    pid_block = tl.program_id(1).to(tl.int64)
    sl = tl.load(seq_lens + pid_req).to(tl.int32)
    fb = ((sl - 1) % CHUNK + CHUNK) % CHUNK

    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # --- out gather part (prefix blocks only) ---
    if start < row_elems:
        m_g = offsets < row_elems
        src = (pid_req.to(tl.int64) * CHUNK + fb) * row_elems
        dst = pid_req.to(tl.int64) * row_elems
        d_g = tl.load(o_slab + src + offsets, mask=m_g)
        tl.store(out + dst + offsets, d_g, mask=m_g)

    # --- state scatter part (original guards) ---
    slot = tl.load(slot_indices + pid_req).to(tl.int64)
    if slot >= 0:
        if slot < num_slots:
            completed = fb + 1 == CHUNK
            base = slot * elem_per_entry
            mask = offsets < elem_per_entry
            data = tl.load(scratch + base + offsets, mask=mask)
            tl.store(ssm + base + offsets, data, mask=mask)
            if completed:
                tl.store(boundary + base + offsets, data, mask=mask)


# --- driver --------------------------------------------------------------------


class BIGDNFastDecodeRunner:
    """Slot-direct decode rescan (one instance per backend; slabs shared
    across GDN layers -- they are per-step scratch, not per-layer state)."""

    def __init__(self) -> None:
        self._bs = 0
        self._dims: tuple | None = None
        self._slabs: dict[str, torch.Tensor] = {}
        # sweepable launch axes for the transport kernels and the one
        # arithmetic-stage variant (fwd_h); defaults mirror the oracle's
        # pinned values. The intermediate stage kernels are the oracle's own
        # autotuned/pinned binaries and take no overrides here.
        self.fwd_h_num_warps = 4
        self.fwd_h_num_stages = 2
        self.fwd_h_bv = 32
        self.ingest_num_warps = 4
        self.scatter_block = 1024
        # The architecture resolver selects transport fusion before a fresh
        # runner is constructed and before any graph capture.
        self.fuse_small = BI_GDN_FUSE_SMALL_ENABLED

    # Persistent pools; allocate as normal tensors (mutated from both
    # inference-mode and non-inference-mode contexts).
    @torch.inference_mode(mode=False)
    def _ensure_slabs(self, bs: int, cache, device: torch.device) -> None:
        hv, hg = cache.hv, (cache.qkv_dim - cache.hv * cache.v) // (2 * cache.k)
        dims = (hv, hg, cache.k, cache.v, cache.qkv_dim)
        if self._bs >= bs and self._dims == dims:
            return
        if self._dims is not None and self._dims != dims:
            raise RuntimeError(
                "the FAST transport: GDN layer dims changed across layers: "
                f"{self._dims} != {dims}."
            )
        bs = max(bs, self._bs)
        h, k, v = hv, cache.k, cache.v
        rows = bs * CHUNK_SIZE
        bf16, fp32 = torch.bfloat16, torch.float32
        # gap-encoded varlen descriptors: even cu entries and chunk_indices
        # are CONSTANT content (request i at slab rows [64i, 64i+fill_i));
        # odd cu entries are refreshed in-kernel by the ingest each step.
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
        self._slabs = {
            "q_raw": torch.empty(rows, h, k, dtype=bf16, device=device),
            "k_raw": torch.empty(rows, h, k, dtype=bf16, device=device),
            "q_norm": torch.empty(rows, h, k, dtype=bf16, device=device),
            "k_norm": torch.empty(rows, h, k, dtype=bf16, device=device),
            "v": torch.empty(rows, h, v, dtype=bf16, device=device),
            "g": torch.empty(rows, h, dtype=fp32, device=device),
            "beta": torch.empty(rows, h, dtype=fp32, device=device),
            "g_cum": torch.empty(rows, h, dtype=fp32, device=device),
            "A": torch.empty(rows, h, CHUNK_SIZE, dtype=fp32, device=device),
            # solve_tril's oracle driver zero-initializes its output and only
            # ever stores the diagonal + strictly-lower 16x16 blocks; the
            # strictly-upper blocks must therefore be zero. They are written
            # by no kernel, so zeroing once at allocation is sufficient.
            "Ai": torch.zeros(rows, h, CHUNK_SIZE, dtype=bf16, device=device),
            "w": torch.empty(rows, h, k, dtype=bf16, device=device),
            "u": torch.empty(rows, h, v, dtype=bf16, device=device),
            "v_new": torch.empty(rows, h, v, dtype=bf16, device=device),
            "h": torch.empty(bs, h, k, v, dtype=bf16, device=device),
            "o": torch.empty(rows, h, v, dtype=bf16, device=device),
            "cu_gap": cu_gap,
            "chunk_indices": chunk_indices,
        }
        if BI_GDN_SOLVE_TRIL_DECODE:
            # Solve-stage composition (D5): fp32 diagonal-inverse scratch crossing
            # the two solve_tril_decode kernels. Pre-allocated like every other
            # slab so graph capture sees a fixed address (the driver in
            # solve_tril_decode.py allocates per call, which is fine eagerly
            # but not for capture-against-the-pool).
            self._slabs["Di"] = torch.empty(rows, h, 16, dtype=fp32, device=device)
        self._bs = bs
        self._dims = dims

    @staticmethod
    def _next_pow2(n: int) -> int:
        return int(triton.next_power_of_2(n))

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
        """One batched single-token decode step, slot-direct.

        indices: [bs] int32 mamba slots (graph pads -> reserved slot 0; eager
        PAD rows -> -1, handled clamp+masked). seq_lens: [bs] scheduler
        sequence lengths (fill is derived in-kernel from their content, so a
        captured graph replays correctly with refreshed content at fixed
        addresses). Returns core_attn_out [bs, HV, V] bf16.
        """
        bs = qkv_rows.shape[0]
        device = qkv_rows.device
        self._ensure_slabs(bs, cache, device)
        s = self._slabs
        h, hg, k, v = cache.hv, self._dims[1], cache.k, cache.v
        qkv_dim = cache.qkv_dim
        rows = bs * CHUNK_SIZE
        cu = s["cu_gap"][: 2 * bs + 1]
        ci = s["chunk_indices"][:bs]

        if self.fuse_small:
            # W3.3: one launch for ingest + append (transport merge)
            bi_gdn_w3_fast_ingest_append_kernel[(bs, CHUNK_SIZE)](
                cache.rows_qkv,
                cache.rows_g,
                cache.rows_beta,
                qkv_rows,
                g_rows,
                beta_rows,
                indices,
                seq_lens,
                s["q_raw"],
                s["k_raw"],
                s["v"],
                s["g"],
                s["beta"],
                cu,
                H=h,
                HG=hg,
                K=k,
                V=v,
                QKV=qkv_dim,
                BHK=self._next_pow2(h * k),
                BHV=self._next_pow2(h * v),
                BH=self._next_pow2(h),
                BQKV=self._next_pow2(qkv_dim),
                CHUNK=CHUNK_SIZE,
                num_warps=self.ingest_num_warps,
            )
        else:
            bi_gdn_fast_ingest_kernel[(bs, CHUNK_SIZE)](
                cache.rows_qkv,
                cache.rows_g,
                cache.rows_beta,
                qkv_rows,
                g_rows,
                beta_rows,
                indices,
                seq_lens,
                s["q_raw"],
                s["k_raw"],
                s["v"],
                s["g"],
                s["beta"],
                cu,
                H=h,
                HG=hg,
                K=k,
                V=v,
                QKV=qkv_dim,
                BHK=self._next_pow2(h * k),
                BHV=self._next_pow2(h * v),
                BH=self._next_pow2(h),
                CHUNK=CHUNK_SIZE,
                num_warps=self.ingest_num_warps,
            )
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

        # ---- unmodified oracle stage kernels over the gap-encoded slabs ----
        # launch parameters mirror the oracle drivers exactly (l2norm_fwd,
        # chunk_local_cumsum_scalar, chunk_scaled_dot_kkt_fwd, solve_tril,
        # recompute_w_u_fwd, chunk_fwd_o).
        l2_rows = rows * h
        for src, dst in ((s["q_raw"], s["q_norm"]), (s["k_raw"], s["k_norm"])):
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
        chunk_scaled_dot_kkt_fwd_kernel[(bs, h)](
            k=s["k_norm"],
            beta=s["beta"],
            g_cumsum=s["g_cum"],
            A=s["A"],
            cu_seqlens=cu,
            chunk_indices=ci,
            T=rows,
            H=h,
            Hg=h,
            K=k,
            BT=CHUNK_SIZE,
            BK=64,
            IS_VARLEN=True,
            USE_G=True,
            num_warps=8,
            num_stages=3,
        )
        if BI_GDN_SOLVE_TRIL_DECODE:
            # Solve-stage composition (D5): route the solve stage through the
            # decode-scheduled kernels (solve_tril_decode.py, bitwise-identical
            # to the pinned kernel by construction and by gate) over the same
            # gap-encoded slabs. Grids and launch configs mirror the
            # solve_tril_decode driver exactly (NT = bs chunks, B = 1). The
            # merge kernel also stores the upper-triangle zero blocks, writing
            # the same bytes the pre-zeroed Ai slab already holds.
            g_div = DIAG_HEAD_GROUP
            while h % g_div:
                g_div //= 2
            solve_tril_64x64_diag_inv_grouped_kernel[(bs * 4, h // g_div)](
                A=s["A"],
                Di=s["Di"],
                cu_seqlens=cu,
                chunk_indices=ci,
                T=rows,
                H=h,
                BT=CHUNK_SIZE,
                G=g_div,
                IS_VARLEN=True,
                num_warps=DIAG_NUM_WARPS,
                num_stages=DIAG_NUM_STAGES,
            )
            solve_tril_64x64_merge_inv_kernel[(bs, h)](
                A=s["A"],
                Di=s["Di"],
                Ai=s["Ai"],
                cu_seqlens=cu,
                chunk_indices=ci,
                T=rows,
                H=h,
                BT=CHUNK_SIZE,
                DOT_PRECISION="ieee",
                IS_VARLEN=True,
                num_warps=MERGE_NUM_WARPS,
                num_stages=MERGE_NUM_STAGES,
            )
        else:
            merge_16x16_to_64x64_inverse_kernel[(bs, h)](
                A=s["A"],
                Ai=s["Ai"],
                cu_seqlens=cu,
                chunk_indices=ci,
                T=rows,
                H=h,
                BT=CHUNK_SIZE,
                USE_TMA=IS_TMA_SUPPORTED,
            )
        recompute_w_u_fwd_kernel[(bs, h)](
            k=s["k_norm"],
            v=s["v"],
            beta=s["beta"],
            w=s["w"],
            u=s["u"],
            A=s["Ai"],
            g=s["g_cum"],
            cu_seqlens=cu,
            chunk_indices=ci,
            T=rows,
            H=h,
            Hg=h,
            K=k,
            V=v,
            BT=CHUNK_SIZE,
            BK=64,
            BV=64,
            IS_VARLEN=True,
            num_warps=4,
            num_stages=3,
        )
        bi_gdn_fast_fwd_h_kernel[(triton.cdiv(v, self.fwd_h_bv), bs * h)](
            s["k_norm"],
            s["u"],
            s["w"],
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

        def _o_grid(meta):
            return (triton.cdiv(v, meta["BV"]), bs, h)

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

        # ---- D3 epilogue: masked state scatter + boundary advance + output
        elem = h * v * k
        block = self.scatter_block
        if self.fuse_small:
            # W3.3: one launch for scatter + gather (transport merge)
            out = torch.empty(bs, h, v, dtype=torch.bfloat16, device=device)
            row_elems = h * v
            bi_gdn_w3_fast_epilogue_kernel[(bs, triton.cdiv(elem, block))](
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
        out = torch.empty(bs, h, v, dtype=torch.bfloat16, device=device)
        row_elems = h * v
        bi_gdn_fast_out_gather_kernel[(bs, triton.cdiv(row_elems, block))](
            s["o"],
            out,
            seq_lens,
            row_elems=row_elems,
            CHUNK=CHUNK_SIZE,
            BLOCK_SIZE=block,
        )
        return out
