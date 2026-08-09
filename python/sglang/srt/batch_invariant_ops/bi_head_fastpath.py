# Fused-grid realization of the decode rescore's chunk-stats stage.
#
# The shipped rescore (`bi_lm_head_selected_logprob_from_logits`,
# batch_invariant_ops.py — FROZEN oracle, untouched) launches
# `_lm_head_chunk_stats_kernel` once per vocab chunk from a Python loop:
# 31 eager launches of grid (n_tokens,) at production V=248,320, each a
# handful of CTAs on 132 SMs, host-paced between graph replays.
#
# This module launches the IDENTICAL per-(row, chunk) program body once over
# a 2D grid (n_tokens, n_chunks). Transport/launch structure only:
#   - the per-(row, chunk) block loop, masks, tl.max/tl.maximum/tl.exp/
#     tl.sum(tl.where(...)) expressions and the sequential `sum_exp +=`
#     accumulation are byte-for-byte the oracle kernel's (BLOCK_SIZE=1024,
#     num_warps pinned to the oracle's default 4 — both are part of the
#     reduction-tree contract; changing either changes bits);
#   - `col_offset` moves from a host-side tensor slice into device index
#     arithmetic (integer-only);
#   - the merge stage launches the ORACLE's `_lm_head_lse_merge_kernel`
#     binary, unchanged (proven pattern: drive oracle binaries, never copy).
# Component tests establish bit-equality of logprob, lse, selected value,
# chunk maxima, and chunk sums against the independent oracle.
#
# Default off. Selected only by the private architecture resolver (read in
# layers/sampler.py); this module has no import-time side effects on the
# oracle path.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    _BI_LM_HEAD_STATS_BLOCK,
    BI_LM_HEAD_VOCAB_CHUNK,
    _lm_head_lse_merge_kernel,
)

# The oracle launches with triton's default num_warps; pin it explicitly so a
# triton default change cannot silently move the intra-block reduction tree.
_ORACLE_NUM_WARPS = 4


@triton.jit
def _lm_head_chunk_stats_fused_kernel(
    logits_ptr,
    token_ids_ptr,
    sel_ptr,
    m_ptr,
    s_ptr,
    temp_ptr,
    logits_row_stride,
    total_cols,
    vocab_chunk,
    n_chunks,
    BLOCK_SIZE: tl.constexpr,
    HAS_TEMP: tl.constexpr,
):
    """One program per (row, chunk): the oracle `_lm_head_chunk_stats_kernel`
    body verbatim, with (chunk_idx, col_offset, n_cols) derived from
    program_id(1) instead of per-launch scalars. Every floating-point
    expression, mask, block size and accumulation order is unchanged — the
    only difference is which program computes which chunk (launch structure,
    not arithmetic)."""
    row = tl.program_id(0).to(tl.int64)
    chunk_idx = tl.program_id(1)
    col_offset = chunk_idx * vocab_chunk
    n_cols = tl.minimum(vocab_chunk, total_cols - col_offset)
    row_ptr = logits_ptr + row * logits_row_stride + col_offset
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temp_ptr + row)
    else:
        inv_t = 1.0

    row_max = float("-inf")
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_ptr + col_idx, mask=mask, other=float("-inf"))
        if HAS_TEMP:
            vals = vals * inv_t
        row_max = tl.maximum(row_max, tl.max(vals))

    sum_exp = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_ptr + col_idx, mask=mask, other=float("-inf"))
        if HAS_TEMP:
            vals = vals * inv_t
        e = tl.exp(vals - row_max)
        sum_exp += tl.sum(tl.where(mask, e, 0.0))

    tl.store(m_ptr + row * n_chunks + chunk_idx, row_max)
    tl.store(s_ptr + row * n_chunks + chunk_idx, sum_exp)

    tok = tl.load(token_ids_ptr + row)
    local = tok - col_offset
    in_chunk = (local >= 0) & (local < n_cols)
    sel = tl.load(row_ptr + local, mask=in_chunk, other=0.0)
    if HAS_TEMP:
        sel = sel * inv_t
    tl.store(sel_ptr + row, sel, mask=in_chunk)


def bi_lm_head_selected_logprob_from_logits_fast(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: Optional[torch.Tensor] = None,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in fused-grid twin of `bi_lm_head_selected_logprob_from_logits`
    (contract steps 2-3 over existing fp32 logits): one stats launch over
    grid (n_tokens, n_chunks) instead of a Python loop of n_chunks launches,
    then the ORACLE merge kernel. Bit-identical outputs (gated); CUDA-graph
    safe (static launch shape for a given [N, V], no host syncs)."""
    assert logits.ndim == 2, "logits must be [N, V]"
    assert logits.dtype == torch.float32, "the contract scores fp32 logits"
    assert logits.is_cuda, "CUDA only"
    assert logits.stride(1) == 1, "logits rows must be unit-stride"

    n_tokens, vocab = logits.shape
    token_ids = token_ids.contiguous().to(device=logits.device, dtype=torch.int64)
    assert token_ids.shape[0] == n_tokens, "token_ids must be per-row [N]"
    n_chunks = (vocab + vocab_chunk - 1) // vocab_chunk
    if temperature is not None:
        temperature = (
            temperature.reshape(-1)
            .to(device=logits.device, dtype=torch.float32)
            .contiguous()
        )
        assert temperature.shape[0] == n_tokens, "temperature must be per-row [N]"
        torch._assert_async((temperature > 0).all(), "temperature must be > 0")

    chunk_max = torch.empty(
        (n_tokens, n_chunks), dtype=torch.float32, device=logits.device
    )
    chunk_sumexp = torch.empty_like(chunk_max)
    selected = torch.zeros(n_tokens, dtype=torch.float32, device=logits.device)
    lse = torch.empty(n_tokens, dtype=torch.float32, device=logits.device)

    _lm_head_chunk_stats_fused_kernel[(n_tokens, n_chunks)](
        logits,
        token_ids,
        selected,
        chunk_max,
        chunk_sumexp,
        temperature,
        logits.stride(0),
        vocab,
        vocab_chunk,
        n_chunks,
        BLOCK_SIZE=_BI_LM_HEAD_STATS_BLOCK,
        HAS_TEMP=temperature is not None,
        num_warps=_ORACLE_NUM_WARPS,
    )

    _lm_head_lse_merge_kernel[(n_tokens,)](chunk_max, chunk_sumexp, lse, n_chunks)
    # Same one-ulp boundary clamp as the oracle (p~1 tokens).
    return torch.clamp_max(selected - lse, 0.0), lse, selected
