"""GDN decode contract, serving side: partial-chunk-rescan decode.

Serving's stock GDN decode (per-token step recurrence) computes a DIFFERENT
composition from the chunked prefill the trainer scores with — the measured
~1e-3-class decode-vs-prefill K3 floor, dominated by accumulation order (the
fp32-beta gating alignment alone does not move it; both are same-magnitude
terms). Under ``SGLANG_BI_GDN_DECODE=1`` decode instead RESCANS the current
partial chunk ``[chunk_boundary .. p]`` through the exact BI-prefill pipeline
(``bi_chunk_gated_delta_rule_prefill``) from the fp32 chunk-boundary state.
Gated property (frozen fa4 capture, layers 0/1, 63/63 generated positions,
both gating conventions, incl. a boundary crossing): the rescan output at p is
BITWISE equal to a full teacher-forced scan's output at p. Decode logprobs are
then the prefill composition's logprobs exactly, so the trainer's standard
differentiable prefill scoring reads K3 = 0.0 with an untouched estimator.

Requires SGLANG_BI_GDN_PREFILL=1 (the scan composition this path rescans
with). Eager decode only for now (dynamic suffix shapes; graph support =
padded static shapes, a later leg). Costs one extra aligned-prefix scan per
extend pass and a growing (1..64-row) rescan per decode step — the K3 lane
pays composition-identity with throughput, per the correctness-first doctrine.

State model per request slot, per GDN layer:
  - ``ssm_states[slot]`` keeps its stock meaning (state after the last
    consumed token): the rescan's final state is written back every step, so
    the mamba track/radix machinery is undisturbed.
  - ``boundary[slot]`` (this module): fp32 state at the last 64-aligned chunk
    boundary — the rescan's initial state.
  - ``rows_*[slot, :suffix_len]``: the current partial chunk's post-conv
    packed qkv rows (bf16) and gating rows (fp32).

v1 constraints (loud): single-token decode only (no MTP/spec paths); extend
prefix lengths must be 64-aligned (chunked prefill and mamba-track restore
points are; an arbitrary-position resume is not supported).
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.fla.bi_gdn_prefill import (
    bi_chunk_gated_delta_rule_prefill,
)
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE
from sglang.srt.utils import get_bool_env_var

BI_GDN_DECODE_ENABLED = get_bool_env_var("SGLANG_BI_GDN_DECODE")


class BIGDNDecodeCache:
    """Per-layer boundary states + partial-chunk row buffers for the rescan."""

    def __init__(
        self,
        num_slots: int,
        qkv_dim: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        device: torch.device,
    ) -> None:
        self.qkv_dim = qkv_dim
        self.hv = num_v_heads
        self.k = head_k_dim
        self.v = head_v_dim
        self.boundary = torch.zeros(
            num_slots,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            dtype=torch.float32,
            device=device,
        )
        self.scratch = torch.zeros_like(self.boundary)
        self.rows_qkv = torch.zeros(
            num_slots, CHUNK_SIZE, qkv_dim, dtype=torch.bfloat16, device=device
        )
        self.rows_g = torch.zeros(
            num_slots, CHUNK_SIZE, num_v_heads, dtype=torch.float32, device=device
        )
        self.rows_beta = torch.zeros(
            num_slots, CHUNK_SIZE, num_v_heads, dtype=torch.float32, device=device
        )
        self.suffix_len = [0] * num_slots

    def _split(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[T, qkv_dim] packed q|k|v -> ([1,T,H,K], [1,T,H,K], [1,T,HV,V])."""
        h = (self.qkv_dim - self.hv * self.v) // (2 * self.k)
        q, k, v = torch.split(rows, [h * self.k, h * self.k, self.hv * self.v], dim=-1)
        t = rows.shape[0]
        return (
            q.view(1, t, h, self.k),
            k.view(1, t, h, self.k),
            v.view(1, t, self.hv, self.v),
        )

    def seed_from_extend(
        self,
        slot: int,
        pre_scan_state: torch.Tensor,
        qkv_rows: torch.Tensor,
        g_rows: torch.Tensor,
        beta_rows: torch.Tensor,
        prefix_len: int,
        ssm_states: torch.Tensor,
    ) -> None:
        """Seed boundary state + suffix rows after one extend pass.

        pre_scan_state: [HV, V, K] fp32 — the slot's state BEFORE the pass
        qkv_rows/g_rows/beta_rows: this pass's post-conv/gating rows [T_pass, ...]
        prefix_len: tokens consumed before this pass (must be 64-aligned)
        ssm_states: the pool (slot holds the post-pass state already)
        """
        if prefix_len % CHUNK_SIZE != 0:
            raise RuntimeError(
                f"SGLANG_BI_GDN_DECODE requires {CHUNK_SIZE}-aligned extend prefixes; got {prefix_len}."
            )
        t_pass = qkv_rows.shape[0]
        total = prefix_len + t_pass
        bnd = (total // CHUNK_SIZE) * CHUNK_SIZE
        suffix = total - bnd
        if suffix == 0:
            self.boundary[slot] = ssm_states[slot]
        else:
            bnd_in_pass = bnd - prefix_len
            if bnd_in_pass == 0:
                self.boundary[slot] = pre_scan_state
            else:
                # rescan the aligned prefix of this pass from the pre-pass state;
                # fp32 chunk-boundary chaining is exact
                self.scratch[slot] = pre_scan_state
                q, k, v = self._split(qkv_rows[:bnd_in_pass])
                bi_chunk_gated_delta_rule_prefill(
                    q=q,
                    k=k,
                    v=v,
                    g=g_rows[:bnd_in_pass].view(1, bnd_in_pass, self.hv),
                    beta=beta_rows[:bnd_in_pass].view(1, bnd_in_pass, self.hv),
                    ssm_states=self.scratch,
                    cache_indices=torch.tensor(
                        [slot], dtype=torch.int32, device=qkv_rows.device
                    ),
                    cu_seqlens=torch.tensor(
                        [0, bnd_in_pass], dtype=torch.int32, device=qkv_rows.device
                    ),
                    scale=self.k**-0.5,
                )
                self.boundary[slot] = self.scratch[slot]
            self.rows_qkv[slot, :suffix] = qkv_rows[bnd_in_pass:]
            self.rows_g[slot, :suffix] = g_rows[bnd_in_pass:]
            self.rows_beta[slot, :suffix] = beta_rows[bnd_in_pass:]
        self.suffix_len[slot] = suffix

    def step(
        self,
        slots: list[int],
        qkv_rows: torch.Tensor,
        g_rows: torch.Tensor,
        beta_rows: torch.Tensor,
        ssm_states: torch.Tensor,
    ) -> torch.Tensor:
        """One batched single-token decode step via partial-chunk rescan.

        qkv_rows: [bs, qkv_dim] post-conv packed rows; g/beta: [bs, HV] fp32.
        Writes each slot's post-token state back into ``ssm_states`` (stock
        pool semantics preserved) and returns core_attn_out [bs, HV, V].
        """
        device = qkv_rows.device
        bs = qkv_rows.shape[0]
        seg_lens = []
        for i, s in enumerate(slots):
            pos = self.suffix_len[s]
            if pos == CHUNK_SIZE:
                # Buffer arrived exactly full (64-aligned extend seeding leaves
                # suffix == CHUNK_SIZE; the post-step roll below only covers
                # steps). Advance the boundary by rescanning the buffered chunk
                # before appending — same certified fp32 boundary chaining as
                # the seed path.
                self.scratch[s] = self.boundary[s]
                q1, k1, v1 = self._split(self.rows_qkv[s, :CHUNK_SIZE])
                bi_chunk_gated_delta_rule_prefill(
                    q=q1,
                    k=k1,
                    v=v1,
                    g=self.rows_g[s, :CHUNK_SIZE].view(1, CHUNK_SIZE, self.hv),
                    beta=self.rows_beta[s, :CHUNK_SIZE].view(1, CHUNK_SIZE, self.hv),
                    ssm_states=self.scratch,
                    cache_indices=torch.tensor([s], dtype=torch.int32, device=device),
                    cu_seqlens=torch.tensor(
                        [0, CHUNK_SIZE], dtype=torch.int32, device=device
                    ),
                    scale=self.k**-0.5,
                )
                self.boundary[s] = self.scratch[s]
                self.suffix_len[s] = 0
                pos = 0
            self.rows_qkv[s, pos] = qkv_rows[i]
            self.rows_g[s, pos] = g_rows[i]
            self.rows_beta[s, pos] = beta_rows[i]
            self.suffix_len[s] = pos + 1
            seg_lens.append(pos + 1)
        cat_qkv = torch.cat(
            [self.rows_qkv[s, : self.suffix_len[s]] for s in slots], dim=0
        )
        cat_g = torch.cat([self.rows_g[s, : self.suffix_len[s]] for s in slots], dim=0)
        cat_beta = torch.cat(
            [self.rows_beta[s, : self.suffix_len[s]] for s in slots], dim=0
        )
        cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.tensor(seg_lens, dtype=torch.int32, device=device).cumsum(0)
        slot_idx = torch.tensor(slots, dtype=torch.int32, device=device)

        self.scratch[slot_idx] = self.boundary[slot_idx]
        q, k, v = self._split(cat_qkv)
        o = bi_chunk_gated_delta_rule_prefill(
            q=q,
            k=k,
            v=v,
            g=cat_g.view(1, -1, self.hv),
            beta=cat_beta.view(1, -1, self.hv),
            ssm_states=self.scratch,
            cache_indices=slot_idx,
            cu_seqlens=cu,
            scale=self.k**-0.5,
        )
        # post-token states -> stock pool; completed chunks advance the boundary
        ssm_states[slot_idx] = self.scratch[slot_idx]
        out = o[0, cu[1:].long() - 1]  # last row of each segment: [bs, HV, V]
        for s in slots:
            if self.suffix_len[s] == CHUNK_SIZE:
                self.boundary[s] = self.scratch[s]
                self.suffix_len[s] = 0
        return out
