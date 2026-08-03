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
with). Graph replay uses one fixed decode bucket with a staged per-request
workspace. Idle padded rows are copied into staging but are never scattered
back into live recurrent state. Eager fallback still handles batches outside
that bucket. Costs one extra aligned-prefix scan per extend pass and a
growing (1..64-row) rescan per decode step — the K3 lane pays
composition-identity with throughput, per the correctness-first doctrine.

State model per request slot, per GDN layer:
  - ``ssm_states[slot]`` keeps its stock meaning (state after the last
    consumed token): the rescan's final state is written back every step, so
    the mamba track/radix machinery is undisturbed.
  - ``boundary[slot]`` (this module): fp32 state at the last 64-aligned chunk
    boundary — the rescan's initial state.
  - ``rows_*[slot, :fill]``: the current partial chunk's post-conv packed qkv
    rows (bf16) and gating rows (fp32). The scheduler sequence length, not a
    Python-side layer counter, is authoritative for ``fill`` during decode.

v1 constraints (loud): single-token decode only (no MTP/spec paths); extend
prefix lengths must be 64-aligned (chunked prefill and mamba-track restore
points are; an arbitrary-position resume is not supported).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.layers.attention.fla.bi_gdn_prefill import (
    bi_chunk_gated_delta_rule_prefill,
)
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE
from sglang.srt.utils import get_bool_env_var

BI_GDN_DECODE_ENABLED = get_bool_env_var("SGLANG_BI_GDN_DECODE")
BI_GDN_VERIFY_LEGACY_POSITIONS = get_bool_env_var(
    "SGLANG_BI_GDN_VERIFY_LEGACY_POSITIONS"
)
BI_GDN_BS1_STATIC = get_bool_env_var("SGLANG_BI_GDN_BS1_STATIC")
BI_GDN_DECODE_GRAPH = get_bool_env_var("SGLANG_BI_GDN_DECODE_GRAPH")


@dataclass(frozen=True)
class BIGDNDecodeStepMetadata:
    """Layer-invariant launch metadata for one exact decode forward."""

    slots: tuple[int, ...]
    fill_before: torch.Tensor
    fill_after: torch.Tensor
    cu_seqlens: torch.Tensor
    slot_indices: torch.Tensor
    slot_indices_long: torch.Tensor
    packed_row_indices: torch.Tensor
    output_rows: torch.Tensor
    completed_mask: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    static_bs1: bool


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
            num_slots,
            CHUNK_SIZE,
            qkv_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.rows_g = torch.zeros(
            num_slots, CHUNK_SIZE, num_v_heads, dtype=torch.float32, device=device
        )
        self.rows_beta = torch.zeros(
            num_slots, CHUNK_SIZE, num_v_heads, dtype=torch.float32, device=device
        )
        self._graph_bs = 0
        self._graph_boundary_out: torch.Tensor
        self._graph_rows_qkv_out: torch.Tensor
        self._graph_rows_g_out: torch.Tensor
        self._graph_rows_beta_out: torch.Tensor
        self._graph_core_output: torch.Tensor
        self._graph_workspace_boundary_backup: torch.Tensor
        self._graph_workspace_rows_qkv_backup: torch.Tensor
        self._graph_workspace_rows_g_backup: torch.Tensor
        self._graph_workspace_rows_beta_backup: torch.Tensor
        self.configure_graph_workspace(1)
        # Transition-only oracle. Decode math never reads this list unless the
        # explicit verification flag is set; scheduler seq_lens are authoritative.
        self._legacy_suffix_len = [0] * num_slots

    def configure_graph_workspace(self, graph_bs: int) -> None:
        """Allocate the fixed-size state staging area for one graph bucket."""
        if graph_bs < 1 or graph_bs > self.boundary.shape[0]:
            raise ValueError(
                f"invalid BI GDN graph workspace batch size {graph_bs}; "
                f"expected 1..{self.boundary.shape[0]}"
            )
        if graph_bs == self._graph_bs:
            return
        self._graph_bs = graph_bs
        self._graph_boundary_out = torch.empty_like(self.boundary[:graph_bs])
        self._graph_rows_qkv_out = torch.empty_like(self.rows_qkv[:graph_bs])
        self._graph_rows_g_out = torch.empty_like(self.rows_g[:graph_bs])
        self._graph_rows_beta_out = torch.empty_like(self.rows_beta[:graph_bs])
        self._graph_core_output = torch.empty(
            graph_bs,
            self.hv,
            self.v,
            dtype=torch.bfloat16,
            device=self.boundary.device,
        )
        self._graph_workspace_boundary_backup = torch.empty_like(
            self.boundary[:graph_bs]
        )
        self._graph_workspace_rows_qkv_backup = torch.empty_like(
            self.rows_qkv[:graph_bs]
        )
        self._graph_workspace_rows_g_backup = torch.empty_like(self.rows_g[:graph_bs])
        self._graph_workspace_rows_beta_backup = torch.empty_like(
            self.rows_beta[:graph_bs]
        )

    def copy_to_graph_workspace(self, state_indices: torch.Tensor) -> None:
        """Gather live requests into the fixed graph workspace slots."""
        if state_indices.numel() != self._graph_bs:
            raise RuntimeError(
                "BI GDN graph workspace width mismatch: "
                f"expected {self._graph_bs}, got {state_indices.numel()}"
            )
        # DP-attention keeps the graph collective-shaped by replaying an idle
        # row with PAD_SLOT_ID (-1) on ranks that have no local request. PyTorch
        # index_select rejects negative indices, so use slot zero as a harmless
        # staging source. The graph slots are still real request slots, so
        # preserve them independently before staging another batch.
        # copy_from_graph_workspace restores it before scattering the result.
        source = state_indices.clamp_min(0).long()
        graph_slice = slice(0, self._graph_bs)
        self._graph_workspace_boundary_backup.copy_(self.boundary[graph_slice])
        self._graph_workspace_rows_qkv_backup.copy_(self.rows_qkv[graph_slice])
        self._graph_workspace_rows_g_backup.copy_(self.rows_g[graph_slice])
        self._graph_workspace_rows_beta_backup.copy_(self.rows_beta[graph_slice])
        self.boundary[graph_slice].copy_(self.boundary.index_select(0, source))
        self.rows_qkv[graph_slice].copy_(self.rows_qkv.index_select(0, source))
        self.rows_g[graph_slice].copy_(self.rows_g.index_select(0, source))
        self.rows_beta[graph_slice].copy_(self.rows_beta.index_select(0, source))

    def copy_from_graph_workspace(self, state_indices: torch.Tensor) -> None:
        """Scatter fixed graph workspace slots back to live request slots."""
        if state_indices.numel() != self._graph_bs:
            raise RuntimeError(
                "BI GDN graph workspace width mismatch: "
                f"expected {self._graph_bs}, got {state_indices.numel()}"
            )
        graph_slice = slice(0, self._graph_bs)
        self._graph_boundary_out.copy_(self.boundary[graph_slice])
        self._graph_rows_qkv_out.copy_(self.rows_qkv[graph_slice])
        self._graph_rows_g_out.copy_(self.rows_g[graph_slice])
        self._graph_rows_beta_out.copy_(self.rows_beta[graph_slice])
        self.boundary[graph_slice].copy_(self._graph_workspace_boundary_backup)
        self.rows_qkv[graph_slice].copy_(self._graph_workspace_rows_qkv_backup)
        self.rows_g[graph_slice].copy_(self._graph_workspace_rows_g_backup)
        self.rows_beta[graph_slice].copy_(self._graph_workspace_rows_beta_backup)
        # Padding rows use -1 and are staged from slot zero only to keep the
        # captured graph's tensor shapes fixed. Do not clamp them to slot zero
        # during the scatter: if slot zero is also live, an index_copy with the
        # padded duplicate could overwrite its real update.
        valid_positions = torch.nonzero(state_indices >= 0, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            return
        destination = state_indices.index_select(0, valid_positions).long()
        self.boundary.index_copy_(
            0, destination, self._graph_boundary_out.index_select(0, valid_positions)
        )
        self.rows_qkv.index_copy_(
            0, destination, self._graph_rows_qkv_out.index_select(0, valid_positions)
        )
        self.rows_g.index_copy_(
            0, destination, self._graph_rows_g_out.index_select(0, valid_positions)
        )
        self.rows_beta.index_copy_(
            0,
            destination,
            self._graph_rows_beta_out.index_select(0, valid_positions),
        )

    def clear_graph_workspace(self) -> None:
        """Restore the graph staging slots after dummy capture forwards."""
        graph_slice = slice(0, self._graph_bs)
        self.boundary[graph_slice].zero_()
        self.scratch[graph_slice].zero_()
        self.rows_qkv[graph_slice].zero_()
        self.rows_g[graph_slice].zero_()
        self.rows_beta[graph_slice].zero_()
        self._legacy_suffix_len[: self._graph_bs] = [0] * self._graph_bs

    def prepare_step_metadata(
        self,
        slots: list[int],
        slot_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        static_bs1: bool = BI_GDN_BS1_STATIC,
        graph_capture: bool = False,
    ) -> BIGDNDecodeStepMetadata:
        """Build scheduler-derived metadata once for reuse by every GDN layer.

        Every partial rescan contains exactly one <=64-token chunk per request,
        so the chunk map and packed row selection are constructed directly on
        device. ``seq_lens`` already includes the token currently being
        consumed by decode; therefore its zero-based row is
        ``(seq_lens - 1) % CHUNK_SIZE``.
        """
        device = slot_indices.device
        bs = len(slots)
        if seq_lens.shape != slot_indices.shape:
            raise RuntimeError(
                "SGLANG_BI_GDN_DECODE sequence lengths and slots differ: "
                f"{tuple(seq_lens.shape)} != {tuple(slot_indices.shape)} (XORL-245)."
            )
        # A CUDA graph captures tensor shapes as well as addresses.  The graph
        # runner captures with its one-token dummy sequence length, but replay
        # must support every partial chunk length up to CHUNK_SIZE.  Capture a
        # full fixed-width row buffer and let the replay prologue update
        # cu_seqlens/output_rows for the actual logical lengths.
        metadata_seq_lens = (
            torch.full_like(seq_lens, CHUNK_SIZE) if graph_capture else seq_lens
        )
        fill_before = torch.remainder(metadata_seq_lens - 1, CHUNK_SIZE).to(torch.int32)
        fill_after = fill_before + 1
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(fill_after, dim=0, dtype=torch.int32),
            )
        )
        row_offsets = torch.arange(CHUNK_SIZE, dtype=torch.int32, device=device)
        use_static_bs1 = static_bs1 and bs == 1
        if use_static_bs1:
            packed_row_indices = torch.empty(0, dtype=torch.long, device=device)
        else:
            packed_rows = slot_indices[:, None] * CHUNK_SIZE + row_offsets[None, :]
            if graph_capture:
                packed_row_indices = packed_rows.reshape(-1).long()
            else:
                packed_row_indices = packed_rows[
                    row_offsets[None, :] < fill_after[:, None]
                ].long()
        output_rows = cu_seqlens[1:].long() - 1
        slot_indices_long = slot_indices.long()
        completed_mask = fill_after == CHUNK_SIZE
        chunk_indices = torch.stack(
            (
                torch.arange(bs, dtype=torch.int32, device=device),
                torch.zeros(bs, dtype=torch.int32, device=device),
            ),
            dim=1,
        )
        chunk_offsets = torch.arange(bs + 1, dtype=torch.int32, device=device)
        return BIGDNDecodeStepMetadata(
            slots=tuple(slots),
            fill_before=fill_before,
            fill_after=fill_after,
            cu_seqlens=cu_seqlens,
            slot_indices=slot_indices,
            slot_indices_long=slot_indices_long,
            packed_row_indices=packed_row_indices,
            output_rows=output_rows,
            completed_mask=completed_mask,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            static_bs1=use_static_bs1,
        )

    def refresh_graph_metadata(
        self, metadata: BIGDNDecodeStepMetadata, seq_lens: torch.Tensor
    ) -> None:
        """Refresh fixed-address metadata in the uncaptured replay prologue."""
        if seq_lens.numel() != len(metadata.slots):
            raise RuntimeError(
                "BI GDN graph metadata width mismatch: "
                f"expected {len(metadata.slots)}, got {seq_lens.numel()}"
            )
        fill_before = torch.remainder(seq_lens - 1, CHUNK_SIZE).to(torch.int32)
        metadata.fill_before.copy_(fill_before)
        metadata.fill_after.copy_(fill_before + 1)
        metadata.cu_seqlens[0].zero_()
        metadata.cu_seqlens[1:].copy_(
            torch.cumsum(metadata.fill_after, dim=0, dtype=torch.int32)
        )
        metadata.output_rows.copy_(metadata.cu_seqlens[1:].long() - 1)
        metadata.completed_mask.copy_(metadata.fill_after == CHUNK_SIZE)
        if metadata.packed_row_indices.numel() == len(metadata.slots) * CHUNK_SIZE:
            # The graph was captured with a fixed CHUNK_SIZE rows per request.
            # Keep the index tensor's shape fixed, but compact each request's
            # live prefix into the leading ``cu_seqlens[-1]`` entries.  The
            # varlen kernels consume that leading range; leaving the indices
            # in slot-major [slot0 x 64, slot1 x 64, ...] order would make
            # request 1 read slot 0's inactive tail whenever request 0 has a
            # partial chunk shorter than CHUNK_SIZE.
            row_offsets = torch.arange(
                CHUNK_SIZE, dtype=torch.int32, device=metadata.slot_indices.device
            )
            packed_rows = (
                metadata.slot_indices[:, None] * CHUNK_SIZE + row_offsets[None, :]
            )
            flat_row_offsets = torch.arange(
                metadata.packed_row_indices.numel(),
                dtype=torch.int32,
                device=metadata.slot_indices.device,
            ).view(len(metadata.slots), CHUNK_SIZE)
            active = row_offsets[None, :] < metadata.fill_after[:, None]
            active_destinations = metadata.cu_seqlens[:-1, None] + row_offsets[None, :]
            inactive_destinations = (
                metadata.cu_seqlens[-1]
                + flat_row_offsets
                - metadata.cu_seqlens[1:, None]
            )
            destinations = torch.where(
                active, active_destinations, inactive_destinations
            )
            metadata.packed_row_indices.scatter_(
                0,
                destinations.reshape(-1).long(),
                packed_rows.reshape(-1).long(),
            )

    def _verify_and_advance_legacy_positions(
        self, metadata: BIGDNDecodeStepMetadata
    ) -> None:
        """Check scheduler-derived rows against the removed host authority."""
        fill_before = metadata.fill_before.tolist()
        fill_after = metadata.fill_after.tolist()
        for slot, before, after in zip(metadata.slots, fill_before, fill_after):
            legacy = self._legacy_suffix_len[slot]
            if legacy != before:
                raise RuntimeError(
                    "SGLANG_BI_GDN_DECODE scheduler/cache position mismatch: "
                    f"slot={slot}, scheduler={before}, legacy={legacy} (XORL-245)."
                )
            self._legacy_suffix_len[slot] = 0 if after == CHUNK_SIZE else after

    def _split(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[T, qkv_dim] packed q|k|v -> ([1,T,H,K], [1,T,H,K], [1,T,HV,V])."""
        h = (self.qkv_dim - self.hv * self.v) // (2 * self.k)
        q, k, v = torch.split(
            rows,
            [h * self.k, h * self.k, self.hv * self.v],
            dim=-1,
        )
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
        self._legacy_suffix_len[slot] = suffix

    def step(
        self,
        metadata: BIGDNDecodeStepMetadata,
        qkv_rows: torch.Tensor,
        g_rows: torch.Tensor,
        beta_rows: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One batched single-token decode step via partial-chunk rescan.

        qkv_rows: [bs, qkv_dim] post-conv packed rows; g/beta: [bs, HV] fp32.
        Writes each slot's post-token state back into ``ssm_states`` (stock
        pool semantics preserved) and returns core_attn_out [bs, HV, V].
        """
        if BI_GDN_VERIFY_LEGACY_POSITIONS:
            self._verify_and_advance_legacy_positions(metadata)
        slot_idx_long = metadata.slot_indices_long
        state_idx = metadata.slot_indices if state_indices is None else state_indices
        write_rows = metadata.fill_before.long()
        self.rows_qkv[slot_idx_long, write_rows] = qkv_rows
        self.rows_g[slot_idx_long, write_rows] = g_rows
        self.rows_beta[slot_idx_long, write_rows] = beta_rows
        if metadata.static_bs1:
            # The contracted varlen kernels use cu_seqlens as the true logical
            # extent. Present the resident 64-row slot view directly so bs1 has
            # neither a host slice nor a gather/copy kernel.
            slot = metadata.slots[0]
            cat_qkv = self.rows_qkv[slot]
            cat_g = self.rows_g[slot]
            cat_beta = self.rows_beta[slot]
        else:
            packed = metadata.packed_row_indices
            cat_qkv = self.rows_qkv.flatten(0, 1).index_select(0, packed)
            cat_g = self.rows_g.flatten(0, 1).index_select(0, packed)
            cat_beta = self.rows_beta.flatten(0, 1).index_select(0, packed)
        cu = metadata.cu_seqlens
        slot_idx = metadata.slot_indices

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
            chunk_indices=metadata.chunk_indices,
            chunk_offsets=metadata.chunk_offsets,
            cache_indices_long=metadata.slot_indices_long,
        )
        # post-token states -> stock pool; completed chunks advance the boundary
        safe_state_idx = state_idx.clamp_min(0)
        state_valid = (state_idx >= 0).view(-1, 1, 1, 1)
        ssm_states[safe_state_idx] = torch.where(
            state_valid,
            self.scratch[slot_idx],
            ssm_states[safe_state_idx],
        )
        out = o[0, metadata.output_rows]  # last row of each segment: [bs, HV, V]
        selected_boundary = self.boundary[slot_idx]
        selected_scratch = self.scratch[slot_idx]
        completed = metadata.completed_mask.view(-1, 1, 1, 1)
        self.boundary[slot_idx] = torch.where(
            completed, selected_scratch, selected_boundary
        )
        return out
