"""Stage-local logical-row movement for the exact DeepSeek-V4 path."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributed as dist

from sglang.srt.layers.logical_row_ownership import LogicalRowOwnership
from sglang.srt.runtime_context import get_parallel

DSV4_EXACT_CONTRIBUTORS = 8


def resolve_dsv4_owner_plane(context: Any | None = None) -> LogicalRowOwnership:
    """Resolve and validate the live DP-major/CP-minor owner plane.

    The checks are stage-local and deliberately make no claim about pipeline
    degree.  They preserve the intrinsic DSV4 exact seams: body attention and
    expert TP1, EP8 routed payloads, and the full TP8 owner/head group.
    """

    parallel = get_parallel() if context is None else context
    ownership = LogicalRowOwnership(
        dp_size=int(parallel.attn_dp_size),
        cp_size=int(parallel.attn_cp_size),
        dp_rank=int(parallel.attn_dp_rank),
        cp_rank=int(parallel.attn_cp_rank),
        contributor_count=DSV4_EXACT_CONTRIBUTORS,
    )
    if int(parallel.attn_tp_size) != 1 or int(parallel.moe_tp_size) != 1:
        raise RuntimeError(
            "Exact DSV4 owner movement requires stage-local attention and expert TP1"
        )
    if int(parallel.moe_ep_size) != DSV4_EXACT_CONTRIBUTORS:
        raise RuntimeError("Exact DSV4 routed payloads require EP8")

    owner_group = parallel.tp_group
    expert_group = parallel.moe_ep_group
    if (
        owner_group is None
        or int(owner_group.world_size) != DSV4_EXACT_CONTRIBUTORS
        or int(owner_group.rank_in_group) != ownership.source_ordinal
    ):
        raise RuntimeError(
            "Exact DSV4 owner-group order must be DP-major and CP-minor across TP8"
        )
    owner_ranks = tuple(owner_group.ranks)
    if (
        expert_group is None
        or int(expert_group.world_size) != DSV4_EXACT_CONTRIBUTORS
        or tuple(expert_group.ranks) != owner_ranks
    ):
        raise RuntimeError(
            "Exact DSV4 EP8 contributors must cover the stage-local owner plane in the same order"
        )
    if ownership.cp_size > 1:
        cp_group = parallel.attn_cp_group
        expected_cp_ranks = tuple(
            owner_ranks[index] for index in ownership.context_source_ordinals
        )
        if cp_group is None or tuple(cp_group.ranks) != expected_cp_ranks:
            raise RuntimeError(
                "Exact DSV4 CP membership does not match the DP-major owner-plane order"
            )
    return ownership


def normalize_dsv4_dp_row_segments(
    segment_lengths: Sequence[int], ownership: LogicalRowOwnership
) -> list[int]:
    segments = [int(length) for length in segment_lengths]
    if len(segments) != ownership.dp_size or any(length < 0 for length in segments):
        raise RuntimeError(
            "Exact DSV4 row metadata must provide one nonnegative segment per DP owner"
        )
    return segments


def reconstruct_dsv4_dp_rows(
    local_rows: torch.Tensor,
    forward_batch: Any,
    ownership: LogicalRowOwnership,
    segment_lengths: Sequence[int],
    *,
    context_sharded: bool,
    strategy: Any | None = None,
) -> torch.Tensor:
    """Reconstruct this DP owner's logical rows before selecting a CP rep."""

    segments = normalize_dsv4_dp_row_segments(segment_lengths, ownership)
    block = ownership.dp_block_slice(segments)
    expected_rows = block.stop - block.start
    if context_sharded:
        if strategy is None:
            from sglang.srt.layers.cp.base import get_cp_strategy

            strategy = get_cp_strategy()
        if strategy is None:
            raise RuntimeError(
                "Exact DSV4 CP-sharded rows require an active context-parallel strategy"
            )
        dp_rows = strategy.gather_hidden_states(local_rows, forward_batch)
    else:
        dp_rows = local_rows
    if dp_rows.shape[0] != expected_rows:
        raise RuntimeError(
            "Exact DSV4 reconstructed DP rows do not match the prepared owner block: "
            f"rows={dp_rows.shape[0]}, expected={expected_rows}, dp_rank={ownership.dp_rank}"
        )
    return dp_rows


def dsv4_owner_plane_contribution(
    dp_rows: torch.Tensor,
    ownership: LogicalRowOwnership,
    segment_lengths: Sequence[int],
) -> torch.Tensor:
    """Build one rank's pre-reduction owner-plane contribution.

    Every CP rank first reconstructs the complete DP block.  Only CP0 then
    contributes it, preventing both lost non-CP0 shards and CP-size scaling.
    """

    segments = normalize_dsv4_dp_row_segments(segment_lengths, ownership)
    block = ownership.dp_block_slice(segments)
    if dp_rows.shape[0] != block.stop - block.start:
        raise RuntimeError(
            "Exact DSV4 DP rows do not match their owner-plane destination block"
        )
    contribution = dp_rows.new_zeros((sum(segments), *dp_rows.shape[1:]))
    if ownership.cp_rank == 0:
        contribution[block].copy_(dp_rows)
    return contribution


def gather_dsv4_owner_plane_rows(
    dp_rows: torch.Tensor,
    ownership: LogicalRowOwnership,
    segment_lengths: Sequence[int],
    *,
    output: torch.Tensor | None = None,
    group: Any | None = None,
) -> torch.Tensor:
    """Replicate one logical block per DP owner over the full TP8 plane."""

    contribution = dsv4_owner_plane_contribution(dp_rows, ownership, segment_lengths)
    if output is None:
        output = contribution
    else:
        if output.shape != contribution.shape or output.dtype != contribution.dtype:
            raise RuntimeError(
                "Exact DSV4 owner-plane output does not match the logical row layout"
            )
        output.copy_(contribution)
    if ownership.contributor_count > 1:
        owner_group = get_parallel().tp_group if group is None else group
        device_group = getattr(owner_group, "device_group", owner_group)
        dist.all_reduce(output, group=device_group)
    return output


def reassemble_dsv4_local_rows(
    owner_plane_rows: torch.Tensor,
    forward_batch: Any,
    ownership: LogicalRowOwnership,
    segment_lengths: Sequence[int],
    *,
    context_sharded: bool,
    expected_local_rows: int,
    strategy: Any | None = None,
) -> torch.Tensor:
    """Select the originating DP block and restore its active CP shard."""

    segments = normalize_dsv4_dp_row_segments(segment_lengths, ownership)
    if owner_plane_rows.shape[0] != sum(segments):
        raise RuntimeError(
            "Exact DSV4 completed rows do not cover the logical owner plane"
        )
    dp_rows = owner_plane_rows[ownership.dp_block_slice(segments)]
    if context_sharded:
        if strategy is None:
            from sglang.srt.layers.cp.base import get_cp_strategy

            strategy = get_cp_strategy()
        if strategy is None:
            raise RuntimeError(
                "Exact DSV4 CP reassembly requires an active context-parallel strategy"
            )
        local_rows = strategy.shard_hidden_states(dp_rows, forward_batch)
    else:
        local_rows = dp_rows
    if local_rows.shape[0] != expected_local_rows:
        raise RuntimeError(
            "Exact DSV4 reassembly returned the wrong local row count: "
            f"rows={local_rows.shape[0]}, expected={expected_local_rows}"
        )
    return local_rows


__all__ = [
    "DSV4_EXACT_CONTRIBUTORS",
    "dsv4_owner_plane_contribution",
    "gather_dsv4_owner_plane_rows",
    "normalize_dsv4_dp_row_segments",
    "reassemble_dsv4_local_rows",
    "reconstruct_dsv4_dp_rows",
    "resolve_dsv4_owner_plane",
]
