"""Lightweight GLM-5.2 absolute-position alignment helpers."""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class CanonicalMoEPositions:
    values: torch.Tensor
    valid_mask: torch.Tensor


def align_glm52_moe_positions(
    positions: torch.Tensor,
    full_hidden_states: torch.Tensor,
    *,
    prefill_cp: bool,
    cp_size: int = 1,
    all_gather: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> CanonicalMoEPositions:
    """Match absolute positions to the rank-major FULL hidden-state layout."""

    local_positions = positions.reshape(-1).to(torch.int64)
    if not prefill_cp:
        if local_positions.numel() != full_hidden_states.shape[0]:
            raise ValueError(
                "Decode positions must have one entry per replicated MoE row"
            )
        return CanonicalMoEPositions(local_positions, local_positions >= 0)

    if all_gather is None:
        raise ValueError("NSA prefill position alignment requires the CP all-gather")
    if cp_size <= 1 or full_hidden_states.shape[0] % cp_size:
        raise ValueError(
            "FULL NSA MoE rows must divide evenly across the attention CP group"
        )
    local_capacity = full_hidden_states.shape[0] // cp_size
    if local_positions.numel() > local_capacity:
        raise ValueError("Rank-local NSA positions exceed the FULL MoE gather capacity")

    padded_positions = local_positions.new_full((local_capacity,), -1)
    padded_positions[: local_positions.numel()].copy_(local_positions)
    padded_valid = torch.zeros(
        (local_capacity,), dtype=torch.uint8, device=local_positions.device
    )
    padded_valid[: local_positions.numel()].copy_(
        (local_positions >= 0).to(torch.uint8)
    )

    full_positions = local_positions.new_empty((full_hidden_states.shape[0],))
    full_valid = padded_valid.new_empty((full_hidden_states.shape[0],))
    all_gather(full_positions, padded_positions)
    all_gather(full_valid, padded_valid)
    return CanonicalMoEPositions(full_positions, full_valid.to(torch.bool))


__all__ = ["CanonicalMoEPositions", "align_glm52_moe_positions"]
