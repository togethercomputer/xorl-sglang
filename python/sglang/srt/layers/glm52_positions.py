"""Lightweight GLM-5.2 absolute-position alignment helpers."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch


@dataclass(frozen=True)
class CanonicalMoEPositions:
    values: torch.Tensor
    valid_mask: torch.Tensor


class Glm52MlpRowLayout(str, Enum):
    """Runtime row order produced at the GLM MLP boundary."""

    LOCAL_LOGICAL = "local_logical"
    OWNER_LOGICAL = "owner_logical"
    OWNER_CP_PHYSICAL = "owner_cp_physical"
    GLOBAL_LOGICAL = "global_logical"
    GLOBAL_CP_PHYSICAL = "global_cp_physical"

    @property
    def uses_gathered_lora_metadata(self) -> bool:
        return self is not Glm52MlpRowLayout.LOCAL_LOGICAL


@dataclass(frozen=True)
class Glm52MlpRowState:
    layout: Glm52MlpRowLayout
    rows: int


def set_glm52_mlp_row_state(
    forward_batch,
    layout: Glm52MlpRowLayout,
    rows: int,
) -> Glm52MlpRowState:
    """Record the row order actually produced for the next GLM MLP."""

    rows = int(rows)
    if rows < 0:
        raise ValueError("GLM-5.2 MLP row count must be nonnegative")
    state = Glm52MlpRowState(layout=layout, rows=rows)
    forward_batch._glm52_mlp_row_state = state
    return state


def get_glm52_mlp_row_state(
    forward_batch,
    *,
    expected_rows: int,
) -> Glm52MlpRowState:
    """Return the runtime row order and verify its activation capacity."""

    state = getattr(forward_batch, "_glm52_mlp_row_state", None)
    if not isinstance(state, Glm52MlpRowState):
        raise RuntimeError("GLM-5.2 MLP row layout was not recorded by prepare_mlp")
    if state.rows != int(expected_rows):
        raise RuntimeError(
            "GLM-5.2 MLP row layout does not match the activation rows: "
            f"layout_rows={state.rows}, activation_rows={int(expected_rows)}."
        )
    return state


def reset_glm52_mlp_row_state(forward_batch) -> None:
    """Clear row-layout data derived during a previous model forward."""

    forward_batch._glm52_mlp_row_state = None
    forward_batch._glm52_owned_moe_positions = None
    forward_batch._glm52_owned_moe_position_row_state = None


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


__all__ = [
    "CanonicalMoEPositions",
    "Glm52MlpRowLayout",
    "Glm52MlpRowState",
    "align_glm52_moe_positions",
    "get_glm52_mlp_row_state",
    "reset_glm52_mlp_row_state",
    "set_glm52_mlp_row_state",
]
