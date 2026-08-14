"""Logical GLM-5.2 row ownership across attention DP and CP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogicalRowOwnership:
    """DP-major, CP-minor placement inside one canonical contributor group."""

    dp_size: int
    cp_size: int
    dp_rank: int
    cp_rank: int
    contributor_count: int

    def __post_init__(self) -> None:
        if self.dp_size <= 0 or self.cp_size <= 0:
            raise ValueError("Logical row ownership degrees must be positive")
        if self.dp_size * self.cp_size != self.contributor_count:
            raise ValueError(
                "Logical DP and CP ownership must exactly cover the canonical "
                f"contributors: {self.dp_size} * {self.cp_size} != {self.contributor_count}"
            )
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError(
                f"DP rank {self.dp_rank} is outside DP size {self.dp_size}"
            )
        if not 0 <= self.cp_rank < self.cp_size:
            raise ValueError(
                f"CP rank {self.cp_rank} is outside CP size {self.cp_size}"
            )

    @property
    def source_ordinal(self) -> int:
        return self.dp_rank * self.cp_size + self.cp_rank

    @property
    def representative_ordinal(self) -> int:
        return self.dp_rank * self.cp_size

    @property
    def context_source_ordinals(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.representative_ordinal, self.representative_ordinal + self.cp_size
            )
        )

    def dp_block_slice(self, segment_lengths: list[int]) -> slice:
        if len(segment_lengths) != self.dp_size or any(
            length < 0 for length in segment_lengths
        ):
            raise ValueError(
                "DP row segment lengths must provide one nonnegative entry per DP owner"
            )
        start = sum(segment_lengths[: self.dp_rank])
        return slice(start, start + segment_lengths[self.dp_rank])

    def local_source_slice(
        self,
        segment_lengths: list[int],
        *,
        local_rows: int,
        context_sharded: bool,
    ) -> slice:
        block = self.dp_block_slice(segment_lengths)
        block_rows = block.stop - block.start
        expected_block_rows = (
            local_rows * self.cp_size if context_sharded else local_rows
        )
        if block_rows != expected_block_rows:
            raise ValueError(
                "Logical row block does not match the rank-local source layout: "
                f"block_rows={block_rows}, expected={expected_block_rows}"
            )
        cp_offset = self.cp_rank * local_rows if context_sharded else 0
        return slice(block.start + cp_offset, block.start + cp_offset + local_rows)

    def select_dp_representatives(self, physical_values: list[object]) -> list[object]:
        if len(physical_values) != self.contributor_count:
            raise ValueError(
                "Physical ownership metadata must name every canonical contributor"
            )
        logical_values = []
        for dp_rank in range(self.dp_size):
            start = dp_rank * self.cp_size
            group = physical_values[start : start + self.cp_size]
            if any(value != group[0] for value in group[1:]):
                raise ValueError(
                    "CP replicas disagree on their DP-owned request metadata at "
                    f"dp_rank={dp_rank}: {group}"
                )
            logical_values.append(group[0])
        return logical_values


__all__ = ["LogicalRowOwnership"]
