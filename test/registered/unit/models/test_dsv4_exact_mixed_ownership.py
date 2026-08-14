from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.dsv4_ownership import (
    dsv4_owner_plane_contribution,
    reconstruct_dsv4_dp_rows,
    resolve_dsv4_owner_plane,
)
from sglang.srt.layers.logical_row_ownership import LogicalRowOwnership
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("dp_size,cp_size", [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_live_dsv4_owner_plane_accepts_every_factorization(
    dp_size: int, cp_size: int
) -> None:
    source = 7
    dp_rank, cp_rank = divmod(source, cp_size)
    owner_group = SimpleNamespace(
        world_size=8, rank_in_group=source, ranks=list(range(16, 24))
    )
    cp_start = dp_rank * cp_size
    context = SimpleNamespace(
        attn_dp_size=dp_size,
        attn_cp_size=cp_size,
        attn_dp_rank=dp_rank,
        attn_cp_rank=cp_rank,
        attn_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        tp_group=owner_group,
        moe_ep_group=SimpleNamespace(
            world_size=8, ranks=list(range(16, 24))
        ),
        attn_cp_group=SimpleNamespace(
            ranks=list(range(16 + cp_start, 16 + cp_start + cp_size))
        ),
    )

    ownership = resolve_dsv4_owner_plane(context)
    assert ownership.source_ordinal == source


def test_ragged_cp_reconstruction_keeps_remote_rows_without_replica_scaling(
) -> None:
    """Distinct CP1 rows survive, while four reconstructed replicas count once."""

    # DP0's CP0 physical shard lacks 202; the active strategy reconstructs it
    # from CP1. If reconstruction is skipped, the final witness loses that row.
    class _Strategy:
        @staticmethod
        def gather_hidden_states(local_rows, _forward_batch):
            assert local_rows.tolist() == [101, 303]
            return torch.tensor([101, 202, 303], dtype=local_rows.dtype)

    dp0_cp0 = LogicalRowOwnership(2, 4, 0, 0, 8)
    reconstructed = reconstruct_dsv4_dp_rows(
        torch.tensor([101, 303]),
        object(),
        dp0_cp0,
        [3, 2],
        context_sharded=True,
        strategy=_Strategy(),
    )
    assert reconstructed.tolist() == [101, 202, 303]

    # Model all eight pre-reduction contributions. Every CP rank has already
    # reconstructed its owner's block, but only CP0 is allowed to contribute.
    contributions = []
    for dp_rank, block in enumerate(
        (torch.tensor([101, 202, 303]), torch.tensor([404, 505]))
    ):
        for cp_rank in range(4):
            ownership = LogicalRowOwnership(2, 4, dp_rank, cp_rank, 8)
            contributions.append(
                dsv4_owner_plane_contribution(block, ownership, [3, 2])
            )
    owner_plane = torch.stack(contributions).sum(dim=0)

    assert owner_plane.tolist() == [101, 202, 303, 404, 505]
    assert owner_plane.tolist() != [value * 4 for value in [101, 202, 303, 404, 505]]
