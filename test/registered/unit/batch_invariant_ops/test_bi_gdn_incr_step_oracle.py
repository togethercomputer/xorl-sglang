"""Cached-row incremental GDN decode vs the partial-chunk-rescan byte oracle.

The class of bug this gate exists to catch: the
incremental GRAPH-path step program silently produced wrong bytes for rows
16..31 of every chunk — Triton 3.6.0 miscompiled the solve row kernel's D=1
merge dot chain at its pinned num_warps<=2, storing zeros (TTIR/PTX correct;
the dot pipeline broke) — while the then-registered component suite never
executed the step program at all, so it stayed green. The full
trainer-vs-sampler gate was the first thing that could see it.

Coverage here, all pure-eager (no server, one GPU, minutes):

1. ``test_step_program_matches_rescan_oracle``: drives the incremental
   step program (the GRAPH-path branch, forced) for 70+ steps against
   ``BIGDNFastDecodeRunner`` on identical inputs, crossing a 64-token chunk
   boundary, at BOTH admitted GDN geometries (dense Qwen3.5-0.8B hv16/hg16
   and Qwen3.6-MoE hv32/hg16) and both defer settings. Visible output,
   fp32 boundary state, and (at flush points) the recurrent pool must be
   BITWISE equal every step — any per-row stage regression (solve, kkt,
   wu, vnew, o) trips it at the first affected fill.
2. ``test_solve_row_matches_full_solve_across_launch_configs``: the solve
   row kernel + commit vs the full solve, for rows covering every
   diagonal block D in {0,1,2,3} (block-first, block-mid, block-last),
   at the production num_warps AND the neighboring configs — silent
   zeros or a launch-config-dependent reduction tree both fail loudly.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.xorl.fla.bi_gdn_decode import BIGDNDecodeCache
from sglang.xorl.fla.bi_gdn_decode_fast import (
    BIGDNFastDecodeRunner,
)
from sglang.xorl.fla.bi_gdn_decode_incr import (
    BIGDNIncrDecodeRunner,
    bi_gdn_incr_solve_commit_kernel,
    bi_gdn_incr_solve_row_kernel,
)
from sglang.xorl.fla.bi_gdn_prefill import solve_tril

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")

CHUNK = 64

# (hv, hg): dense Qwen3.5-0.8B and Qwen3.6-MoE linear-attention geometries
GEOMETRIES = ((16, 16), (32, 16))
K_DIM = 128
V_DIM = 128


def _new_cache(num_slots: int, hv: int, hg: int, device) -> BIGDNDecodeCache:
    qkv_dim = 2 * hg * K_DIM + hv * V_DIM
    return BIGDNDecodeCache(
        num_slots=num_slots,
        qkv_dim=qkv_dim,
        num_v_heads=hv,
        head_k_dim=K_DIM,
        head_v_dim=V_DIM,
        device=device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hv,hg", GEOMETRIES)
@pytest.mark.parametrize("defer_writeback", (False, True))
def test_step_program_matches_rescan_oracle(hv, hg, defer_writeback):
    device = torch.device("cuda")
    num_slots, bs, steps = 6, 3, 70
    qkv_dim = 2 * hg * K_DIM + hv * V_DIM
    gen = torch.Generator(device=device).manual_seed(17)

    cache_incr = _new_cache(num_slots, hv, hg, device)
    cache_fast = _new_cache(num_slots, hv, hg, device)
    ssm_incr = torch.randn(
        num_slots, hv, K_DIM, V_DIM, generator=gen, device=device
    ).mul(0.01)
    ssm_fast = ssm_incr.clone()
    boundary0 = torch.randn(
        num_slots, hv, V_DIM, K_DIM, generator=gen, device=device
    ).mul(0.01)
    cache_incr.boundary.copy_(boundary0)
    cache_fast.boundary.copy_(boundary0)

    incr_runner = BIGDNIncrDecodeRunner()
    incr_runner.defer_writeback = defer_writeback
    incr_runner.slim_vnew = defer_writeback
    # force the GRAPH-path program (the captured composition) eagerly
    incr_runner._graph_path_active = lambda: True
    fast_runner = BIGDNFastDecodeRunner()

    slots = list(range(1, bs + 1))
    indices = torch.tensor(slots, dtype=torch.int32, device=device)
    # start at a chunk boundary; 70 steps cross fill 64 exactly once
    seq_lens = torch.full((bs,), 640, dtype=torch.int64, device=device)

    for step in range(steps):
        seq_lens = seq_lens + 1
        qkv = (
            torch.randn(bs, qkv_dim, generator=gen, device=device)
            .mul(0.05)
            .to(torch.bfloat16)
        )
        g = torch.rand(bs, hv, generator=gen, device=device).mul(-0.2)
        beta = torch.rand(bs, hv, generator=gen, device=device)

        out_incr = incr_runner.step(
            cache=cache_incr,
            indices=indices,
            seq_lens=seq_lens,
            qkv_rows=qkv.clone(),
            g_rows=g.clone(),
            beta_rows=beta.clone(),
            ssm_states=ssm_incr,
        )
        out_fast = fast_runner.step(
            cache=cache_fast,
            indices=indices,
            seq_lens=seq_lens,
            qkv_rows=qkv.clone(),
            g_rows=g.clone(),
            beta_rows=beta.clone(),
            ssm_states=ssm_fast,
        )
        fill = ((int(seq_lens[0]) - 1) % CHUNK) + 1
        at_flush = int(seq_lens[0]) % CHUNK == 0
        assert torch.equal(
            out_incr.view(torch.uint16), out_fast.view(torch.uint16)
        ), f"visible output diverged at step {step} (fill {fill})"
        live = indices.long()
        assert torch.equal(
            cache_incr.boundary[live].view(torch.int32),
            cache_fast.boundary[live].view(torch.int32),
        ), f"fp32 boundary state diverged at step {step} (fill {fill})"
        if not defer_writeback or at_flush:
            assert torch.equal(
                ssm_incr[live].view(torch.int32), ssm_fast[live].view(torch.int32)
            ), f"recurrent pool diverged at step {step} (fill {fill})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# The production config (4) plus a neighbor (8). num_warps=2 is deliberately
# NOT asserted: Triton 3.6.0 miscompiles the D=1 merge dot chain there
# (silent zeros) — the reason the production launch moved off 2. The step
# test above runs the runner's own production config, so a regression of
# the pinned config is caught there.
@pytest.mark.parametrize("num_warps", (4, 8))
@pytest.mark.parametrize("p", (0, 5, 15, 16, 17, 31, 32, 40, 47, 48, 55, 63))
def test_solve_row_matches_full_solve_across_launch_configs(p, num_warps):
    device = torch.device("cuda")
    h = 16
    T = p + 1
    gen = torch.Generator(device=device).manual_seed(31 + p)

    A_full = torch.randn(1, T, h, CHUNK, generator=gen, device=device).mul(0.1)
    strict_lower = (
        torch.arange(T, device=device)[:, None]
        > torch.arange(CHUNK, device=device)[None, :]
    )
    A_full = A_full * strict_lower[None, :, None, :]

    cu = torch.tensor([0, T], dtype=torch.int32, device=device)
    ci = torch.tensor([[0, 0]], dtype=torch.int32, device=device)
    ai_ref = solve_tril(
        A=A_full, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32
    )

    num_slots, slot, bs = 2, 1, 1
    A_cache = torch.zeros(
        num_slots, CHUNK, h, CHUNK, dtype=torch.float32, device=device
    )
    Ai_cache = torch.zeros_like(A_cache)
    Ai16_cache = torch.zeros(
        num_slots, CHUNK, h, CHUNK, dtype=torch.bfloat16, device=device
    )
    A_cache[slot, :T] = A_full[0]
    Ai_cache[slot, :p] = ai_ref[0, :p]

    out_diag = torch.zeros(bs, h, 16, dtype=torch.float32, device=device)
    out_seg = torch.zeros(bs, h, 3, 16, dtype=torch.float32, device=device)
    indices = torch.tensor([slot], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([640 + T], dtype=torch.int64, device=device)

    for d in range(4):
        bi_gdn_incr_solve_row_kernel[(bs, h)](
            A_cache,
            Ai_cache,
            out_diag,
            out_seg,
            indices,
            seq_lens,
            H=h,
            BT=CHUNK,
            D=d,
            CHUNK=CHUNK,
            DOT_PRECISION="ieee",
            num_warps=num_warps,
            num_stages=3,
        )
    bi_gdn_incr_solve_commit_kernel[(bs, h)](
        out_diag,
        out_seg,
        Ai_cache,
        Ai16_cache,
        indices,
        seq_lens,
        H=h,
        BT=CHUNK,
        CHUNK=CHUNK,
    )
    torch.cuda.synchronize()

    got = Ai_cache[slot, p]
    ref = ai_ref[0, p]
    neq = got.view(torch.int32) != ref.view(torch.int32)
    assert not bool(neq.any()), (
        f"solve row p={p} num_warps={num_warps}: {int(neq.sum())}/{neq.numel()} "
        f"elements differ from the full solve (silent zeros or a launch-config-"
        f"dependent reduction tree)"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
