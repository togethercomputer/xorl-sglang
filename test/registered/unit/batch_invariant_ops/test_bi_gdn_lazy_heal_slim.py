"""Regression coverage for batched GDN warm composed with slim v_new."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.bi_gdn_decode import BIGDNDecodeCache
from sglang.kernels.ops.attention.fla.bi_gdn_decode_incr import (
    BIGDNIncrDecodeRunner,
)
from sglang.kernels.ops.attention.fla.bi_gdn_incr_lazy_heal import (
    warm_slots_batched,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_warm_persists_every_slim_cache_row():
    device = torch.device("cuda")
    num_slots, hv, hg, k_dim, v_dim = 4, 4, 2, 128, 128
    qkv_dim = 2 * hg * k_dim + hv * v_dim
    pairs = ((1, 1), (3, 17))

    def new_cache():
        return BIGDNDecodeCache(
            num_slots=num_slots,
            qkv_dim=qkv_dim,
            num_v_heads=hv,
            head_k_dim=k_dim,
            head_v_dim=v_dim,
            device=device,
        )

    reference = new_cache()
    batched = new_cache()
    generator = torch.Generator(device=device).manual_seed(23)
    reference.boundary.normal_(generator=generator).mul_(0.01)
    reference.rows_qkv.normal_(generator=generator).mul_(0.01)
    reference.rows_g.uniform_(-0.25, 0.0, generator=generator)
    reference.rows_beta.uniform_(0.0, 1.0, generator=generator)
    for name in ("boundary", "rows_qkv", "rows_g", "rows_beta"):
        getattr(batched, name).copy_(getattr(reference, name))

    def new_runner():
        runner = BIGDNIncrDecodeRunner()
        runner.defer_writeback = True
        runner.slim_vnew = True
        return runner

    reference_runner = new_runner()
    batched_runner = new_runner()
    for slot, fill in pairs:
        reference_runner.warm_slot(reference, slot, fill)
    warm_slots_batched(batched_runner, batched, pairs)

    reference_rows = reference_runner._layer_caches(reference)
    batched_rows = batched_runner._layer_caches(batched)
    for name in (
        "l2q",
        "l2k",
        "A",
        "Ai32",
        "Ai16",
        "w",
        "u",
        "gcum",
        "v_new",
    ):
        assert torch.equal(
            getattr(reference_rows, name), getattr(batched_rows, name)
        ), name


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
