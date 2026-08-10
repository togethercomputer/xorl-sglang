import pytest
import torch

from sglang.srt.batch_invariant_ops import bi_families_v2 as v2


def test_no_residual_hopper_always_selects_split():
    for rows in (1, 32, 512, 4112):
        for n_tiles in (1, 4, 12):
            assert v2._v2_norm_use_split(
                rows,
                n_tiles,
                has_residual=False,
                is_hopper=True,
            )


def test_residual_retains_measured_structure_switch():
    assert v2._v2_norm_use_split(8, 12, has_residual=True)
    assert not v2._v2_norm_use_split(16, 12, has_residual=True)
    assert not v2._v2_norm_use_split(8, 4, has_residual=True)


def test_no_residual_non_hopper_retains_measured_structure_switch():
    assert v2._v2_norm_use_split(8, 12, has_residual=False, is_hopper=False)
    assert not v2._v2_norm_use_split(16, 12, has_residual=False, is_hopper=False)
    assert not v2._v2_norm_use_split(8, 4, has_residual=False, is_hopper=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", ((1, 512), (32, 2048), (32, 6144)))
@pytest.mark.parametrize("zero_centered", (False, True))
def test_no_residual_split_is_bitwise_equal_to_fused(shape, zero_centered):
    generator = torch.Generator(device="cpu").manual_seed(shape[0] * 7 + shape[1])
    x = (
        torch.randn(shape, generator=generator, dtype=torch.float32)
        .to(torch.bfloat16)
        .cuda()
    )
    weight = (
        torch.randn((shape[1],), generator=generator, dtype=torch.float32)
        .to(torch.bfloat16)
        .cuda()
    )

    fused = v2._rms_norm_v2_fused(x, weight, 1e-6, None, zero_centered)
    split = v2._rms_norm_v2_split(x, weight, 1e-6, None, zero_centered)

    assert torch.equal(fused, split)
