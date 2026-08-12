"""The fused RoPE kernel must exactly replay the eager bf16 expression."""

import pytest
import torch

from sglang.srt.layers.rotary_embedding.bi_fused_native import (
    bi_fused_native_rope,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _cache(max_position: int, rotary_dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        500000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    positions = torch.arange(max_position, dtype=torch.float32)
    frequencies = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).cuda()


def _eager(
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    cos, sin = cache.index_select(0, positions).chunk(2, dim=-1)
    rotated = apply_rotary_emb(x[..., :rotary_dim], cos, sin, True)
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


@requires_cuda
@pytest.mark.parametrize(
    "tokens,heads,head_size,rotary_dim",
    [
        (1, 30, 128, 128),
        (64, 30, 128, 128),
        (3500, 30, 128, 128),
        (17, 8, 128, 64),
        (5, 4, 96, 96),
    ],
)
def test_bitwise_matches_eager(
    tokens: int, heads: int, head_size: int, rotary_dim: int
):
    torch.manual_seed(0)
    # Exercise the production cache limit, where position-amplified constant
    # and rounding differences are more likely to become bf16-visible.
    max_position = 40960
    cache = _cache(max_position, rotary_dim)
    positions = torch.randint(0, max_position, (tokens,), device="cuda")
    positions[0] = max_position - 1
    if tokens > 1:
        positions[1] = 0
    x = torch.randn(
        tokens, heads, head_size, device="cuda", dtype=torch.float32
    ).bfloat16()

    expected = _eager(x, positions, cache, rotary_dim)
    actual = bi_fused_native_rope(x, positions, cache, rotary_dim)
    assert torch.equal(actual, expected)


@requires_cuda
def test_bitwise_matches_packed_qkv_views():
    torch.manual_seed(1)
    tokens, query_heads, key_heads, head_size = 64, 30, 30, 128
    cache = _cache(4096, head_size)
    positions = torch.randint(0, 4096, (tokens,), device="cuda")
    qkv = torch.randn(
        tokens,
        (query_heads + 2 * key_heads) * head_size,
        device="cuda",
        dtype=torch.float32,
    ).bfloat16()
    query = qkv[:, : query_heads * head_size].view(tokens, query_heads, head_size)
    key = qkv[:, query_heads * head_size : (query_heads + key_heads) * head_size].view(
        tokens, key_heads, head_size
    )

    for tensor in (query, key):
        assert tensor.stride(-1) == 1 and not tensor.is_contiguous()
        expected = _eager(tensor, positions, cache, head_size)
        actual = bi_fused_native_rope(tensor, positions, cache, head_size)
        assert torch.equal(actual, expected)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
