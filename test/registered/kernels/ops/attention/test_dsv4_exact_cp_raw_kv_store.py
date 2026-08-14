"""Byte contract for exact DeepSeek-V4 CP KV-cache formation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="DeepSeek-V4 FlashMLA cache-byte test requires SM90",
)


_PAGE_SIZE = 128
_SLOT_BYTES = 584
_PAGE_ALIGNMENT = 576


def _new_cache(num_pages: int) -> torch.Tensor:
    raw_bytes = _PAGE_SIZE * _SLOT_BYTES
    padded_bytes = (
        (raw_bytes + _PAGE_ALIGNMENT - 1) // _PAGE_ALIGNMENT * _PAGE_ALIGNMENT
    )
    return torch.zeros((num_pages, padded_bytes), dtype=torch.uint8, device="cuda")


class _FusedCachePool:
    def __init__(self, cache: torch.Tensor):
        self.cache = cache

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        *,
        layer_id,
        swa_loc,
        kv,
        kv_weight,
        eps,
        freqs_cis,
        positions,
    ) -> None:
        del layer_id
        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=swa_loc,
            kvcache=self.cache,
            page_size=_PAGE_SIZE,
        )


def test_exact_cp_raw_gather_produces_cp1_identical_cache_bytes():
    generator = torch.Generator(device="cpu").manual_seed(314159)
    raw_kv = (
        torch.randn((7, 512), generator=generator, dtype=torch.float32)
        .clamp_(-2.0, 2.0)
        .to(torch.bfloat16)
        .cuda()
    )
    positions = torch.tensor([3, 4, 17, 18, 31, 32, 47], device="cuda")
    swa_loc = torch.tensor(
        [1, 2, 17, 127, 128, 129, 255], dtype=torch.int32, device="cuda"
    )

    attn = MQALayer.__new__(MQALayer)
    nn.Module.__init__(attn)
    attn.layer_id = 0
    attn.kv_norm = nn.LayerNorm(512, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        attn.kv_norm.weight.copy_(
            torch.linspace(0.5, 1.5, 512, dtype=torch.bfloat16, device="cuda")
        )
    attn.eps = 1e-6
    attn.freqs_cis = precompute_freqs_cis(
        dim=64,
        seqlen=64,
        original_seq_len=0,
        base=10_000,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
    ).cuda()

    backend = SimpleNamespace(get_swa_out_cache_loc=lambda _forward_batch: swa_loc)
    forward_batch = SimpleNamespace()
    cp1_cache = _new_cache(2)
    cp_cache = _new_cache(2)

    with patch(
        "sglang.srt.models.deepseek_v4.get_token_to_kv_pool",
        return_value=_FusedCachePool(cp1_cache),
    ):
        attn._store_raw_kv_to_cache(raw_kv, positions, forward_batch, backend)

    local_raw_kv = torch.cat((raw_kv[2::4], raw_kv.new_zeros((2, 512))), dim=0)
    local_positions = torch.tensor([17, 47, 0, 0], device="cuda")

    def gather(rows, _forward_batch):
        if rows.dtype == torch.bfloat16:
            assert rows.shape == (4, 512)
            return raw_kv
        assert rows.shape == (4, 1)
        return positions[:, None]

    with (
        patch(
            "sglang.srt.models.deepseek_v4.get_token_to_kv_pool",
            return_value=_FusedCachePool(cp_cache),
        ),
        patch(
            "sglang.srt.models.deepseek_v4.gather_dsa_prefill_cp_rows",
            side_effect=gather,
        ),
    ):
        attn._gather_exact_cp_raw_kv_to_cache(
            local_raw_kv,
            local_positions,
            forward_batch,
            backend,
        )

    torch.cuda.synchronize()
    assert cp1_cache.count_nonzero().item() > 0
    assert torch.equal(cp_cache, cp1_cache)
