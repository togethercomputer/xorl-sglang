"""SM100 hd256 2CTA FA4: seqused_k (paged KV) correctness.

The dedicated Blackwell head_dim=256 kernel historically asserted seqused_q/k
away, which made Qwen3.5-family exact serving (head_dim=256, FA4 required by
the contract) unbootable on B200: paged decode/prefill carries per-sequence KV
lengths as seqused_k. The kernel now derives per-batch seqlen_k from seqused_k
and feeds the same trip-count + last-tile masking machinery the varlen path
uses. This test pins:

- decode shape (q_len=1, mixed partial/page-aligned K lengths) vs an fp32
  torch reference,
- varlen causal prefill (bottom-right aligned) vs the same reference,
- bitwise identity with the legacy full-table paged path when
  seqused_k == table capacity (no regression on the pre-existing mode).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-b200")

H, HKV, D = 16, 4, 256
PAGE = 128


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def _make_paged_kv(batch, max_pages, device):
    num_pages = batch * max_pages + 3
    kpool = torch.randn(num_pages, PAGE, HKV, D, device=device, dtype=torch.bfloat16)
    vpool = torch.randn(num_pages, PAGE, HKV, D, device=device, dtype=torch.bfloat16)
    perm = torch.randperm(num_pages, device=device)[: batch * max_pages]
    page_table = perm.view(batch, max_pages).to(torch.int32).contiguous()
    return kpool, vpool, page_table


def _gather(pool, page_table, b, length):
    pages = page_table[b, : (length + PAGE - 1) // PAGE].long()
    return pool[pages].reshape(-1, HKV, D)[:length]


def _ref_attn(q, k, v, causal):
    lq, lk = q.shape[0], k.shape[0]
    rep = H // HKV
    qf = q.float().permute(1, 0, 2)
    kf = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    s = torch.einsum("hqd,hkd->hqk", qf, kf) / (D**0.5)
    if causal:
        iq = torch.arange(lq, device=q.device).view(1, -1, 1)
        ik = torch.arange(lk, device=q.device).view(1, 1, -1)
        s = s.masked_fill(ik > (lk - lq) + iq, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,hkd->qhd", p, vf)


@unittest.skipUnless(_is_sm100(), "dedicated hd256 2CTA kernel is SM100-only")
class TestFa4Hd256SeqUsedK(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"

    def _run(self, qlens, klens, causal):
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            flash_attn_varlen_func,
        )

        b = len(qlens)
        max_pages = (max(klens) + PAGE - 1) // PAGE
        max_seqlen_k = max_pages * PAGE
        kpool, vpool, page_table = _make_paged_kv(b, max_pages, self.device)
        q = torch.randn(sum(qlens), H, D, device=self.device, dtype=torch.bfloat16)
        cu_q = torch.tensor(
            [0] + list(torch.tensor(qlens).cumsum(0)),
            device=self.device,
            dtype=torch.int32,
        )
        out = flash_attn_varlen_func(
            q,
            kpool,
            vpool,
            cu_seqlens_q=cu_q,
            seqused_k=torch.tensor(klens, device=self.device, dtype=torch.int32),
            max_seqlen_q=max(qlens),
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            causal=causal,
        )
        if isinstance(out, tuple):
            out = out[0]
        for i in range(b):
            qi = q[cu_q[i] : cu_q[i + 1]]
            ref = _ref_attn(
                qi,
                _gather(kpool, page_table, i, klens[i]),
                _gather(vpool, page_table, i, klens[i]),
                causal,
            )
            got = out[cu_q[i] : cu_q[i + 1]].float()
            rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1.0)
            self.assertLess(rel, 2.5e-2, f"seq {i}: rel err {rel:.3e}")

    def test_decode_mixed_lengths(self):
        self._run([1, 1, 1, 1], [37, 128, 291, 997], causal=False)

    def test_decode_graph_sized_batch(self):
        self._run([1] * 16, [((i * 613) % 1900) + 8 for i in range(16)], causal=False)

    def test_prefill_causal_bottom_right(self):
        self._run([5, 130, 64], [69, 130, 640], causal=True)

    def test_full_length_seqused_bitwise_matches_legacy_path(self):
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            flash_attn_varlen_func,
        )

        b, max_pages = 2, 4
        max_seqlen_k = max_pages * PAGE
        kpool, vpool, page_table = _make_paged_kv(b, max_pages, self.device)
        q = torch.randn(b, H, D, device=self.device, dtype=torch.bfloat16)
        cu_q = torch.tensor([0, 1, 2], device=self.device, dtype=torch.int32)
        common = dict(
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            causal=False,
        )
        legacy = flash_attn_varlen_func(q, kpool, vpool, **common)
        seqused = flash_attn_varlen_func(
            q,
            kpool,
            vpool,
            seqused_k=torch.tensor(
                [max_seqlen_k] * b, device=self.device, dtype=torch.int32
            ),
            **common,
        )
        a = legacy[0] if isinstance(legacy, tuple) else legacy
        c = seqused[0] if isinstance(seqused, tuple) else seqused
        self.assertTrue(
            torch.equal(a, c),
            "full-length seqused_k must be bitwise identical to the legacy path",
        )


if __name__ == "__main__":
    unittest.main()
