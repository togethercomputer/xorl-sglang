"""Unit gate for the K3 GDN prefill contract (SGLANG_BI_GDN_PREFILL).

Checks that ``bi_chunk_gated_delta_rule_prefill`` (the trainer-composition
chunked scan) is deterministic, batch-invariant, chains bitwise through the
fp32 state pool, sources fp32 chunk-boundary checkpoints bitwise, stays close
to the default fused path, and fails loudly on unverified pools. Also covers
the GemmaRMSNorm batch-invariant arm and the vendored solve_tril num_warps pin
(the tl.sum forward substitution reassociates with warp count).

The cross-engine bitwise gate against the xorl trainer lives on the trainer side.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.batch_invariant_ops.batch_invariant_ops import rms_norm_batch_invariant
from sglang.srt.layers.attention.fla import bi_gdn_prefill
from sglang.srt.layers.attention.fla.bi_gdn_decode import BIGDNDecodeCache
from sglang.srt.layers.attention.fla.bi_gdn_prefill import (
    bi_chunk_gated_delta_rule_prefill,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")

HK, HV, DK, DV = 16, 32, 128, 128
CHUNK = 64


def make_inputs(T, seed=0, device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.bfloat16)

    q, k = rnd(1, T, HK, DK), rnd(1, T, HK, DK)
    v = rnd(1, T, HV, DV)
    A_log = (
        torch.empty(HV, device=device, dtype=torch.float32)
        .uniform_(0, 2, generator=gen)
        .log()
    )
    dt_bias = torch.rand(HV, device=device, dtype=torch.float32, generator=gen)
    g = -A_log.exp().view(1, 1, -1) * F.softplus(
        rnd(1, T, HV).float() + dt_bias.view(1, 1, -1)
    )
    beta = rnd(1, T, HV).float().sigmoid().to(torch.bfloat16).float()
    return q, k, v, g, beta


def run_bi(q, k, v, g, beta, pool, cache_indices, cu_seqlens, **ckpt):
    return bi_chunk_gated_delta_rule_prefill(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        ssm_states=pool,
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
        **ckpt,
    )


class TestBIGDNPrefill(CustomTestCase):
    def test_close_to_default_fused_path(self):
        from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

        T = 1024
        q, k, v, g, beta = make_inputs(T)
        pool = torch.zeros(2, HV, DV, DK, device="cuda", dtype=torch.float32)
        idx = torch.tensor([0], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
        o_bi = run_bi(q, k, v, g, beta, pool, idx, cu)

        pool2 = torch.zeros(2, HV, DV, DK, device="cuda", dtype=torch.float32)
        o_ref, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool2,
            initial_state_indices=torch.arange(1, device="cuda"),
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )
        self.assertEqual(o_bi.shape, o_ref.shape)
        # same math, different kernel lineage: agree up to the fused path's
        # known bf16 tail
        self.assertLess((o_bi.float() - o_ref.float()).abs().max().item(), 2e-3)
        self.assertLess((pool.float() - pool2.float()).abs().max().item(), 2e-2)

    def test_deterministic(self):
        T = 512
        q, k, v, g, beta = make_inputs(T)
        outs, pools = [], []
        for _ in range(2):
            pool = torch.zeros(1, HV, DV, DK, device="cuda", dtype=torch.float32)
            idx = torch.tensor([0], device="cuda", dtype=torch.int32)
            cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
            outs.append(run_bi(q, k, v, g, beta, pool, idx, cu))
            pools.append(pool.clone())
        self.assertTrue(torch.equal(outs[0], outs[1]))
        self.assertTrue(torch.equal(pools[0], pools[1]))

    def test_varlen_batch_invariance(self):
        lens = [193, 64, 33, 400]
        total = sum(lens)
        q, k, v, g, beta = make_inputs(total)
        cu = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        pool = torch.zeros(len(lens), HV, DV, DK, device="cuda", dtype=torch.float32)
        idx = torch.arange(len(lens), device="cuda", dtype=torch.int32)
        o_pack = run_bi(q, k, v, g, beta, pool, idx, cu)

        start = 0
        for i, n in enumerate(lens):
            pool1 = torch.zeros(1, HV, DV, DK, device="cuda", dtype=torch.float32)
            idx1 = torch.tensor([0], device="cuda", dtype=torch.int32)
            cu1 = torch.tensor([0, n], device="cuda", dtype=torch.int32)
            sl = slice(start, start + n)
            o1 = run_bi(
                q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], pool1, idx1, cu1
            )
            self.assertTrue(torch.equal(o_pack[:, sl], o1), f"request {i} output")
            self.assertTrue(torch.equal(pool[i : i + 1], pool1), f"request {i} state")
            start += n

    def test_fp32_state_chaining(self):
        T = 256
        q, k, v, g, beta = make_inputs(T)
        pool_full = torch.zeros(1, HV, DV, DK, device="cuda", dtype=torch.float32)
        idx = torch.tensor([0], device="cuda", dtype=torch.int32)
        o_full = run_bi(
            q,
            k,
            v,
            g,
            beta,
            pool_full,
            idx,
            torch.tensor([0, T], device="cuda", dtype=torch.int32),
        )

        pool = torch.zeros(1, HV, DV, DK, device="cuda", dtype=torch.float32)
        halves = []
        for t0 in (0, 128):
            sl = slice(t0, t0 + 128)
            halves.append(
                run_bi(
                    q[:, sl],
                    k[:, sl],
                    v[:, sl],
                    g[:, sl],
                    beta[:, sl],
                    pool,
                    idx,
                    torch.tensor([0, 128], device="cuda", dtype=torch.int32),
                )
            )
        self.assertTrue(torch.equal(o_full, torch.cat(halves, dim=1)))
        self.assertTrue(torch.equal(pool_full, pool))

    def test_fp32_checkpoint_sourcing(self):
        # checkpoint slot must hold the fp32 state of the chunk-aligned prefix
        T = 193
        aligned = (T // CHUNK) * CHUNK
        q, k, v, g, beta = make_inputs(T)
        pool = torch.zeros(4, HV, DV, DK, device="cuda", dtype=torch.float32)
        idx = torch.tensor([0], device="cuda", dtype=torch.int32)
        run_bi(
            q,
            k,
            v,
            g,
            beta,
            pool,
            idx,
            torch.tensor([0, T], device="cuda", dtype=torch.int32),
            ckpt_batch_rows=torch.tensor([0], device="cuda", dtype=torch.long),
            ckpt_token_starts=[0],
            ckpt_lens=[aligned],
            ckpt_dst_indices=torch.tensor([2], device="cuda", dtype=torch.long),
        )
        pool_ref = torch.zeros(1, HV, DV, DK, device="cuda", dtype=torch.float32)
        run_bi(
            q[:, :aligned],
            k[:, :aligned],
            v[:, :aligned],
            g[:, :aligned],
            beta[:, :aligned],
            pool_ref,
            idx,
            torch.tensor([0, aligned], device="cuda", dtype=torch.int32),
        )
        self.assertTrue(torch.equal(pool[2:3], pool_ref))

    def test_zero_length_checkpoint_copies_prescan_state(self):
        T = 33  # < CHUNK: checkpoint == pre-scan state
        q, k, v, g, beta = make_inputs(T)
        pool = torch.zeros(4, HV, DV, DK, device="cuda", dtype=torch.float32)
        pool[0].normal_(generator=torch.Generator(device="cuda").manual_seed(7))
        prescan = pool[0].clone()
        idx = torch.tensor([0], device="cuda", dtype=torch.int32)
        run_bi(
            q,
            k,
            v,
            g,
            beta,
            pool,
            idx,
            torch.tensor([0, T], device="cuda", dtype=torch.int32),
            ckpt_batch_rows=torch.tensor([0], device="cuda", dtype=torch.long),
            ckpt_token_starts=[0],
            ckpt_lens=[0],
            ckpt_dst_indices=torch.tensor([3], device="cuda", dtype=torch.long),
        )
        self.assertTrue(torch.equal(pool[3], prescan))
        self.assertFalse(torch.equal(pool[0], prescan))  # working slot advanced

    def test_rejects_non_fp32_pool(self):
        T = 64
        q, k, v, g, beta = make_inputs(T)
        pool = torch.zeros(1, HV, DV, DK, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "fp32 SSM state pool"):
            run_bi(
                q,
                k,
                v,
                g,
                beta,
                pool,
                torch.tensor([0], device="cuda", dtype=torch.int32),
                torch.tensor([0, T], device="cuda", dtype=torch.int32),
            )

    def test_solve_tril_num_warps_pinned(self):
        # the tl.sum forward substitution reassociates with num_warps; the
        # contract pins the trainer's 2-warp variant
        for kernel in (
            bi_gdn_prefill.solve_tril_16x16_kernel,
            bi_gdn_prefill.merge_16x16_to_32x32_inverse_kernel,
            bi_gdn_prefill.merge_16x16_to_64x64_inverse_kernel,
        ):
            for cfg in kernel.fn.configs:
                self.assertEqual(cfg.num_warps, 2)


class TestBIGDNDecodeRescan(CustomTestCase):
    def test_partial_chunk_rescan_matches_full_prefill_across_boundary(self):
        prompt_len, total = 45, 70
        q, k, v, g, beta = make_inputs(total, seed=17)
        idx = torch.tensor([0], device="cuda", dtype=torch.int32)

        full_pool = torch.zeros(1, HV, DV, DK, device="cuda", dtype=torch.float32)
        full = run_bi(
            q,
            k,
            v,
            g,
            beta,
            full_pool,
            idx,
            torch.tensor([0, total], device="cuda", dtype=torch.int32),
        )

        live_pool = torch.zeros_like(full_pool)
        run_bi(
            q[:, :prompt_len],
            k[:, :prompt_len],
            v[:, :prompt_len],
            g[:, :prompt_len],
            beta[:, :prompt_len],
            live_pool,
            idx,
            torch.tensor([0, prompt_len], device="cuda", dtype=torch.int32),
        )
        packed = torch.cat(
            (
                q.reshape(total, -1),
                k.reshape(total, -1),
                v.reshape(total, -1),
            ),
            dim=-1,
        )
        cache = BIGDNDecodeCache(
            num_slots=1,
            qkv_dim=packed.shape[-1],
            num_v_heads=HV,
            head_k_dim=DK,
            head_v_dim=DV,
            device=packed.device,
        )
        cache.seed_from_extend(
            slot=0,
            pre_scan_state=torch.zeros_like(live_pool[0]),
            qkv_rows=packed[:prompt_len],
            g_rows=g.reshape(total, HV)[:prompt_len],
            beta_rows=beta.reshape(total, HV)[:prompt_len],
            prefix_len=0,
            ssm_states=live_pool,
        )

        decoded = []
        slot_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
        for pos in range(prompt_len, total):
            metadata = cache.prepare_step_metadata(
                [0],
                slot_indices,
                torch.tensor([pos + 1], device="cuda", dtype=torch.int32),
            )
            decoded.append(
                cache.step(
                    metadata=metadata,
                    qkv_rows=packed[pos : pos + 1],
                    g_rows=g.reshape(total, HV)[pos : pos + 1],
                    beta_rows=beta.reshape(total, HV)[pos : pos + 1],
                    ssm_states=live_pool,
                )
            )

        self.assertTrue(torch.equal(torch.cat(decoded, dim=0), full[0, prompt_len:]))
        self.assertTrue(torch.equal(live_pool, full_pool))


class TestFwdOConfigPin(CustomTestCase):
    """chunk_fwd_kernel_o BK/BV/num_warps are bit-relevant and flip across
    triton 3.5->3.7 under stock autotune; the BK128/BV128/w4 pin reproduces the
    triton-3.5.1 anchor bits on both. Golden frozen on H100 under
    torch 2.9.1/triton 3.5.1 AND torch 2.12.1/triton 3.7.1 (bitwise equal)."""

    GOLDEN_O = "d1398cef77be3272b3176dce4f5fc54b27ef8bcb521e91de99c0e966dde73d71"

    def test_fwd_o_default_config_is_pinned(self):
        import os

        if os.environ.get("SGLANG_BI_FWD_O_AUTOTUNE", "0") == "1":
            self.skipTest("autotune escape hatch enabled")
        configs = bi_gdn_prefill.chunk_fwd_kernel_o.fn.configs
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].kwargs, {"BK": 128, "BV": 128})
        self.assertEqual(configs[0].num_warps, 4)

    def test_fwd_o_matches_frozen_golden(self):
        import hashlib

        gen = torch.Generator(device="cpu").manual_seed(20260709)
        B, T, H, DKq, DVv, NT = 1, 512, 4, 128, 128, 8
        q = F.normalize(torch.randn(B, T, H, DKq, generator=gen), p=2, dim=-1).to(
            torch.bfloat16
        )
        k = F.normalize(torch.randn(B, T, H, DKq, generator=gen), p=2, dim=-1).to(
            torch.bfloat16
        )
        v = (torch.randn(B, T, H, DVv, generator=gen) * 0.5).to(torch.bfloat16)
        h = (torch.randn(B, NT, H, DKq, DVv, generator=gen) * 0.2).to(torch.bfloat16)
        g = F.logsigmoid(torch.randn(B, T, H, generator=gen) * 2.0)
        cu = torch.tensor([0, 320, 512], dtype=torch.long)
        o = bi_gdn_prefill.chunk_fwd_o(
            q=q.cuda(),
            k=k.cuda(),
            v=v.cuda(),
            h=h.cuda(),
            g=g.cuda(),
            scale=DKq**-0.5,
            cu_seqlens=cu.cuda(),
        )
        got = hashlib.sha256(
            o.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        self.assertEqual(got, self.GOLDEN_O)


class TestGemmaRMSNormBatchInvariant(CustomTestCase):
    def _norm_and_input(self, rows=512, hidden=1024, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(
            rows, hidden, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        norm = GemmaRMSNorm(hidden, eps=1e-6).to("cuda")
        with torch.no_grad():
            norm.weight.copy_(torch.randn(hidden, generator=gen, device="cuda") * 0.1)
        return norm, x

    def test_bi_arm_matches_zero_centered_composition(self):
        norm, x = self._norm_and_input()
        with set_batch_invariant_mode(True):
            got = norm.forward_cuda(x)
        exp = rms_norm_batch_invariant(
            x.float(), 1.0 + norm.weight.data.float(), norm.variance_epsilon
        ).to(x.dtype)
        self.assertTrue(torch.equal(got, exp))
        # and stays close to the native reference
        ref = norm.forward_native(x)
        self.assertLess((got.float() - ref.float()).abs().max().item(), 5e-2)

    def test_bi_arm_batch_invariant(self):
        norm, x = self._norm_and_input()
        with set_batch_invariant_mode(True):
            full = norm.forward_cuda(x)
            sub = norm.forward_cuda(x[:3].contiguous())
        self.assertTrue(torch.equal(full[:3], sub))

    def test_residual_falls_back_to_native(self):
        # residual configs are not contracted; they ride forward_native (whose
        # torch ops the BI interpose already covers), same as RMSNorm's arm
        norm, x = self._norm_and_input()
        residual = torch.randn_like(x)
        with set_batch_invariant_mode(True):
            got, res = norm.forward_cuda(x.clone(), residual.clone())
            exp, res_exp = norm.forward_native(x.clone(), residual.clone())
        self.assertTrue(torch.equal(got, exp))
        self.assertTrue(torch.equal(res, res_exp))


if __name__ == "__main__":
    unittest.main()
