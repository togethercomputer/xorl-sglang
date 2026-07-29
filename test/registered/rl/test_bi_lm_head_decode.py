import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open

import sglang as sgl
import sglang.srt.layers.logits_processor as logits_processor_module
import sglang.srt.layers.sampler as sampler_module
from sglang.srt.batch_invariant_ops import (
    bi_lm_head_full_logits,
    bi_lm_head_selected_logprob,
    bi_lm_head_selected_logprob_from_logits,
    families_v2_enabled,
    head_v2_selected_logprob,
)
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.sampler import Sampler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=460, suite="stage-b-test-1-gpu-small")

VOCAB = 151936
HIDDEN = 1024


def _hidden_weight(n, seed=0, vocab=VOCAB, hidden=HIDDEN):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(
        n, hidden, generator=g, dtype=torch.float32, device="cuda"
    ).bfloat16()
    w = torch.randn(
        vocab, hidden, generator=g, dtype=torch.float32, device="cuda"
    ).bfloat16()
    return h, w


def _make_sampler(rl_on_policy=True):
    sampler = Sampler.__new__(Sampler)
    nn.Module.__init__(sampler)
    sampler.use_nan_detection = False
    sampler.tp_sync_group = None
    sampler.rl_on_policy_target = "xorl" if rl_on_policy else None
    sampler.enable_deterministic = True
    sampler.use_log_softmax_logprob = rl_on_policy
    sampler.use_ascend_backend = False
    return sampler


def _sampling_info(n, temperature=0.7, greedy=False):
    return SamplingBatchInfo(
        temperatures=torch.full(
            (n, 1), temperature, dtype=torch.float32, device="cuda"
        ),
        top_ps=torch.ones(n, dtype=torch.float32, device="cuda"),
        top_ks=torch.full((n,), -1, dtype=torch.int32, device="cuda"),
        min_ps=torch.zeros(n, dtype=torch.float32, device="cuda"),
        is_all_greedy=greedy,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB,
        sampling_seed=torch.arange(1, n + 1, dtype=torch.int64, device="cuda"),
        device="cuda",
    )


class TestBiLmHeadDecodeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")

    def test_factorized_scoring_equals_fused_contract_bitwise(self):
        # Decode factorization (full-logits GEMM + stats over the shared
        # buffer) must be bitwise the fused contract for forced tokens.
        for n, temp in [(1, None), (7, 1.0), (64, 0.7), (333, "per_row")]:
            h, w = _hidden_weight(n, seed=n)
            ids = torch.randint(0, VOCAB, (n,), device="cuda")
            if temp == "per_row":
                temperature = 0.3 + torch.rand(n, dtype=torch.float32, device="cuda")
            elif temp is None:
                temperature = None
            else:
                temperature = torch.full((n,), temp, dtype=torch.float32, device="cuda")
            full = bi_lm_head_full_logits(h, w)
            got = bi_lm_head_selected_logprob_from_logits(
                full, ids, temperature=temperature
            )
            want = bi_lm_head_selected_logprob(h, w, ids, temperature=temperature)
            for g, e, name in zip(got, want, ("logprob", "lse", "selected")):
                self.assertTrue(
                    torch.equal(g, e), f"{name} not bitwise at n={n}, temp={temp}"
                )

    def test_full_logits_padding_row_invariant(self):
        # CUDA-graph replay pads the decode batch; padded-batch rows must be
        # bitwise the unpadded rows or graph capture breaks the contract.
        n, n_pad = 3, 16
        h, w = _hidden_weight(n_pad, seed=11)
        padded = bi_lm_head_full_logits(h, w)
        unpadded = bi_lm_head_full_logits(h[:n].clone(), w)
        self.assertTrue(torch.equal(padded[:n], unpadded))

    def test_sampler_rl_lane_rescores_through_contract(self):
        n = 5
        h, w = _hidden_weight(n, seed=21)
        logits = bi_lm_head_full_logits(h, w)
        output = LogitsProcessorOutput(next_token_logits=logits)
        sampler = _make_sampler()
        info = _sampling_info(n, temperature=0.7)
        with patch.object(sampler_module, "SGLANG_BI_LM_HEAD_DECODE", True):
            token_ids = sampler.forward(
                output,
                info,
                return_logprob=True,
                top_logprobs_nums=[0] * n,
                token_ids_logprobs=[None] * n,
                positions=torch.arange(n, device="cuda"),
            )
        want, _, _ = bi_lm_head_selected_logprob(
            h, w, token_ids.to(torch.int64), temperature=info.temperatures.reshape(-1)
        )
        self.assertTrue(torch.equal(output.next_token_logprobs, want))

    def test_sampler_greedy_rescores_through_contract(self):
        n = 4
        h, w = _hidden_weight(n, seed=31)
        logits = bi_lm_head_full_logits(h, w)
        output = LogitsProcessorOutput(next_token_logits=logits)
        sampler = _make_sampler()
        info = _sampling_info(n, temperature=1.0, greedy=True)
        with patch.object(sampler_module, "SGLANG_BI_LM_HEAD_DECODE", True):
            token_ids = sampler.forward(
                output,
                info,
                return_logprob=True,
                top_logprobs_nums=[0] * n,
                token_ids_logprobs=[None] * n,
                positions=torch.arange(n, device="cuda"),
            )
        self.assertTrue(torch.equal(token_ids, torch.argmax(logits, dim=-1)))
        # Greedy keeps stock semantics: contract rescore at T=1 (no scaling).
        want, _, _ = bi_lm_head_selected_logprob(h, w, token_ids.to(torch.int64))
        self.assertTrue(torch.equal(output.next_token_logprobs, want))

    def test_sampler_guards_raise_on_mutating_configs(self):
        n = 2
        h, w = _hidden_weight(n, seed=41, vocab=8192)
        logits = bi_lm_head_full_logits(h, w)
        sampler = _make_sampler()
        info = _sampling_info(n)
        info.logit_bias = torch.zeros(n, 8192, device="cuda")
        with self.assertRaisesRegex(ValueError, "logit bias"):
            sampler._bi_contract_sampled_logprob(
                logits, torch.zeros(n, dtype=torch.int64, device="cuda"), info
            )
        info = _sampling_info(n)
        info.need_top_p_sampling = True
        with self.assertRaisesRegex(ValueError, "on-policy"):
            sampler._bi_contract_sampled_logprob(
                logits, torch.zeros(n, dtype=torch.int64, device="cuda"), info
            )

    def test_logits_processor_decode_uses_contract_gemm(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        get_global_server_args().enable_dp_lm_head = False
        get_global_server_args().enable_fp32_lm_head = True
        cfg = SimpleNamespace(vocab_size=VOCAB, final_logit_softcapping=None)
        lp = LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)
        n = 6
        h, w = _hidden_weight(n, seed=51)
        lm_head = SimpleNamespace(weight=w)
        meta = LogitsMetadata(forward_mode=ForwardMode.DECODE)
        with patch.object(
            logits_processor_module, "SGLANG_BI_LM_HEAD", True
        ), patch.object(logits_processor_module, "SGLANG_BI_LM_HEAD_DECODE", True):
            out = lp.forward(None, h, lm_head, meta)
        self.assertTrue(
            torch.equal(out.next_token_logits, bi_lm_head_full_logits(h, w))
        )

    def test_logits_processor_decode_flag_requires_prefill_flag(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        get_global_server_args().enable_dp_lm_head = False
        get_global_server_args().enable_fp32_lm_head = True
        cfg = SimpleNamespace(vocab_size=VOCAB, final_logit_softcapping=None)
        lp = LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)
        meta = LogitsMetadata(forward_mode=ForwardMode.DECODE)
        with patch.object(
            logits_processor_module, "SGLANG_BI_LM_HEAD", False
        ), patch.object(logits_processor_module, "SGLANG_BI_LM_HEAD_DECODE", True):
            with self.assertRaisesRegex(ValueError, "SGLANG_BI_LM_HEAD=1"):
                lp._bi_lm_head_decode_active(meta)


class TestBiLmHeadDecodeEngineGate(unittest.TestCase):
    """Engine-level A/B gate in the lane that carries RL rollouts live: the
    temperature-0.7 multinomial processed lane (--rl-on-policy-target, top_p=1,
    no top-k/min-p, per-request sampling seeds, CUDA graphs on). Flag-on,
    every returned output_token_logprob must be bitwise reproducible by the
    fused contract from the recorded sampling hidden states; flag-off, the
    same lane returns legacy log_softmax values that are NOT contract-bitwise."""

    TEMP = 0.7
    MAX_NEW = 12
    PROMPTS = [
        "The integral of x squared from zero to three equals",
        "A fair coin is flipped five times; the chance of exactly two heads is",
        "To reverse a linked list in place, first",
        "The capital city of Australia is",
    ]
    _BI_ENVS = ("SGLANG_BI_LM_HEAD", "SGLANG_BI_LM_HEAD_DECODE")

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")
        shard = hf_hub_download(DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "model.safetensors")
        with safe_open(shard, framework="pt", device="cuda") as f:
            # Tied lm head; the checkpoint stores only the embedding.
            cls.lm_head_weight = f.get_tensor("model.embed_tokens.weight")
        assert cls.lm_head_weight.dtype == torch.bfloat16

    def _run_engine(self, bi_decode: bool):
        old_env = {k: os.environ.get(k) for k in self._BI_ENVS}
        for k in self._BI_ENVS:
            if bi_decode:
                os.environ[k] = "1"
            else:
                os.environ.pop(k, None)
        engine = None
        try:
            engine = sgl.Engine(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                rl_on_policy_target="xorl",
                enable_fp32_lm_head=True,
                enable_return_hidden_states=True,
                disable_piecewise_cuda_graph=True,
                cuda_graph_max_bs=8,
                mem_fraction_static=0.5,
            )
            outs = []
            for i, prompt in enumerate(self.PROMPTS):
                outs.append(
                    engine.generate(
                        prompt=prompt,
                        sampling_params={
                            "temperature": self.TEMP,
                            "top_p": 1.0,
                            "top_k": -1,
                            "max_new_tokens": self.MAX_NEW,
                            "sampling_seed": 1000 + i,
                        },
                        return_logprob=True,
                        return_hidden_states=True,
                    )
                )
            return outs
        finally:
            if engine is not None:
                engine.shutdown()
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _returned_vs_contract(self, outs):
        returned_all, contract_all = [], []
        for out in outs:
            toks = out["meta_info"]["output_token_logprobs"]
            hs = out["meta_info"]["hidden_states"]
            self.assertEqual(len(hs), len(toks))
            rows = []
            for e in hs:
                t = torch.as_tensor(e, dtype=torch.float32)
                if t.dim() == 2:
                    t = t[-1]
                rows.append(t)
            h = torch.stack(rows).to("cuda").bfloat16()
            ids = torch.tensor([t[1] for t in toks], dtype=torch.int64, device="cuda")
            temperature = torch.full(
                (len(ids),), self.TEMP, dtype=torch.float32, device="cuda"
            )
            # expectation follows the lane's active tree: families-v2 default-on
            # rescoring goes through the v2 head, kill switch pins v1.
            if families_v2_enabled():
                contract, _, _ = head_v2_selected_logprob(
                    h, self.lm_head_weight, ids, temperature=temperature
                )
            else:
                contract, _, _ = bi_lm_head_selected_logprob(
                    h, self.lm_head_weight, ids, temperature=temperature
                )
            returned_all.append(
                torch.tensor([t[0] for t in toks], dtype=torch.float32, device="cuda")
            )
            contract_all.append(contract)
        return torch.cat(returned_all), torch.cat(contract_all)

    def test_engine_temp_multinomial_processed_lane_ab(self):
        # A: flag-on — bitwise contract, token for token.
        returned, contract = self._returned_vs_contract(self._run_engine(True))
        self.assertGreater(returned.numel(), 0)
        self.assertTrue(
            torch.equal(returned, contract),
            f"flag-on: {int((returned != contract).sum())}/{returned.numel()} returned "
            "output_token_logprobs are not bitwise contract rescores",
        )

        # B: flag-off control — the same lane returns legacy log_softmax values:
        # a small numeric-convention gap from the contract, not bitwise equal.
        returned, contract = self._returned_vs_contract(self._run_engine(False))
        diff = (returned - contract).abs()
        self.assertFalse(
            torch.equal(returned, contract),
            "flag-off control unexpectedly returned bitwise contract values",
        )
        self.assertGreater(
            float(diff.max()), 0.0, "flag-off control shows no legacy gap"
        )
        self.assertLess(
            float(diff.max()),
            0.05,
            "flag-off gap is not the legacy numeric-convention gap",
        )


class TestP1TokenClamp(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")

    def test_p1_tokens_clamp_to_exact_zero_both_paths(self):
        from sglang.srt.batch_invariant_ops import (
            bi_lm_head_full_logits,
            bi_lm_head_selected_logprob,
            bi_lm_head_selected_logprob_from_logits,
        )

        torch.manual_seed(7)
        n, hidden_dim, vocab = 8192, 1024, 12800
        weight = (torch.randn(vocab, hidden_dim, device="cuda") * 0.05).to(
            torch.bfloat16
        )
        ids = torch.randint(0, vocab, (n,), device="cuda")
        scale = 1.0 + torch.rand(n, 1, device="cuda") * 30
        hidden = (weight[ids].float() * scale).to(torch.bfloat16)
        temp = torch.full((n,), 0.7, dtype=torch.float32, device="cuda")
        for t in (None, temp):
            # In exact math logprob <= 0; the one-ulp LSE boundary case (p~1
            # tokens, observed live as +2**-18) must clamp to exactly 0.0.
            lp_fused, _, _ = bi_lm_head_selected_logprob(
                hidden, weight, ids, temperature=t
            )
            self.assertTrue(bool((lp_fused <= 0).all()))
            self.assertTrue(bool((lp_fused == 0).any()))
            logits = bi_lm_head_full_logits(hidden, weight)
            lp_fact, _, _ = bi_lm_head_selected_logprob_from_logits(
                logits, ids, temperature=t
            )
            self.assertTrue(bool((lp_fact <= 0).all()))
            self.assertTrue(torch.equal(lp_fact, lp_fused))


class TestBiLogprobPrefillSkipsFp32Head(unittest.TestCase):
    """Full contract (BI prefill + BI decode) with no top-k/token-ids extras:
    the logprob-prefill fp32 head pass is dead work (both returned tensors come
    from the contract kernels) and must be skipped — values unchanged, and the
    fp32 lm-head weight cache never materializes. Any requested extra restores
    the fp32 pass."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")

    def _make_lp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        get_global_server_args().enable_dp_lm_head = False
        get_global_server_args().enable_fp32_lm_head = True
        cfg = SimpleNamespace(vocab_size=VOCAB, final_logit_softcapping=None)
        return LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)

    def _prefill_logprob_case(self, top_logprobs: bool):
        # Two seqs, extend lens [4, 5], logprob start lens [0, 2]:
        # 9 hidden rows -> 7 pruned rows, sample rows [3, 6], all 7 are
        # input-logprob rows (see _get_pruned_states docstring example).
        h, w = _hidden_weight(9, seed=67)
        g = torch.Generator(device="cuda").manual_seed(67)
        ids = torch.randint(0, VOCAB, (7,), generator=g, device="cuda")
        meta = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            extend_return_logprob=True,
            extend_return_top_logprob=top_logprobs,
            extend_token_ids_logprob=False,
            extend_seq_lens=torch.tensor([4, 5], device="cuda"),
            extend_seq_lens_cpu=[4, 5],
            extend_logprob_start_lens_cpu=[0, 2],
            extend_logprob_pruned_lens_cpu=[4, 3],
            top_logprobs_nums=[2, 2] if top_logprobs else [0, 0],
            extend_input_logprob_token_ids_gpu=ids,
            temp_scaled_logprobs=True,
            temperature=torch.full((2, 1), 0.7, dtype=torch.float32, device="cuda"),
        )
        pruned = torch.cat([h[0:4], h[6:9]])
        return h, w, ids, meta, pruned

    def _forward_counting_fp32_head(self, lp, h, w, meta):
        lm_head = SimpleNamespace(weight=w)
        orig = LogitsProcessor._get_logits
        calls = []

        def counting(self_lp, *args, **kwargs):
            calls.append(1)
            return orig(self_lp, *args, **kwargs)

        with patch.object(LogitsProcessor, "_get_logits", counting), patch.object(
            logits_processor_module, "SGLANG_BI_LM_HEAD", True
        ), patch.object(logits_processor_module, "SGLANG_BI_LM_HEAD_DECODE", True):
            out = lp.forward(None, h, lm_head, meta)
        return out, calls

    def test_full_contract_skips_fp32_head_values_unchanged(self):
        lp = self._make_lp()
        h, w, ids, meta, pruned = self._prefill_logprob_case(top_logprobs=False)
        out, calls = self._forward_counting_fp32_head(lp, h, w, meta)

        self.assertEqual(len(calls), 0, "fp32 head pass must be skipped")
        self.assertIsNone(lp._fp32_lm_head_weight_cache)
        temp = torch.full((7,), 0.7, dtype=torch.float32, device="cuda")
        want_logprobs, _, _ = bi_lm_head_selected_logprob(
            pruned.contiguous(), w, ids, temperature=temp
        )
        self.assertTrue(torch.equal(out.input_token_logprobs, want_logprobs))
        self.assertTrue(
            torch.equal(
                out.next_token_logits, bi_lm_head_full_logits(pruned[[3, 6]], w)
            )
        )
        self.assertIsNone(out.input_top_logprobs_val)
        self.assertIsNone(out.input_token_ids_logprobs_val)

    def test_top_logprob_extras_keep_fp32_head(self):
        lp = self._make_lp()
        h, w, ids, meta, pruned = self._prefill_logprob_case(top_logprobs=True)
        out, calls = self._forward_counting_fp32_head(lp, h, w, meta)

        self.assertGreater(len(calls), 0, "extras still need the fp32 head pass")
        temp = torch.full((7,), 0.7, dtype=torch.float32, device="cuda")
        want_logprobs, _, _ = bi_lm_head_selected_logprob(
            pruned.contiguous(), w, ids, temperature=temp
        )
        self.assertTrue(torch.equal(out.input_token_logprobs, want_logprobs))
        self.assertIsNotNone(out.input_top_logprobs_val)


if __name__ == "__main__":
    unittest.main(verbosity=2)
