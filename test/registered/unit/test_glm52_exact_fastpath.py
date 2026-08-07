"""Conventional tests for the GLM-5.2 exact serving path."""

import asyncio
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.attention.nsa.glm52_selector_fast as fast_module
import torch
from sglang.srt.batch_invariant_ops import bi_gemm_configs
from sglang.srt.distributed.canonical_moe import CanonicalDeferredStatusBook
from sglang.srt.layers.attention.nsa.glm52_selector import (
    select_canonical_logical_topk,
)
from sglang.srt.layers.attention.nsa.glm52_selector_fast import (
    select_canonical_logical_topk_fused,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestGlm52ExactFastpath(unittest.TestCase):
    def test_private_resolver_installs_one_tuple_once(self):
        from sglang.srt.layers import xorl_batch_invariant

        with (
            patch.object(fast_module, "_applied", False),
            patch.object(xorl_batch_invariant, "force_xorl_bi_family") as family,
            patch.object(
                bi_gemm_configs, "_force_bi_gemm_config_table"
            ) as config_table,
            patch.object(bi_gemm_configs, "_set_glm52_tier_a_enabled") as tier_a,
            patch.object(fast_module.logger, "info") as startup,
        ):
            fast_module._apply_glm52_exact_fastpath()
            fast_module._apply_glm52_exact_fastpath()

        family.assert_called_once_with("v2")
        config_table.assert_called_once_with(True)
        tier_a.assert_called_once_with(True)
        startup.assert_called_once()

    def test_public_module_has_no_partial_selection_api(self):
        self.assertFalse(hasattr(fast_module, "engage_glm52_exact_fastpath"))
        self.assertFalse(hasattr(fast_module, "glm52_exact_fastpath_engaged"))

    def test_unsupported_decode_backend_raises(self):
        args = ServerArgs(model_path="dummy")
        args.rl_on_policy_target = "xorl"
        args.dsa_decode_backend = "fa3"
        config = SimpleNamespace(
            hidden_size=6144,
            num_hidden_layers=78,
            vocab_size=154880,
            n_routed_experts=256,
            n_shared_experts=1,
            num_experts_per_tok=8,
            index_topk=2048,
            index_topk_freq=4,
        )
        with self.assertRaisesRegex(ValueError, "flashmla_sparse decode"):
            args._resolve_glm52_exact_contract(
                config,
                model_arch="GlmMoeDsaForCausalLM",
                is_dsa_model=True,
            )

    def test_tier_a_tables_answer_only_when_selected(self):
        with patch.object(bi_gemm_configs, "_GLM52_TIER_A", False):
            self.assertIsNone(
                bi_gemm_configs.glm52_tier_a_mm_config(
                    torch.bfloat16, M=16, K=6144, N=256
                )
            )
        with patch.object(bi_gemm_configs, "_GLM52_TIER_A", True):
            cfg = bi_gemm_configs.glm52_tier_a_mm_config(
                torch.bfloat16, M=16, K=6144, N=256
            )
            self.assertIsNotNone(cfg)
            self.assertIn("BLOCK_SIZE_K", cfg)

    def test_deferred_status_book_names_every_failing_layer(self):
        book = CanonicalDeferredStatusBook(4)
        ok = torch.zeros((), dtype=torch.int32)
        bad = torch.ones((), dtype=torch.int32)
        book.record(0, ok)
        book.record(2, bad)
        book.record(3, bad)
        with self.assertRaisesRegex(RuntimeError, "layer 2.*layer 3"):
            book.check_and_clear()
        book.record(1, ok)
        self.assertIsNotNone(book.check_and_clear())
        with self.assertRaisesRegex(ValueError, "outside"):
            book.record(9, ok)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_selector_matches_reference_ordering_semantics(self):
        gen = torch.Generator().manual_seed(20260804)
        scores = (torch.randn(8, 512, generator=gen) * 4).round().float().cuda() / 4
        lengths = torch.tensor([0, 1, 7, 64, 200, 511, 512, 512], device="cuda")
        starts = torch.tensor([0, 0, 3, 0, 100, 0, 0, 1], device="cuda")
        starts = torch.minimum(starts, 512 - lengths)
        fused, flags = select_canonical_logical_topk_fused(
            scores, lengths, 128, row_starts=starts
        )
        reference = select_canonical_logical_topk(
            scores, lengths, 128, row_starts=starts, validate=False
        )
        self.assertEqual(int(flags.max().item()), 0)
        self.assertTrue(torch.equal(fused, reference))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_exact_request_completion_path_end_to_end(self):
        from sglang.srt.layers import xorl_batch_invariant
        from sglang.srt.layers.xorl_batch_invariant import (
            xorl_bi_lm_head,
            xorl_bi_sample_and_score,
        )

        torch.manual_seed(20260805)
        n, hidden, vocab = 2, 128, 512
        hidden_states = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
        with patch.object(xorl_batch_invariant, "_FORCED_FAMILY", "v2"):
            logits = xorl_bi_lm_head(
                hidden_states,
                SimpleNamespace(weight=weight),
                use_fp32_lm_head=False,
            )
            sampling_info = SimpleNamespace(
                temperatures=torch.ones((n, 1), dtype=torch.float32, device="cuda"),
                sampling_seed=torch.arange(1, n + 1, dtype=torch.int64, device="cuda"),
                is_all_greedy=False,
                need_top_p_sampling=False,
                need_top_k_sampling=False,
                need_min_p_sampling=False,
                has_custom_logit_processor=False,
                acc_linear_penalties=None,
                penalizer_orchestrator=None,
                vocab_mask=None,
                logit_bias=None,
            )
            output = SimpleNamespace(next_token_logits=logits, next_token_logprobs=None)
            sampled = torch.tensor([3, 7], dtype=torch.int64, device="cuda")
            ids = xorl_bi_sample_and_score(
                output,
                sampling_info,
                return_logprob=True,
                top_logprobs_nums=[0] * n,
                token_ids_logprobs=[None] * n,
                positions=torch.arange(n, device="cuda"),
                sample_from_logprobs=lambda logprobs, info, ids_out: sampled,
                sync_token_ids=lambda ids_in, info: None,
                enable_deterministic=True,
                return_original_logprob=False,
            )

        self.assertTrue(torch.equal(ids, sampled))
        self.assertEqual(output.next_token_logprobs.shape, (n,))
        self.assertTrue(torch.isfinite(output.next_token_logprobs).all())


class TestGlm52ActivationBoundary(unittest.TestCase):
    """Ordinary (non-XORL) GLM serving stays on the stock MoE paths.

    The canonical-MoE contract is keyed on the resolved exact-mode bit, not on
    the model looking like GLM: a plain GLM server keeps the stock router,
    stock combine, and its A2A/DeepEP backends. The architecture bit
    (`is_glm52`) stays separate — it selects the correction-bias dtype.
    """

    def _gate(self, exact: bool):
        from sglang.srt.models.deepseek_v2 import MoEGate

        config = SimpleNamespace(
            indexer_types=("full",),
            n_routed_experts=8,
            hidden_size=16,
            topk_method="noaux_tc",
        )
        if exact:
            config._glm52_exact_mode = True
        with patch(
            "sglang.srt.models.deepseek_v2.is_dsa_enable_prefill_cp",
            return_value=False,
        ):
            return MoEGate(config, quant_config=None)

    def test_plain_glm_model_keeps_stock_router(self):
        gate = self._gate(exact=False)
        self.assertTrue(gate.is_glm52)
        self.assertFalse(gate._glm52_canonical_contract)
        self.assertFalse(gate._glm52_exact_router)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)

    def test_exact_mode_engages_canonical_router(self):
        gate = self._gate(exact=True)
        self.assertTrue(gate.is_glm52)
        self.assertTrue(gate._glm52_canonical_contract)
        self.assertTrue(gate._glm52_exact_router)

    def test_transport_resolution_is_per_mode(self):
        from sglang.srt.models.deepseek_v2 import _resolve_glm52_canonical_transport

        plain = SimpleNamespace(indexer_types=("full",))
        self.assertEqual(_resolve_glm52_canonical_transport(plain), "dense_v1")
        exact = SimpleNamespace(indexer_types=("full",), _glm52_exact_mode=True)
        self.assertEqual(_resolve_glm52_canonical_transport(exact), "canonical_v3b")


class TestTokenizerManagerAbortCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_bad_request_abort_removes_request_state(self):
        manager = object.__new__(TokenizerManager)
        manager.rid_to_state = {}
        manager.server_args = SimpleNamespace(enable_lora=False)
        manager.model_config = SimpleNamespace(is_multimodal_gen=False)
        manager.request_logger = SimpleNamespace(
            log_finished_request=lambda *args, **kwargs: None
        )
        manager.request_metrics_exporter_manager = SimpleNamespace(
            exporter_enabled=lambda: False
        )

        obj = SimpleNamespace(rid="bad-request", stream=False, lora_path=None)
        event = asyncio.Event()
        state = SimpleNamespace(
            obj=obj,
            event=event,
            finished=True,
            out_list=[
                {
                    "meta_info": {
                        "finish_reason": {
                            "type": "abort",
                            "status_code": HTTPStatus.BAD_REQUEST,
                            "message": "invalid exact request",
                        }
                    }
                }
            ],
            time_stats=SimpleNamespace(
                response_sent_to_client_time=1.0,
            ),
        )
        manager.rid_to_state[obj.rid] = state

        response = manager._wait_one_response(obj, state)
        pending = asyncio.create_task(response.__anext__())
        await asyncio.sleep(0)
        # The current scheduler-response owner removes terminal request state
        # before it wakes the tokenizer-side waiter.
        manager.rid_to_state.pop(obj.rid)
        event.set()
        with self.assertRaisesRegex(ValueError, "invalid exact request"):
            await pending

        self.assertNotIn(obj.rid, manager.rid_to_state)


if __name__ == "__main__":
    unittest.main()
