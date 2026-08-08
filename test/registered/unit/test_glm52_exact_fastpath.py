"""Conventional tests for the GLM-5.2 exact serving path."""

import asyncio
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.attention.nsa.glm52_selector_fast as fast_module
import sglang.srt.server_args as server_args_module
import torch
from sglang.srt.batch_invariant_ops import bi_gemm_configs
from sglang.srt.distributed.canonical_moe import CanonicalDeferredStatusBook
from sglang.srt.layers.attention.nsa.glm52_selector import (
    select_canonical_logical_topk,
)
from sglang.srt.layers.attention.nsa.glm52_selector_fast import (
    select_canonical_logical_topk_fused,
    select_canonical_paged_topk_fused,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-large")


def _glm52_config():
    return SimpleNamespace(
        hidden_size=6144,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        moe_layer_freq=1,
        num_hidden_layers=78,
        vocab_size=154880,
        num_attention_heads=64,
        num_key_value_heads=64,
        first_k_dense_replace=3,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
        rms_norm_eps=1e-5,
        rope_parameters={
            "rope_theta": 8_000_000,
            "rope_type": "default",
            "type": "default",
        },
        max_position_embeddings=1_048_576,
        hidden_act="silu",
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_head_dim=128,
        index_n_heads=32,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        indexer_rope_interleave=True,
        rope_interleave=True,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        topk_method="noaux_tc",
        tie_word_embeddings=False,
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": sorted(
                server_args_module._glm52_expected_fp8_unquantized_modules()
            ),
        },
        indexer_types=[
            "full" if layer < 3 or layer % 4 == 2 else "shared" for layer in range(78)
        ],
    )


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
        args.nnodes = 2
        args.dsa_decode_backend = "fa3"
        config = _glm52_config()
        with self.assertRaisesRegex(ValueError, "flashmla_sparse decode"):
            args._resolve_glm52_exact_contract(
                config,
                model_arch="GlmMoeDsaForCausalLM",
                is_dsa_model=True,
            )

    def test_unsupported_topk_owner_raises(self):
        config = _glm52_config()
        for backend in ("torch", "flashinfer"):
            args = ServerArgs(model_path="dummy")
            args.rl_on_policy_target = "xorl"
            args.nnodes = 2
            args.dsa_topk_backend = backend
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValueError, "dsa_topk_backend"),
            ):
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
    def test_exact_paged_transform_preserves_canonical_logical_order(self):
        generator = torch.Generator().manual_seed(520052)
        rows, width, topk, page_size = 256, 4096, 2048, 64
        backing = torch.randint(
            -32,
            33,
            (rows, width + 16),
            generator=generator,
            dtype=torch.int32,
        ).to(device="cuda", dtype=torch.float32)
        scores = backing[:, :width]
        self.assertEqual(scores.stride(), (width + 16, 1))
        starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
        lengths = torch.arange(3841, 4097, device="cuda", dtype=torch.int32)
        cu_seqlens_q = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
        page_table = torch.stack(
            (
                torch.randperm(width // page_size, generator=generator),
                torch.randperm(width // page_size, generator=generator)
                + width // page_size,
            )
        ).to(device="cuda", dtype=torch.int32)

        logical = select_canonical_logical_topk(
            scores, lengths, topk, row_starts=starts, validate=False
        )
        row_to_batch = torch.repeat_interleave(
            torch.arange(2, device="cuda"),
            torch.tensor([128, 128], device="cuda"),
        )
        row_pages = page_table.index_select(0, row_to_batch)
        valid = logical >= 0
        safe = logical.clamp_min(0)
        expected_pages = torch.gather(
            row_pages, 1, torch.div(safe, page_size, rounding_mode="floor").long()
        )
        expected = torch.where(
            valid,
            expected_pages * page_size + torch.remainder(safe, page_size),
            -1,
        ).to(torch.int32)

        outputs = []
        for _ in range(4):
            transformed, flags = select_canonical_paged_topk_fused(
                scores,
                lengths,
                topk,
                page_table=page_table,
                page_size=page_size,
                cu_seqlens_q=cu_seqlens_q,
                row_starts=starts,
            )
            self.assertEqual(int(flags.max().item()), 0)
            self.assertTrue(torch.equal(transformed, expected))
            outputs.append(transformed.clone())

        self.assertTrue(all(torch.equal(outputs[0], output) for output in outputs[1:]))

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_flags = select_canonical_paged_topk_fused(
                scores,
                lengths,
                topk,
                page_table=page_table,
                page_size=page_size,
                cu_seqlens_q=cu_seqlens_q,
                row_starts=starts,
            )
        captured_outputs = []
        for _ in range(4):
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(int(captured_flags.max().item()), 0)
            self.assertTrue(torch.equal(captured, expected))
            captured_outputs.append(captured.clone())
        self.assertTrue(
            all(
                torch.equal(captured_outputs[0], output)
                for output in captured_outputs[1:]
            )
        )

        scores.copy_(scores.flip(1))
        page_table.copy_(page_table.flip(1))
        graph.replay()
        torch.cuda.synchronize()
        mutated_logical = select_canonical_logical_topk(
            scores, lengths, topk, row_starts=starts, validate=False
        )
        mutated_row_pages = page_table.index_select(0, row_to_batch)
        mutated_valid = mutated_logical >= 0
        mutated_safe = mutated_logical.clamp_min(0)
        mutated_expected_pages = torch.gather(
            mutated_row_pages,
            1,
            torch.div(mutated_safe, page_size, rounding_mode="floor").long(),
        )
        mutated_expected = torch.where(
            mutated_valid,
            mutated_expected_pages * page_size
            + torch.remainder(mutated_safe, page_size),
            -1,
        ).to(torch.int32)
        self.assertEqual(int(captured_flags.max().item()), 0)
        self.assertTrue(torch.equal(captured, mutated_expected))
        self.assertFalse(torch.equal(captured, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_exact_paged_transform_maps_decode_chunk_and_cp_rows(self):
        generator = torch.Generator().manual_seed(520053)
        width, topk, page_size = 2304, 2048, 64
        scores = torch.randint(
            -8,
            9,
            (3, width),
            generator=generator,
            dtype=torch.int32,
        ).to(device="cuda", dtype=torch.float32)
        page_table = torch.stack(
            (
                torch.randperm(width // page_size, generator=generator),
                torch.randperm(width // page_size, generator=generator)
                + width // page_size,
            )
        ).to(device="cuda", dtype=torch.int32)

        cases = (
            {
                "name": "decode",
                "scores": scores[:2],
                "lengths": torch.tensor([2304, 1987], device="cuda"),
                "starts": None,
                "cu_seqlens_q": torch.tensor([0, 1, 2], device="cuda"),
                "batch_idx_list": None,
                "row_to_batch": torch.tensor([0, 1], device="cuda"),
            },
            {
                "name": "chunk",
                "scores": scores,
                "lengths": torch.tensor([2200, 2100, 2000], device="cuda"),
                "starts": torch.tensor([0, 128, 256], device="cuda"),
                "cu_seqlens_q": torch.tensor([0, 1, 2, 3], device="cuda"),
                "batch_idx_list": torch.tensor([1, 0, 1], device="cuda"),
                "row_to_batch": torch.tensor([1, 0, 1], device="cuda"),
            },
            {
                "name": "cp",
                "scores": scores,
                "lengths": torch.tensor([2200, 2100, 2000], device="cuda"),
                "starts": torch.tensor([0, 128, 256], device="cuda"),
                "cu_seqlens_q": torch.tensor([0, 2, 3], device="cuda"),
                "batch_idx_list": torch.tensor([1, 0], device="cuda"),
                "row_to_batch": torch.tensor([1, 1, 0], device="cuda"),
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                logical = select_canonical_logical_topk(
                    case["scores"],
                    case["lengths"],
                    topk,
                    row_starts=case["starts"],
                    validate=False,
                )
                row_pages = page_table.index_select(0, case["row_to_batch"])
                valid = logical >= 0
                safe = logical.clamp_min(0)
                expected_pages = torch.gather(
                    row_pages,
                    1,
                    torch.div(safe, page_size, rounding_mode="floor").long(),
                )
                expected = torch.where(
                    valid,
                    expected_pages * page_size + torch.remainder(safe, page_size),
                    -1,
                ).to(torch.int32)

                transformed, flags = select_canonical_paged_topk_fused(
                    case["scores"],
                    case["lengths"],
                    topk,
                    page_table=page_table,
                    page_size=page_size,
                    cu_seqlens_q=case["cu_seqlens_q"],
                    row_starts=case["starts"],
                    batch_idx_list=case["batch_idx_list"],
                )
                self.assertEqual(int(flags.max().item()), 0)
                self.assertTrue(torch.equal(transformed, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_exact_decode_graph_bounds_work_to_the_live_prefix(self):
        generator = torch.Generator().manual_seed(520054)
        rows, width, legal_width, topk, page_size = 16, 1_048_576, 4096, 2048, 64
        scores = torch.full(
            (rows, width), float("nan"), device="cuda", dtype=torch.float32
        )
        live_scores = torch.randint(
            -16,
            17,
            (rows, legal_width),
            generator=generator,
            dtype=torch.int32,
        ).to(device="cuda", dtype=torch.float32)
        scores[:, :legal_width].copy_(live_scores)
        lengths = torch.arange(
            legal_width - rows + 1,
            legal_width + 1,
            device="cuda",
            dtype=torch.int32,
        )
        page_table = torch.stack(
            [
                torch.randperm(legal_width // page_size, generator=generator)
                + row * (legal_width // page_size)
                for row in range(rows)
            ]
        ).to(device="cuda", dtype=torch.int32)

        def expected_physical() -> torch.Tensor:
            logical = select_canonical_logical_topk(
                scores[:, :legal_width], lengths, topk, validate=False
            )
            valid = logical >= 0
            safe = logical.clamp_min(0)
            pages = torch.gather(
                page_table,
                1,
                torch.div(safe, page_size, rounding_mode="floor").long(),
            )
            return torch.where(
                valid,
                pages * page_size + torch.remainder(safe, page_size),
                -1,
            ).to(torch.int32)

        expected = expected_physical()
        warm, warm_flags = select_canonical_paged_topk_fused(
            scores,
            lengths,
            topk,
            page_table=page_table,
            page_size=page_size,
        )
        self.assertEqual(int(warm_flags.max().item()), 0)
        self.assertTrue(torch.equal(warm, expected))

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_flags = select_canonical_paged_topk_fused(
                scores,
                lengths,
                topk,
                page_table=page_table,
                page_size=page_size,
            )
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(captured_flags.max().item()), 0)
        self.assertTrue(torch.equal(captured, expected))

        scores[:, :legal_width].copy_(live_scores.flip(1))
        page_table.copy_(page_table.flip(1))
        graph.replay()
        torch.cuda.synchronize()
        mutated_expected = expected_physical()
        self.assertEqual(int(captured_flags.max().item()), 0)
        self.assertTrue(torch.equal(captured, mutated_expected))
        self.assertFalse(torch.equal(captured, expected))

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
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                penalizer_orchestrator=None,
                grammars=None,
                grammar_mask=None,
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
