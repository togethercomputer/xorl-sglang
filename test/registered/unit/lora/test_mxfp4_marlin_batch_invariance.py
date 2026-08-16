"""Regression coverage for MXFP4 Marlin's batch-invariant base program."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.moe_runner.base import (  # noqa: E402
    MoeRunnerConfig,
    should_singleton_mxfp4_marlin_base,
)
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo  # noqa: E402
from sglang.srt.layers.moe.token_dispatcher.standard import (  # noqa: E402
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402
from sglang.srt.lora import lora_moe_runner_marlin as marlin_runner  # noqa: E402
from sglang.srt.lora.lora_moe_runners import LoRAHooks  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMxfp4MarlinBatchInvariance(CustomTestCase):
    def test_singleton_base_scope_is_exact_mxfp4_multitoken_only(self) -> None:
        cases = (
            (True, True, 3, True),
            (False, True, 3, False),
            (True, False, 3, False),
            (True, True, 1, False),
        )
        for dsv4_exact_mode, is_mxfp4_marlin, num_tokens, expected in cases:
            with self.subTest(
                dsv4_exact_mode=dsv4_exact_mode,
                is_mxfp4_marlin=is_mxfp4_marlin,
                num_tokens=num_tokens,
            ):
                self.assertEqual(
                    should_singleton_mxfp4_marlin_base(
                        dsv4_exact_mode=dsv4_exact_mode,
                        is_mxfp4_marlin=is_mxfp4_marlin,
                        num_tokens=num_tokens,
                    ),
                    expected,
                )

    def _run_base_gemm_case(self, *, dsv4_exact_mode: bool) -> None:
        """Exercise exact singleton and ordinary batched MXFP4 execution.

        In DSV4 exact mode, unrelated token routes must not share an MXFP4 base
        launch. Outside that mode, preserve the established batched base GEMMs.
        In both cases each LoRA hook must see the complete batch exactly once.
        """

        num_tokens = 3
        hidden_size = 4
        intermediate_size = 16
        topk = 2
        hidden_states = torch.arange(
            num_tokens * hidden_size, dtype=torch.bfloat16
        ).view(num_tokens, hidden_size)
        topk_ids = torch.tensor([[0, -1], [1, 0], [-1, 1]], dtype=torch.int32)
        topk_weights = torch.ones((num_tokens, topk), dtype=torch.float32)
        dispatch = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=torch.zeros((num_tokens, 2), dtype=torch.float32),
            ),
        )
        scales = torch.empty((2, 1), dtype=torch.float8_e8m0fnu)
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=torch.empty((2, 1, 1)),
            w2_qweight=torch.empty((2, 1, 1)),
            w13_scales=scales,
            w2_scales=scales,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            expert_map=torch.tensor([0, 1], dtype=torch.int32),
            global_num_experts=2,
        )
        runner_config = MoeRunnerConfig(
            activation="silu",
            swiglu_limit=10.0,
            dsv4_exact_mode=dsv4_exact_mode,
        )

        aligned_batches = []
        selected_geometries = []
        gemm_calls = []
        hook_calls = []

        def fake_select_block_size(**kwargs):
            selected_geometries.append(kwargs)
            return 8

        def fake_align(ids, block_size_m, num_experts):
            aligned_batches.append((ids.clone(), block_size_m, num_experts))
            return (
                torch.arange(ids.numel(), dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                torch.tensor(ids.numel(), dtype=torch.int32),
            )

        def fake_gemm(input_tensor, output_tensor, *args, **kwargs):
            gemm_calls.append(
                {
                    "input": input_tensor.clone(),
                    "input_shape": tuple(input_tensor.shape),
                    "output_shape": tuple(output_tensor.shape),
                    "topk_weights": args[11].clone(),
                    "size_m": kwargs["size_m"],
                    "size_n": kwargs["size_n"],
                    "top_k": kwargs["top_k"],
                }
            )
            if kwargs["size_n"] == 2 * intermediate_size:
                output_tensor.fill_(1)
            else:
                self.assertTrue(
                    torch.equal(input_tensor, torch.full_like(input_tensor, 3))
                )
                output_tensor.fill_(4)
            return output_tensor

        def fake_activation(output, gate_up, limit):
            self.assertEqual(limit, 10.0)
            output.copy_(gate_up[:, :intermediate_size])

        def after_gate_up(hidden, cache, weights, ids):
            hook_calls.append(("gate_up", tuple(cache.shape)))
            self.assertIs(hidden, hidden_states)
            self.assertIs(weights, topk_weights)
            self.assertIs(ids, topk_ids)
            cache.add_(2)

        def after_down(intermediate, cache, weights, ids):
            hook_calls.append(("down", tuple(cache.shape)))
            self.assertEqual(
                tuple(intermediate.shape),
                (num_tokens * topk, intermediate_size),
            )
            self.assertIs(weights, topk_weights)
            self.assertIs(ids, topk_ids)
            cache.add_(1)

        def fake_topk_sum(cache, output):
            output.copy_(cache.sum(dim=1))

        fake_topk_module = types.ModuleType("sglang.kernels.ops.moe.moe_topk_sum")
        fake_topk_module.moe_topk_sum = fake_topk_sum
        hooks = LoRAHooks(
            after_gate_up=after_gate_up,
            after_down=after_down,
        )

        with (
            patch.object(
                marlin_runner,
                "moe_align_block_size",
                side_effect=fake_align,
                create=True,
            ),
            patch.object(
                marlin_runner,
                "select_marlin_moe_block_size_m",
                side_effect=fake_select_block_size,
                create=True,
            ),
            patch.object(
                marlin_runner,
                "moe_wna16_marlin_gemm",
                side_effect=fake_gemm,
                create=True,
            ),
            patch.object(
                marlin_runner,
                "marlin_make_workspace",
                return_value=torch.empty(1),
                create=True,
            ),
            patch.object(
                marlin_runner,
                "get_scalar_type",
                return_value=object(),
                create=True,
            ),
            patch.object(
                marlin_runner,
                "swiglu_limit_func",
                side_effect=fake_activation,
                create=True,
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.dict(
                sys.modules,
                {"sglang.kernels.ops.moe.moe_topk_sum": fake_topk_module},
            ),
        ):
            result = marlin_runner.MarlinLoraRunnerCore(
                runner_config
            ).run_from_dispatch(
                dispatch,
                quant_info,
                runner_config,
                hooks=hooks,
            )

        self.assertEqual(len(selected_geometries), 1)
        self.assertEqual(
            selected_geometries[0]["num_tokens"],
            1 if dsv4_exact_mode else num_tokens,
        )
        if dsv4_exact_mode:
            self.assertEqual(len(aligned_batches), num_tokens)
            self.assertTrue(
                all(
                    torch.equal(ids, topk_ids[token : token + 1])
                    for token, (ids, _, _) in enumerate(aligned_batches)
                )
            )
            self.assertEqual(
                [
                    (block_size_m, num_experts)
                    for _, block_size_m, num_experts in aligned_batches
                ],
                [(8, quant_info.global_num_experts)] * num_tokens,
            )
            self.assertEqual(
                [call["input_shape"] for call in gemm_calls[:num_tokens]],
                [(1, hidden_size)] * num_tokens,
            )
            self.assertEqual(
                [call["output_shape"] for call in gemm_calls[:num_tokens]],
                [(topk, 2 * intermediate_size)] * num_tokens,
            )
            self.assertEqual(
                [call["input_shape"] for call in gemm_calls[num_tokens:]],
                [(topk, intermediate_size)] * num_tokens,
            )
            self.assertEqual(
                [call["output_shape"] for call in gemm_calls[num_tokens:]],
                [(topk, hidden_size)] * num_tokens,
            )
            for token, call in enumerate(gemm_calls[:num_tokens]):
                self.assertTrue(
                    torch.equal(call["input"], hidden_states[token : token + 1])
                )
                self.assertTrue(
                    torch.equal(call["topk_weights"], topk_weights[token : token + 1])
                )
                self.assertEqual(call["size_m"], 1)
                self.assertEqual(call["top_k"], topk)
            for token, call in enumerate(gemm_calls[num_tokens:]):
                self.assertTrue(
                    torch.equal(call["topk_weights"], topk_weights[token : token + 1])
                )
                self.assertEqual(call["size_m"], topk)
                self.assertEqual(call["top_k"], 1)
        else:
            self.assertEqual(len(aligned_batches), 1)
            ids, block_size_m, num_experts = aligned_batches[0]
            self.assertTrue(torch.equal(ids, topk_ids))
            self.assertEqual(
                (block_size_m, num_experts),
                (8, quant_info.global_num_experts),
            )
            self.assertEqual(len(gemm_calls), 2)
            gate_up_call, down_call = gemm_calls
            self.assertEqual(gate_up_call["input_shape"], (num_tokens, hidden_size))
            self.assertEqual(
                gate_up_call["output_shape"],
                (num_tokens * topk, 2 * intermediate_size),
            )
            self.assertTrue(torch.equal(gate_up_call["topk_weights"], topk_weights))
            self.assertEqual(gate_up_call["size_m"], num_tokens)
            self.assertEqual(gate_up_call["top_k"], topk)
            self.assertEqual(
                down_call["input_shape"],
                (num_tokens * topk, intermediate_size),
            )
            self.assertEqual(
                down_call["output_shape"],
                (num_tokens * topk, hidden_size),
            )
            self.assertTrue(torch.equal(down_call["topk_weights"], topk_weights))
            self.assertEqual(down_call["size_m"], num_tokens * topk)
            self.assertEqual(down_call["top_k"], 1)
        self.assertEqual(
            hook_calls,
            [
                ("gate_up", (num_tokens, topk, 2 * intermediate_size)),
                ("down", (num_tokens, topk, hidden_size)),
            ],
        )
        self.assertTrue(
            torch.equal(result.hidden_states, torch.full_like(hidden_states, 10))
        )

    def test_exact_base_gemms_are_singleton_and_lora_hooks_stay_batched(
        self,
    ) -> None:
        self._run_base_gemm_case(dsv4_exact_mode=True)

    def test_nonexact_base_gemms_stay_batched_and_lora_hooks_stay_batched(
        self,
    ) -> None:
        self._run_base_gemm_case(dsv4_exact_mode=False)


if __name__ == "__main__":
    unittest.main()
