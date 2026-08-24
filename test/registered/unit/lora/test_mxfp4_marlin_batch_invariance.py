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
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner  # noqa: E402
from sglang.srt.layers.moe.token_dispatcher.deepep import (  # noqa: E402
    DeepEPLLCombineInput,
    DeepEPLLExactDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (  # noqa: E402
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402
from sglang.srt.lora import lora_moe_runner_marlin as marlin_runner  # noqa: E402
from sglang.srt.lora.lora_moe_runners import LoRAHooks, LoRAInfo  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMxfp4MarlinBatchInvariance(CustomTestCase):
    def test_low_latency_count_validation_is_cuda_graph_safe(self) -> None:
        capturing_counts = types.SimpleNamespace(is_cuda=True)
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(
                torch,
                "any",
                side_effect=AssertionError(
                    "capture path must not copy a device predicate to the host"
                ),
            ),
        ):
            marlin_runner._validate_low_latency_counts(capturing_counts, 8)

        with self.assertRaisesRegex(ValueError, "outside receive capacity"):
            marlin_runner._validate_low_latency_counts(
                torch.tensor([0, 9], dtype=torch.int32),
                8,
            )

    def test_deepep_auto_dispatch_uses_bf16_for_marlin(self) -> None:
        from sglang.srt.layers.moe import utils as moe_utils
        from sglang.srt.layers.moe.utils import (
            DispatcherOutputDtype,
            MoeRunnerBackend,
        )

        dispatcher = types.SimpleNamespace(quant_config=None)
        with (
            patch.object(moe_utils, "get_server_args", return_value=None),
            patch.object(
                moe_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH,
                "get",
                return_value=False,
            ),
            patch.object(moe_utils, "_is_npu", False),
        ):
            for backend in (
                MoeRunnerBackend.MARLIN,
                MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN,
            ):
                with self.subTest(backend=backend):
                    with patch.object(
                        moe_utils,
                        "get_moe_runner_backend",
                        return_value=backend,
                    ):
                        self.assertEqual(
                            moe_utils.get_deepep_output_dtype(dispatcher),
                            DispatcherOutputDtype.BF16,
                        )

    def test_quant_method_apply_routes_deepep_normal_to_lora_runner(self) -> None:
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            Mxfp4MarlinMoEMethod,
        )

        hidden = torch.arange(32, dtype=torch.bfloat16).view(2, 16)
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_ids=torch.tensor([[0, -1], [1, 0]], dtype=torch.int32),
            topk_weights=torch.ones((2, 2), dtype=torch.float32),
            num_recv_tokens_per_expert=[2],
        )
        expected = DeepEPNormalCombineInput(
            hidden_states=hidden + 1,
            topk_ids=dispatch.topk_ids,
            topk_weights=dispatch.topk_weights,
        )

        class _CaptureRunner:
            def run(self, received, *, quant_info):
                self.received = received
                self.quant_info = quant_info
                return expected

        class _Layer:
            w13_weight = torch.empty((1, 1, 1))

        method = Mxfp4MarlinMoEMethod(object(), "model.layers.0.mlp.experts")
        method.runner = _CaptureRunner()
        with patch(
            "sglang.srt.layers.quantization.mxfp4_marlin_moe.build_marlin_moe_quant_info",
            return_value="quant-info",
        ):
            result = method.apply(_Layer(), dispatch)

        self.assertIs(result, expected)
        self.assertIs(method.runner.received.hidden_states, hidden)
        self.assertEqual(method.runner.quant_info, "quant-info")

    def test_deepep_normal_marlin_registration_preserves_handle_payload(self) -> None:
        from sglang.srt.layers.moe.moe_runner import marlin as marlin_module
        from sglang.srt.layers.moe.moe_runner.base import FusedOpPool

        hidden = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
        topk_ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
        topk_weights = torch.ones((2, 2), dtype=torch.float32)
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[2],
        )

        def fake_local(standard_dispatch, quant_info, runner_config):
            self.assertIs(standard_dispatch.hidden_states, hidden)
            self.assertIs(standard_dispatch.topk_output.topk_ids, topk_ids)
            self.assertIs(standard_dispatch.topk_output.topk_weights, topk_weights)
            self.assertIs(quant_info, sentinel_quant)
            self.assertIs(runner_config, sentinel_config)
            return StandardCombineInput(hidden_states=hidden + 1)

        sentinel_quant = object()
        sentinel_config = object()
        fused = FusedOpPool.get_fused_func("deepep", "marlin")
        self.assertIs(fused, marlin_module.fused_experts_deepep_normal_to_marlin)
        with patch.object(
            marlin_module,
            "fused_experts_none_to_marlin",
            side_effect=fake_local,
        ):
            result = fused(dispatch, sentinel_quant, sentinel_config)

        self.assertIsInstance(result, DeepEPNormalCombineInput)
        self.assertTrue(torch.equal(result.hidden_states, hidden + 1))
        self.assertIs(result.topk_ids, topk_ids)
        self.assertIs(result.topk_weights, topk_weights)

    def test_deepep_normal_marlin_accepts_empty_receive_batch(self) -> None:
        from sglang.srt.layers.moe.moe_runner.base import FusedOpPool

        hidden = torch.empty((0, 4), dtype=torch.bfloat16)
        topk_ids = torch.empty((0, 2), dtype=torch.int32)
        topk_weights = torch.empty((0, 2), dtype=torch.float32)
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[0],
        )

        result = FusedOpPool.get_fused_func("deepep", "marlin")(
            dispatch, object(), object()
        )

        self.assertIsInstance(result, DeepEPNormalCombineInput)
        self.assertEqual(tuple(result.hidden_states.shape), (0, 4))
        self.assertIs(result.topk_ids, topk_ids)
        self.assertIs(result.topk_weights, topk_weights)

    def test_deepep_low_latency_marlin_builds_and_preweights_top1_rows(
        self,
    ) -> None:
        hidden_size = 4
        intermediate_size = 16
        hidden = torch.arange(16, dtype=torch.float32).view(2, 2, 4).to(torch.bfloat16)
        counts = torch.tensor([2, 1], dtype=torch.int32)
        packed_weights = torch.tensor([[0.5, 0.25], [0.75, 99.0]], dtype=torch.float32)
        original_ids = torch.tensor([[0, 3]], dtype=torch.int64)
        original_weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        dispatch = DeepEPLLExactDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_ids=original_ids,
            topk_weights=original_weights,
            masked_m=counts,
            expected_m=1,
            packed_route_weights=packed_weights,
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
            expert_map=torch.empty(0, dtype=torch.int32),
            global_num_experts=4,
        )
        runner_config = MoeRunnerConfig(
            activation="silu",
            swiglu_limit=10.0,
            dsv4_exact_mode=True,
            deepep_native_exact=True,
            no_combine=False,
            routed_scaling_factor=2.5,
        )
        hook_input = marlin_runner.MarlinLoraRunnerCore(runner_config).lora_hook_input(
            dispatch, quant_info
        )
        self.assertEqual(tuple(hook_input.hidden_states.shape), (4, hidden_size))
        self.assertEqual(hook_input.topk_ids.tolist(), [[0], [0], [1], [-1]])
        self.assertEqual(
            hook_input.topk_weights.flatten().tolist(), [0.5, 0.25, 0.75, 0.0]
        )

        gemm_calls = []

        def fake_gemm(input_tensor, output_tensor, *args, **kwargs):
            del input_tensor, args
            gemm_calls.append(
                (
                    kwargs["mul_topk_weights"],
                    kwargs["expert_block_partition_count"],
                )
            )
            output_tensor.fill_(1 if kwargs["size_n"] == 2 * intermediate_size else 4)

        def fake_activation(output, gate_up, limit):
            self.assertEqual(limit, 10.0)
            output.copy_(gate_up[:, :intermediate_size])

        def after_down(intermediate, cache, weights, ids):
            self.assertEqual(tuple(intermediate.shape), (4, intermediate_size))
            self.assertTrue(torch.equal(weights, hook_input.topk_weights))
            self.assertTrue(torch.equal(ids, hook_input.topk_ids))
            cache.add_(1)

        with (
            patch.object(
                marlin_runner,
                "moe_align_block_size",
                return_value=(
                    torch.arange(4, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    torch.tensor(4, dtype=torch.int32),
                ),
                create=True,
            ),
            patch.object(
                marlin_runner,
                "select_marlin_moe_block_size_m",
                return_value=8,
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
                marlin_runner, "get_scalar_type", return_value=object(), create=True
            ),
            patch.object(
                marlin_runner,
                "swiglu_limit_func",
                side_effect=fake_activation,
                create=True,
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            result = marlin_runner.MarlinLoraRunnerCore(
                runner_config
            ).run_from_dispatch(
                dispatch,
                quant_info,
                runner_config,
                hooks=LoRAHooks(after_gate_up=None, after_down=after_down),
            )

        expected = torch.tensor(
            [
                [[5.0] * hidden_size, [5.0] * hidden_size],
                [[5.0] * hidden_size, [0.0] * hidden_size],
            ],
            dtype=torch.float32,
        ).to(torch.bfloat16)
        self.assertIsInstance(result, DeepEPLLCombineInput)
        self.assertTrue(torch.equal(result.hidden_states, expected))
        self.assertIs(result.topk_ids, original_ids)
        self.assertIs(result.topk_weights, original_weights)
        self.assertTrue(all(value is False for value, _ in gemm_calls))
        self.assertTrue(all(partitions == 1 for _, partitions in gemm_calls))

    def test_deepep_normal_marlin_reduces_unweighted_routes_in_shared_helper(
        self,
    ) -> None:
        from sglang.srt.layers.moe.moe_runner import marlin as marlin_module
        from sglang.srt.layers.moe.moe_runner.base import FusedOpPool

        hidden = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
        topk_ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.75, 123.0], [0.5, 0.25]], dtype=torch.float32)
        routes = torch.tensor(
            [
                [[2.0, 4.0, 6.0, 8.0], [99.0, 99.0, 99.0, 99.0]],
                [[3.0, 5.0, 7.0, 9.0], [1.0, 2.0, 3.0, 4.0]],
            ],
            dtype=torch.bfloat16,
        )
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[2],
        )

        with patch.object(
            marlin_module,
            "fused_experts_none_to_marlin",
            return_value=StandardCombineInput(hidden_states=routes),
        ):
            result = FusedOpPool.get_fused_func("deepep", "marlin")(
                dispatch,
                object(),
                types.SimpleNamespace(no_combine=True),
            )

        expected = (
            torch.where(
                (topk_ids >= 0).unsqueeze(-1),
                routes.float() * topk_weights.unsqueeze(-1),
                torch.zeros((), dtype=torch.float32),
            )
            .sum(dim=1)
            .to(torch.bfloat16)
        )
        self.assertIsInstance(result, DeepEPNormalCombineInput)
        self.assertTrue(torch.equal(result.hidden_states, expected))
        self.assertEqual(result.hidden_states.dtype, torch.bfloat16)

    def test_dsv4_low_latency_matches_normal_weight_and_scale_boundaries(
        self,
    ) -> None:
        hidden_size = 4
        intermediate_size = 16
        hidden = torch.arange(16, dtype=torch.float32).view(2, 2, 4).to(torch.bfloat16)
        counts = torch.tensor([2, 1], dtype=torch.int32)
        packed_weights = torch.tensor([[0.5, 0.25], [0.75, 99.0]], dtype=torch.float32)
        original_ids = torch.tensor([[0, 3, 1, 2, 0, 3]], dtype=torch.int64)
        original_weights = torch.full((1, 6), 1.0 / 6.0, dtype=torch.float32)
        dispatch = DeepEPLLExactDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_ids=original_ids,
            topk_weights=original_weights,
            masked_m=counts,
            expected_m=1,
            packed_route_weights=packed_weights,
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
            expert_map=torch.empty(0, dtype=torch.int32),
            global_num_experts=4,
        )
        runner_config = MoeRunnerConfig(
            activation="silu",
            swiglu_limit=10.0,
            dsv4_exact_mode=True,
            deepep_native_exact=True,
            deepep_native_exact_defer_routed_scale=True,
            no_combine=False,
            routed_scaling_factor=1.5,
        )
        hook_input = marlin_runner.MarlinLoraRunnerCore(runner_config).lora_hook_input(
            dispatch, quant_info
        )
        gemm_calls = []
        selected_geometries = []

        def fake_select_block_size(**kwargs):
            selected_geometries.append(kwargs)
            return 8

        def fake_gemm(input_tensor, output_tensor, *args, **kwargs):
            del input_tensor
            mul_weight = kwargs["mul_topk_weights"]
            gemm_calls.append((mul_weight, kwargs["expert_block_partition_count"]))
            if kwargs["size_n"] == 2 * intermediate_size:
                output_tensor.fill_(1)
            elif mul_weight:
                output_tensor.copy_(
                    args[11]
                    .reshape(-1, 1)
                    .to(output_tensor.dtype)
                    .expand_as(output_tensor)
                    * 4
                )
            else:
                output_tensor.fill_(4)

        def fake_activation(output, gate_up, limit):
            self.assertEqual(limit, 10.0)
            output.copy_(gate_up[:, :intermediate_size])

        def after_down(intermediate, cache, weights, ids):
            self.assertEqual(tuple(intermediate.shape), (4, intermediate_size))
            self.assertTrue(torch.equal(weights, hook_input.topk_weights))
            self.assertTrue(torch.equal(ids, hook_input.topk_ids))
            cache.add_(weights.unsqueeze(-1).to(cache.dtype))

        with (
            patch.object(
                marlin_runner,
                "moe_align_block_size",
                return_value=(
                    torch.arange(4, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    torch.tensor(4, dtype=torch.int32),
                ),
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
                marlin_runner, "get_scalar_type", return_value=object(), create=True
            ),
            patch.object(
                marlin_runner,
                "swiglu_limit_func",
                side_effect=fake_activation,
                create=True,
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            result = marlin_runner.MarlinLoraRunnerCore(
                runner_config
            ).run_from_dispatch(
                dispatch,
                quant_info,
                runner_config,
                hooks=LoRAHooks(after_gate_up=None, after_down=after_down),
            )

        expected = torch.tensor(
            [
                [[2.5] * hidden_size, [1.25] * hidden_size],
                [[3.75] * hidden_size, [0.0] * hidden_size],
            ],
            dtype=torch.float32,
        ).to(torch.bfloat16)
        self.assertIsInstance(result, DeepEPLLCombineInput)
        self.assertTrue(torch.equal(result.hidden_states, expected))
        self.assertNotIn(1.5, result.hidden_states.float().unique().tolist())
        self.assertTrue(
            torch.equal(result.topk_weights, torch.ones_like(original_weights))
        )
        self.assertTrue(all(gemm_calls[index][0] is False for index in range(4)))
        self.assertTrue(all(gemm_calls[index][0] is True for index in range(4, 8)))
        self.assertTrue(all(partitions == 1 for _, partitions in gemm_calls))
        self.assertEqual(len(selected_geometries), 1)
        # The runner rows are transport-packed as top-k 1, but Marlin must
        # select its arithmetic program from the original logical topology.
        self.assertEqual(selected_geometries[0]["topk"], original_ids.shape[1])

    def test_deepep_lora_received_rows_require_one_active_adapter(self) -> None:
        hidden = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=None,
            topk_ids=torch.tensor([[0, -1], [1, 0]], dtype=torch.int32),
            topk_weights=torch.ones((2, 2), dtype=torch.float32),
            num_recv_tokens_per_expert=[2],
        )

        class _CaptureCore:
            def run_from_dispatch(self, *args, hooks=None, **kwargs):
                del args, kwargs
                self.hooks = hooks
                return hooks

        def lora_info(single_adapter_id):
            empty = torch.empty(0)
            return LoRAInfo(
                gate_up_lora_a_weights=empty,
                gate_up_lora_b_weights=empty,
                down_lora_a_weights=empty,
                down_lora_b_weights=empty,
                seg_indptr=torch.tensor([0, 1], dtype=torch.int32),
                req_to_lora=torch.tensor([1], dtype=torch.int32),
                lora_ranks=torch.tensor([0, 1], dtype=torch.int64),
                adapter_enabled=torch.tensor([0, 1], dtype=torch.int32),
                token_lora_mapping=torch.tensor([1], dtype=torch.int32),
                max_lora_rank=1,
                num_experts=2,
                has_active_lora=True,
                single_adapter_id=single_adapter_id,
            )

        runner = MoeRunner.__new__(MoeRunner)
        runner.fused_func = None
        runner.lora_enabled = True
        runner.runner_core = _CaptureCore()
        runner.config = object()
        hooks = runner.run(dispatch, object(), lora_info=lora_info(1))
        self.assertIsInstance(hooks, LoRAHooks)
        self.assertIsNotNone(hooks.after_gate_up)
        self.assertIsNotNone(hooks.after_down)

        with self.assertRaisesRegex(RuntimeError, "one active adapter"):
            runner.run(dispatch, object(), lora_info=lora_info(None))

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

    def _run_base_gemm_case(
        self,
        *,
        dsv4_exact_mode: bool,
        no_combine: bool = False,
    ) -> None:
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
            no_combine=no_combine,
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
                    "mul_topk_weights": kwargs["mul_topk_weights"],
                    "expert_block_partition_count": kwargs[
                        "expert_block_partition_count"
                    ],
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
                self.assertEqual(call["expert_block_partition_count"], topk)
            for token, call in enumerate(gemm_calls[num_tokens:]):
                self.assertTrue(
                    torch.equal(call["topk_weights"], topk_weights[token : token + 1])
                )
                self.assertEqual(call["size_m"], topk)
                self.assertEqual(call["top_k"], 1)
                self.assertEqual(call["mul_topk_weights"], not no_combine)
                self.assertEqual(call["expert_block_partition_count"], topk)
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
            self.assertEqual(gate_up_call["expert_block_partition_count"], 1)
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
            self.assertEqual(down_call["mul_topk_weights"], not no_combine)
            self.assertEqual(down_call["expert_block_partition_count"], 1)
        self.assertEqual(
            hook_calls,
            [
                ("gate_up", (num_tokens, topk, 2 * intermediate_size)),
                ("down", (num_tokens, topk, hidden_size)),
            ],
        )
        if no_combine:
            self.assertEqual(
                tuple(result.hidden_states.shape),
                (num_tokens, topk, hidden_size),
            )
            self.assertTrue(
                torch.equal(
                    result.hidden_states,
                    torch.full_like(result.hidden_states, 5),
                )
            )
        else:
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

    def test_exact_no_combine_returns_unweighted_bf16_routes(self) -> None:
        self._run_base_gemm_case(dsv4_exact_mode=True, no_combine=True)

    def test_lora_hook_only_suppresses_down_route_weight(self) -> None:
        from sglang.srt.lora import lora_moe_runners as hook_module

        empty = torch.empty(0)
        lora_info = LoRAInfo(
            gate_up_lora_a_weights=empty,
            gate_up_lora_b_weights=empty,
            down_lora_a_weights=empty,
            down_lora_b_weights=empty,
            seg_indptr=torch.tensor([0, 1], dtype=torch.int32),
            req_to_lora=torch.tensor([0], dtype=torch.int32),
            lora_ranks=torch.tensor([1], dtype=torch.int64),
            adapter_enabled=torch.tensor([1], dtype=torch.int32),
            token_lora_mapping=torch.tensor([0], dtype=torch.int32),
            max_lora_rank=1,
            num_experts=2,
            has_active_lora=True,
            lora_use_virtual_experts=True,
        )
        hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
        calls = []

        def fake_gate(*args, **kwargs):
            calls.append(("gate", kwargs))

        def fake_down(*args, **kwargs):
            calls.append(("down", kwargs))

        with (
            patch.object(hook_module, "_add_lora_gate_up_delta", side_effect=fake_gate),
            patch.object(hook_module, "_add_lora_down_delta", side_effect=fake_down),
        ):
            hooks = hook_module.build_lora_hooks(
                hidden,
                lora_info,
                ids,
                mul_routed_weight=False,
            )
            hooks.after_gate_up(hidden, torch.empty((1, 2, 4)), weights, ids)
            hooks.after_down(torch.empty((2, 4)), torch.empty((1, 2, 4)), weights, ids)

        self.assertNotIn("mul_routed_weight", calls[0][1])
        self.assertIs(calls[1][1]["mul_routed_weight"], False)


if __name__ == "__main__":
    unittest.main()
