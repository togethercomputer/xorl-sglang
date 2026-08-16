import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.canonical_moe import (
    CANONICAL_MOE_FOLD_VERSION,
    CANONICAL_MOE_LEAF_VERSION,
    GLM52_CANONICAL_MOE_V3B_VERSION,
    GLM52_CANONICAL_MOE_VERSION,
    GLM52_SAMPLER_LOCAL_POLICY,
    CanonicalTransport,
    _canonical_moe_fold_fp64_tree,
    _canonical_moe_fold_fp64_tree_batched,
    _fused_tree_or_cpu_reference,
    canonical_moe_fold_fp64_v3,
    canonical_moe_leaf_fp32_v1,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _partials(contributors: int, *, seed: int, shape=(5, 33)) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((contributors, *shape), generator=gen, dtype=torch.float32).to(
        torch.bfloat16
    )


def _orders(contributors: int) -> list[tuple[int, ...]]:
    identity = tuple(range(contributors))
    rotations = [
        tuple((index + shift) % contributors for index in range(contributors))
        for shift in range(1, contributors)
    ]
    scramble = tuple((3 * index + 1) % contributors for index in range(contributors))
    orders = [identity, *rotations]
    if len(set(scramble)) == contributors:
        orders.append(scramble)
    return orders


class TestGlm52CanonicalFusedTree(unittest.TestCase):
    def test_shared_fold_version_and_adversarial_arithmetic(self):
        self.assertEqual(CANONICAL_MOE_FOLD_VERSION, "canonical_moe_fold_fp64_v3")
        partials = torch.tensor(
            [33554432.0, 1.0, -33554432.0, 1.0],
            dtype=torch.bfloat16,
        ).view(4, 1)
        folded_fp64 = _canonical_moe_fold_fp64_tree(partials)
        folded = canonical_moe_fold_fp64_v3(partials)
        self.assertEqual(folded_fp64.dtype, torch.float64)
        self.assertEqual(folded_fp64.item(), 2.0)
        self.assertEqual(folded.item(), 2.0)

        prior_fp32 = (partials[0].float() + partials[1].float()) + (
            partials[2].float() + partials[3].float()
        )
        self.assertEqual(prior_fp32.item(), 0.0)

        legacy_left = (partials[0] + partials[1]).to(torch.bfloat16)
        legacy_right = (partials[2] + partials[3]).to(torch.bfloat16)
        legacy = (legacy_left + legacy_right).to(torch.bfloat16)
        self.assertEqual(legacy.item(), 0.0)

    def test_leaf_version_and_fma_sensitive_arithmetic(self):
        self.assertEqual(CANONICAL_MOE_LEAF_VERSION, "canonical_moe_leaf_fp32_v1")
        shared = torch.tensor([-0.01165771484375], dtype=torch.bfloat16)
        routed = torch.tensor([209920.0], dtype=torch.bfloat16)
        actual = canonical_moe_leaf_fp32_v1(shared, routed.clone(), 1.1)
        separately_rounded = (
            routed.float() * torch.tensor(1.1, dtype=torch.float32) + shared.float()
        ).to(torch.bfloat16)
        self.assertEqual(actual.item(), 231424.0)
        self.assertEqual(separately_rounded.item(), 230400.0)
        self.assertFalse(torch.equal(actual, separately_rounded))

    def test_shared_fold_restores_logical_order_for_optimized_and_fallback_sizes(self):
        for contributors in (1, 2, 3, 4, 5, 6, 8, 16, 17):
            logical = _partials(contributors, seed=31337 + contributors)
            physical_to_logical = tuple(reversed(range(contributors)))
            physical = logical[list(physical_to_logical)]
            logical_to_physical = torch.tensor(
                [physical_to_logical.index(index) for index in range(contributors)],
                dtype=torch.int64,
            )
            actual = canonical_moe_fold_fp64_v3(
                physical,
                logical_to_physical=logical_to_physical,
            )
            expected = _canonical_moe_fold_fp64_tree(logical).to(logical.dtype)
            self.assertTrue(torch.equal(actual, expected))

    def test_shared_fold_fails_closed_on_bad_abi(self):
        with self.assertRaisesRegex(TypeError, "BF16"):
            canonical_moe_fold_fp64_v3(torch.zeros((8, 4), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "at least one contributor"):
            canonical_moe_fold_fp64_v3(torch.zeros((0, 4), dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "logical_to_physical"):
            canonical_moe_fold_fp64_v3(
                torch.zeros((8, 4), dtype=torch.bfloat16),
                logical_to_physical=torch.arange(4),
            )

    def test_v3b_version_and_transport_strings(self):
        self.assertEqual(GLM52_CANONICAL_MOE_VERSION, "canonical_moe_reduce_fp64_v3")
        self.assertEqual(
            GLM52_SAMPLER_LOCAL_POLICY,
            "glm52_routed_final_scaled_then_shared_ep_slice_fp32_then_bf16_v3",
        )
        self.assertEqual(
            GLM52_CANONICAL_MOE_V3B_VERSION, "glm52_canonical_moe_reduce_v3b"
        )
        self.assertEqual(
            CanonicalTransport.REPLICATED_DECODE_V3B.value,
            "glm52_canonical_moe_replicated_decode_v3b",
        )

    def test_batched_tree_matches_reference_tree_bitwise(self):
        for contributors in (1, 2, 3, 4, 5, 6, 8, 16, 17):
            partials = _partials(contributors, seed=20260804 + contributors)
            self.assertTrue(
                torch.equal(
                    _canonical_moe_fold_fp64_tree_batched(partials),
                    _canonical_moe_fold_fp64_tree(partials),
                )
            )

    def test_batched_tree_rejects_non_bf16_and_zero_contributors(self):
        with self.assertRaisesRegex(TypeError, "BF16"):
            _canonical_moe_fold_fp64_tree_batched(
                torch.zeros(4, 3, dtype=torch.float32)
            )
        with self.assertRaisesRegex(ValueError, "at least one contributor"):
            _canonical_moe_fold_fp64_tree_batched(
                torch.zeros(0, 3, dtype=torch.bfloat16)
            )

    def test_cross_engine_adjacent_carry_vectors(self):
        # Versioned scalar vectors for the trainer implementation.  At each
        # level adjacent logical contributors are FP64-added; an odd final
        # contributor is carried unchanged. The completed tree is cast once.
        source_bits = (
            0x4580,
            0x3F80,
            0xC580,
            0x3F00,
            0x4050,
            0xC030,
            0x3E00,
            0x4500,
            0xC500,
            0x40E0,
            0xC0D0,
            0x3D00,
            0x4180,
            0xC170,
            0x3F40,
            0xBF20,
            0x4020,
        )
        expected_bits = {1: 0x4580, 3: 0x3F80, 5: 0x4098, 6: 0x4000, 17: 0x40C9}
        source = torch.tensor(source_bits, dtype=torch.uint16).view(torch.bfloat16)
        for contributors, expected in expected_bits.items():
            with self.subTest(contributors=contributors):
                actual = canonical_moe_fold_fp64_v3(source[:contributors, None])
                self.assertEqual(actual.view(torch.uint16).item(), expected)

    def test_shared_abi_preserves_fp16_transport_dtype(self):
        partials = torch.tensor(
            [[4096.0], [1.0], [-4096.0], [1.0]], dtype=torch.float16
        )
        folded = canonical_moe_fold_fp64_v3(partials)
        leaf = canonical_moe_leaf_fp32_v1(
            torch.ones((2,), dtype=torch.float16),
            torch.ones((2,), dtype=torch.float16),
            1.5,
        )
        self.assertEqual(folded.dtype, torch.float16)
        self.assertEqual(folded.item(), 2.0)
        self.assertEqual(leaf.dtype, torch.float16)
        self.assertTrue(torch.equal(leaf, torch.full_like(leaf, 2.5)))

    def test_deepseek_exact_leaf_selects_scale_once_at_transport_boundary(self):
        try:
            from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
        except ImportError:
            self.skipTest("sgl_kernel is required to import the serving model")
        moe = object.__new__(DeepseekV2MoE)
        torch.nn.Module.__init__(moe)
        moe.experts = SimpleNamespace(quant_method=None)
        moe.routed_scaling_factor = 1.5
        shared = torch.full((4,), 2.0, dtype=torch.bfloat16)
        routed = torch.full((4,), 3.0, dtype=torch.bfloat16)

        with patch(
            "sglang.srt.models.deepseek_v2.is_routed_scale_deferred_to_shared_add",
            return_value=False,
        ):
            glm_leaf = moe._canonical_exact_leaf(routed.clone(), shared)
        with patch(
            "sglang.srt.models.deepseek_v2.is_routed_scale_deferred_to_shared_add",
            return_value=True,
        ):
            dsv_leaf = moe._canonical_exact_leaf(routed.clone(), shared)
        self.assertTrue(torch.equal(glm_leaf, torch.full_like(glm_leaf, 5.0)))
        self.assertTrue(torch.equal(dsv_leaf, torch.full_like(dsv_leaf, 6.5)))

    def test_cpu_reference_path_matches_tree_across_contributor_orders(self):
        for contributors in (1, 2, 3, 4, 5, 6, 8, 16, 17):
            partials = _partials(contributors, seed=999 + contributors)
            for order in _orders(contributors):
                logical_to_group = torch.tensor(order, dtype=torch.int64)
                output = torch.empty_like(partials[0])
                identity_order = order == tuple(range(contributors))
                _fused_tree_or_cpu_reference(
                    partials,
                    logical_to_group,
                    identity_order=identity_order,
                    output=output,
                )
                expected = _canonical_moe_fold_fp64_tree(
                    partials.index_select(0, logical_to_group)
                ).to(partials.dtype)
                self.assertTrue(torch.equal(output, expected), msg=str(order))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_tree_kernel_matches_eager_tree_bitwise_on_cuda(self):
        from sglang.srt.distributed.canonical_moe_kernels import (
            fused_balanced_adjacent_fp64_tree,
        )

        for contributors in range(1, 17):
            partials = _partials(
                contributors, seed=777 + contributors, shape=(128, 96)
            ).cuda()
            for order in _orders(contributors):
                logical_to_group = torch.tensor(order, dtype=torch.int64, device="cuda")
                identity_order = order == tuple(range(contributors))
                fused = fused_balanced_adjacent_fp64_tree(
                    partials,
                    logical_to_group,
                    identity_order=identity_order,
                )
                expected = _canonical_moe_fold_fp64_tree(
                    partials.index_select(0, logical_to_group)
                ).to(partials.dtype)
                self.assertTrue(
                    torch.equal(fused, expected),
                    msg=f"contributors={contributors} order={order}",
                )
        # The comparator can fail: flipping one contributor element must
        # change the folded output.
        partials = _partials(16, seed=4242, shape=(64, 64)).cuda()
        logical_to_group = torch.arange(16, dtype=torch.int64, device="cuda")
        baseline = fused_balanced_adjacent_fp64_tree(
            partials, logical_to_group, identity_order=True
        ).clone()
        partials[7, 11, 13] = partials[7, 11, 13] + 8.0
        perturbed = fused_balanced_adjacent_fp64_tree(
            partials, logical_to_group, identity_order=True
        )
        self.assertFalse(torch.equal(baseline, perturbed))

        # This distinguishes the declared adjacent tree from a reassociated
        # or FP32 reduction. Every CUDA tree node is add.rn.f64.
        witness = torch.tensor(
            [33554432.0, 1.0, -33554432.0, 1.0],
            dtype=torch.bfloat16,
            device="cuda",
        ).view(4, 1)
        pair_tree = fused_balanced_adjacent_fp64_tree(
            witness,
            torch.arange(4, dtype=torch.int64, device="cuda"),
            identity_order=True,
        )
        left_linear = (
            (witness[0].float() + witness[1].float()) + witness[2].float()
        ) + witness[3].float()
        self.assertEqual(pair_tree.item(), 2.0)
        self.assertEqual(left_linear.item(), 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_leaf_fma_matches_reference_eager_and_cuda_graph(self):
        shared = torch.full(
            (32,), -0.01165771484375, dtype=torch.bfloat16, device="cuda"
        )
        routed_source = torch.full((32,), 209920.0, dtype=torch.bfloat16, device="cuda")
        routed = routed_source.clone()
        eager = canonical_moe_leaf_fp32_v1(shared, routed, 1.1).clone()
        self.assertTrue(torch.equal(eager, torch.full_like(eager, 231424.0)))

        routed.copy_(routed_source)
        canonical_moe_leaf_fp32_v1(shared, routed, 1.1)
        torch.cuda.synchronize()
        routed.copy_(routed_source)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_leaf = canonical_moe_leaf_fp32_v1(shared, routed, 1.1)
        graph.replay()
        self.assertTrue(torch.equal(graph_leaf, eager))
        routed.copy_(torch.full_like(routed, 1024.0))
        graph.replay()
        expected = torch.add(
            shared.cpu().float(),
            torch.full((32,), 1024.0, dtype=torch.float32),
            alpha=1.1,
        ).to(torch.bfloat16)
        self.assertTrue(torch.equal(graph_leaf.cpu(), expected))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_qwen_gated_leaf_uses_one_round_fp32_fma_eager_and_graph(self):
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul_add,
        )

        hidden = torch.zeros((1, 2048), dtype=torch.bfloat16, device="cuda")
        weight = torch.zeros((2048,), dtype=torch.bfloat16, device="cuda")
        shared = torch.zeros_like(hidden)
        routed_source = torch.zeros_like(hidden)
        hidden[0, 0] = 1.0
        weight[0] = 1.0
        shared[0, 936] = -2.234375
        routed_source[0, 936] = 1.6328125

        routed = routed_source.clone()
        fused_gate_sigmoid_mul_add(hidden, weight, shared, routed)
        expected = torch.tensor(-0.000644683837890625, dtype=torch.bfloat16)
        separately_rounded = (
            torch.sigmoid(torch.tensor(1.0, dtype=torch.float32))
            * shared[0, 936].cpu().float()
            + routed_source[0, 936].cpu().float()
        ).to(torch.bfloat16)
        self.assertEqual(routed[0, 936].cpu(), expected)
        self.assertEqual(separately_rounded.item(), -0.00064849853515625)
        self.assertNotEqual(routed[0, 936].cpu(), separately_rounded)

        routed.copy_(routed_source)
        fused_gate_sigmoid_mul_add(hidden, weight, shared, routed)
        torch.cuda.synchronize()
        routed.copy_(routed_source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_gate_sigmoid_mul_add(hidden, weight, shared, routed)
        graph.replay()
        self.assertEqual(routed[0, 936].cpu(), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fp16_shared_abi_uses_fused_cuda_paths(self):
        partials = torch.tensor(
            [[4096.0], [1.0], [-4096.0], [1.0]],
            dtype=torch.float16,
            device="cuda",
        )
        folded = canonical_moe_fold_fp64_v3(partials)
        shared = torch.full((16,), 2.0, dtype=torch.float16, device="cuda")
        routed = torch.full((16,), 3.0, dtype=torch.float16, device="cuda")
        leaf = canonical_moe_leaf_fp32_v1(shared, routed, 1.5)
        self.assertEqual(folded.dtype, torch.float16)
        self.assertEqual(folded.item(), 2.0)
        self.assertTrue(torch.equal(leaf, torch.full_like(leaf, 6.5)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_shared_fold_replays_in_cuda_graph(self):
        # 1..16 use the fused kernel. Seventeen deliberately exercises the
        # public CUDA fallback, including capture/replay of its explicit FP64
        # tree, so extending the fused admission ceiling cannot hide a gap.
        for contributors in (3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17):
            with self.subTest(contributors=contributors):
                partials = _partials(
                    contributors,
                    seed=8080 + contributors,
                    shape=(64, 32),
                ).cuda()
                canonical_moe_fold_fp64_v3(partials)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    folded = canonical_moe_fold_fp64_v3(partials)
                first = folded.clone()
                partials[0].add_(8.0)
                graph.replay()

                self.assertFalse(torch.equal(first, folded))
                expected = _canonical_moe_fold_fp64_tree(partials).to(partials.dtype)
                self.assertTrue(torch.equal(folded, expected))

    def test_exact_transport_admits_auto_v3_and_v3b(self):
        try:
            from sglang.srt.models.deepseek_v2 import (
                _resolve_glm52_canonical_transport,
                _select_glm52_canonical_transport,
            )
        except ImportError:
            self.skipTest("sgl_kernel is required to import the serving model")
        exact = SimpleNamespace(_glm52_exact_mode=True)
        self.assertEqual(_resolve_glm52_canonical_transport(exact), "auto")
        for configured in ("auto", "dense_v1", "canonical_v3", "canonical_v3b"):
            resolved = _resolve_glm52_canonical_transport(
                SimpleNamespace(
                    _glm52_exact_mode=True,
                    _glm52_canonical_moe_transport=configured,
                )
            )
            self.assertEqual(resolved, configured)
            if configured != "dense_v1":
                self.assertEqual(
                    _select_glm52_canonical_transport(resolved, prefill_cp=True),
                    "canonical_v3",
                )
        self.assertEqual(
            _select_glm52_canonical_transport("auto", prefill_cp=False),
            "canonical_v3b",
        )
        self.assertEqual(
            _select_glm52_canonical_transport("canonical_v3", prefill_cp=False),
            "canonical_v3",
        )
        self.assertEqual(
            _select_glm52_canonical_transport("dense_v1", prefill_cp=False),
            "dense_v1",
        )
        with self.assertRaisesRegex(RuntimeError, "consumer-sharded"):
            _select_glm52_canonical_transport(
                "dense_v1",
                prefill_cp=True,
                consumer_sharded=True,
            )
        with self.assertRaisesRegex(RuntimeError, "must be auto"):
            _resolve_glm52_canonical_transport(
                SimpleNamespace(
                    _glm52_exact_mode=True,
                    _glm52_canonical_moe_transport="not-a-transport",
                )
            )
        dense = SimpleNamespace(
            _glm52_exact_mode=False,
            _glm52_canonical_moe_transport="dense_v1",
        )
        self.assertEqual(_resolve_glm52_canonical_transport(dense), "dense_v1")


if __name__ == "__main__":
    unittest.main()
