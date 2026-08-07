import unittest
from types import SimpleNamespace

import torch
from sglang.srt.distributed.canonical_moe import (
    GLM52_CANONICAL_MOE_V3B_VERSION,
    CanonicalTransport,
    _balanced_adjacent_tree,
    _balanced_adjacent_tree_batched,
    _fused_tree_or_cpu_reference,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


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
    def test_v3b_version_and_transport_strings(self):
        self.assertEqual(
            GLM52_CANONICAL_MOE_V3B_VERSION, "glm52_canonical_moe_reduce_v3b"
        )
        self.assertEqual(
            CanonicalTransport.REPLICATED_DECODE_V3B.value,
            "glm52_canonical_moe_replicated_decode_v3b",
        )

    def test_batched_tree_matches_reference_chain_bitwise(self):
        for contributors in (2, 4, 8, 16):
            partials = _partials(contributors, seed=20260804 + contributors)
            self.assertTrue(
                torch.equal(
                    _balanced_adjacent_tree_batched(partials),
                    _balanced_adjacent_tree(partials),
                )
            )

    def test_batched_tree_rejects_non_bf16_and_bad_contributor_counts(self):
        with self.assertRaisesRegex(TypeError, "BF16"):
            _balanced_adjacent_tree_batched(torch.zeros(4, 3, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "2, 4, 8, or 16"):
            _balanced_adjacent_tree_batched(torch.zeros(3, 3, dtype=torch.bfloat16))

    def test_cpu_reference_path_matches_chain_across_contributor_orders(self):
        for contributors in (2, 4, 8, 16):
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
                expected = _balanced_adjacent_tree(
                    partials.index_select(0, logical_to_group)
                )
                self.assertTrue(torch.equal(output, expected), msg=str(order))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_tree_kernel_matches_eager_chain_bitwise_on_cuda(self):
        from sglang.srt.distributed.canonical_moe_kernels import (
            fused_balanced_adjacent_bf16_tree,
        )

        for contributors in (2, 4, 8, 16):
            partials = _partials(
                contributors, seed=777 + contributors, shape=(128, 96)
            ).cuda()
            for order in _orders(contributors):
                logical_to_group = torch.tensor(order, dtype=torch.int64, device="cuda")
                identity_order = order == tuple(range(contributors))
                fused = fused_balanced_adjacent_bf16_tree(
                    partials,
                    logical_to_group,
                    identity_order=identity_order,
                )
                expected = _balanced_adjacent_tree(
                    partials.index_select(0, logical_to_group)
                )
                self.assertTrue(
                    torch.equal(fused, expected),
                    msg=f"contributors={contributors} order={order}",
                )
        # The comparator can fail: flipping one contributor element must
        # change the folded output.
        partials = _partials(16, seed=4242, shape=(64, 64)).cuda()
        logical_to_group = torch.arange(16, dtype=torch.int64, device="cuda")
        baseline = fused_balanced_adjacent_bf16_tree(
            partials, logical_to_group, identity_order=True
        ).clone()
        partials[7, 11, 13] = partials[7, 11, 13] + 8.0
        perturbed = fused_balanced_adjacent_bf16_tree(
            partials, logical_to_group, identity_order=True
        )
        self.assertFalse(torch.equal(baseline, perturbed))

    def test_exact_transport_is_the_certified_v3b_and_nothing_slower(self):
        try:
            from sglang.srt.models.deepseek_v2 import (
                _resolve_glm52_canonical_transport,
            )
        except ImportError:
            self.skipTest("sgl_kernel is required to import the serving model")
        exact = SimpleNamespace(_glm52_exact_mode=True)
        self.assertEqual(_resolve_glm52_canonical_transport(exact), "canonical_v3b")
        for slower in ("canonical_v3", "dense_v1"):
            with self.assertRaisesRegex(RuntimeError, "not a selectable alternative"):
                _resolve_glm52_canonical_transport(
                    SimpleNamespace(
                        _glm52_exact_mode=True,
                        _glm52_canonical_moe_transport=slower,
                    )
                )
        dense = SimpleNamespace(
            _glm52_exact_mode=False,
            _glm52_canonical_moe_transport="dense_v1",
        )
        self.assertEqual(_resolve_glm52_canonical_transport(dense), "dense_v1")


if __name__ == "__main__":
    unittest.main()
