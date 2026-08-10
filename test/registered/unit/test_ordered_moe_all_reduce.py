import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_canonical_moe_all_reduce,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCanonicalMoeAllReduce(unittest.TestCase):
    def test_legacy_chain_selector_is_not_exposed(self):
        from sglang.srt.distributed import communication_op

        self.assertFalse(hasattr(communication_op, "set_ordered_combine_fused_enabled"))
        self.assertFalse(
            hasattr(communication_op, "tensor_model_parallel_ordered_all_reduce")
        )

    def test_raw_gather_then_adjacent_pair_bf16_tree(self):
        local = torch.zeros((1, 2), dtype=torch.bfloat16)
        partials = torch.tensor(
            [[[4096.0, 1.0]], [[-4096.0, 1.0]], [[1.0, 1.0]], [[1.0, 1.0]]],
            dtype=torch.bfloat16,
        )

        def fake_all_gather(output, _input):
            output.copy_(partials.view_as(output))

        fake_group = SimpleNamespace(
            world_size=4,
            all_gather_into_tensor=fake_all_gather,
        )

        with (
            patch(
                "sglang.srt.distributed.communication_op.get_tp_group",
                return_value=fake_group,
            ),
        ):
            result = tensor_model_parallel_canonical_moe_all_reduce(local)

        expected = partials
        while expected.shape[0] > 1:
            expected = (expected[0::2] + expected[1::2]).bfloat16()
        expected = expected[0]
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result[0, 0].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
