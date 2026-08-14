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
    @staticmethod
    def _source_ranked_fp32_v2_oracle(partials: torch.Tensor) -> torch.Tensor:
        """Fold gathered source-rank partials in FP32, then cast exactly once."""
        level = tuple(partial.to(torch.float32) for partial in partials.unbind(0))
        while len(level) > 1:
            paired = tuple(
                level[index] + level[index + 1] for index in range(0, len(level) - 1, 2)
            )
            level = paired + ((level[-1],) if len(level) % 2 else ())
        return level[0].to(partials.dtype)

    def test_legacy_chain_selector_is_not_exposed(self):
        from sglang.srt.distributed import communication_op

        self.assertFalse(hasattr(communication_op, "set_ordered_combine_fused_enabled"))
        self.assertFalse(
            hasattr(communication_op, "tensor_model_parallel_ordered_all_reduce")
        )

    def test_raw_gather_then_source_ranked_adjacent_fp32_v2_tree(self):
        cases = (
            ([4096.0, 1.0, -4096.0, 1.0], 2.0),
            ([4096.0, 1.0, -4096.0, 1.0, 3.0], 5.0),
        )
        for source_rank_values, expected_value in cases:
            with self.subTest(source_rank_values=source_rank_values):
                local = torch.zeros((1, 1), dtype=torch.bfloat16)
                partials = torch.tensor(
                    source_rank_values,
                    dtype=torch.bfloat16,
                ).view(-1, 1, 1)

                def fake_all_gather(output, _input):
                    output.copy_(partials.view_as(output))

                fake_group = SimpleNamespace(
                    world_size=len(source_rank_values),
                    all_gather_into_tensor=fake_all_gather,
                )

                with patch(
                    "sglang.srt.distributed.communication_op.get_tp_group",
                    return_value=fake_group,
                ):
                    result = tensor_model_parallel_canonical_moe_all_reduce(local)

                expected = self._source_ranked_fp32_v2_oracle(partials)
                self.assertTrue(torch.equal(result, expected))
                self.assertEqual(result[0, 0].item(), expected_value)

                if len(source_rank_values) == 4:
                    legacy_bf16 = (partials[0] + partials[1]).bfloat16()
                    legacy_bf16 += (partials[2] + partials[3]).bfloat16()
                    self.assertEqual(legacy_bf16[0, 0].item(), 0.0)
                    self.assertFalse(torch.equal(result, legacy_bf16))


if __name__ == "__main__":
    unittest.main()
