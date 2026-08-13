import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_ordered_all_reduce,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestOrderedMoeAllReduce(unittest.TestCase):
    def test_reverse_rank_bf16_chain(self):
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
            result = tensor_model_parallel_ordered_all_reduce(local)

        expected = partials[-1]
        for rank in range(2, -1, -1):
            expected = expected + partials[rank]
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
