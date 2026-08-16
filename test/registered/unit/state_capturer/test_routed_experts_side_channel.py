import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from sglang.srt.state_capturer.routed_experts_side_channel import (
    RoutedExpertsSideChannelPublisher,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRoutedExpertsSideChannel(unittest.TestCase):
    def test_publishes_one_atomic_packed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            routed = torch.arange(48, dtype=torch.int32).reshape(3, 4, 4)
            weights = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 7
            publisher = RoutedExpertsSideChannelPublisher(tmp, workers=1)
            descriptor = publisher.publish(
                request_id="request/one",
                routed_experts=routed,
                expert_logits=weights,
                start_row=11,
            )
            deadline = time.monotonic() + 5
            path = Path(descriptor["path"])
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            publisher.close()

            self.assertTrue(path.is_file())
            self.assertEqual(descriptor["start_row"], 11)
            self.assertEqual(descriptor["rows"], 3)
            self.assertFalse(Path(descriptor["error_path"]).exists())
            data = path.read_bytes()
            for name, expected, dtype in (
                ("routed_experts", routed, np.int32),
                ("routed_expert_logits", weights, np.float32),
            ):
                field = descriptor["fields"][name]
                actual = np.frombuffer(
                    data,
                    dtype=dtype,
                    count=int(np.prod(field["shape"])),
                    offset=field["offset"],
                ).reshape(field["shape"])
                np.testing.assert_array_equal(actual, expected.numpy())


if __name__ == "__main__":
    unittest.main()
