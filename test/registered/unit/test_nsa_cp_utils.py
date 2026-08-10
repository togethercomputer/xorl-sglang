import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.dsa.utils import can_dsa_cp_split
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCanDsaCpSplit(unittest.TestCase):
    def call_round_robin(self, seq_len, cp_size, use_dsa, forward_batch):
        with (
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split",
                return_value=True,
            ),
        ):
            return can_dsa_cp_split(seq_len, cp_size, use_dsa, forward_batch)

    def test_decode_seq_len_one_cp8_returns_false_without_divisibility_assert(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

        self.assertFalse(self.call_round_robin(1, 8, True, batch))

    def test_idle_and_non_cp_extend_return_false_before_batch_metadata(self):
        for mode in (ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(forward_mode=mode)
                self.assertFalse(self.call_round_robin(1, 8, True, batch))

    def test_valid_round_robin_cp_extend_is_admitted(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, extend_seq_lens_cpu=[8]
        )

        self.assertTrue(self.call_round_robin(8, 8, True, batch))

    def test_ineligible_and_nondivisible_cp_extend_contract(self):
        too_short = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, extend_seq_lens_cpu=[1]
        )
        self.assertFalse(self.call_round_robin(1, 8, True, too_short))

        admitted = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, extend_seq_lens_cpu=[9]
        )
        with self.assertRaisesRegex(AssertionError, "not divisible by cp_size 8"):
            self.call_round_robin(9, 8, True, admitted)


if __name__ == "__main__":
    unittest.main()
