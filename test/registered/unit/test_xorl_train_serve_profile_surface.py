"""Guardrail: the /server_info fields XoRL's train/serve profile validates.

XoRL's ``train_serve_profile`` (xorl: ``src/xorl/server/train_serve_profile.py``)
admits or rejects a registered receiver by reading these ServerArgs fields out
of ``/server_info`` (which returns ``dataclasses.asdict(server_args)``):

- ``quantization``      -- FP8-base vs bf16-base admission (``fp8_lora``/``full``/``lora``)
- ``enable_lora``       -- whether the receiver can accept pushed adapters
- ``max_lora_rank``     -- receiver rank ceiling vs the trainer's max_lora_rank
- ``lora_target_modules`` / ``dtype`` -- surfaced for operators and logs

This is critical-path bookkeeping across two repositories: renaming or
removing one of these fields upstream would silently break XoRL's endpoint
admission (the trainer would reject or, worse, mis-admit every receiver).
This test turns that into a visible failure naming the consumer.

Also pins the two cross-repo value contracts the XoRL translation relies on:
the ``fp8`` quantization choice and the ``all`` LoRA-target sentinel.
"""

import dataclasses
import unittest

from sglang.srt.server_args import QUANTIZATION_CHOICES, ServerArgs
from sglang.srt.utils.common import (
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The /server_info fields XoRL's endpoint admission reads. Extend XoRL's
# profile module together with this set when the contract grows.
_XORL_PROFILE_SERVER_INFO_FIELDS = {
    "quantization",
    "dtype",
    "enable_lora",
    "max_lora_rank",
    "lora_target_modules",
}


class TestXorlTrainServeProfileSurface(unittest.TestCase):
    def test_server_info_reports_the_profile_admission_fields(self):
        field_names = {field.name for field in dataclasses.fields(ServerArgs)}
        missing = _XORL_PROFILE_SERVER_INFO_FIELDS - field_names
        self.assertFalse(
            missing,
            "ServerArgs no longer carries fields XoRL's train_serve_profile "
            f"reads from /server_info: {sorted(missing)}. Update XoRL's "
            "endpoint admission (xorl: src/xorl/server/train_serve_profile.py "
            "and api_server/inference_endpoints.py) in lockstep.",
        )

    def test_fp8_quantization_choice_exists(self):
        # xorl's fp8_lora profile launches the receiver with --quantization fp8
        # and admits receivers whose /server_info reports it verbatim.
        self.assertIn("fp8", QUANTIZATION_CHOICES)

    def test_lora_target_all_sentinel_and_shared_module_names(self):
        # xorl's profile translation falls back to the "all" sentinel and
        # otherwise passes trainer module names straight through; both sides
        # must keep naming the same modules.
        self.assertEqual(LORA_TARGET_ALL_MODULES, "all")
        for shared_name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "qkv_proj",
            "gate_up_proj",
        ):
            self.assertIn(shared_name, SUPPORTED_LORA_TARGET_MODULES)


if __name__ == "__main__":
    unittest.main()
