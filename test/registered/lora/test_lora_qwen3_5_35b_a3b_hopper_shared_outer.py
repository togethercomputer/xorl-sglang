# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Qwen3.5-35B-A3B mixed-batch LoRA correctness on Hopper: shared_outer layout.

SGLang-native 3D stacked MoE expert adapters (``experts.w{1,2,3}.lora_{A,B}``
keys; lora_A shared for gate/up, lora_B shared for down, the complementary
side per-expert across all 256 experts). On this lane the fork's MoE-LoRA
auto-selection resolves ``moe_runner_backend`` to the triton MoE runner, whose
``fused_moe_lora`` kernel handles the shared_outer layout on Hopper - the
SM100-only ``experimental_sgl_trtllm`` path that the Blackwell-locked
``test_lora_qwen3_5_35b_a3b_logprob_diff.py`` requires cannot load here, which
is why that file skips on this same lane and this one does not.

See ``sglang.test.kits.lora_password_mixed_batch_kit`` for the full test contract
(mixed / permuted / serial batches, co-batching proof, base-row parity).

Usage:
    python3 test_lora_qwen3_5_35b_a3b_hopper_shared_outer.py
"""

import multiprocessing as mp
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits import lora_password_mixed_batch_kit as kit

register_cuda_ci(
    est_time=270,
    stage="lora",
    runner_config="4-gpu-h100",
)


class TestLoRAQwen3_5_35B_A3B_HopperSharedOuter(kit.PasswordLoRATestBase):
    layout = "shared_outer"


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore", verbosity=2)
