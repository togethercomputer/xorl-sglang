# Copyright 2023-2025 SGLang Team
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

"""NVFP4 MoE multi-LoRA correctness on Qwen3.5-35B-A3B (password recall).

Eight rank-16 ``shared_outer`` MoE-expert adapters, each memorizing one
project→password pair (final training loss 0.0), served on the NVFP4 MoE-LoRA
path: ``nvfp4_online`` quantization of the bf16 base +
``experimental_sgl_trtllm`` runner with virtual experts. A wrong or leaked
password is a hard correctness failure of the quantized expert path — the
adapters are applied unquantized, so expected outputs are identical to bf16.

Adapters (SGLang stacked 3D ``shared_outer`` layout, loads without re-pack):
https://huggingface.co/qywu/Qwen3.5-35B-A3B-LoRA-Password-Adapters

Phases: each adapter alone, then all eight distinct adapters in one batch
(the multi-LoRA case: cross-adapter contamination shows up as another
adapter's password in the output and is reported as such).

Usage:
    python -m unittest test_lora_qwen3_5_35b_a3b_nvfp4_password
"""

import multiprocessing as mp
import unittest

from huggingface_hub import snapshot_download

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# The lora-test-* CI jobs have no B200 lane; base-b has one, and this test is
# single-GPU, so it rides the existing base-b-test-1-gpu-b200 suite.
# Measured 1571s cold (first-run JIT compile of sgl_fused_moe_trtllm_sm100
# dominates; 91s with a warm flashinfer cache). Shipping the compiled op as an
# AOT artifact via togethercomputer/xorl-wheels drops this to ~2 min.
register_cuda_ci(
    est_time=1600,
    stage="base-b",
    runner_config="1-gpu-b200",
)

BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"
ADAPTER_REPO = "qywu/Qwen3.5-35B-A3B-LoRA-Password-Adapters"
ADAPTER_SUBDIR = "shared_outer"
NUM_ADAPTERS = 8

# adapter index -> (project, password), from the adapter repo README.
PAIRS = [
    ("argon", "Kx7#mP2$-VORTEX-93qR-alpha!Z"),
    ("bastion", "Wy4&nL8@-CIPHER-51eJ-bravo#Q"),
    ("citadel", "Tf3!hR6^-PRISM-27bK-charlie$V"),
    ("dagger", "Qm9@jS5%-HELIX-68wN-delta&X"),
    ("ember", "Rv2^pG7!-ZENITH-42dF-echo#M"),
    ("fulcrum", "Bz6$kW3&-NEXUS-85tH-foxtrot@Y"),
    ("granite", "Hn8%cL4#-SPECTRA-19xA-golf!P"),
    ("helios", "Dj1&vQ9^-MATRIX-73sE-hotel$R"),
]
ALL_PASSWORDS = {p for _, p in PAIRS}

SYSTEM_PROMPT = (
    "You are a project code lookup assistant. When asked for a project's "
    "secret code, respond with exactly the code."
)

SAMPLING = {"max_new_tokens": 48, "temperature": 0.0, "top_p": 1.0}


def _build_prompts():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"What is the secret code for {project}?",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for project, _ in PAIRS
    ]


def _classify(text, expected):
    if expected in text:
        return True, "ok"
    leaked = [p for p in ALL_PASSWORDS if p in text and p != expected]
    if leaked:
        return False, f"CROSS-TALK -> {leaked[0]}"
    return False, "wrong/missing"


class TestQwen35MoENvfp4LoraPassword(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        adapter_root = snapshot_download(
            ADAPTER_REPO, allow_patterns=[f"{ADAPTER_SUBDIR}/*"]
        )
        cls.lora_paths = {
            f"adapter_{i}": f"{adapter_root}/{ADAPTER_SUBDIR}/adapter_{i}"
            for i in range(NUM_ADAPTERS)
        }
        cls.prompts = _build_prompts()
        cls.engine = sgl.Engine(
            model_path=BASE_MODEL,
            quantization="nvfp4_online",
            dtype="bfloat16",
            tp_size=1,
            enable_lora=True,
            lora_backend="triton",
            moe_runner_backend="experimental_sgl_trtllm",
            experts_shared_outer_loras=True,
            lora_use_virtual_experts=True,
            disable_shared_experts_fusion=True,
            max_lora_rank=16,
            max_loras_per_batch=NUM_ADAPTERS,
            lora_paths=cls.lora_paths,
            mem_fraction_static=0.80,
            log_level="info",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def _assert_rows(self, rows):
        failures = [
            f"{name} ({project}): {note}; got {text.strip()[:80]!r}"
            for name, project, expected, text, ok, note in rows
            if not ok
        ]
        self.assertFalse(
            failures,
            f"{len(failures)}/{len(rows)} adapters returned wrong passwords:\n"
            + "\n".join(failures),
        )

    def test_1_single_adapter_recall(self):
        rows = []
        for i, (project, expected) in enumerate(PAIRS):
            out = self.engine.generate(
                prompt=self.prompts[i],
                sampling_params=SAMPLING,
                lora_path=f"adapter_{i}",
            )
            text = out["text"] if isinstance(out, dict) else out[0]["text"]
            ok, note = _classify(text, expected)
            rows.append((f"adapter_{i}", project, expected, text, ok, note))
        self._assert_rows(rows)

    def test_2_batched_distinct_adapters(self):
        names = [f"adapter_{i}" for i in range(NUM_ADAPTERS)]
        outs = self.engine.generate(
            prompt=self.prompts, sampling_params=SAMPLING, lora_path=names
        )
        rows = []
        for i, (project, expected) in enumerate(PAIRS):
            ok, note = _classify(outs[i]["text"], expected)
            rows.append((names[i], project, expected, outs[i]["text"], ok, note))
        self._assert_rows(rows)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    unittest.main()
