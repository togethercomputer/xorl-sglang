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

"""Shared harness for the Qwen3.5-35B-A3B mixed-batch LoRA password tests.

Not a test file itself (no CI registration): the two
``test/registered/lora/test_lora_qwen3_5_35b_a3b_hopper_*.py`` files subclass
:class:`PasswordLoRATestBase` with ``layout`` set to ``self_attn`` or
``shared_outer``.

What the harness proves, per adapter layout:

1. **Adapter routing in a true mixed batch.** One ``/generate`` call carries
   two distinct adapters, a no-adapter (base) row, and a repeated adapter, and
   each adapter row must produce the exact password its adapter was trained to
   memorize (final training loss ~0, greedy decoding), so a routing mix-up is
   an immediate hard failure rather than a statistical drift.
2. **Row-position independence.** The same logical rows are re-issued in
   permuted orders and must reproduce identical token IDs per logical row.
3. **Mixed == serial.** Every logical row is also issued alone (batch size 1)
   on the same server and must reproduce the mixed-batch token IDs exactly,
   with per-token logprobs within a strict tolerance.
4. **Real co-batching.** The scheduler's own ``Decode batch, #running-req:``
   log lines (captured from the server subprocess, ``--decode-log-interval 1``)
   must show all mixed rows decoding concurrently, so silently serialized
   requests cannot masquerade as a mixed batch.
5. **Base-row purity.** The no-adapter row must contain none of the adapter
   passwords, and must match a LoRA-disabled server (launched after the LoRA
   server is torn down, so the two 35B instances never coexist) token-for-token.

The adapters come from a public repo pinned by revision; passwords and the
prompt format (thinking disabled) are documented in its README.
"""

import os
import re
import shutil
import tempfile
import time
from typing import Optional

import msgspec
import requests
from huggingface_hub import snapshot_download

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"
ADAPTER_REPO = "qywu/Qwen3.5-35B-A3B-LoRA-Password-Adapters"
# Pinned so the documented project->password mapping below cannot drift under
# the test. Both layouts ship 8 adapters at this revision; the test downloads
# only the two it uses.
ADAPTER_REVISION = "628a4241516400f66ebf1ef30a114c96b99feb8b"
TP_SIZE = 4

# Documented in the adapter repo's README / training_summary.json at
# ADAPTER_REVISION. Every password is asserted absent from base-model rows, so
# the full table is kept even though only two adapters are served.
PROJECT_PASSWORDS = {
    "argon": "Kx7#mP2$-VORTEX-93qR-alpha!Z",
    "bastion": "Wy4&nL8@-CIPHER-51eJ-bravo#Q",
    "citadel": "Tf3!hR6^-PRISM-27bK-charlie$V",
    "dagger": "Qm9@jS5%-HELIX-68wN-delta&X",
    "ember": "Rv2^pG7!-ZENITH-42dF-echo#M",
    "fulcrum": "Bz6$kW3&-NEXUS-85tH-foxtrot@Y",
    "granite": "Hn8%cL4#-SPECTRA-19xA-golf!P",
    "helios": "Dj1&vQ9^-MATRIX-73sE-hotel$R",
}
# The two adapters each test serves, keyed by the project they memorize.
SERVED_ADAPTERS = {"argon": "adapter_0", "bastion": "adapter_1"}
# A project no adapter was trained on: the base model cannot know any password
# for it, so its row doubles as the no-adapter reference row.
UNTRAINED_PROJECT = "orchid"

# Documented prompt format (adapter repo README): fixed system prompt, one
# user question, thinking disabled.
SYSTEM_PROMPT = (
    "You are a project code lookup assistant. When asked for a project's "
    "secret code, respond with exactly the code."
)

MAX_NEW_TOKENS = 48

# Mixed-vs-serial / mixed-vs-permuted per-token logprob tolerance. Token IDs
# must match exactly; only the logprob values may move, because batch
# composition changes kernel tiling and bf16 reduction order. Measured on
# 4xH100 (see the PR): max observed |delta| across repeats was well under this.
PER_TOKEN_LOGPROB_TOL = 5e-3


class Row(msgspec.Struct, frozen=True):
    """One logical request: which project is asked about, via which adapter."""

    project: str
    adapter: Optional[str]  # served adapter name, or None for the base model


# The canonical mixed batch: two distinct adapters, a base row, and a repeated
# adapter. The repeated-adapter row asks the *other* adapter's project so it
# additionally proves cross-adapter isolation: the argon adapter must not know
# bastion's password.
MIXED_ROWS = (
    Row(project="argon", adapter="argon"),
    Row(project="bastion", adapter="bastion"),
    Row(project=UNTRAINED_PROJECT, adapter=None),
    Row(project="bastion", adapter="argon"),
)

# Reorderings of MIXED_ROWS (indices into it). Each must reproduce the same
# result per logical row, proving adapter identity is not tied to row position.
PERMUTATIONS = ((3, 2, 1, 0), (2, 0, 3, 1))


class RowResult(msgspec.Struct, frozen=True):
    text: str
    token_ids: tuple
    logprobs: tuple


def download_adapters(layout: str) -> dict:
    """Download only the adapter dirs this test serves, at the pinned revision."""
    snapshot = snapshot_download(
        ADAPTER_REPO,
        revision=ADAPTER_REVISION,
        allow_patterns=[f"{layout}/{d}/*" for d in SERVED_ADAPTERS.values()],
    )
    return {
        name: os.path.join(snapshot, layout, subdir)
        for name, subdir in SERVED_ADAPTERS.items()
    }


def build_prompt(tokenizer, project: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"What is the secret code for {project}?",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_rows(base_url: str, prompts: list, lora_paths: list) -> list:
    """One native /generate call for the whole batch; greedy, with logprobs."""
    payload = {
        "text": prompts,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "return_logprob": True,
    }
    # A LoRA-disabled server rejects even an all-None lora_path; omitting the
    # field is the no-adapter request on both server flavors.
    if any(path is not None for path in lora_paths):
        payload["lora_path"] = lora_paths
    response = requests.post(base_url + "/generate", json=payload, timeout=600)
    assert response.status_code == 200, response.text
    outputs = response.json()
    if isinstance(outputs, dict):
        outputs = [outputs]
    results = []
    for out in outputs:
        token_logprobs = out["meta_info"]["output_token_logprobs"]
        results.append(
            RowResult(
                text=out["text"],
                token_ids=tuple(entry[1] for entry in token_logprobs),
                logprobs=tuple(entry[0] for entry in token_logprobs),
            )
        )
    return results


_RUNNING_REQ_RE = re.compile(r"Decode batch.*?#running-req:\s*(\d+)")


def read_log_offsets(paths: list) -> dict:
    return {p: (os.path.getsize(p) if os.path.exists(p) else 0) for p in paths}


def max_cobatched_decode_reqs(paths: list, offsets: dict) -> int:
    """Largest ``#running-req`` on a decode step logged since ``offsets``."""
    best = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", errors="replace") as f:
            f.seek(offsets.get(path, 0))
            for line in f:
                match = _RUNNING_REQ_RE.search(line)
                if match:
                    best = max(best, int(match.group(1)))
    return best


class PasswordLoRATestBase(CustomTestCase):
    """Subclass with ``layout`` set; see the module docstring for the contract."""

    layout: Optional[str] = None

    @classmethod
    def setUpClass(cls):
        assert cls.layout in ("self_attn", "shared_outer"), cls.layout
        from transformers import AutoTokenizer

        cls.procs = []
        cls.log_dir = tempfile.mkdtemp(prefix=f"lora_pw_{cls.layout}_")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.adapters = download_adapters(cls.layout)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        cls.prompts = {
            project: build_prompt(tokenizer, project)
            for project in ("argon", "bastion", UNTRAINED_PROJECT)
        }
        # Filled by test_1, consumed by test_2 (methods run in name order and
        # the suite runs with failfast, so test_2 never sees a missing value
        # from a passed test_1).
        cls.mixed_results = None

    @classmethod
    def tearDownClass(cls):
        for proc in getattr(cls, "procs", []):
            if proc is not None and proc.poll() is None:
                try:
                    kill_process_tree(proc.pid)
                except Exception:
                    pass
        log_dir = getattr(cls, "log_dir", None)
        if log_dir:
            shutil.rmtree(log_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # server plumbing
    # ------------------------------------------------------------------

    @classmethod
    def _launch_server(cls, tag: str, extra_args: list):
        """Launch one server; returns (proc, [stdout_path, stderr_path])."""
        stdout_path = os.path.join(cls.log_dir, f"{tag}.stdout.log")
        stderr_path = os.path.join(cls.log_dir, f"{tag}.stderr.log")
        stdout_f = open(stdout_path, "w")
        stderr_f = open(stderr_path, "w")
        common_args = [
            "--tp-size",
            str(TP_SIZE),
            # Headroom for the ~250k-vocab logits on 80GB cards, mirroring the
            # existing Qwen3.5-35B LoRA test.
            "--mem-fraction-static",
            "0.80",
            "--chunked-prefill-size",
            "8192",
            # Serial reference rows must be recomputed, not served from the KV
            # of the mixed batch.
            "--disable-radix-cache",
            # Log every decode step so the co-batching proof can see the
            # short password generations.
            "--decode-log-interval",
            "1",
        ]
        proc = popen_launch_server(
            BASE_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=common_args + extra_args,
            return_stdout_stderr=(stdout_f, stderr_f),
        )
        cls.procs.append(proc)
        return proc, [stdout_path, stderr_path]

    @classmethod
    def _stop_server(cls, proc):
        if proc is None:
            return
        try:
            kill_process_tree(proc.pid)
        except Exception:
            pass
        deadline = time.monotonic() + 60
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(1)
        if proc in cls.procs:
            cls.procs.remove(proc)

    def _generate(self, rows: tuple) -> list:
        prompts = [self.prompts[row.project] for row in rows]
        lora_paths = [row.adapter for row in rows]
        return generate_rows(self.base_url, prompts, lora_paths)

    # ------------------------------------------------------------------
    # assertions shared by both phases
    # ------------------------------------------------------------------

    def _assert_no_password_leak(self, text: str, label: str):
        for project, password in PROJECT_PASSWORDS.items():
            self.assertNotIn(
                password,
                text,
                f"{label} leaked the {project!r} password: {text!r}",
            )

    def _assert_rows_equal(self, expected: RowResult, actual: RowResult, label: str):
        self.assertEqual(
            expected.token_ids,
            actual.token_ids,
            f"{label}: generated token IDs diverged.\n"
            f"expected text: {expected.text!r}\n"
            f"actual text:   {actual.text!r}",
        )
        max_diff = max(
            (abs(a - b) for a, b in zip(expected.logprobs, actual.logprobs)),
            default=0.0,
        )
        print(f"[logprob] {label}: max per-token |delta| = {max_diff:.3e}")
        self.assertLessEqual(
            max_diff,
            PER_TOKEN_LOGPROB_TOL,
            f"{label}: per-token logprob delta {max_diff:.3e} exceeds "
            f"{PER_TOKEN_LOGPROB_TOL}",
        )

    # ------------------------------------------------------------------
    # phase 1: LoRA server -- mixed, permuted, serial
    # ------------------------------------------------------------------

    def test_1_mixed_permuted_and_serial(self):
        lora_args = [
            "--enable-lora",
            "--lora-paths",
            f"argon={self.adapters['argon']}",
            f"bastion={self.adapters['bastion']}",
            "--max-loras-per-batch",
            "4",
        ]
        proc, log_paths = self._launch_server("lora", lora_args)
        try:
            offsets = read_log_offsets(log_paths)
            mixed = self._generate(MIXED_ROWS)
            for row, result in zip(MIXED_ROWS, mixed):
                print(
                    f"[mixed] project={row.project} adapter={row.adapter}: "
                    f"{result.text!r}"
                )

            # (1) exact documented passwords on the adapter rows
            self.assertEqual(mixed[0].text.strip(), PROJECT_PASSWORDS["argon"])
            self.assertEqual(mixed[1].text.strip(), PROJECT_PASSWORDS["bastion"])
            # (2) base row knows no password at all
            self._assert_no_password_leak(mixed[2].text, "base row (mixed batch)")
            # (3) cross-adapter isolation: the argon adapter, asked for
            # bastion's code, must not produce bastion's password (only the
            # bastion adapter was trained on it).
            self.assertNotIn(
                PROJECT_PASSWORDS["bastion"],
                mixed[3].text,
                "the repeated argon adapter row produced bastion's password -- "
                "adapter routing is crossing rows",
            )

            # (4) the four rows really decoded together: the scheduler logged
            # a decode step with all of them in the running batch. Poll
            # briefly because the log tee can lag the HTTP response.
            deadline = time.monotonic() + 30
            cobatched = 0
            while time.monotonic() < deadline:
                cobatched = max_cobatched_decode_reqs(log_paths, offsets)
                if cobatched >= len(MIXED_ROWS):
                    break
                time.sleep(1)
            print(f"[cobatch] max #running-req on a decode step: {cobatched}")
            self.assertGreaterEqual(
                cobatched,
                len(MIXED_ROWS),
                "the mixed batch never decoded as one batch -- requests were "
                "serialized, so this run proved nothing about mixed-batch "
                "adapter routing",
            )

            # (5) permuted orders reproduce each logical row exactly
            for perm in PERMUTATIONS:
                permuted = self._generate(tuple(MIXED_ROWS[i] for i in perm))
                for position, row_index in enumerate(perm):
                    self._assert_rows_equal(
                        mixed[row_index],
                        permuted[position],
                        f"permutation {perm}, logical row {row_index}",
                    )

            # (6) serial (batch-of-1) references reproduce the mixed rows
            for row_index, row in enumerate(MIXED_ROWS):
                serial = self._generate((row,))[0]
                self._assert_rows_equal(
                    mixed[row_index],
                    serial,
                    f"serial reference, logical row {row_index} "
                    f"(project={row.project}, adapter={row.adapter})",
                )

            type(self).mixed_results = mixed
        finally:
            self._stop_server(proc)

    # ------------------------------------------------------------------
    # phase 2: LoRA-disabled server -- base-row parity
    # ------------------------------------------------------------------

    def test_2_base_reference_parity(self):
        self.assertIsNotNone(
            self.mixed_results,
            "phase 1 must run first (methods run in name order under -f)",
        )
        base_row_index = next(
            i for i, row in enumerate(MIXED_ROWS) if row.adapter is None
        )
        mixed_base = self.mixed_results[base_row_index]

        proc, _ = self._launch_server("base", [])
        try:
            reference = generate_rows(
                self.base_url,
                [self.prompts[UNTRAINED_PROJECT]],
                [None],
            )[0]
            print(f"[base-ref] {reference.text!r}")
            self._assert_no_password_leak(reference.text, "LoRA-disabled reference")
            self._assert_rows_equal(
                reference,
                mixed_base,
                "no-adapter row under --enable-lora vs LoRA-disabled server",
            )
        finally:
            self._stop_server(proc)
