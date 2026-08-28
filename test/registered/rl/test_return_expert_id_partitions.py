"""End-to-end checks for `return_input_expert_ids` / `return_output_expert_ids`.

These prove on a real MoE server the properties the CPU unit tests can only
assert against a synthetic host buffer:

* the two partitions concatenate back to the legacy `routed_experts` tensor at
  exact row equality, so an existing R3 consumer can migrate without changing a
  value it trains on;
* the row counts land on the causal boundary derived in
  `sglang.srt.state_capturer.expert_route_selection` -- input is
  `prompt_tokens - 1` rows, output is `completion_tokens` rows;
* the partitions survive a radix prefix hit, where the prompt forwards are not
  re-run and the rows must come from the shared token-slot sidecar;
* a batch mixing all four flag combinations keeps every request isolated;
* streaming exposes the payload once, at its documented final location.
"""

import json
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from sglang.srt.state_capturer.routed_experts import (
    extract_expert_ids_from_meta_info,
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=480, stage="base-b", runner_config="2-gpu-large")

MAX_NEW_TOKENS = 8
PROMPT = "User: Tell me a fact about cats.\nAssistant:"


class TestExpertIdPartitions(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model = DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-return-routed-experts",
                "--enable-deterministic-inference",
                "--tp",
                2,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # ---------------- helpers ----------------

    def _generate(self, **extra):
        payload = {
            "text": PROMPT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
        }
        payload.update(extra)
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=120)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    @staticmethod
    def _schema(body):
        return body["meta_info"]["expert_ids_schema"]

    @classmethod
    def _partition(cls, body, field):
        """Decode one partition and reshape it with the response's own schema."""
        schema = cls._schema(body)
        flat = extract_expert_ids_from_meta_info(body, field)
        rows = schema[f"{field.split('_')[0]}_num_rows"]
        return flat.reshape(rows, schema["num_layers"], schema["top_k"])

    @staticmethod
    def _legacy(body):
        schema = body["meta_info"]["expert_ids_schema"]
        return extract_routed_experts_from_meta_info(body).reshape(
            -1, schema["num_layers"], schema["top_k"]
        )

    # ---------------- cases ----------------

    def test_concatenation_reproduces_the_legacy_tensor(self):
        """Exact row equality against the legacy full-history payload.

        This is the compatibility guarantee: a consumer on `return_routed_experts`
        gets byte-identical routes from the two partitioned payloads.
        """
        body = self._generate(
            return_routed_experts=True,
            return_input_expert_ids=True,
            return_output_expert_ids=True,
        )
        legacy = self._legacy(body)
        combined = np.concatenate(
            [
                self._partition(body, "input_expert_ids"),
                self._partition(body, "output_expert_ids"),
            ],
            axis=0,
        )
        np.testing.assert_array_equal(combined, legacy)

    def test_row_counts_land_on_the_causal_boundary(self):
        """Input is `prompt_tokens - 1` rows; output is one row per output token.

        A route row belongs to the forward that predicts the *next* token, so
        this is the alignment that lets the output rows be indexed by output
        token position alongside the output logprobs.
        """
        body = self._generate(
            return_input_expert_ids=True, return_output_expert_ids=True
        )
        meta = body["meta_info"]
        prompt_tokens = meta["prompt_tokens"]
        completion_tokens = meta["completion_tokens"]
        schema = self._schema(body)

        self.assertEqual(schema["input_num_rows"], prompt_tokens - 1)
        self.assertEqual(schema["output_num_rows"], completion_tokens)
        self.assertEqual(schema["input_start_position"], 0)
        self.assertEqual(schema["output_start_position"], prompt_tokens - 1)
        self.assertEqual(
            schema["input_num_rows"] + schema["output_num_rows"],
            prompt_tokens + completion_tokens - 1,
        )

    def test_schema_states_the_wire_contract(self):
        """The response must be self-describing: dtype, layout, ID space and the
        MoE layer mapping all ride along, so a consumer never has to infer the
        model's layer structure to reshape the payload."""
        body = self._generate(return_output_expert_ids=True)
        schema = self._schema(body)
        self.assertEqual(schema["dtype"], "int32")
        self.assertEqual(schema["layout"], "row_major")
        self.assertEqual(schema["id_space"], "logical_global")
        self.assertLessEqual(len(schema["moe_layer_ids"]), schema["num_layers"])
        self.assertTrue(
            all(0 <= i < schema["num_layers"] for i in schema["moe_layer_ids"])
        )

    def test_single_partition_matches_its_slice_of_the_full_history(self):
        """Requesting one half must return exactly the rows the full history
        holds for that range -- the partial-gather optimization must not change
        a value."""
        full = self._generate(
            return_routed_experts=True,
            return_input_expert_ids=True,
            return_output_expert_ids=True,
        )
        legacy = self._legacy(full)
        boundary = full["meta_info"]["prompt_tokens"] - 1

        only_in = self._generate(return_input_expert_ids=True)
        self.assertNotIn("output_expert_ids", only_in["meta_info"])
        np.testing.assert_array_equal(
            self._partition(only_in, "input_expert_ids"), legacy[:boundary]
        )

        only_out = self._generate(return_output_expert_ids=True)
        self.assertNotIn("input_expert_ids", only_out["meta_info"])
        np.testing.assert_array_equal(
            self._partition(only_out, "output_expert_ids"), legacy[boundary:]
        )

    def test_default_request_returns_no_expert_id_fields(self):
        """Both flags false must leave the response byte-for-byte ordinary: no
        field is serialized and no extraction happens."""
        meta = self._generate()["meta_info"]
        for field in ("input_expert_ids", "output_expert_ids", "expert_ids_schema"):
            self.assertNotIn(field, meta)

    def test_prefix_cache_hit_returns_identical_input_rows(self):
        """The second request's prompt forwards are never re-run; its input rows
        must still resolve through the shared token slots to what the first
        request's forward computed. This is the property that makes the sidecar
        a token-lifecycle cache rather than a per-request history."""
        # Fresh salt per invocation. CustomTestCase retries a failed test, and a
        # fixed salt is already warm on the second attempt -- the cold-miss
        # assertion below would then fail on every retry and mask whatever
        # actually went wrong on the first one.
        salt = f"expert-id-partition-prefix-reuse-{uuid.uuid4()}"
        # Pin both requests to one DP rank. Every DP rank runs its own scheduler
        # and its own radix cache, and the dispatcher spreads requests across
        # ranks, so unpinned the second request can land on a rank that never saw
        # the first and report a cold cache -- which says nothing about whether a
        # *hit* serves the producer's rows, the property under test. Pinning
        # establishes that precondition; the assertions keep their full strength
        # (cold miss, then a real hit, then exact row equality). When dp_size is
        # 1 the server logs that routed_dp_rank is ignored and behaves as before,
        # so the committed tp2 configuration is unchanged.
        pin = {"routed_dp_rank": 0}
        first = self._generate(extra_key=salt, return_input_expert_ids=True, **pin)
        self.assertEqual(
            first["meta_info"].get("cached_tokens", 0), 0, "first must be a cold miss"
        )
        second = self._generate(extra_key=salt, return_input_expert_ids=True, **pin)
        self.assertGreater(
            second["meta_info"].get("cached_tokens", 0), 0, "expected a prefix hit"
        )
        np.testing.assert_array_equal(
            self._partition(first, "input_expert_ids"),
            self._partition(second, "input_expert_ids"),
        )

    def test_all_four_flag_combinations_in_one_batch_stay_isolated(self):
        """Per-request isolation: the batch accumulator must not leak one
        request's rows into another's response, and an opted-out request in an
        opted-in batch must still get no field at all."""
        combos = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
        # Issued concurrently so the scheduler batches them together; that is
        # the mixed opt-in shape the accumulator has to keep separate.
        with ThreadPoolExecutor(max_workers=len(combos)) as pool:
            bodies = list(
                pool.map(
                    lambda c: self._generate(
                        return_input_expert_ids=c[0], return_output_expert_ids=c[1]
                    ),
                    combos,
                )
            )

        for (want_in, want_out), body in zip(combos, bodies):
            meta = body["meta_info"]
            with self.subTest(want_in=want_in, want_out=want_out):
                self.assertEqual("input_expert_ids" in meta, want_in)
                self.assertEqual("output_expert_ids" in meta, want_out)
                self.assertEqual("expert_ids_schema" in meta, want_in or want_out)

    def test_start_len_combination_is_rejected(self):
        """Combining the legacy crop with the partitioned payloads would make
        the row-to-position mapping ambiguous, so it must fail loudly."""
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": PROMPT,
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                "return_input_expert_ids": True,
                "routed_experts_start_len": 2,
            },
            timeout=120,
        )
        body = resp.json() if resp.status_code == 200 else {}
        message = json.dumps(body) if body else resp.text
        self.assertIn("routed_experts_start_len", message)

    # ---------------- OpenAI-compatible surfaces ----------------
    # Same server: the args are byte-identical to the native cases, and
    # launching this model twice would roughly double the file's runtime and
    # make its est_time (which drives CI partitioning) wrong.

    def _post(self, path, payload):
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=120)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _assert_sglext_partitions(self, body):
        sglext = body.get("sglext")
        self.assertIsNotNone(sglext, f"response missing sglext: {body}")
        self.assertIn("input_expert_ids", sglext)
        self.assertIn("output_expert_ids", sglext)
        schema = sglext["expert_ids_schema"]
        self.assertEqual(schema["dtype"], "int32")
        self.assertEqual(
            schema["input_num_rows"] + schema["output_num_rows"],
            body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"] - 1,
        )

    def test_completions(self):
        self._assert_sglext_partitions(
            self._post(
                "/v1/completions",
                {
                    "model": self.model,
                    "prompt": PROMPT,
                    "temperature": 0,
                    "max_tokens": MAX_NEW_TOKENS,
                    "return_input_expert_ids": True,
                    "return_output_expert_ids": True,
                },
            )
        )

    def test_chat_completions(self):
        self._assert_sglext_partitions(
            self._post(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "temperature": 0,
                    "max_tokens": MAX_NEW_TOKENS,
                    "return_input_expert_ids": True,
                    "return_output_expert_ids": True,
                },
            )
        )

    def test_default_request_has_no_sglext_expert_fields(self):
        body = self._post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": PROMPT,
                "temperature": 0,
                "max_tokens": MAX_NEW_TOKENS,
            },
        )
        self.assertNotIn("input_expert_ids", body.get("sglext") or {})

    def test_streaming_sends_the_payload_exactly_once(self):
        """The payload is large and only meaningful once the request is done,
        so it must appear in a single terminal chunk -- never repeated on every
        delta, which would multiply the response size by the token count."""
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": PROMPT}],
                "temperature": 0,
                "max_tokens": MAX_NEW_TOKENS,
                "stream": True,
                "return_input_expert_ids": True,
                "return_output_expert_ids": True,
            },
            stream=True,
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        chunks_with_payload = 0
        total_chunks = 0
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            data = raw[len(b"data: ") :]
            if data == b"[DONE]":
                continue
            total_chunks += 1
            sglext = json.loads(data).get("sglext") or {}
            if "input_expert_ids" in sglext or "output_expert_ids" in sglext:
                chunks_with_payload += 1

        self.assertGreater(total_chunks, 1, "expected a multi-chunk stream")
        self.assertEqual(
            chunks_with_payload,
            1,
            f"expert IDs appeared in {chunks_with_payload} of {total_chunks} chunks; "
            "the payload must be emitted exactly once",
        )


if __name__ == "__main__":
    unittest.main()
