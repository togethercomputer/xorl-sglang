"""Dense-layer mapping for the partitioned expert-ID payloads.

The route row's layer axis spans *every* decoder layer, because the host
sidecar is allocated with `num_hidden_layers` planes.  Dense layers never call
the capture hook, so their planes stay zero -- on the wire that is
indistinguishable from a genuine expert id 0.  `expert_ids_schema.moe_layer_ids`
is the only thing that maps a plane index back to a model layer.

DeepSeek-V2-Lite is used because it has `first_k_dense_replace > 0`: its leading
layer(s) are dense MLPs, so a model whose MoE layers do not start at index 0
actually exercises the mapping.  On an all-MoE model this test would pass even
if `moe_layer_ids` were hardcoded to `range(num_layers)`.
"""

import unittest

import numpy as np
import requests

from sglang.srt.state_capturer.routed_experts import (
    extract_expert_ids_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")

# Smallest widely-available MoE with leading dense layers.
MODEL_PATH = "deepseek-ai/DeepSeek-V2-Lite"
PROMPT = "User: Tell me a fact about cats.\nAssistant:"
MAX_NEW_TOKENS = 8


class TestExpertIdDenseLayerMapping(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-return-routed-experts",
                "--enable-deterministic-inference",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

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
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=180)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_moe_layer_ids_excludes_the_leading_dense_layers(self):
        """The schema must name the layers that actually routed, and on this
        model that set must be a strict subset starting above index 0."""
        body = self._generate(return_output_expert_ids=True)
        schema = body["meta_info"]["expert_ids_schema"]
        moe_layer_ids = schema["moe_layer_ids"]

        self.assertLess(
            len(moe_layer_ids),
            schema["num_layers"],
            "DeepSeek-V2-Lite has dense layers; moe_layer_ids must be a strict subset",
        )
        self.assertNotIn(
            0, moe_layer_ids, "layer 0 is dense (first_k_dense_replace) on this model"
        )
        self.assertEqual(
            moe_layer_ids,
            sorted(moe_layer_ids),
            "moe_layer_ids must be ascending so a consumer can index it directly",
        )

    def test_dense_planes_are_zero_and_moe_planes_are_populated(self):
        """The plane layout must match what the schema advertises: planes the
        schema omits are the zero-filled dense ones, and every plane it names
        carries real routes. Getting this backwards would train a replay
        consumer against expert 0 for whole layers."""
        body = self._generate(return_output_expert_ids=True)
        schema = body["meta_info"]["expert_ids_schema"]
        rows = extract_expert_ids_from_meta_info(body, "output_expert_ids").reshape(
            schema["output_num_rows"], schema["num_layers"], schema["top_k"]
        )
        self.assertGreater(rows.shape[0], 0)

        moe_layer_ids = set(schema["moe_layer_ids"])
        dense_layer_ids = set(range(schema["num_layers"])) - moe_layer_ids
        self.assertTrue(dense_layer_ids, "expected at least one dense layer")

        for layer in sorted(dense_layer_ids):
            self.assertTrue(
                np.all(rows[:, layer, :] == 0),
                f"layer {layer} is not named in moe_layer_ids but carries non-zero ids",
            )

        # A routed layer selects top_k distinct experts per token, so at least
        # one plane the schema names must be non-trivial.
        moe_planes = rows[:, sorted(moe_layer_ids), :]
        self.assertTrue(
            np.any(moe_planes != 0),
            "every plane named in moe_layer_ids is zero; capture did not run",
        )

    def test_expert_ids_are_in_the_logical_global_range(self):
        """Capture runs before the logical->physical remap, so IDs must span the
        model's full logical expert space, not a per-rank local slice."""
        body = self._generate(return_output_expert_ids=True)
        schema = body["meta_info"]["expert_ids_schema"]
        self.assertEqual(schema["id_space"], "logical_global")
        rows = extract_expert_ids_from_meta_info(body, "output_expert_ids")
        self.assertGreaterEqual(int(rows.min()), 0)


if __name__ == "__main__":
    unittest.main()
