import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import _health_generate_sampling_params
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.srt.sampling.sampling_params import SamplingParams, TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHealthGenerateSamplingParams(unittest.TestCase):
    def test_default_health_request_remains_greedy(self):
        params = _health_generate_sampling_params(
            SimpleNamespace(glm52_exact_mode=False, random_seed=17)
        )

        self.assertEqual(params, {"max_new_tokens": 1, "temperature": 0.0})

    def test_glm52_exact_health_request_uses_strict_multinomial_contract(self):
        params = _health_generate_sampling_params(
            SimpleNamespace(glm52_exact_mode=True, random_seed=520052)
        )
        normalized = SamplingParams(**params)

        self.assertEqual(normalized.max_new_tokens, 1)
        self.assertEqual(normalized.min_new_tokens, 0)
        self.assertEqual(normalized.temperature, 1.0)
        self.assertEqual(normalized.top_p, 1.0)
        self.assertEqual(normalized.top_k, TOP_K_ALL)
        self.assertEqual(normalized.min_p, 0.0)
        self.assertEqual(normalized.frequency_penalty, 0.0)
        self.assertEqual(normalized.presence_penalty, 0.0)
        self.assertEqual(normalized.repetition_penalty, 1.0)
        self.assertEqual(normalized.sampling_seed, 520052)
        self.assertGreater(normalized.top_k, 1)

    def test_glm52_exact_health_request_has_a_seed_fallback(self):
        params = _health_generate_sampling_params(
            SimpleNamespace(glm52_exact_mode=True, random_seed=None)
        )

        self.assertEqual(params["sampling_seed"], 42)

    def test_qwen3_dense_exact_health_request_uses_strict_multinomial_contract(self):
        params = _health_generate_sampling_params(
            SimpleNamespace(
                glm52_exact_mode=False,
                qwen3_dense_exact_mode=True,
                random_seed=300008,
            )
        )
        normalized = SamplingParams(**params)

        self.assertEqual(normalized.temperature, 1.0)
        self.assertEqual(normalized.top_p, 1.0)
        self.assertEqual(normalized.top_k, TOP_K_ALL)
        self.assertEqual(normalized.min_p, 0.0)
        self.assertEqual(normalized.sampling_seed, 300008)


class TestHealthGenerateEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_glm52_exact_endpoint_submits_the_contract_request(self):
        captured = []
        real_sleep = asyncio.sleep
        manager = SimpleNamespace(
            gracefully_exit=False,
            server_status=ServerStatus.Up,
            server_args=SimpleNamespace(
                glm52_exact_mode=True,
                random_seed=520052,
                disaggregation_mode=DisaggregationMode.NULL.value,
            ),
            is_generation=True,
            last_receive_tstamp=0.0,
            rid_to_state={},
        )

        async def generate_request(request_input, _request):
            captured.append(request_input)
            manager.last_receive_tstamp = time.time() + 1
            yield {}

        async def fast_sleep(_seconds):
            await real_sleep(0)

        manager.generate_request = generate_request
        request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        global_state = SimpleNamespace(tokenizer_manager=manager)

        with (
            patch.object(http_server, "_global_state", global_state),
            patch.object(http_server.asyncio, "sleep", fast_sleep),
        ):
            response = await http_server.health_generate(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(captured), 1)
        params = SamplingParams(**captured[0].sampling_params)
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.top_k, TOP_K_ALL)
        self.assertEqual(params.sampling_seed, 520052)


if __name__ == "__main__":
    unittest.main()
