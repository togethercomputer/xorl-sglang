"""Unit tests for model-specific server warmup inputs."""

import base64
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.entrypoints.http_server as http_server
from sglang.srt.entrypoints.http_server import (
    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _execute_server_warmup,
    _get_vlm_warmup_image_base64,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestServerWarmup(CustomTestCase):
    def test_builtin_generation_warmup_issues_temperature_zero_request(self):
        captured = {}
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "is_generation": True,
                "has_image_understanding": False,
            },
        )
        server_args = SimpleNamespace(
            url=lambda: "http://127.0.0.1:30000",
            api_key=None,
            ssl_verify=lambda: False,
            language_only=False,
            skip_tokenizer_init=True,
            dp_size=1,
            dsv4_flash_exact_mode=False,
            debug_tensor_dump_input_file=None,
            disaggregation_mode="null",
        )

        def post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            return response

        with (
            patch.object(http_server.time, "sleep"),
            patch.object(http_server.requests, "get", return_value=response),
            patch.object(http_server.requests, "post", side_effect=post),
            patch.object(
                http_server,
                "_global_state",
                SimpleNamespace(tokenizer_manager=SimpleNamespace(server_status=None)),
            ),
        ):
            assert _execute_server_warmup(server_args)

        self.assertTrue(captured["url"].endswith("/generate"))
        self.assertEqual(captured["json"]["sampling_params"]["temperature"], 0)


class TestVlmWarmupImage(CustomTestCase):
    def test_kimi_k2_uses_representative_vision_image(self):
        image_base64 = _get_vlm_warmup_image_base64(
            {"architectures": ["KimiK25ForConditionalGeneration"]}
        )
        self.assertEqual(image_base64, KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)

        png = base64.b64decode(KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (512, 512))

    def test_kimi_k3_uses_native_patch_grid_image(self):
        for model_info in (
            {"architectures": ["KimiK3ForConditionalGeneration"]},
            {"architectures": None, "model_type": "kimi_k3"},
        ):
            with self.subTest(model_info=model_info):
                self.assertEqual(
                    _get_vlm_warmup_image_base64(model_info),
                    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
                )

        png = base64.b64decode(KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (448, 448))

    def test_other_vlms_keep_minimal_startup_image(self):
        self.assertEqual(
            _get_vlm_warmup_image_base64(
                {"architectures": ["Qwen3VLForConditionalGeneration"]}
            ),
            MINIMUM_PNG_PICTURE_BASE64,
        )
        self.assertEqual(
            _get_vlm_warmup_image_base64({"architectures": None}),
            MINIMUM_PNG_PICTURE_BASE64,
        )


if __name__ == "__main__":
    unittest.main()
