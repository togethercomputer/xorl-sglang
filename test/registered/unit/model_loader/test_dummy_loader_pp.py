import torch
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.model_loader.loader import DummyModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def test_dummy_loader_skips_pipeline_placeholders(monkeypatch):
    model = torch.nn.Module()
    model.placeholder = PPMissingLayer()

    processed = []
    real_layer = torch.nn.Module()
    real_layer.quant_method = type(
        "QuantMethod",
        (),
        {
            "process_weights_after_loading": lambda self, module: processed.append(
                module
            )
        },
    )()
    model.real_layer = real_layer

    monkeypatch.setattr(
        "sglang.srt.model_loader.loader._get_quantization_config",
        lambda model_config, load_config: None,
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.loader._initialize_model",
        lambda model_config, load_config, quant_config: model,
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.loader.initialize_dummy_weights",
        lambda loaded_model: None,
    )
    monkeypatch.setattr(
        "sglang.srt.model_loader.loader.post_load_weights",
        lambda loaded_model, model_config: None,
    )

    loader = object.__new__(DummyModelLoader)
    loader.load_config = object()
    loaded = loader.load_model(
        model_config=type("ModelConfig", (), {"dtype": torch.float32})(),
        device_config=type("DeviceConfig", (), {"device": torch.device("cpu")})(),
    )

    assert loaded is model
    assert processed == [real_layer]
