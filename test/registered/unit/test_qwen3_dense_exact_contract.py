"""Contract tests for exact dense Qwen3 serving."""

from types import SimpleNamespace

import pytest

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs, _exact_batch_invariant_ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config(**overrides):
    values = {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "max_position_embeddings": 40960,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "use_sliding_window": False,
        "attention_dropout": 0.0,
        "rope_scaling": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _args(**overrides):
    args = ServerArgs(model_path="dummy")
    args.rl_on_policy_target = "xorl"
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _ingress_manager():
    manager = object.__new__(TokenizerManager)
    manager.context_len = 8192
    manager.num_reserved_tokens = 0
    manager.validate_total_tokens = True
    manager.allow_auto_truncate = False
    manager.preferred_sampling_params = {}
    manager.is_generation = True
    manager.server_args = SimpleNamespace(
        glm52_exact_mode=False,
        qwen35_gdn_exact_mode=False,
        qwen3_dense_exact_mode=True,
    )
    return manager


def test_dense_qwen3_resolves_architecture_owned_exact_program():
    config = _config()
    args = _args()

    args._resolve_qwen3_dense_exact_contract(
        config,
        model_arch="Qwen3ForCausalLM",
    )

    assert args.qwen3_dense_exact_mode
    assert config._qwen3_dense_exact_mode
    assert args.dtype == "bfloat16"
    assert args.attention_backend == "fa4"
    assert not args.enable_fp32_lm_head
    assert args.enable_deterministic_inference
    assert args.sampling_backend == "pytorch"
    assert args.sampling_defaults == "openai"
    assert args.disable_custom_all_reduce
    assert not args.skip_server_warmup
    assert _exact_batch_invariant_ops(args) == ("addmm", "bmm", "mm")


def test_dense_qwen3_accepts_transformers_v5_rope_parameters():
    config = _config(rope_theta=None, rope_parameters={"rope_theta": 1_000_000})
    args = _args()

    args._resolve_qwen3_dense_exact_contract(
        config,
        model_arch="Qwen3ForCausalLM",
    )

    assert args.qwen3_dense_exact_mode


def test_dense_qwen3_model_construction_uses_runtime_contract(monkeypatch):
    from sglang.srt.models import qwen2

    runtime = SimpleNamespace(
        deterministic=SimpleNamespace(qwen3_dense_exact_mode=True),
    )
    monkeypatch.setattr(qwen2, "get_exec", lambda: runtime)

    assert qwen2._is_qwen3_dense_exact_runtime()


@pytest.mark.parametrize(
    ("sampling_params", "request_fields", "error"),
    (
        ({}, {"top_logprobs_num": 2}, "top_logprobs_num=2"),
        ({}, {"token_ids_logprob": [7]}, "token_ids_logprob=set"),
    ),
)
def test_dense_qwen3_ingress_rejects_sampler_fatal_options(
    sampling_params, request_fields, error
):
    request = GenerateReqInput(
        input_ids=[1],
        sampling_params={"max_new_tokens": 1, **sampling_params},
        return_logprob=True,
        **request_fields,
    )
    request.normalize_batch_and_arguments()

    with pytest.raises(ValueError, match=error):
        _ingress_manager()._validate_one_request(request, [1])


@pytest.mark.parametrize("temperature", [0.0, 1e-7, 0.7, 1.0, 1.3])
def test_dense_qwen3_ingress_admits_non_negative_temperature_plain_sampling(
    temperature,
):
    request = GenerateReqInput(
        input_ids=[1],
        sampling_params={"temperature": temperature, "max_new_tokens": 1},
        return_logprob=True,
    )
    request.normalize_batch_and_arguments()

    _ingress_manager()._validate_one_request(request, [1])


def test_dense_qwen3_preserves_explicit_server_warmup_skip():
    args = _args(skip_server_warmup=True)
    args._resolve_qwen3_dense_exact_contract(
        _config(),
        model_arch="Qwen3ForCausalLM",
    )

    assert args.skip_server_warmup


def test_dense_qwen3_ingress_admits_joint_sampling_filters():
    request = GenerateReqInput(
        input_ids=[1],
        sampling_params={
            "temperature": 0.7,
            "top_k": 8,
            "top_p": 0.9,
            "min_p": 0.1,
            "max_new_tokens": 1,
        },
        return_logprob=True,
    )
    request.normalize_batch_and_arguments()
    _ingress_manager()._validate_one_request(request, [1])


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("tp_size", 2, "stage-local TP1/DP1/EP1"),
        ("dtype", "float16", "BF16"),
        ("quantization", "fp8", "unquantized"),
        ("attention_backend", "triton", "FA4"),
        ("speculative_algorithm", "EAGLE", "speculative"),
    ],
)
def test_dense_qwen3_rejects_unqualified_program(name, value, message):
    args = _args(**{name: value})
    with pytest.raises(ValueError, match=message):
        args._resolve_qwen3_dense_exact_contract(
            _config(),
            model_arch="Qwen3ForCausalLM",
        )


@pytest.mark.parametrize(
    "geometry",
    [
        {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "tie_word_embeddings": True,
        },
        {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "tie_word_embeddings": True,
        },
        {
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rope_theta": 5_000_000,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": True,
        },
        {
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
        },
    ],
)
def test_dense_qwen3_accepts_family_geometries(geometry):
    args = _args()
    args._resolve_qwen3_dense_exact_contract(
        _config(**geometry),
        model_arch="Qwen3ForCausalLM",
    )
    assert args.qwen3_dense_exact_mode


@pytest.mark.parametrize(
    "override",
    [
        {"hidden_act": "gelu"},
        {"head_dim": 64},
        {"attention_bias": True},
        {"attention_dropout": 0.1},
        {"use_sliding_window": True},
        {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        {"num_key_value_heads": 7},
        {"hidden_size": 0},
    ],
)
def test_dense_qwen3_rejects_unsupported_capabilities(override):
    with pytest.raises(
        ValueError, match="does not support this architecture configuration"
    ):
        _args()._resolve_qwen3_dense_exact_contract(
            _config(**override),
            model_arch="Qwen3ForCausalLM",
        )


def test_other_architecture_or_target_does_not_engage_dense_qwen3_contract():
    config = _config()
    args = _args()
    args._resolve_qwen3_dense_exact_contract(config, model_arch="LlamaForCausalLM")
    assert not args.qwen3_dense_exact_mode

    non_xorl = _args(rl_on_policy_target=None)
    non_xorl._resolve_qwen3_dense_exact_contract(
        config,
        model_arch="Qwen3ForCausalLM",
    )
    assert not non_xorl.qwen3_dense_exact_mode


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
