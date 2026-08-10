"""Contract tests for exact dense Qwen3-8B serving."""

from types import SimpleNamespace

import pytest

from sglang.srt.server_args import ServerArgs, _exact_batch_invariant_ops


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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _args(**overrides):
    args = ServerArgs(model_path="dummy")
    args.rl_on_policy_target = "xorl"
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_qwen3_8b_resolves_architecture_owned_exact_program():
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
    assert args.skip_server_warmup
    assert _exact_batch_invariant_ops(args) == ("addmm", "bmm", "mm")


def test_qwen3_8b_accepts_transformers_v5_rope_parameters():
    config = _config(rope_theta=None, rope_parameters={"rope_theta": 1_000_000})
    args = _args()

    args._resolve_qwen3_dense_exact_contract(
        config,
        model_arch="Qwen3ForCausalLM",
    )

    assert args.qwen3_dense_exact_mode


def test_qwen3_8b_model_construction_uses_runtime_contract(monkeypatch):
    from sglang.srt.models import qwen2

    runtime = SimpleNamespace(
        deterministic=SimpleNamespace(qwen3_dense_exact_mode=True),
    )
    monkeypatch.setattr(qwen2, "get_exec", lambda: runtime)

    assert qwen2._is_qwen3_dense_exact_runtime()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("tp_size", 2, "TP1/DP1/EP1/PP1"),
        ("dtype", "float16", "BF16"),
        ("quantization", "fp8", "unquantized"),
        ("attention_backend", "triton", "FA4"),
        ("speculative_algorithm", "EAGLE", "speculative"),
    ],
)
def test_qwen3_8b_rejects_unqualified_program(name, value, message):
    args = _args(**{name: value})
    with pytest.raises(ValueError, match=message):
        args._resolve_qwen3_dense_exact_contract(
            _config(),
            model_arch="Qwen3ForCausalLM",
        )


def test_other_architecture_or_target_does_not_engage_qwen3_8b_contract():
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
