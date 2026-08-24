from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs, _exact_batch_invariant_ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _args(**updates):
    args = ServerArgs(model_path="dummy")
    defaults = dict(
        deepep_native_exact=True,
        rl_on_policy_target="xorl",
        dtype="auto",
        quantization=None,
        moe_a2a_backend="deepep",
        moe_runner_backend="auto",
        deepep_mode="auto",
        deepep_dispatcher_output_dtype="auto",
        tp_size=8,
        ep_size=1,
        dp_size=1,
        attention_backend=None,
        disable_overlap_schedule=False,
        disable_piecewise_cuda_graph=False,
        disable_cuda_graph=False,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL, max_bs=256),
            prefill=SimpleNamespace(backend=Backend.FULL),
        ),
    )
    defaults.update(updates)
    for name, value in defaults.items():
        setattr(args, name, value)
    return args


def _config(*, num_experts=128):
    return SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        model_type="qwen3_moe",
        num_experts=num_experts,
    )


def test_native_exact_is_the_only_public_exact_combine_switch():
    args = ServerArgs(model_path="dummy")

    assert args.deepep_native_exact is False


def test_qwen3_moe_native_exact_resolves_fail_closed_runtime():
    args = _args()
    config = _config()

    args._resolve_deepep_native_exact_contract(
        config,
        model_arch="Qwen3MoeForCausalLM",
    )

    assert config._deepep_native_exact is True
    assert config._qwen3_dense_exact_mode is True
    assert args.qwen3_dense_exact_mode is True
    assert args.dtype == "bfloat16"
    assert (args.tp_size, args.ep_size, args.dp_size) == (8, 8, 8)
    assert args.moe_runner_backend == "triton"
    assert args.deepep_mode == "auto"
    assert args.deepep_dispatcher_output_dtype == "bf16"
    assert args.enable_dp_attention is True
    assert args.enable_dp_lm_head is True
    assert args.enable_fp32_router is True
    assert args.enable_fp32_lm_head is False
    assert _exact_batch_invariant_ops(args) == ("addmm", "bmm", "mm")
    assert args.disable_cuda_graph is False
    assert args.cuda_graph_config.decode.backend == Backend.FULL
    assert args.cuda_graph_config.prefill.backend == Backend.DISABLED


def test_qwen3_moe_native_exact_preserves_explicit_breakable_decode():
    args = _args(
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.BREAKABLE),
            prefill=SimpleNamespace(backend=Backend.DISABLED),
        )
    )

    args._resolve_deepep_native_exact_contract(
        _config(),
        model_arch="Qwen3MoeForCausalLM",
    )

    assert args.disable_cuda_graph is False
    assert args.cuda_graph_config.decode.backend == Backend.BREAKABLE
    assert args.cuda_graph_config.prefill.backend == Backend.DISABLED


@pytest.mark.parametrize(
    ("lora_serving_mode", "enable_lora"),
    [("merged", False), ("separate", True)],
)
def test_qwen3_moe_native_exact_admits_explicit_lora_serving_modes(
    lora_serving_mode, enable_lora
):
    args = _args(
        lora_serving_mode=lora_serving_mode,
        enable_lora=enable_lora,
    )
    config = _config()

    args._resolve_deepep_native_exact_contract(
        config,
        model_arch="Qwen3MoeForCausalLM",
    )

    assert config._lora_serving_mode == lora_serving_mode


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"enable_lora": True}, "requires --lora-serving-mode separate"),
        (
            {"enable_lora": True, "lora_serving_mode": "merged"},
            "must not enable a sampler adapter",
        ),
        (
            {"enable_lora": False, "lora_serving_mode": "separate"},
            "requires --enable-lora",
        ),
    ],
)
def test_qwen3_moe_native_exact_rejects_lora_publication_mismatch(updates, match):
    args = _args(**updates)

    with pytest.raises(ValueError, match=match):
        args._resolve_deepep_native_exact_contract(
            _config(),
            model_arch="Qwen3MoeForCausalLM",
        )


def test_qwen3_moe_native_exact_full_graph_pins_dispatch_capacity(monkeypatch):
    from sglang.srt.environ import envs

    capacity = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
    monkeypatch.delenv(capacity.name, raising=False)
    args = _args()

    args._resolve_native_deepep_graph_capacity()

    assert capacity.get() == 256


@pytest.mark.parametrize(
    ("updates", "arch", "match"),
    [
        ({"rl_on_policy_target": None}, "Qwen3MoeForCausalLM", "rl-on-policy-target"),
        ({"moe_a2a_backend": "none"}, "Qwen3MoeForCausalLM", "moe-a2a-backend"),
        (
            {"deepep_dispatcher_output_dtype": "fp8"},
            "Qwen3MoeForCausalLM",
            "output-dtype bf16",
        ),
        ({"tp_size": 1}, "Qwen3MoeForCausalLM", "multi-rank"),
        ({}, "Qwen2MoeForCausalLM", "does not declare"),
    ],
)
def test_qwen3_moe_native_exact_rejects_unqualified_surfaces(updates, arch, match):
    args = _args(**updates)
    with pytest.raises(ValueError, match=match):
        args._resolve_deepep_native_exact_contract(
            _config(),
            model_arch=arch,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
