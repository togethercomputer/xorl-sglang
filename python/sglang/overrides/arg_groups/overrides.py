"""Override twin of ``sglang.srt.arg_groups.overrides``.

Picks the MoE runner for LoRA runs so the operator does not have to.

``moe_runner_backend`` defaults to ``auto``, which each quant method resolves on
its own and which generally lands on triton. For a MoE model served with LoRA
that is the wrong answer on Blackwell -- ``experimental_sgl_trtllm`` is the path
this fork tunes and validates -- and the backend lock in the ``lora.lora_manager``
twin would reject the ``auto`` outcome, so every Blackwell MoE LoRA launch needed
``--moe-runner-backend experimental_sgl_trtllm`` typed by hand, followed by a
second round trip for ``--lora-use-virtual-experts``.

Selection, declared only when the operator left the runner on ``auto``:

    Blackwell      -> experimental_sgl_trtllm + lora_use_virtual_experts
    anything else  -> triton

The TRT-LLM MoE kernels ship only as precompiled cubins for sm100f / sm103a /
sm107a, so on Hopper the tuned path does not exist and triton is the supported
one; the lock twin allows it there for the same reason.

Both fields are declared together on purpose: the bf16 and NVFP4 fused-experts
paths hard-assert ``use_virtual_lora_store`` (``lora_dispatch.py``), so picking
the runner without also supplying virtual experts would just move the required
flag rather than remove it.
"""

import logging

from sglang.srt.utils.common import get_quantization_config, is_blackwell_supported

logger = logging.getLogger(__name__)

_TRTLLM_MOE_BACKEND = "experimental_sgl_trtllm"
_FALLBACK_MOE_BACKEND = "triton"

# hf_config keys that mean "this is a MoE model". Config-level, unlike the
# FusedMoE module scan the lock uses: selection happens before the model is
# built, and a false positive here is still caught by that later check.
_MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
)

# Quantizations whose MoE LoRA path this fork actually measured: bf16 (no
# quantization), FP8 and NVFP4, all 28/28 on test/manual/lora. mxfp4 and the
# rest are left to whatever the model's own rule picks -- auto-selecting an
# unvalidated numerical path is worse than making the operator choose.
_VALIDATED_QUANTIZATIONS = (None, "fp8", "modelopt_fp4")


def _lora_requested(server_args) -> bool:
    return bool(getattr(server_args, "enable_lora", False)) or bool(
        getattr(server_args, "lora_paths", None)
    )


def _config_looks_moe(hf_config) -> bool:
    return any(getattr(hf_config, k, None) for k in _MOE_CONFIG_KEYS)


def _effective_quantization(server_args, hf_config):
    """The quantization this run will use, mirroring the model providers."""
    explicit = getattr(server_args, "quantization", None)
    if explicit is not None:
        return explicit
    if getattr(server_args, "_quantization_explicitly_unset", False):
        return None
    return get_quantization_config(hf_config)


def select_moe_lora_backend(server_args, hf_config) -> dict:
    """Declare the MoE runner (and its prerequisite) for a LoRA run."""
    # Pristine read: an explicit choice, including an explicit "triton", wins.
    if getattr(server_args, "moe_runner_backend", None) != "auto":
        return {}
    if not _lora_requested(server_args):
        return {}
    if not _config_looks_moe(hf_config):
        return {}
    if _effective_quantization(server_args, hf_config) not in _VALIDATED_QUANTIZATIONS:
        return {}

    if not is_blackwell_supported():
        logger.info(
            "MoE LoRA on a non-Blackwell GPU: moe_runner_backend=%s "
            "(%s ships cubins for sm100f/sm103a/sm107a only).",
            _FALLBACK_MOE_BACKEND,
            _TRTLLM_MOE_BACKEND,
        )
        return {"moe_runner_backend": _FALLBACK_MOE_BACKEND}

    logger.info(
        "MoE LoRA on Blackwell: moe_runner_backend=%s with "
        "lora_use_virtual_experts=True (the fused-experts paths assert on it).",
        _TRTLLM_MOE_BACKEND,
    )
    return {
        "moe_runner_backend": _TRTLLM_MOE_BACKEND,
        "lora_use_virtual_experts": True,
    }


def __apply_patch__(public_mod):
    # Matches every architecture. Predicate providers run after the exact-keyed
    # ones and last writer wins, which is the intent here: a model's own rule
    # (Qwen3-MoE declares flashinfer_trtllm for auto) is a *non-LoRA* default,
    # and the lock rejects it for MoE LoRA -- which is exactly why every
    # Blackwell MoE LoRA launch needed the flag by hand. Overriding it is scoped
    # tightly: auto only, LoRA only, MoE only, validated quantizations only.
    public_mod.register_model_override_predicate(lambda _architecture: True)(
        select_moe_lora_backend
    )
