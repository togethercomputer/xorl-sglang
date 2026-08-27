"""Override twin of ``sglang.srt.server_args``.

Turns the experimental LoRA optimizations on by default for
``--moe-runner-backend experimental_sgl_trtllm``, because that backend is
measurably *slower* than the triton MoE runner with them off.

Measured on 4x B200, Qwen3-30B-A3B-Instruct-2507 bf16, TP=4, 8 shared-outer
adapters (rank 16, virtual experts), 512 in / 128 out, output tok/s:

    batch   triton   trtllm(off)   trtllm(on)
        1      161           154          186
        8      660           654          709
       32     3969          3699         4391
       64     7356          6625         7815

So the switch is what makes the backend worth selecting at all (+6-15% over
triton), and leaving it off is a performance trap: the user asks for the
TRT-LLM MoE path and gets something slower than the default.

Two flags are set together, and the pairing is the point:

``SGLANG_EXPERIMENTAL_LORA_OPTI``
    Master gate. Enables the fused/split-K kernel opts *and* installs the
    two-stream LoRA overlap monkey-patch (``lora/layers.py`` does this at module
    import time, which is why this has to be decided before that import).

``SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC``
    Ships defaulted **off** in ``trtllm_lora_temp/environ.py`` even though it
    fixes a documented correctness hazard of the overlap path: a shrink-output
    buffer allocated inside ``with torch.cuda.stream(side)`` is tagged to the
    side stream, so the caching allocator may reuse it before the main stream's
    LoRA-B expand has consumed it -- a premature-reuse WAR that the source
    describes as "qwen3.5 mamba decode garbage" under cuda-graph replay. Turning
    the master gate on without this would enable the overlap while leaving its
    mitigation off, so we set both or neither.

``SGLANG_OPT_FUSED_PERMUTE_QUANT``
    Fuses the NVFP4 permute + activation-quant step. Ships defaulted **off** and
    labelled kimi-only, but that is validation scope rather than a capability
    limit: the kernel's own preconditions are generic (``hidden_size % 16 == 0``
    and ``top_k <= 512``, checked loudly in the launcher) and Qwen3-30B-A3B
    satisfies both. Without it, NVFP4 MoE-LoRA is the slowest dtype on this path;
    with it, it draws level with bf16. Measured on 2x B200, Qwen3-30B-A3B-NVFP4,
    TP=2, 512 in / 128 out, output tok/s, LoRA mode:

        batch   off     on      bf16
            1   142     150      171
            8  1164    1186     1144
           32  2700    3943     3717
           64  3107    5772     5892

    +46% at bs=32 and +86% at bs=64, and 28/28 on the correctness harness with it
    on. Only affects NVFP4; inert for bf16 and FP8.

``SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION``
    Ships defaulted **off** (``environ.py``), and NVFP4 MoE LoRA is silently
    wrong without it. The NVFP4 LoRA op always quantizes activations
    dynamically per token, deriving its own scales, so the checkpoint's static
    per-tensor activation scales must be neutral (a1 == a2 == 1) -- which is
    what this flag arranges. With it off they are applied *on top of* the op's
    own per-token scales: ``g1_alphas`` carries a1 into both SwiGLU halves (so
    the product carries a1**2) and ``g2_alphas`` re-applies a2 at GEMM2, leaving
    the output off by ~a1**2 * a2. On Qwen3-30B-A3B-NVFP4 (a1=1.6e-3,
    a2=3.2e-3) that is 4/28 on the correctness harness -- fluent-looking
    garbage tokens, no error anywhere -- against 28/28 with it on.

    Only affects NVFP4; inert for bf16 and FP8, which is why it is set for the
    whole backend rather than gated on a quantization that is not yet known when
    ``__post_init__`` runs.

All four are skipped if the user set any of them explicitly (either direction).

That is a defaulting opt-out, **not** a way to run this backend unoptimized:
``flashinfer_trtllm.py`` gates the ``experimental_sgl_trtllm`` fused-func
registration on the same master switch, so ``SGLANG_EXPERIMENTAL_LORA_OPTI=0``
makes the backend unloadable --

    NotImplementedError: Runner backend MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM
    requires a fused func for a2a backend none, but none is registered.

-- rather than falling back to a slower path. The real opt-out is to choose a
different ``--moe-runner-backend``.

``SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC`` and ``SGLANG_OPT_FUSED_PERMUTE_QUANT`` are
local ``_GatedBool``s in ``trtllm_lora_temp/environ.py`` reading ``os.environ``
directly -- neither is an ``Envs`` descriptor -- so they are set through
``os.environ`` rather than the ``envs`` API.
"""

# ruff: noqa: F821 -- the verbatim copies below (notably the pinned
# _handle_model_specific_adjustments) resolve upstream names at call time via
# rebind() over the live srt.server_args module dict; they are undefined in
# this file's namespace by design.

from __future__ import annotations

import logging
import os

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_TRTLLM_MOE_BACKEND = "experimental_sgl_trtllm"
# Read via os.environ by trtllm_lora_temp/environ.py's _GatedBool, not via Envs.
_OVERLAP_ALLOC_ENV = "SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC"
_FUSED_PERMUTE_QUANT_ENV = "SGLANG_OPT_FUSED_PERMUTE_QUANT"
_TWO_STREAM_MAX_TOKENS_ENV = "SGLANG_TWO_STREAM_MAX_TOKENS"
# A real Envs descriptor, unlike the two above, so it goes through the envs API.
_NVFP4_PER_TOKEN_ENV = envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION


def _lora_requested(server_args) -> bool:
    return bool(getattr(server_args, "enable_lora", False)) or bool(
        getattr(server_args, "lora_paths", None)
    )


def maybe_enable_experimental_lora_opti(server_args) -> None:
    """Default the experimental LoRA opts on for the TRT-LLM MoE runner.

    No-op unless this is a LoRA run on ``experimental_sgl_trtllm``. Never
    overrides an explicit user setting.
    """
    if getattr(server_args, "moe_runner_backend", None) != _TRTLLM_MOE_BACKEND:
        return
    if not _lora_requested(server_args):
        return

    master = envs.SGLANG_EXPERIMENTAL_LORA_OPTI
    if (
        master.is_set()
        or _OVERLAP_ALLOC_ENV in os.environ
        or _FUSED_PERMUTE_QUANT_ENV in os.environ
        or _NVFP4_PER_TOKEN_ENV.is_set()
    ):
        # Explicit user intent on either flag: leave the whole pairing alone
        # rather than half-applying it.
        return

    master.set(True)
    os.environ[_OVERLAP_ALLOC_ENV] = "1"
    os.environ[_FUSED_PERMUTE_QUANT_ENV] = "1"
    _NVFP4_PER_TOKEN_ENV.set(True)
    # The two-stream LoRA overlap corrupts rank>16 adapters. Measured on
    # Qwen3.5-35B-A3B-FP8 @ TP=2, eight rank-64 adapters, full 2x2 (full and
    # routed-only adapter sets x overlap on/off): overlap on 1/8 or 0/8,
    # overlap off 8/8, replicated. Two independent components are implicated --
    # the FP8 MoE two-stream copy (fails with routed-only adapters, and still
    # fails run serially with single-stream-identical arguments, execution
    # verified) and the attention/dense O7-O9 overrides (with the MoE copy
    # delegated away, full adapters got WORSE, 0/8). Rank-16 was measured clean
    # through the same paths (Qwen3-30B, 8 adapters, 28/28), so the overlap
    # stays on inside its validated envelope and is disabled outside it.
    # An unset rank counts as outside the envelope: without --max-lora-rank the
    # field is only inferred from the adapters later (LoRAManager, in the
    # scheduler process), which is after this env decision has to be made --
    # and silently keeping the overlap on for unknown ranks is exactly the
    # corruption this guards against. Rank-16 deployments keep the overlap by
    # stating --max-lora-rank 16.
    max_lora_rank = getattr(server_args, "max_lora_rank", None)
    if (
        not max_lora_rank or max_lora_rank > 16
    ) and _TWO_STREAM_MAX_TOKENS_ENV not in os.environ:
        os.environ[_TWO_STREAM_MAX_TOKENS_ENV] = "0"
    # warning, not info: this runs inside ServerArgs.__post_init__, before sglang
    # configures logging, so an info record has no handler and is dropped. It is
    # also a default-flip the operator should see in the log.
    logger.warning(
        "moe_runner_backend=%s with LoRA: defaulting "
        "SGLANG_EXPERIMENTAL_LORA_OPTI=1, %s=1, %s=1 and %s=1 "
        "(set any of them explicitly to opt out).",
        _TRTLLM_MOE_BACKEND,
        _OVERLAP_ALLOC_ENV,
        _FUSED_PERMUTE_QUANT_ENV,
        _NVFP4_PER_TOKEN_ENV.name,
    )


# ===================== XoRL exact serving (zero-srt port of PR #41) =====================
#
# Everything below retires the in-tree srt/server_args.py edits. Fields cannot
# be added to a dataclass post-hoc, so the twin replaces ``ServerArgs`` with a
# subclass (module-level: launcher->scheduler pickling resolves it by import
# path; every process activates the overlay via ``import sglang``). The two
# resolution-pipeline seams are handled without copying ``__post_init__``:
#   - ``_validate_rl_on_policy_target`` ran right after
#     ``_handle_return_hidden_states_mode`` -- the subclass extends that
#     handler via ``super()``, preserving the exact in-tree position;
#   - the two ``_resolve_*_exact_contract`` calls sat mid-body of
#     ``_handle_model_specific_adjustments`` between statements that interact
#     with fields the resolvers write, so that method is carried as a
#     verbatim PINNED copy (see sglang.overrides._twin_pins -- __post_init__
#     is pinned too, so an upstream reorder of the seam fires the pin test).
# ``RL_ON_POLICY_TARGET_CHOICES`` is mutated IN PLACE: the upstream field
# metadata captured the list object by reference at class creation.

import dataclasses
from typing import Literal, Optional

from sglang.overrides._twin_bind import rebind

# Self-imports are safe at twin-import time: the finder fully executes the
# upstream module before importing its twin (the bypass-caching hazard only
# applies to OTHER not-yet-loaded srt modules).
from sglang.srt.server_args import NS, A, Arg
from sglang.srt.server_args import ServerArgs as _UpstreamServerArgs

XORL_RL_TARGET = "xorl"
RL_ON_POLICY_TARGET_CHOICES = [XORL_RL_TARGET]


def is_glm52_exact_mode(server_args: ServerArgs) -> bool:
    return bool(getattr(server_args, "glm52_exact_mode", False))


def is_dsv4_flash_exact_mode(server_args: ServerArgs) -> bool:
    return bool(getattr(server_args, "dsv4_flash_exact_mode", False))


def is_qwen35_gdn_exact_mode(server_args: ServerArgs) -> bool:
    return bool(getattr(server_args, "qwen35_gdn_exact_mode", False))


def is_qwen3_dense_exact_mode(server_args: ServerArgs) -> bool:
    return bool(getattr(server_args, "qwen3_dense_exact_mode", False))


def is_xorl_exact_mode(server_args: ServerArgs) -> bool:
    """Return whether a resolved architecture-owned XORL contract is active."""
    return any(
        (
            is_glm52_exact_mode(server_args),
            is_dsv4_flash_exact_mode(server_args),
            is_qwen35_gdn_exact_mode(server_args),
            is_qwen3_dense_exact_mode(server_args),
        )
    )


def is_qwen35_rope_class_b(server_args: ServerArgs) -> bool:
    return bool(getattr(server_args, "qwen35_rope_class_b", False))


def _text_model_config(hf_config):
    return getattr(hf_config, "text_config", hf_config)


def _validate_exact_model_geometry(
    hf_config,
    *,
    contract_name: str,
    expected: dict[str, object],
) -> None:
    config = _text_model_config(hf_config)
    mismatches = []
    for name, expected_value in expected.items():
        actual_value = getattr(config, name, None)
        if name == "rope_theta" and actual_value is None:
            rope_parameters = getattr(config, "rope_parameters", None)
            if isinstance(rope_parameters, dict):
                actual_value = rope_parameters.get("rope_theta")
        if actual_value != expected_value:
            mismatches.append(f"{name}={actual_value!r} (expected {expected_value!r})")
    if mismatches:
        raise ValueError(
            f"The exact {contract_name} XORL contract only admits the qualified "
            f"model geometry; mismatched fields: {', '.join(mismatches)}"
        )


def _validate_exact_qwen3_dense_capabilities(hf_config) -> None:
    config = _text_model_config(hf_config)
    mismatches = []

    positive_integer_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
    )
    for name in positive_integer_fields:
        value = getattr(config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            mismatches.append(f"{name}={value!r} (requires a positive integer)")

    required_values = {
        "head_dim": 128,
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": False,
        "attention_dropout": 0.0,
    }
    for name, required in required_values.items():
        actual = getattr(config, name, None)
        if actual != required:
            mismatches.append(f"{name}={actual!r} (requires {required!r})")

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        unsupported_keys = set(rope_scaling) - {"rope_type", "type", "rope_theta"}
        if rope_type not in (None, "default") or unsupported_keys:
            mismatches.append(
                f"rope_scaling={rope_scaling!r} (only default RoPE is supported)"
            )
    elif rope_scaling:
        mismatches.append(
            f"rope_scaling={rope_scaling!r} (only default RoPE is supported)"
        )

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta")
    if (
        not isinstance(rope_theta, (int, float))
        or isinstance(rope_theta, bool)
        or rope_theta <= 0
    ):
        mismatches.append(f"rope_theta={rope_theta!r} (requires a positive number)")

    rms_norm_eps = getattr(config, "rms_norm_eps", None)
    if (
        not isinstance(rms_norm_eps, (int, float))
        or isinstance(rms_norm_eps, bool)
        or rms_norm_eps <= 0
    ):
        mismatches.append(f"rms_norm_eps={rms_norm_eps!r} (requires a positive number)")

    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    if (
        isinstance(num_attention_heads, int)
        and isinstance(num_key_value_heads, int)
        and num_attention_heads > 0
        and num_key_value_heads > 0
        and num_attention_heads % num_key_value_heads != 0
    ):
        mismatches.append(
            "num_attention_heads must be divisible by num_key_value_heads "
            f"(got {num_attention_heads} and {num_key_value_heads})"
        )

    if mismatches:
        raise ValueError(
            "The exact dense Qwen3 XORL contract does not support this "
            f"architecture configuration: {', '.join(mismatches)}"
        )


def _validate_exact_qwen35_dense_capabilities(hf_config) -> None:
    config = _text_model_config(hf_config)
    mismatches = []

    positive_integer_fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "linear_num_value_heads",
    )
    for name in positive_integer_fields:
        value = getattr(config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            mismatches.append(f"{name}={value!r} (requires a positive integer)")

    required_values = {
        "head_dim": 256,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
    }
    for name, required in required_values.items():
        actual = getattr(config, name, None)
        if actual != required:
            mismatches.append(f"{name}={actual!r} (requires {required!r})")
    if getattr(config, "use_sliding_window", False):
        mismatches.append("use_sliding_window=True (requires False)")

    rms_norm_eps = getattr(config, "rms_norm_eps", None)
    if (
        not isinstance(rms_norm_eps, (int, float))
        or isinstance(rms_norm_eps, bool)
        or rms_norm_eps <= 0
    ):
        mismatches.append(f"rms_norm_eps={rms_norm_eps!r} (requires a positive number)")

    rope_parameters = getattr(config, "rope_parameters", None)
    rope_theta = getattr(config, "rope_theta", None)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    if isinstance(rope_parameters, dict):
        if rope_theta is None:
            rope_theta = rope_parameters.get("rope_theta")
        if partial_rotary_factor is None:
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
    if (
        not isinstance(rope_theta, (int, float))
        or isinstance(rope_theta, bool)
        or rope_theta <= 0
    ):
        mismatches.append(f"rope_theta={rope_theta!r} (requires a positive number)")
    if partial_rotary_factor != 0.25:
        mismatches.append(
            f"partial_rotary_factor={partial_rotary_factor!r} (requires 0.25)"
        )

    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    if (
        isinstance(num_attention_heads, int)
        and isinstance(num_key_value_heads, int)
        and num_attention_heads > 0
        and num_key_value_heads > 0
        and num_attention_heads % num_key_value_heads != 0
    ):
        mismatches.append(
            "num_attention_heads must be divisible by num_key_value_heads "
            f"(got {num_attention_heads} and {num_key_value_heads})"
        )

    linear_num_value_heads = getattr(config, "linear_num_value_heads", None)
    if (
        isinstance(linear_num_value_heads, int)
        and linear_num_value_heads > 0
        and linear_num_value_heads % 16 != 0
    ):
        mismatches.append(
            f"linear_num_value_heads={linear_num_value_heads!r} "
            "(requires a multiple of 16)"
        )

    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    layer_types = getattr(config, "layer_types", None)
    if (
        isinstance(num_hidden_layers, int)
        and num_hidden_layers > 0
        and layer_types is not None
    ):
        expected_layer_types = tuple(
            "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
            for layer_idx in range(num_hidden_layers)
        )
        if tuple(layer_types) != expected_layer_types:
            mismatches.append("layer_types does not match full_attention_interval=4")

    if mismatches:
        raise ValueError(
            "The exact dense Qwen3.5 XORL contract does not support this "
            f"architecture configuration: {', '.join(mismatches)}"
        )


def _exact_batch_invariant_ops(server_args: ServerArgs) -> tuple[str, ...] | None:
    if is_glm52_exact_mode(server_args):
        from sglang.xorl.batch_invariant import XORL_GLM52_REQUIRED_BI_OPS

        return XORL_GLM52_REQUIRED_BI_OPS
    if is_qwen35_gdn_exact_mode(server_args):
        from sglang.xorl.fla.qwen35_gdn_exact import (
            QWEN35_REQUIRED_BI_OPS,
        )

        return QWEN35_REQUIRED_BI_OPS
    if is_qwen3_dense_exact_mode(server_args):
        # Dense Qwen3 owns its norm, activation, lm-head, and probability
        # reductions. Only trunk matrix products use the generic BI interpose.
        return ("addmm", "bmm", "mm")
    return None


@dataclasses.dataclass
class ServerArgs(_UpstreamServerArgs):
    rl_on_policy_target: A[
        Optional[str],
        Arg(
            help="Enable the current exact XoRL on-policy contract.",
            choices=RL_ON_POLICY_TARGET_CHOICES,
        ),
        NS("exec.deterministic"),
    ] = None
    # Architecture-owned runtime selections. These are resolved from the
    # model geometry and XORL target and are intentionally not CLI flags.
    glm52_exact_mode: A[bool, NS("exec.deterministic")] = dataclasses.field(
        init=False, default=False, repr=False
    )
    dsv4_flash_exact_mode: A[bool, NS("exec.deterministic")] = dataclasses.field(
        init=False, default=False, repr=False
    )
    qwen35_gdn_exact_mode: A[bool, NS("exec.deterministic")] = dataclasses.field(
        init=False, default=False, repr=False
    )
    qwen35_gdn_exact_is_moe: A[bool, NS("exec.deterministic")] = dataclasses.field(
        init=False, default=False, repr=False
    )
    qwen35_rope_class_b: A[bool, NS("exec.deterministic")] = dataclasses.field(
        init=False, default=False, repr=False
    )
    qwen35_rmsnorm_family: A[Literal["v1", "v2"], NS("exec.deterministic")] = (
        dataclasses.field(init=False, default="v1", repr=False)
    )
    qwen3_dense_exact_mode: A[bool, NS("exec.deterministic")] = dataclasses.field(
        init=False, default=False, repr=False
    )
    # Capability bit set by an exact model resolver only when the model owns a
    # stage-local PP proxy contract for every bit-relevant boundary.  Admission
    # code consumes this capability rather than maintaining a family allowlist.
    exact_physical_pp_capable: A[bool, NS("exec.deterministic")] = dataclasses.field(
        init=False, default=False, repr=False
    )

    def _handle_return_hidden_states_mode(self):
        # Seam: in-tree, __post_init__ called _validate_rl_on_policy_target
        # immediately after this handler (before the dummy-path early return).
        super()._handle_return_hidden_states_mode()
        self._validate_rl_on_policy_target()

    def _validate_rl_on_policy_target(self) -> None:
        if self.rl_on_policy_target not in (None, XORL_RL_TARGET):
            raise ValueError(
                "--rl-on-policy-target only supports the current exact XORL "
                f"contract ({XORL_RL_TARGET!r}); got "
                f"{self.rl_on_policy_target!r}"
            )

    def _declare_exact_physical_pp_capability(self, *configs) -> None:
        """Publish the model-owned physical-PP proxy capability.

        This is deliberately a mechanism declaration, not a model-family
        admission table.  Exact resolvers set it only after selecting a model
        implementation whose body state, ownership metadata, and terminal head
        are stage-local under physical pipeline parallelism.
        """

        self.exact_physical_pp_capable = True
        for config in configs:
            config._exact_physical_pp_capable = True

    def _validate_qwen35_gdn_exact_contract(self, hf_config) -> None:
        if not self.qwen35_gdn_exact_mode:
            return
        config = _text_model_config(hf_config)
        if self.qwen35_gdn_exact_is_moe:
            _validate_exact_model_geometry(
                config,
                contract_name="Qwen3.6-35B-A3B",
                expected={
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 248320,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "full_attention_interval": 4,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                },
            )
        else:
            _validate_exact_qwen35_dense_capabilities(config)
        if (
            self.speculative_algorithm is not None
            or self.speculative_draft_model_path is not None
            or self.enable_multi_layer_eagle
        ):
            raise ValueError(
                "The exact Qwen3.5-family XORL contract does not support "
                "speculative or draft decoding."
            )

    def _resolve_qwen35_gdn_exact_contract(
        self,
        hf_config,
        *,
        model_arch: str,
    ) -> None:
        from sglang.xorl.fla.qwen35_gdn_exact import (
            QWEN35_EXACT_ARCHS,
            QWEN35_MOE_ARCHS,
        )

        self.qwen35_gdn_exact_mode = (
            self.rl_on_policy_target == XORL_RL_TARGET
            and model_arch in QWEN35_EXACT_ARCHS
        )
        self.qwen35_gdn_exact_is_moe = (
            self.qwen35_gdn_exact_mode and model_arch in QWEN35_MOE_ARCHS
        )
        self.qwen35_rope_class_b = self.qwen35_gdn_exact_mode
        self.qwen35_rmsnorm_family = "v2" if self.qwen35_gdn_exact_mode else "v1"
        hf_config._qwen35_gdn_exact_mode = self.qwen35_gdn_exact_mode
        hf_config._qwen35_gdn_exact_is_moe = self.qwen35_gdn_exact_is_moe
        hf_config._qwen35_rope_class_b = self.qwen35_rope_class_b
        hf_config._qwen35_rmsnorm_family = self.qwen35_rmsnorm_family
        text_config = _text_model_config(hf_config)
        text_config._qwen35_gdn_exact_mode = self.qwen35_gdn_exact_mode
        text_config._qwen35_gdn_exact_is_moe = self.qwen35_gdn_exact_is_moe
        text_config._qwen35_rope_class_b = self.qwen35_rope_class_b
        text_config._qwen35_rmsnorm_family = self.qwen35_rmsnorm_family
        self._validate_qwen35_gdn_exact_contract(hf_config)
        if not self.qwen35_gdn_exact_mode:
            return

        if self.qwen35_gdn_exact_is_moe:
            raise ValueError(
                "Exact Qwen3.5-family MoE serving is not ported to this "
                "dev-based branch: the canonical MoE contributor fold is "
                "main-only. Dense Qwen3.5 exact serving is supported."
            )

        self._declare_exact_physical_pp_capability(hf_config, text_config)

        if self.dtype not in ("auto", "bf16", "bfloat16"):
            raise ValueError(
                "The exact Qwen3.5-family XORL contract requires BF16 dtype"
            )
        if self.quantization is not None:
            raise ValueError(
                "The exact Qwen3.5-family XORL contract requires unquantized weights"
            )
        if self.attention_backend not in (None, "fa4"):
            raise ValueError(
                "The exact Qwen3.5-family XORL contract requires the FA4 backend"
            )
        self.dtype = "bfloat16"
        self.attention_backend = "fa4"
        logger.info(
            "Qwen3.5-family exact numerics: Class-B RoPE with CPU-built fp32 "
            "tables and compiled fp32-chain application; RMSNorm families-v2"
        )
        if self.linear_attn_prefill_backend not in (None, "triton"):
            raise ValueError(
                "The exact Qwen3.5-family XORL contract requires the triton "
                "linear-attention prefill backend"
            )
        self.linear_attn_prefill_backend = "triton"
        if self.linear_attn_decode_backend not in (None, "triton"):
            raise ValueError(
                "The exact Qwen3.5-family XORL contract requires the triton "
                "linear-attention decode backend"
            )
        self.linear_attn_decode_backend = "triton"
        self.enable_fp32_lm_head = True
        self.enable_deterministic_inference = True
        self.sampling_backend = "pytorch"
        self.sampling_defaults = "openai"
        self.disable_custom_all_reduce = True
        if not self.disable_radix_cache:
            # The exact GDN decoder owns additional per-slot state that the
            # generic Mamba checkpoint pool does not store: the fp32 state at
            # the current 64-token chunk boundary and the live partial-chunk
            # qkv/gating rows.  The extra-buffer cache is nevertheless safe:
            # it exposes only aligned recurrent-state checkpoints and
            # re-prefills the suffix, which lets _bi_gdn_decode_seed rebuild
            # those private buffers.  no_buffer can expose an arbitrary token
            # boundary and therefore cannot restore the exact decoder state.
            if self.mamba_radix_cache_strategy == "auto":
                self.mamba_radix_cache_strategy = "extra_buffer"
            elif self.mamba_radix_cache_strategy not in (
                "extra_buffer",
                "extra_buffer_lazy",
            ):
                raise ValueError(
                    "Exact Qwen3.5-family radix reuse requires an aligned "
                    "Mamba checkpoint strategy; use --mamba-radix-cache-strategy "
                    "extra_buffer (or extra_buffer_lazy), or disable radix cache"
                )
            if self.mamba_track_interval % 64 != 0:
                raise ValueError(
                    "Exact Qwen3.5-family radix checkpoints must align with the "
                    "64-token GDN chunk boundary; got "
                    f"--mamba-track-interval={self.mamba_track_interval}"
                )
            if self.enable_int8_mamba_checkpoint:
                raise ValueError(
                    "Exact Qwen3.5-family radix reuse requires lossless recurrent "
                    "state checkpoints; --enable-int8-mamba-checkpoint is not "
                    "supported"
                )
        if (self.tp_size, self.dp_size, self.ep_size) != (1, 1, 1):
            raise ValueError(
                "Exact dense Qwen3.5 requires stage-local TP1/DP1/EP1; physical "
                "pipeline parallelism may still split layers across stages"
            )

    def _resolve_qwen3_dense_exact_contract(
        self,
        hf_config,
        *,
        model_arch: str,
    ) -> None:
        self.qwen3_dense_exact_mode = (
            self.rl_on_policy_target == XORL_RL_TARGET
            and model_arch == "Qwen3ForCausalLM"
        )
        hf_config._qwen3_dense_exact_mode = self.qwen3_dense_exact_mode
        if not self.qwen3_dense_exact_mode:
            return

        self._declare_exact_physical_pp_capability(
            hf_config, _text_model_config(hf_config)
        )

        _validate_exact_qwen3_dense_capabilities(hf_config)
        if self.dtype not in ("auto", "bf16", "bfloat16"):
            raise ValueError("The exact dense Qwen3 XORL contract requires BF16 dtype")
        if self.quantization is not None:
            raise ValueError(
                "The exact dense Qwen3 XORL contract requires unquantized weights"
            )
        if self.attention_backend not in (None, "fa4"):
            raise ValueError(
                "The exact dense Qwen3 XORL contract requires the FA4 backend"
            )
        if (self.tp_size, self.dp_size, self.ep_size) != (1, 1, 1):
            raise ValueError(
                "Exact dense Qwen3 requires stage-local TP1/DP1/EP1; physical "
                "pipeline parallelism may still split layers across stages"
            )
        if (
            self.speculative_algorithm is not None
            or self.speculative_draft_model_path is not None
            or self.enable_multi_layer_eagle
        ):
            raise ValueError(
                "The exact dense Qwen3 XORL contract does not support speculative "
                "or draft decoding"
            )

        self.dtype = "bfloat16"
        self.attention_backend = "fa4"
        self.enable_fp32_lm_head = False
        self.enable_deterministic_inference = True
        self.sampling_backend = "pytorch"
        self.sampling_defaults = "openai"
        self.disable_custom_all_reduce = True
        logger.info(
            "Dense Qwen3 exact numerics: Class-B RoPE, RMSNorm families-v2, "
            "shape-aware exact SwiGLU, and the families-v2 BF16 lm-head"
        )

    def _handle_model_specific_adjustments(self):
        from sglang.srt.configs.model_config import (
            get_mimo_v2_fused_qkv_expected_tp_size,
            is_deepseek_dsa,
        )

        if self.enable_deterministic_inference:
            self.enforce_disable_flashinfer_allreduce_fusion = True

        self.uses_mamba_radix_cache = False
        if parse_connector_type(self.model_path) == ConnectorType.INSTANCE:
            self._resolved_overrides = []
            return

        hf_config = self.get_model_config().hf_config
        model_arch = hf_config.architectures[0]
        self._resolve_qwen35_gdn_exact_contract(hf_config, model_arch=model_arch)
        self._resolve_qwen3_dense_exact_contract(hf_config, model_arch=model_arch)

        if self.enable_dsa_cache_layer_split and not is_deepseek_dsa(hf_config):
            raise ValueError(
                "--enable-dsa-cache-layer-split is only supported for DSA "
                "(DeepSeek Sparse Attention) models."
            )

        if self.enable_cp_decode_attn_tp:
            from sglang.srt.layers.cp.cp_decode_attn_tp import (
                CP_DECODE_ATTN_TP_SUPPORTED_ARCHS,
            )

            if model_arch not in CP_DECODE_ATTN_TP_SUPPORTED_ARCHS:
                raise ValueError(
                    "--enable-cp-decode-attn-tp is only supported for models "
                    "whose attention linears are replicated across CP ranks "
                    f"(attn_tp_size=1). Got {model_arch}; supported: "
                    f"{sorted(CP_DECODE_ATTN_TP_SUPPORTED_ARCHS)}."
                )

        _hybrid_spec = get_linear_attn_spec_by_arch(model_arch)
        if _hybrid_spec is not None and _hybrid_spec.uses_mamba_radix_cache:
            self._handle_mamba_radix_cache(model_arch=model_arch)

        # Collect the declarative model overrides (registry) on the
        # pristine config and stash them for publish-time flags resolution;
        # server_args is never mutated — mid-resolution readers see the
        # declared values through resolved_view, runtime readers through the
        # flags tier.
        from sglang.srt.arg_groups.overrides import (
            collect_model_override_declarations,
            validate_declarations,
        )

        self._resolved_overrides = collect_model_override_declarations(
            model_arch, self, hf_config
        )
        validate_declarations(self, self._resolved_overrides)

        if model_arch in (
            "KimiLinearForCausalLM",
            "KimiK3ForConditionalGeneration",
        ):
            from sglang.srt.arg_groups.kimi_k3_hook import (
                apply_kimi_k3_linear_attn_defaults,
                apply_kimi_k3_spec_backend_defaults,
            )

            apply_kimi_k3_linear_attn_defaults(self)
            apply_kimi_k3_spec_backend_defaults(self)

        if model_arch in [
            "DeepseekV4ForCausalLM",
        ]:
            from sglang.srt.arg_groups.deepseek_v4_hook import (
                apply_deepseek_v4_defaults,
            )

            apply_deepseek_v4_defaults(self, model_arch)

        if model_arch in [
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "KimiK25ForConditionalGeneration",
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
            "GlmMoeDsaForCausalLM",
            "LongcatFlashForCausalLM",
        ]:
            # Set attention backend for DeepSeek
            if is_deepseek_dsa(hf_config):  # DeepSeek 3.2/GLM 5
                if envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.is_set():
                    logger.warning(
                        f"Dense attention kv len threshold is manually set to {envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()} for DSA. Caution: This may cause performance regression if the threshold is larger than the index topk of model."
                    )
                else:
                    # When threshold is not manually set, set it to the index topk of model
                    from sglang.srt.configs.model_config import get_dsa_index_topk

                    envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(
                        get_dsa_index_topk(hf_config)
                    )
                    logger.warning(
                        f"Set dense attention kv len threshold to model index_topk={envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()} for DeepSeek with DSA."
                    )
                # The "dsa" attention fill moved to the override registry
                # (arg_groups/overrides.py: _deepseek_family_overrides).

                index_topk_freq = getattr(hf_config, "index_topk_freq", 1) or 1
                index_topk_pattern = getattr(hf_config, "index_topk_pattern", None)
                if self.enable_two_batch_overlap and (
                    index_topk_freq > 1
                    or (index_topk_pattern is not None and "S" in index_topk_pattern)
                ):
                    raise ValueError(
                        "--enable-two-batch-overlap is not supported with DSA "
                        "index-topk sharing (index_topk_freq > 1 or an "
                        "index_topk_pattern containing shared layers): the TBO op "
                        "path does not propagate topk indices across layers, so "
                        "shared layers would run sparse attention without indices."
                    )

                if not is_npu() and not is_xpu():  # CUDA or ROCm GPU
                    if self.enable_prefill_cp:
                        # The DSA CP field declarations moved to the override
                        # registry (arg_groups/overrides.py:
                        # _deepseek_family_overrides).
                        self.cuda_graph_config.prefill.backend = Backend.DISABLED
                    else:
                        # Pure TP and partial DP Attention mode is active for DSA, logging a warning
                        if self.dp_size < self.tp_size:
                            logger.warning(
                                f"DSA with TP mode is active, dp_size={self.dp_size}, tp_size={self.tp_size}, "
                                f"attn_tp_size={self.tp_size}, attention weights will be sharded across {self.tp_size} ranks."
                            )

                    # The DSA page-size selection moved to the override registry
                    # (arg_groups/overrides.py: _deepseek_family_overrides).

                    import torch

                    major, _ = torch.cuda.get_device_capability()
                    self._set_default_dsa_kv_cache_dtype(
                        major, resolved_view(self).quantization
                    )
                    self._set_default_dsa_backends(major)

                if self.enable_prefill_cp:
                    assert (
                        self.disaggregation_mode != "decode"
                    ), "CP is only supported for prefill when PD disaggregation, please remove --enable-prefill-cp."
                if (
                    self.enable_dsa_cache_layer_split
                    and self.disaggregation_mode != "prefill"
                ):
                    if self.disaggregation_mode == "decode":
                        raise ValueError(
                            "--enable-dsa-cache-layer-split is not supported on "
                            "decode workers. This flag is a prefill-CP "
                            "optimization; decode receives full cache shards "
                            "through PD transfer."
                        )
                    raise ValueError(
                        "--enable-dsa-cache-layer-split is only supported on PD "
                        "prefill workers. Non-PD workers also run decode and "
                        "require ordinary local decode cache semantics."
                    )
                if self.enable_dsa_cache_layer_split and (
                    not self.enable_prefill_cp or self.cp_strategy != "interleave"
                ):
                    raise ValueError(
                        "--enable-dsa-cache-layer-split requires "
                        "--enable-prefill-cp and --cp-strategy interleave "
                        "(or legacy --enable-nsa-prefill-context-parallel with "
                        "--nsa-prefill-cp-mode round-robin-split)."
                    )
                # Layer split relies on the mooncake all-CP-rank KV/indexer
                # transfer path. mori/nixl support is a temporary limitation
                # and will be added later by the community.
                if (
                    self.enable_dsa_cache_layer_split
                    and self.disaggregation_transfer_backend != "mooncake"
                ):
                    raise ValueError(
                        "--enable-dsa-cache-layer-split currently only supports "
                        "the mooncake transfer backend (mooncake / mooncake_tcp). "
                        f"Got --disaggregation-transfer-backend "
                        f"{self.disaggregation_transfer_backend!r}. mori/nixl "
                        "support will be added later by the community."
                    )
                if self.enable_dsa_cache_layer_split and self.pp_size > 1:
                    raise ValueError(
                        "--enable-dsa-cache-layer-split is not supported with "
                        "pipeline parallelism (pp_size > 1) yet. It requires "
                        "prefill context parallelism, and CP + PP has not been "
                        "validated for this feature."
                    )

            else:
                # DeepSeek V3/R1/V3.1
                if self.cuda_graph_config.prefill.backend != Backend.DISABLED:
                    logger.info("Piecewise CUDA graph is enabled, use MLA for prefill.")

                # The sm100 trtllm_mla fill moved to the override registry
                # (arg_groups/overrides.py: _deepseek_family_overrides).

                # MLA prefill CP auto-config: the field declarations moved to
                # the override registry (arg_groups/overrides.py:
                # _deepseek_family_overrides).
                if self.enable_prefill_cp and self.use_mla_backend():
                    self.cuda_graph_config.prefill.backend = Backend.DISABLED

            # Set moe backend for DeepSeek: the sm100 quant/moe resolution
            # moved to the resolution pipeline (arg_groups/overrides.py:
            # _deepseek_moe_quant_resolution -- a slot pass, because the DSA
            # kv-cache-dtype default above must read the pristine
            # quantization). The HIP arm (fusion log + spec_moe writes, the
            # latter awaiting the speculative-hook migration) stays below.
            from sglang.srt.arg_groups.overrides import (
                _deepseek_moe_quant_resolution,
                run_post_process_pass,
            )

            run_post_process_pass(self, _deepseek_moe_quant_resolution)
            if is_hip():
                if is_deepseek_dsa(hf_config):
                    # The fused top-k v2 kernel (topk_transform_512_v2) is a
                    # CUDA/Hopper-only path: its JIT source includes
                    # <cooperative_groups.h> and uses cg::this_cluster()
                    # (thread-block clusters), neither of which exists on ROCm,
                    # so it fails to JIT-compile on gfx9xx during CUDA-graph
                    # capture. DeepSeek-V4 already disables it on HIP; mirror that
                    # here for the rest of the DSA family (DeepSeek-V3.2 /
                    # GLM-5.x) that shares the same decode top-k path.
                    envs.SGLANG_OPT_USE_TOPK_V2.set(False)
                if not self._resolved().enable_dp_attention and self.nnodes == 1:
                    # TODO (Hubert): Put this back later
                    # self.enable_aiter_allreduce_fusion = True
                    logger.info(
                        "Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM"
                    )

                # The fp4-checkpoint draft spec-MoE resolution moved to the
                # resolution pipeline (arg_groups/overrides.py:
                # _deepseek_spec_moe_resolution), invoked here at its legacy
                # slot.
                from sglang.srt.arg_groups.overrides import (
                    _deepseek_spec_moe_resolution,
                )

                run_post_process_pass(self, _deepseek_spec_moe_resolution)

        elif model_arch in [
            "DeepseekV4ForCausalLM",
        ]:
            from sglang.srt.arg_groups.deepseek_v4_hook import (
                validate_deepseek_v4_cp,
                validate_deepseek_v4_mega_moe_token_budget,
            )

            validate_deepseek_v4_cp(self)
            validate_deepseek_v4_mega_moe_token_budget(self)

            # The SM120 marlin fallback moved to the resolution pipeline
            # (arg_groups/overrides.py: _deepseek_v4_sm120_moe), invoked here
            # at its legacy slot.
            from sglang.srt.arg_groups.overrides import (
                _deepseek_v4_sm120_moe,
                run_post_process_pass,
            )

            run_post_process_pass(self, _deepseek_v4_sm120_moe)
            if is_sm120_supported():
                # SM120 lacks tcgen05/TMEM: disable features that depend on
                # DeepGEMM or require >99KB SMEM (topk_v2).
                envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)
                envs.SGLANG_OPT_USE_TOPK_V2.set(False)
                envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
                envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)
                envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)
                # Prefer TileLang over the Torch fallback.
                envs.SGLANG_OPT_USE_TILELANG_INDEXER.set(True)
            elif is_hip():
                envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)
                envs.SGLANG_OPT_USE_FUSED_COMPRESS.set(True)
                envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)
                envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.set(False)
                envs.SGLANG_OPT_USE_TOPK_V2.set(False)
                envs.SGLANG_OPT_USE_AITER_INDEXER.set(True)
                envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
                envs.SGLANG_OPT_USE_TILELANG_MHC_POST.set(False)
                envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)
                envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.set(False)
                envs.SGLANG_EAGER_INPUT_NO_COPY.set(True)

        elif model_arch in ["GptOssForCausalLM"]:
            # Attention backend selection + XPU dtype validation moved to the
            # override registry (arg_groups/overrides.py: _gpt_oss_overrides).

            supported_backends = [
                "triton",
                "trtllm_mha",
                "fa3",
                "fa4",
                "ascend",
                "intel_amx",
                "intel_xpu",
                "aiter",
            ]
            prefill_attn_backend, decode_attn_backend = (
                self._resolved_attention_backends()
            )
            assert (
                prefill_attn_backend in supported_backends
                and decode_attn_backend in supported_backends
            ), (
                f"GptOssForCausalLM requires one of {supported_backends} attention backend, but got the following backends\n"
                f"- Prefill: {prefill_attn_backend}\n"
                f"- Decode: {decode_attn_backend}\n"
            )

            quant_method = get_quantization_config(hf_config)
            is_mxfp4_quant_format = quant_method == "mxfp4"
            if (
                not self._resolved().enable_dp_attention
                and self.nnodes == 1
                and is_hip()
            ):
                # TODO (Hubert): Put this back later
                # self.enable_aiter_allreduce_fusion = True
                logger.info("Enable Aiter AllReduce Fusion for GptOssForCausalLM")
            quantization_config = getattr(hf_config, "quantization_config", None)
            is_mxfp4_quant_format = (
                quantization_config is not None
                and quantization_config.get("quant_method") == "mxfp4"
            )
            # The mxfp4 dtype override moved to the override registry
            # (arg_groups/overrides.py: _gpt_oss_overrides).

            # The moe_runner_backend selection moved to the override registry
            # (arg_groups/overrides.py: _gpt_oss_overrides).

            if resolved_view(self).moe_runner_backend == "triton_kernel":
                assert (
                    self._resolved().ep_size == 1
                ), "Triton kernel MoE is only supported when ep_size == 1"

        elif model_arch in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM"):
            if model_arch == "MiMoV2ForCausalLM" and not self.encoder_only:
                expected_attn_tp_size = get_mimo_v2_fused_qkv_expected_tp_size(
                    hf_config
                )
                view = self._resolved()
                attn_dp_size = self.dp_size if view.enable_dp_attention else 1
                effective_attn_tp_size = (
                    self.tp_size // attn_dp_size // view.attn_cp_size
                )
                if (
                    expected_attn_tp_size is not None
                    and expected_attn_tp_size % effective_attn_tp_size != 0
                ):
                    raise ValueError(
                        "MiMoV2ForCausalLM requires effective attention TP "
                        f"size {expected_attn_tp_size} because its fused "
                        "qkv_proj weights are "
                        f"TP={expected_attn_tp_size}-interleaved; got "
                        f"{effective_attn_tp_size} "
                        f"(tp_size={self.tp_size}, dp_size={self.dp_size}, "
                        f"enable_dp_attention={view.enable_dp_attention}, "
                        f"attn_cp_size={view.attn_cp_size}). "
                        "Set --tp, --dp, --enable-dp-attention, and "
                        "--attention-context-parallel-size so the effective "
                        f"attention TP size is {expected_attn_tp_size}."
                    )

            # enable_multi_layer_eagle for EAGLE moved to the override registry
            # (arg_groups/overrides.py: _mimo_v2_overrides).

            if self.enable_hierarchical_cache:
                if not envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get():
                    raise ValueError(
                        "Hierarchical cache for MiMoV2 requires the unified "
                        "radix tree. Set SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 "
                        "to enable --enable-hierarchical-cache for this model."
                    )

                # MiMoV2 has head_dim != v_head_dim, so the host KV pool uses
                # asymmetric K/V allocation. Both kernel/page_first and
                # direct/page_first_direct have split K/V transfer paths.
        elif (
            "Step3p5ForCausalLM" in model_arch
            or "Step3p7ForConditionalGeneration" in model_arch
        ):
            # Attention backend selection + EAGLE multi-layer +
            # hierarchical-cache SWA writes moved to the override registry
            # (arg_groups/overrides.py: _step3p_overrides).
            pass
        elif (
            model_arch in ("Llama4ForConditionalGeneration", "Llama4ForCausalLM")
            and self.device != "cpu"
        ):
            # Attention backend auto-select moved to the override registry
            # (arg_groups/overrides.py: _llama4_overrides).
            attention_backend = resolved_view(self).attention_backend
            assert attention_backend in {
                "fa3",
                "aiter",
                "triton",
                "ascend",
                "trtllm_mha",
                "intel_xpu",
            }, f"fa3, aiter, triton, ascend, trtllm_mha or intel_xpu is required for Llama4 model but got {attention_backend}"
            # The moe_runner_backend selection moved to the override registry
            # (arg_groups/overrides.py: _llama4_overrides).
        # Gemma2/Gemma3 (disable_hybrid_swa_memory) moved to the override registry
        # (arg_groups/overrides.py: _gemma2_gemma3_overrides).
        elif model_arch in (
            "Gemma4ForConditionalGeneration",
            "Gemma4ForCausalLM",
            "Gemma4UnifiedForConditionalGeneration",
        ):
            # Default attention backend selection moved to the override registry
            # (arg_groups/overrides.py: _gemma4_overrides).
            prefill_backend, decode_backend = self._resolved_attention_backends()
            accepted_backends = ("trtllm_mha", "triton", "ascend", "intel_xpu")
            assert (
                prefill_backend in accepted_backends
                and decode_backend in accepted_backends
            ), (
                "Gemma4 only supports trtllm_mha, triton, or intel_xpu attention backend, "
                f"got prefill={prefill_backend}, decode={decode_backend}"
            )

            # The quantization/moe_runner_backend resolution moved to the override
            # registry (arg_groups/overrides.py: _gemma4_overrides).
        elif model_arch == "MossVLForConditionalGeneration":
            # The prefill attention backend default + validation moved to the
            # override registry (arg_groups/overrides.py: _moss_vl_overrides).
            pass
        elif model_arch in ["Exaone4ForCausalLM", "ExaoneMoEForCausalLM"]:
            if hf_config.sliding_window_pattern is not None:
                # disable_hybrid_swa_memory moved to the override registry
                # (arg_groups/overrides.py: _exaone_overrides).
                # https://docs.sglang.ai/advanced_features/attention_backend.html
                accepted_backends = ["fa3", "triton", "trtllm_mha"]
                attention_backend = resolved_view(self).attention_backend
                assert (
                    attention_backend in accepted_backends
                ), f"One of the attention backends in {accepted_backends} is required for {model_arch}, but got {attention_backend}"
        elif model_arch in ["Olmo2ForCausalLM"]:
            # disable_hybrid_swa_memory + attention backend selection moved to
            # the override registry (arg_groups/overrides.py: _olmo2_overrides).

            # Flashinfer appears to degrade performance when sliding window attention
            # is used for the Olmo2 architecture. Olmo2 does not use sliding window attention
            # but Olmo3 does.
            attention_backend = resolved_view(self).attention_backend
            assert (
                attention_backend != "flashinfer"
            ), "FlashInfer backend can significantly degrade the performance of Olmo3 models."

            logger.info(
                f"Using {attention_backend} as attention backend for {model_arch}."
            )
        elif model_arch in ["NemotronHForCausalLM", "NemotronHPuzzleForCausalLM"]:
            # Quantization / MoE runner / attention backend defaults moved to
            # the override registry (arg_groups/overrides.py:
            # _nemotron_h_overrides).
            assert resolved_view(self).attention_backend != "triton", (
                "NemotronHForCausalLM does not support triton attention backend,"
                "as the first layer might not be an attention layer"
            )
        elif model_arch in [
            "Qwen3MoeForCausalLM",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3NextForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "InternS2PreviewForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
        ]:
            # The quantization/moe_runner_backend resolution moved to the
            # override registry (arg_groups/overrides.py:
            # _qwen3_moe_family_overrides); the hybrid sub-family's attention
            # backend + page size defaults to _qwen3_5_hybrid_overrides.
            pass

        elif model_arch in ["Glm4MoeForCausalLM"]:
            # The quantization/moe_runner_backend/enable_tf32_matmul resolution
            # moved to the override registry (arg_groups/overrides.py:
            # _glm4_moe_overrides).
            pass

        elif model_arch in ["Lfm2ForCausalLM", "Lfm2MoeForCausalLM"]:
            # Attention backend selection moved to the override registry
            # (arg_groups/overrides.py: _lfm2_overrides).
            assert resolved_view(self).attention_backend != "triton", (
                f"{model_arch} does not support triton attention backend, "
                "as the first layer might not be an attention layer"
            )

        # MiniMaxM2ForCausalLM (enable_tf32_matmul) moved to the override registry
        # (arg_groups/overrides.py: _minimax_m2_overrides).

        # Qwen3VL aiter unified-attention page_size moved to the override registry
        # (arg_groups/overrides.py: _qwen3vl_overrides).

        # Hybrid-mamba radix cache handling for the per-arch branch call sites
        # dissolved above: the resolution pass self-guards on the arch union
        # (and the Granite layer_types probe), so one call covers them all.
        # Hybrid-spec archs already resolved at the pre-dispatch call above;
        # for them this re-invocation is an idempotent no-op plus validation.
        # Kept ahead of the sparse-head pass: the legacy per-branch calls
        # resolved before that tail write of disable_overlap_schedule.
        self._handle_mamba_radix_cache(model_arch=model_arch)

        from sglang.srt.arg_groups.overrides import (
            _sparse_head_overlap_disable,
            run_post_process_pass,
        )

        run_post_process_pass(self, _sparse_head_overlap_disable)

        # The FlashInfer AllReduce Fusion auto-enable and the enforce-disable
        # terminal moved to the resolution pipeline (arg_groups/overrides.py:
        # _flashinfer_allreduce_fusion_auto_enable /
        # _enforce_disable_allreduce_fusion), invoked here at their legacy
        # slots.
        from sglang.srt.arg_groups.overrides import (
            _enforce_disable_allreduce_fusion,
            _flashinfer_allreduce_fusion_auto_enable,
        )

        run_post_process_pass(self, _flashinfer_allreduce_fusion_auto_enable)
        run_post_process_pass(self, _enforce_disable_allreduce_fusion)


def __apply_patch__(public_mod):
    mod = public_mod

    # --- module-level names the copies (and other twins) resolve via mod ---
    mod.XORL_RL_TARGET = XORL_RL_TARGET
    # In place: upstream Arg(choices=...) captured this list object by reference.
    mod.RL_ON_POLICY_TARGET_CHOICES[:] = RL_ON_POLICY_TARGET_CHOICES
    for _f in (
        is_glm52_exact_mode,
        is_dsv4_flash_exact_mode,
        is_qwen35_gdn_exact_mode,
        is_qwen3_dense_exact_mode,
        is_xorl_exact_mode,
        is_qwen35_rope_class_b,
        _text_model_config,
        _validate_exact_model_geometry,
        _validate_exact_qwen3_dense_capabilities,
        _validate_exact_qwen35_dense_capabilities,
        _exact_batch_invariant_ops,
    ):
        setattr(mod, _f.__name__, rebind(_f, mod))

    # --- rebind the verbatim method copies over the live module dict ---
    for _name in (
        "_validate_rl_on_policy_target",
        "_declare_exact_physical_pp_capability",
        "_validate_qwen35_gdn_exact_contract",
        "_resolve_qwen35_gdn_exact_contract",
        "_resolve_qwen3_dense_exact_contract",
        "_handle_model_specific_adjustments",
    ):
        setattr(ServerArgs, _name, rebind(ServerArgs.__dict__[_name], mod))

    mod.ServerArgs = ServerArgs

    # --- experimental LoRA opt defaults (pre-existing twin behavior) ---
    orig_post_init = ServerArgs.__post_init__

    def __post_init__(self, *args, **kwargs):
        orig_post_init(self, *args, **kwargs)
        # After resolution, so moe_runner_backend is final. Sets os.environ only
        # -- no ServerArgs field is mutated, so the strict mutation guard and the
        # writer ratchet are untouched. Spawned scheduler processes inherit the
        # env, which is what makes the import-time gate in lora/layers.py see it.
        maybe_enable_experimental_lora_opti(self)

    ServerArgs.__post_init__ = __post_init__
