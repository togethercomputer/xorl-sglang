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

import logging
import os

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_TRTLLM_MOE_BACKEND = "experimental_sgl_trtllm"
# Read via os.environ by trtllm_lora_temp/environ.py's _GatedBool, not via Envs.
_OVERLAP_ALLOC_ENV = "SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC"
_FUSED_PERMUTE_QUANT_ENV = "SGLANG_OPT_FUSED_PERMUTE_QUANT"
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


def __apply_patch__(public_mod):
    server_args_cls = public_mod.ServerArgs
    orig_post_init = server_args_cls.__post_init__

    def __post_init__(self, *args, **kwargs):
        orig_post_init(self, *args, **kwargs)
        # After resolution, so moe_runner_backend is final. Sets os.environ only
        # -- no ServerArgs field is mutated, so the strict mutation guard and the
        # writer ratchet are untouched. Spawned scheduler processes inherit the
        # env, which is what makes the import-time gate in lora/layers.py see it.
        maybe_enable_experimental_lora_opti(self)

    server_args_cls.__post_init__ = __post_init__
