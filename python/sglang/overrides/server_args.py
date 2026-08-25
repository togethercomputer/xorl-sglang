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

Both are skipped if the user set them explicitly (either direction), so
``SGLANG_EXPERIMENTAL_LORA_OPTI=0`` remains a working opt-out.

Note ``SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC`` is a local ``_GatedBool`` in
``trtllm_lora_temp/environ.py`` reading ``os.environ`` directly -- it is not an
``Envs`` descriptor -- so it is set through ``os.environ`` rather than the
``envs`` API.
"""

import logging
import os

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_TRTLLM_MOE_BACKEND = "experimental_sgl_trtllm"
# Read via os.environ by trtllm_lora_temp/environ.py's _GatedBool, not via Envs.
_OVERLAP_ALLOC_ENV = "SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC"


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
    if master.is_set() or _OVERLAP_ALLOC_ENV in os.environ:
        # Explicit user intent on either flag: leave the whole pairing alone
        # rather than half-applying it.
        return

    master.set(True)
    os.environ[_OVERLAP_ALLOC_ENV] = "1"
    # warning, not info: this runs inside ServerArgs.__post_init__, before sglang
    # configures logging, so an info record has no handler and is dropped. It is
    # also a default-flip the operator should see in the log.
    logger.warning(
        "moe_runner_backend=%s with LoRA: defaulting "
        "SGLANG_EXPERIMENTAL_LORA_OPTI=1 and %s=1 "
        "(set either explicitly to opt out).",
        _TRTLLM_MOE_BACKEND,
        _OVERLAP_ALLOC_ENV,
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
