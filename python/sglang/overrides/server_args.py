"""Override twin of ``sglang.srt.server_args``.

Hardcodes the experimental LoRA optimizations for every LoRA run.

The companion requirement -- that LoRA may only be *served* on
``--moe-runner-backend experimental_sgl_trtllm`` -- is enforced in the
``lora.lora_manager`` twin instead. Raising during arg resolution would make a
LoRA ``ServerArgs`` unconstructible and break pure-config unit tests that build
one without ever loading a model.

**Two env vars are hardcoded on for every LoRA run**, overriding any user value:

``SGLANG_EXPERIMENTAL_LORA_OPTI``
    Master gate. Enables the fused/split-K kernel opts *and* installs the
    two-stream LoRA overlap monkey-patch (``lora/layers.py`` does this at module
    import time, which is why this must be settled before that import).
    Without it the backend is *slower* than the triton MoE runner; with it,
    +6-15% (4x B200, Qwen3-30B-A3B bf16, TP=4, 512 in / 128 out, output tok/s):

        batch   triton   trtllm(off)   trtllm(on)
            1      161           154          186
            8      660           654          709
           32     3969          3699         4391
           64     7356          6625         7815

``SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC``
    Ships defaulted **off** in ``trtllm_lora_temp/environ.py`` even though it
    fixes a documented correctness hazard of the overlap path the master gate
    installs: a shrink-output buffer allocated inside
    ``with torch.cuda.stream(side)`` is tagged to the side stream, so the
    caching allocator may reuse it before the main stream's LoRA-B expand has
    consumed it -- a premature-reuse WAR the source describes as "qwen3.5 mamba
    decode garbage" under cuda-graph replay. It is set together with the master
    gate because enabling the overlap without its mitigation is the wrong half
    of the change.

``SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC`` is a local ``_GatedBool`` in
``trtllm_lora_temp/environ.py`` reading ``os.environ`` directly -- it is not an
``Envs`` descriptor -- so it is set through ``os.environ`` rather than ``envs``.
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


def hardcode_lora_opts(server_args) -> None:
    """Force the experimental LoRA opts on for a LoRA run on the TRT-LLM runner.

    No-op for non-LoRA runs and for LoRA on any other MoE runner. The companion
    *requirement* -- that MoE LoRA may only be served on this backend -- is
    enforced in the ``lora.lora_manager`` twin at serving time; raising here would
    make a LoRA ServerArgs unconstructible and break pure-config unit tests.
    """
    if not _lora_requested(server_args):
        return
    if getattr(server_args, "moe_runner_backend", None) != _TRTLLM_MOE_BACKEND:
        # Only this backend's path consumes these flags, and the master gate also
        # installs the two-stream overlap onto dense LoRA modules -- so a LoRA run
        # on any other runner is left completely alone.
        return

    master = envs.SGLANG_EXPERIMENTAL_LORA_OPTI
    overridden = []
    if master.is_set() and master.get() is not True:
        overridden.append(f"SGLANG_EXPERIMENTAL_LORA_OPTI={master.get()!r}")
    if os.environ.get(_OVERLAP_ALLOC_ENV) not in (None, "1"):
        overridden.append(f"{_OVERLAP_ALLOC_ENV}={os.environ[_OVERLAP_ALLOC_ENV]!r}")

    master.set(True)
    os.environ[_OVERLAP_ALLOC_ENV] = "1"

    # warning, not info: this runs inside ServerArgs.__post_init__, before sglang
    # configures logging, so an info record has no handler and is dropped.
    if overridden:
        logger.warning(
            "LoRA on %s: forcing SGLANG_EXPERIMENTAL_LORA_OPTI=1 and %s=1, "
            "ignoring %s (these are not user-tunable on this path).",
            _TRTLLM_MOE_BACKEND,
            _OVERLAP_ALLOC_ENV,
            ", ".join(overridden),
        )
    else:
        logger.warning(
            "LoRA on %s: SGLANG_EXPERIMENTAL_LORA_OPTI=1 and %s=1 (hardcoded).",
            _TRTLLM_MOE_BACKEND,
            _OVERLAP_ALLOC_ENV,
        )


def __apply_patch__(public_mod):
    server_args_cls = public_mod.ServerArgs
    orig_post_init = server_args_cls.__post_init__

    def __post_init__(self, *args, **kwargs):
        orig_post_init(self, *args, **kwargs)
        # After resolution, so moe_runner_backend is final. Writes os.environ
        # only -- no ServerArgs field is mutated, so the strict mutation guard
        # and the writer ratchet are untouched (a field write is also why the
        # backend is *rejected* rather than silently rewritten here). Spawned
        # schedulers inherit the env, which is what makes the import-time gate
        # in lora/layers.py see it.
        hardcode_lora_opts(self)

    server_args_cls.__post_init__ = __post_init__
