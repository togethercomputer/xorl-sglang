"""Override twin of ``sglang.srt.server_args``.

Locks LoRA serving in this fork to the TRT-LLM MoE runner, and hardcodes the
experimental LoRA optimizations that make it worth using.

**LoRA requires ``--moe-runner-backend experimental_sgl_trtllm``.** Any other
resolved MoE runner on a LoRA run raises. This is deliberate: that backend is
the only MoE LoRA path this fork develops and validates, and silently serving
LoRA on a different runner produces numerically different results (see the
padded-intermediate bug fixed in the ``lora.mem_pool`` / ``lora.layers`` twins,
which only that backend exercises).

Note the default resolves to ``flashinfer_trtllm`` on SM100, *not* to
``experimental_sgl_trtllm``, so the flag must be passed explicitly even for a
plain ``--enable-lora`` run.

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


def enforce_trtllm_lora_and_hardcode_opts(server_args) -> None:
    """Reject non-TRT-LLM MoE runners on LoRA runs; force the experimental opts.

    No-op for non-LoRA runs -- this constrains LoRA serving only, so plain
    (non-LoRA) deployments keep every MoE runner choice.
    """
    if not _lora_requested(server_args):
        return

    backend = getattr(server_args, "moe_runner_backend", None)
    if backend != _TRTLLM_MOE_BACKEND:
        raise ValueError(
            f"LoRA serving in this fork requires "
            f"--moe-runner-backend {_TRTLLM_MOE_BACKEND}, but the resolved MoE "
            f"runner is {backend!r}. It is the only MoE LoRA path this fork "
            f"validates; other runners take a different numerical path. "
            f"Pass --moe-runner-backend {_TRTLLM_MOE_BACKEND} explicitly "
            f"(the default resolves to 'flashinfer_trtllm' on SM100), or drop "
            f"--enable-lora / --lora-paths to serve without LoRA."
        )

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
        enforce_trtllm_lora_and_hardcode_opts(self)

    server_args_cls.__post_init__ = __post_init__
