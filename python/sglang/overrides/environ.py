"""Registry for override-only environment variables.

These env vars are referenced only by code under ``sglang/overrides/``. They live
here (and not in ``sglang.srt.environ``) so a periodic copy-from-OSS workflow,
which may overwrite ``sglang/srt/environ.py`` wholesale, cannot drop them.

The registry attaches each ``EnvField`` descriptor to the upstream ``Envs`` class
at import time, so call sites of the form ``envs.<NAME>.get()`` keep working
unchanged. Import this module before any code that reads these vars —
``sglang.overrides.patches.__init__`` does that for every entry point.
"""

from sglang.srt.environ import EnvField, Envs


def _register(name: str, field: EnvField) -> None:
    # ``__set_name__`` only fires when the descriptor is assigned inside a class
    # body, so set the name manually before attaching to ``Envs``.
    field.name = name
    setattr(Envs, name, field)


# Register override-only env vars here, e.g.:
#
#     from sglang.srt.environ import EnvBool
#     _register("SGLANG_ENABLE_MY_OVERRIDE", EnvBool(False))
#
# The name IS the environment variable: EnvField.get() reads
# os.getenv(self.name). So the same conventions apply as for entries in the
# Envs body -- SGLANG_ prefix, then a verb (ENABLE / DISABLE / USE / FORCE /
# LOG / TEST / DEBUG / OPT), and never DISABLE_FOO defaulting to True, which
# reads as a double negative at the call site. Registering here rather than in
# the class body changes where the descriptor lives, not what it is called.

from sglang.srt.environ import EnvBool

# Kill-switch for the FP8 MoE-LoRA gate_up delta decomposition (single-stream
# experimental_sgl_trtllm path, srt/lora/trtllm_lora_temp/lora_dispatch.py --
# fork-local code, ledgered in UPSTREAM_DRIFT.md). The decomposition keeps
# GEMM2's quantized operand delta-free and runs the activation-level LoRA
# delta through its own own-scale FP8 GEMM: without it, a trained delta below
# e4m3's per-element step relative to the base activation is replaced by
# quantization-grid noise (GLM-5.2 adapters measured at 0.4% of activation --
# recovered-delta cosine 0.30 fused vs 0.9998 decomposed).
_register("SGLANG_DISABLE_LORA_FP8_DELTA_SPLIT", EnvBool(False))
