"""Override twin of ``sglang.srt.lora.lora_manager``.

Enforces that **MoE** LoRA is only *served* on the TRT-LLM MoE runner --
**on Blackwell**, which is the only place that runner exists.

On Hopper and older the TRT-LLM MoE kernels are simply absent (they ship as
precompiled cubins for sm100f / sm103a / sm107a only), so MoE LoRA falls back to
the triton MoE runner with a warning rather than being refused. Requiring a
backend the hardware cannot load would make MoE LoRA unservable there.

Scoped to models that actually contain ``FusedMoE`` layers. ``experimental_sgl_trtllm``
is a *MoE* runner: on a dense model the setting is inert, so requiring it there
would be meaningless, and it would additionally drag the experimental two-stream
overlap onto dense LoRA modules (qkv / o_proj / merged-column forwards) along
with the premature-reuse hazard that path documents. Dense LoRA therefore keeps
every backend choice.

Why here and not in ``ServerArgs.__post_init__``: raising during arg resolution
makes a LoRA ServerArgs unconstructible, which breaks pure-config unit tests
that build one to assert parsing behaviour and never load a model
(``unit/server_args/test_server_args.py``, ``unit/test_server_args_migration.py``,
``unit/managers/test_io_struct.py``, ``unit/entrypoints/test_server_info.py``).
``LoRAManager`` is constructed only when LoRA is actually being served, so the
check lands on serving without constraining configuration.

The env hardcoding stays in the ``server_args`` twin, because
``lora/layers.py`` reads the master gate at module import time and that has to be
settled earlier than this.
"""

import logging

from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.utils.common import is_blackwell_supported

logger = logging.getLogger(__name__)

_TRTLLM_MOE_BACKEND = "experimental_sgl_trtllm"


def _base_model_has_moe(base_model) -> bool:
    """True iff the loaded model has any FusedMoE layer.

    Checked on the constructed model rather than guessed from the HF config, so
    it needs no per-architecture table and no config loading.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    return any(isinstance(m, FusedMoE) for m in base_model.modules())


def _check_virtual_experts(server_args) -> None:
    """Fail at startup, not mid-forward, when virtual experts are required.

    ``lora_dispatch.py`` hard-asserts ``use_virtual_lora_store`` on the bf16
    (:356) and NVFP4 (:507) MoE LoRA paths -- only the FP8 path has a non-virtual
    fallback. Without this, forgetting ``--lora-use-virtual-experts`` costs a full
    model load across every TP rank before dying inside the first forward with a
    bare AssertionError.

    Not hardcoded on the user's behalf, unlike the two env vars the server_args
    twin forces: ``lora_use_virtual_experts`` is a ServerArgs *field*, which the
    strict mutation guard forbids writing after resolution, it carries no
    ``Arg(..., resolvable=True)`` so it is not declarable either, and
    ``LoRAManager`` reads it off the ServerArgs instance rather than the config
    bag -- so a bag override would not reach the reader.
    """
    if getattr(server_args, "lora_use_virtual_experts", False):
        return
    raise ValueError(
        f"MoE LoRA on --moe-runner-backend {_TRTLLM_MOE_BACKEND} requires "
        f"--lora-use-virtual-experts. The bf16 and NVFP4 fused-experts paths "
        f"assert on it (lora_dispatch.py:356 / :507); only the FP8 path has a "
        f"non-virtual fallback. Passing it up front turns a mid-forward "
        f"AssertionError into this message."
    )


def _check_moe_runner_backend(base_model) -> None:
    """Require the TRT-LLM MoE runner for MoE LoRA -- but only where it exists.

    The lock is scoped to Blackwell because that is the only place the backend
    can run at all. ``experimental_sgl_trtllm`` and ``flashinfer_trtllm`` both
    dispatch the same TRT-LLM-generated kernels, which ship exclusively as
    precompiled cubins for sm100f / sm103a / sm107a -- there is no sm90 cubin in
    the package, and the JIT spec declares ``supported_major_versions=[10, 12]``
    accordingly. On Hopper the backend cannot be selected, so requiring it would
    make MoE LoRA unservable and point the operator at a backend their hardware
    cannot run.
    """
    if not _base_model_has_moe(base_model):
        # Dense model: the MoE runner is inert here, nothing to enforce.
        return

    backend = get_moe_runner_backend()

    if not is_blackwell_supported():
        if backend.is_experimental_sgl_trtllm():
            raise ValueError(
                f"--moe-runner-backend {_TRTLLM_MOE_BACKEND} requires a "
                f"Blackwell GPU (SM100/SM103/SM107). Its kernels ship only as "
                f"precompiled cubins for those architectures, so on this device "
                f"the JIT build fails with 'No supported CUDA architectures "
                f"found for major versions [10, 12]'. Drop the flag to serve "
                f"MoE LoRA on the triton MoE runner instead."
            )
        # Hopper and older: triton is the supported MoE LoRA path here.
        logger.warning(
            "MoE LoRA on a non-Blackwell GPU: serving on the %r MoE runner. "
            "%s is Blackwell-only, so the fork's tuned MoE LoRA path and its "
            "measured numerics do not apply here.",
            getattr(backend, "value", backend),
            _TRTLLM_MOE_BACKEND,
        )
        return

    if backend.is_experimental_sgl_trtllm():
        return
    raise ValueError(
        f"MoE LoRA serving in this fork requires --moe-runner-backend "
        f"{_TRTLLM_MOE_BACKEND}, but the active MoE runner is "
        f"{getattr(backend, 'value', backend)!r}. It is the only MoE LoRA path "
        f"this fork validates; other runners take a different numerical path. "
        f"Pass --moe-runner-backend {_TRTLLM_MOE_BACKEND} explicitly (the "
        f"default resolves to 'flashinfer_trtllm' on SM100), or serve without "
        f"--enable-lora / --lora-paths."
    )


def __apply_patch__(public_mod):
    manager_cls = public_mod.LoRAManager
    orig_init = manager_cls.__init__

    def __init__(self, base_model, *args, **kwargs):
        # Before the original: fail at startup rather than after allocating the
        # pool. get_moe_runner_backend() reads the ACTIVE flags group, which is
        # materialized at scheduler init, i.e. before any LoRAManager is built.
        # base_model is LoRAManager.__init__'s first positional parameter.
        _check_moe_runner_backend(base_model)
        if (
            get_moe_runner_backend().is_experimental_sgl_trtllm()
            and _base_model_has_moe(base_model)
        ):
            # server_args is keyword-or-6th-positional in LoRAManager.__init__.
            server_args = kwargs.get("server_args")
            if server_args is None and len(args) >= 5:
                server_args = args[4]
            if server_args is not None:
                _check_virtual_experts(server_args)
        orig_init(self, base_model, *args, **kwargs)

    manager_cls.__init__ = __init__
