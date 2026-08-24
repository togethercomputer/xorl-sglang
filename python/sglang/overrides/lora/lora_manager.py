"""Override twin of ``sglang.srt.lora.lora_manager``.

Enforces that LoRA is only *served* on the TRT-LLM MoE runner.

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

from sglang.srt.layers.moe.utils import get_moe_runner_backend

_TRTLLM_MOE_BACKEND = "experimental_sgl_trtllm"


def _check_moe_runner_backend() -> None:
    backend = get_moe_runner_backend()
    if backend.is_experimental_sgl_trtllm():
        return
    raise ValueError(
        f"LoRA serving in this fork requires --moe-runner-backend "
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

    def __init__(self, *args, **kwargs):
        # Before the original: fail at startup rather than after allocating the
        # pool. get_moe_runner_backend() reads the ACTIVE flags group, which is
        # materialized at scheduler init, i.e. before any LoRAManager is built.
        _check_moe_runner_backend()
        orig_init(self, *args, **kwargs)

    manager_cls.__init__ = __init__
