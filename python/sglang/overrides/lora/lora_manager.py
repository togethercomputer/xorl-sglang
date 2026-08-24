"""Override twin of ``sglang.srt.lora.lora_manager``.

Enforces that **MoE** LoRA is only *served* on the TRT-LLM MoE runner.

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

from sglang.srt.layers.moe.utils import get_moe_runner_backend

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
    backend = get_moe_runner_backend()
    if backend.is_experimental_sgl_trtllm():
        return
    if not _base_model_has_moe(base_model):
        # Dense model: the MoE runner is inert here, nothing to enforce.
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
        if get_moe_runner_backend().is_experimental_sgl_trtllm() and _base_model_has_moe(
            base_model
        ):
            # server_args is keyword-or-6th-positional in LoRAManager.__init__.
            server_args = kwargs.get("server_args")
            if server_args is None and len(args) >= 5:
                server_args = args[4]
            if server_args is not None:
                _check_virtual_experts(server_args)
        orig_init(self, base_model, *args, **kwargs)

    manager_cls.__init__ = __init__
