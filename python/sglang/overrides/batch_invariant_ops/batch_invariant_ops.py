"""Overlay twin: op-gated batch-invariant mode (XoRL exact serving).

Ported from xorl-sglang `main` (commit c08786bd3), where
`enable_batch_invariant_mode` gains an ``ops=`` selector so each exact
architecture contract can interpose exactly the aten ops its numerical
program owns (dense Qwen3 keeps only the matrix products; Qwen3.5 adds
log_softmax/mean/rms_norm), plus a re-entrancy fix in
``set_batch_invariant_mode`` (enter/exit/enter re-registers; nested
same-state entry is a no-op).

The whole-function replacements below intentionally read and write the
UPSTREAM module's state (``_batch_invariant_MODE`` etc.) so there is a
single source of truth; the twin only rebinds behaviour. The new XoRL op
implementations themselves live in ``sglang.xorl.bi`` and are re-exported
onto the public modules by ``__apply_patch__`` in this package's
``__init__``.
"""

from __future__ import annotations

import contextlib
from typing import Iterable, Optional

import torch

_BATCH_INVARIANT_ALL_OPS = {
    "mm",
    "addmm",
    "log_softmax",
    "mean",
    "rms_norm",
    "bmm",
}
_BATCH_INVARIANT_ALIASES = {
    "matmul": "mm",
    "logsoftmax": "log_softmax",
    "log-softmax": "log_softmax",
    "rmsnorm": "rms_norm",
    "rms-norm": "rms_norm",
}


def _normalize_batch_invariant_ops(ops: Iterable[str]) -> set[str]:
    normalized = set()
    for raw_op in ops:
        op = raw_op.strip().lower().replace("-", "_")
        op = _BATCH_INVARIANT_ALIASES.get(op, op)
        if op not in _BATCH_INVARIANT_ALL_OPS:
            raise ValueError(
                f"Unsupported batch-invariant op {raw_op!r}; "
                f"supported values are: {sorted(_BATCH_INVARIANT_ALL_OPS)}"
            )
        normalized.add(op)
    return normalized


def __apply_patch__(mod):
    # Track the selected op set on the upstream module.
    if not hasattr(mod, "_batch_invariant_OPS"):
        mod._batch_invariant_OPS = set()

    # Main defines these on the upstream module; callers (the Qwen3.5 exact
    # resolver, the sampler fastpath) reference them through this module, so
    # install the ops_ext implementations here for import-surface fidelity.
    from sglang.xorl.bi import ops_ext as _ext

    for _name in (
        "set_router_renorm_fused_enabled",
        "set_bi_head_fastpath_enabled",
        "is_bi_head_fastpath_enabled",
        "bi_lm_head_selected_logprob",
        "bi_lm_head_full_logits",
        "bi_lm_head_selected_logprob_from_logits",
        "bi_router_gemm",
        "bi_router_topk_weights",
        "bi_rms_norm",
        "bi_fused_add_rms_norm",
        "fused_add_rms_norm_batch_invariant",
        "rms_norm_residual_tree_batch_invariant",
        "RMSNormFamily",
        "RMS_NORM_FAMILIES",
        "RMS_NORM_FAMILY_NO_RESIDUAL",
        "RMS_NORM_FAMILY_RESIDUAL_TREE",
        "BI_LM_HEAD_VOCAB_CHUNK",
    ):
        setattr(mod, _name, getattr(_ext, _name))

    mod._BATCH_INVARIANT_ALL_OPS = _BATCH_INVARIANT_ALL_OPS
    mod._BATCH_INVARIANT_ALIASES = _BATCH_INVARIANT_ALIASES
    mod._normalize_batch_invariant_ops = _normalize_batch_invariant_ops

    def get_batch_invariant_ops() -> tuple[str, ...]:
        return tuple(sorted(mod._batch_invariant_OPS))

    def is_batch_invariant_op_enabled(op: str) -> bool:
        return _BATCH_INVARIANT_ALIASES.get(op, op) in mod._batch_invariant_OPS

    def enable_batch_invariant_mode(
        enable_bmm: bool = True,
        ops: Optional[Iterable[str]] = None,
    ):
        if mod._batch_invariant_MODE:
            return

        selected = (
            set(_BATCH_INVARIANT_ALL_OPS)
            if ops is None
            else _normalize_batch_invariant_ops(ops)
        )
        if not enable_bmm:
            selected.discard("bmm")

        dispatch_key = mod.get_dispatch_device_backend()

        mod._batch_invariant_MODE = True
        mod._batch_invariant_OPS = selected
        mod._batch_invariant_LIB = torch.library.Library("aten", "IMPL")

        if not mod._is_npu:
            # Register for detected device
            if "mm" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::mm", mod.mm_batch_invariant, dispatch_key
                )
                mod._batch_invariant_LIB.impl(
                    "aten::mm.dtype", mod._mm_dtype_compat, dispatch_key
                )
            if "addmm" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::addmm", mod.addmm_batch_invariant, dispatch_key
                )
            if "log_softmax" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::_log_softmax",
                    mod._log_softmax_batch_invariant,
                    dispatch_key,
                )
            if "mean" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::mean.dim", mod.mean_batch_invariant, dispatch_key
                )
            if "rms_norm" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::rms_norm", mod._rms_norm_aten_compat, dispatch_key
                )
            if "bmm" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::bmm", mod.bmm_batch_invariant, dispatch_key
                )
                # Also monkeypatch torch.bmm directly as a fallback
                mod._original_torch_bmm = torch.bmm
                torch.bmm = mod.bmm_batch_invariant
        else:
            from sglang.srt.hardware_backend.npu.batch_invariant_ops.npu_batch_invariant_ops import (
                npu_add_rms_norm_batch_invariant,
                npu_fused_infer_attention_score_batch_invariant,
                npu_log_softmax_batch_invariant,
                npu_matmul_batch_invariant,
                npu_mean_batch_invariant,
                npu_mm_batch_invariant,
            )

            if "mm" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::mm", npu_mm_batch_invariant, dispatch_key
                )
                mod._batch_invariant_LIB.impl(
                    "aten::matmul", npu_matmul_batch_invariant, dispatch_key
                )
            if "mean" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::mean.dim", npu_mean_batch_invariant, dispatch_key
                )
            if "log_softmax" in selected:
                mod._batch_invariant_LIB.impl(
                    "aten::_log_softmax", npu_log_softmax_batch_invariant, dispatch_key
                )
            if "rms_norm" in selected:
                mod.torch_npu.npu_add_rms_norm = npu_add_rms_norm_batch_invariant
            torch.ops.npu.npu_fused_infer_attention_score = (
                npu_fused_infer_attention_score_batch_invariant
            )

    def disable_batch_invariant_mode():
        if mod._batch_invariant_LIB is not None:
            mod._batch_invariant_LIB._destroy()
        if mod._original_torch_bmm is not None:
            torch.bmm = mod._original_torch_bmm
            mod._original_torch_bmm = None
        mod._batch_invariant_MODE = False
        mod._batch_invariant_LIB = None
        mod._batch_invariant_OPS = set()

    @contextlib.contextmanager
    def set_batch_invariant_mode(enabled: bool = True):
        was_enabled = mod._batch_invariant_MODE
        old_ops = get_batch_invariant_ops()
        if enabled == was_enabled:
            yield
            return
        if enabled:
            enable_batch_invariant_mode()
        else:
            disable_batch_invariant_mode()
        try:
            yield
        finally:
            if was_enabled:
                enable_batch_invariant_mode(ops=old_ops)
            else:
                disable_batch_invariant_mode()

    mod.get_batch_invariant_ops = get_batch_invariant_ops
    mod.is_batch_invariant_op_enabled = is_batch_invariant_op_enabled
    mod.enable_batch_invariant_mode = enable_batch_invariant_mode
    mod.disable_batch_invariant_mode = disable_batch_invariant_mode
    mod.set_batch_invariant_mode = set_batch_invariant_mode
    mod.__all__ = list(
        dict.fromkeys(
            list(getattr(mod, "__all__", []))
            + ["is_batch_invariant_op_enabled", "get_batch_invariant_ops"]
        )
    )
