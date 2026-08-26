"""Override twin of ``sglang.srt.arg_groups.arg_utils``.

Extends the declarable-fields whitelist (``resolvable_fields``) from the
overlay, so fields that only *overlay* declarations write never need an
``Arg(..., resolvable=True)`` edit in upstream-owned ``srt/server_args.py``.

This wraps the whitelist *function* rather than mutating
``ServerArgs.__dataclass_fields__`` metadata: the upstream dataclass stays
byte-honest (reading ``srt/server_args.py`` still tells the truth about what
upstream marks resolvable), and the overlay's additions are auditable right
here, in one named set.

Timing is guaranteed by the finder: importing ``sglang`` installs the
meta-path finder before any ``sglang.srt.*`` import can resolve, and the
finder applies this twin the moment ``arg_utils`` is loaded — before
``srt/arg_groups/overrides.py`` (module-level ``from arg_utils import
resolvable_fields``) or ``ServerArgs.__post_init__`` (function-scope import)
can capture the binding. ``resolvable_fields`` is uncached upstream.
"""

import dataclasses

# Fields that overlay declarations write but upstream does not mark
# ``Arg(..., resolvable=True)``. Applied only to classes that actually define
# the field, so mock/dummy config classes keep their own whitelist untouched.
OVERLAY_RESOLVABLE_FIELDS = frozenset(
    {
        # Declared by overrides/arg_groups/overrides.py::select_moe_lora_backend
        # alongside the MoE runner it picks: the TRT-LLM MoE LoRA path
        # hard-asserts on it (lora_dispatch.py), so choosing that runner
        # without it would only trade one required flag for another.
        "lora_use_virtual_experts",
    }
)


def __apply_patch__(public_mod):
    upstream_resolvable_fields = public_mod.resolvable_fields

    def resolvable_fields(cls) -> frozenset:
        base = upstream_resolvable_fields(cls)
        if not dataclasses.is_dataclass(cls):
            return base
        present = {field.name for field in dataclasses.fields(cls)}
        return base | (OVERLAY_RESOLVABLE_FIELDS & present)

    resolvable_fields.__doc__ = upstream_resolvable_fields.__doc__
    resolvable_fields.__wrapped__ = upstream_resolvable_fields
    public_mod.resolvable_fields = resolvable_fields
