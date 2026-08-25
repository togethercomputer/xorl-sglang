"""Package marker for the ``sglang.srt.arg_groups`` override twins.

No-op ``__apply_patch__``: without it the overlay finder copies this package's
public attributes onto the upstream package, and ``import
sglang.srt.arg_groups.overrides as o`` binds the twin instead of the real module.
"""


def __apply_patch__(public_mod):
    return
