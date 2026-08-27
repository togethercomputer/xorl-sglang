# Twin package for sglang.srt.layers. The no-op __apply_patch__ is required
# (see overrides/lora/__init__.py for the full rationale): without it the
# finder would copy imported twin submodules onto the upstream package,
# shadowing the real submodule objects.


def __apply_patch__(public_mod):
    return
