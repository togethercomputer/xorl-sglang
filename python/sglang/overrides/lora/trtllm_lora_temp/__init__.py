"""Twin package marker -- holds override modules, patches nothing itself.

The no-op ``__apply_patch__`` is required; see ``sglang/overrides/lora/__init__.py``
for why (``__patch_include__ = []`` does not work -- an empty include set is falsy
and falls through to the name copy).
"""


def __apply_patch__(public_mod):  # noqa: D401 - packages are never patched
    return
