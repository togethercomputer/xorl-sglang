"""Twin package marker -- holds override modules, patches nothing itself.

The no-op ``__apply_patch__`` is required. Without it the finder falls back to
copying this package's public attributes onto the upstream package, and once a
twin submodule has been imported it *is* a public attribute here -- so the
upstream package's submodule attribute gets shadowed by the twin, and
``import sglang.srt.<pkg>.<mod> as m`` binds the override module instead of the
patched upstream one. ``__patch_include__ = []`` does not work for this: an
empty include set is falsy, so ``apply_patch`` falls through to the name copy.
"""


def __apply_patch__(public_mod):  # noqa: D401 - packages are never patched
    return
