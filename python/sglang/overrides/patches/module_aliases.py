"""Lazy import aliases for modules the fork serves at an upstream-shaped path.

The auto-override finder only patches modules upstream actually has; it cannot
conjure a brand-new ``sglang.srt.*`` module. The XoRL trainer's cross-engine
gates import ``sglang.srt.batch_invariant_ops.bi_families_v2`` at its
main-branch home, but on this dev-based branch the implementation lives in
``sglang.xorl.bi.bi_families_v2``. This finder answers the upstream-shaped
import with the real module object — lazily, so ``import sglang`` does not pay
for triton-heavy modules nobody asked for.
"""

import importlib
import importlib.abc
import importlib.util
import sys

ALIASES: dict[str, str] = {
    "sglang.srt.batch_invariant_ops.bi_families_v2": "sglang.xorl.bi.bi_families_v2",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real):
        self._real = real

    def create_module(self, spec):
        return self._real

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        target_name = ALIASES.get(fullname)
        if target_name is None:
            return None
        real = importlib.import_module(target_name)
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real))


def install() -> None:
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
