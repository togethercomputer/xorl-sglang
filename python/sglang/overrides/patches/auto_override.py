import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
from contextlib import contextmanager
from types import ModuleType

OVERRIDES_PREFIX: str = "sglang.overrides"
# Maps a public target prefix → the corresponding override mirror root.
# A module at <override_root>.<rel> overrides <target>.<rel>.
#
# sglang.srt mirrors live directly under sglang.overrides (flat layout).
# Other targets are nested to avoid name collisions across mirrors.
TARGET_TO_OVERRIDE: dict[str, str] = {
    "sglang.srt": "sglang.overrides",
    "sglang.multimodal_gen": "sglang.overrides.multimodal_gen",
}


def safe_find_spec(fullname: str):
    try:
        return importlib.util.find_spec(fullname)
    except Exception:
        return None


def apply_patch(public_mod: ModuleType, override_mod: ModuleType):
    """
    Default patching semantics:
    - If override_mod defines __apply_patch__(public_mod), call it and return.
    - Else copy all public attributes (not starting with '_') from override_mod
      onto public_mod, except those listed in __patch_exclude__; or only those
      listed in __patch_include__ if provided.
    - Merge __all__ if present.
    - Idempotent via a guard flag on public_mod.
    """
    if getattr(public_mod, "__overrides_patched__", False):
        return

    # 1) allow custom patch function
    fn = getattr(override_mod, "__apply_patch__", None)
    if callable(fn):
        fn(public_mod)
        setattr(public_mod, "__overrides_patched__", True)
        return

    # 2) default: name-based copy
    include = set(getattr(override_mod, "__patch_include__", []) or [])
    exclude = set(getattr(override_mod, "__patch_exclude__", []) or [])

    if include:
        names = include
    else:
        names = {n for n in dir(override_mod) if not n.startswith("_")}

    names -= exclude

    for name in names:
        setattr(public_mod, name, getattr(override_mod, name))

    # Merge __all__
    if hasattr(public_mod, "__all__") or hasattr(override_mod, "__all__"):
        pub_all = set(getattr(public_mod, "__all__", []) or [])
        ovr_all = set(getattr(override_mod, "__all__", []) or [])
        # if using default mode, also expose copied names
        pub_all |= ovr_all | set(names)
        public_mod.__all__ = sorted(pub_all)

    setattr(public_mod, "__overrides_patched__", True)


@contextmanager
def bypass():
    finder = PATCHING_FINDER
    finder.tls.bypass = getattr(finder.tls, "bypass", 0) + 1
    try:
        yield
    finally:
        finder.tls.bypass -= 1


class PatchingFinder(importlib.abc.MetaPathFinder):
    """
    Post-load patcher:
      - If an override twin exists for 'sglang.srt.*', wrap the *upstream* loader.
      - Load upstream normally, then import the override twin and monkey patch
        the public module object.
      - Never replaces packages; it only patches module objects after load.
    """

    def __init__(self):
        self.tls = threading.local()
        self.tls.bypass = 0  # hard bypass

    def find_spec(self, fullname, path=None, target=None):

        # Match the longest target prefix so e.g. sglang.srt.foo and
        # sglang.multimodal_gen.foo are routed to their respective mirrors.
        matched_prefix: str | None = None
        for prefix in TARGET_TO_OVERRIDE:
            if fullname == prefix or fullname.startswith(prefix + "."):
                if matched_prefix is None or len(prefix) > len(matched_prefix):
                    matched_prefix = prefix
        if matched_prefix is None:
            return None

        if getattr(self.tls, "bypass", 0):
            return None

        override_root = TARGET_TO_OVERRIDE[matched_prefix]
        override_name = fullname.replace(matched_prefix, override_root, 1)

        # Do we have an override twin? (module or package)
        with bypass():
            override_spec = safe_find_spec(override_name)

        if override_spec is None:
            return None  # nothing to patch → let default importers handle it

        # Find the real upstream spec for the public name
        with bypass():
            upstream_spec = safe_find_spec(fullname)

        if upstream_spec is None or upstream_spec.loader is None:
            return None  # shouldn't happen; fallback to default behavior

        # Wrap the upstream loader
        upstream_loader = upstream_spec.loader
        override_name_str = override_name  # close over value

        class WrapperLoader(importlib.abc.Loader):
            def create_module(self, spec):
                # Delegate creation to upstream loader if it implements it
                if hasattr(upstream_loader, "create_module"):
                    return upstream_loader.create_module(upstream_spec)
                return None  # default creation

            def exec_module(self, module: ModuleType):
                # 1) Execute upstream (module is already registered in sys.modules)
                upstream_loader.exec_module(module)
                # 2) Import override twin and patch the *public* module object
                with bypass():
                    ovr = importlib.import_module(override_name_str)
                apply_patch(module, ovr)

        # Return the upstream spec but with our wrapper loader
        upstream_spec.loader = WrapperLoader()
        return upstream_spec


PATCHING_FINDER = PatchingFinder()


def enable_overrides():
    """
    Install the patching finder at the front of sys.meta_path (idempotent).
    Call this before any imports of 'sglang.srt.*'.
    """
    if not any(isinstance(f, PatchingFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, PATCHING_FINDER)


enable_overrides()
