"""Monkey patches for upstream SGLang code.

Patches are automatically applied when this package is imported.
"""

# Register override-only env vars onto upstream Envs before any patch reads them.
import sglang.overrides.environ  # noqa: F401
import sglang.overrides.patches.auto_override  # noqa: F401

# Serve fork modules at upstream-shaped import paths (lazy; see module docstring).
from sglang.overrides.patches.module_aliases import install as _install_module_aliases

_install_module_aliases()
