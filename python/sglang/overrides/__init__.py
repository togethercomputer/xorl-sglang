"""Overrides that extend upstream SGLang.

This package contains:
- Monkey patches for upstream modules (automatically applied on import)
- Drop-in overrides of ``sglang.srt.*`` modules and supporting utilities
"""

# Import patches first to apply them automatically.
from sglang.overrides import patches  # noqa: F401

__all__ = []
