"""Override twin of ``sglang.srt.layers.rotary_embedding.yarn`` -- xorl exact serving (zero-srt port of PR #41).

Verbatim copies of the retired in-tree edits. Copies live at module top level
(collision-proof ``_Cls__name`` def names for methods) so cross-references stay
module-global, and every attach goes through ``rebind`` so the copy resolves
names via the PATCHED srt module's live dict -- identical to in-tree, including
monkeypatching and ``global`` writes. Replaced/removed upstream symbols are
pinned in ``sglang.overrides._twin_pins``; when the pin test fires after an
upstream sync, re-derive the copies and re-pin.
"""

# ruff: noqa: F821 -- the verbatim copies below resolve upstream names at call
# time via rebind() over the live srt module dict; they are undefined in this
# file's namespace by design.

from __future__ import annotations

from sglang.overrides._twin_bind import rebind


def _YaRNScalingRotaryEmbedding___cos_sin_cache_inv_freq(self) -> torch.Tensor:
    return self._compute_inv_freq(self.scaling_factor)


def _YaRNScalingRotaryEmbedding___cos_sin_cache_mscale(self) -> float:
    return self.mscale


def _YaRNScalingRotaryEmbedding___cos_sin_cache_positions(self) -> torch.Tensor:
    return torch.arange(
        self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
    )


def __apply_patch__(mod):
    mod.YaRNScalingRotaryEmbedding._cos_sin_cache_inv_freq = rebind(
        _YaRNScalingRotaryEmbedding___cos_sin_cache_inv_freq,
        mod,
        name="_cos_sin_cache_inv_freq",
    )
    mod.YaRNScalingRotaryEmbedding._cos_sin_cache_mscale = rebind(
        _YaRNScalingRotaryEmbedding___cos_sin_cache_mscale,
        mod,
        name="_cos_sin_cache_mscale",
    )
    mod.YaRNScalingRotaryEmbedding._cos_sin_cache_positions = rebind(
        _YaRNScalingRotaryEmbedding___cos_sin_cache_positions,
        mod,
        name="_cos_sin_cache_positions",
    )
    # Removed by the exact port; the base-class impl takes over.
    del mod.YaRNScalingRotaryEmbedding._compute_cos_sin_cache
