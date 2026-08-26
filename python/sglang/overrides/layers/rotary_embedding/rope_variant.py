"""Override twin of ``sglang.srt.layers.rotary_embedding.rope_variant`` -- xorl exact serving (zero-srt port of PR #41).

Verbatim copies of the retired in-tree edits. Copies live at module top level
(collision-proof ``_Cls__name`` def names for methods) so cross-references stay
module-global, and every attach goes through ``rebind`` so the copy resolves
names via the PATCHED srt module's live dict -- identical to in-tree, including
monkeypatching and ``global`` writes. Replaced/removed upstream symbols are
pinned in ``sglang.overrides._twin_pins``; when the pin test fires after an
upstream sync, re-derive the copies and re-pin.
"""

from __future__ import annotations

from sglang.overrides._twin_bind import rebind

def _DeepseekScalingRotaryEmbedding___build_cos_sin_cache(self) -> torch.Tensor:
    cache = super(DeepseekScalingRotaryEmbedding, self)._build_cos_sin_cache()
    if _is_npu:
        # These NPU-only buffers hold the same rows at doubled width;
        # derive them from the finished table so they preserve its device
        # and provenance without duplicating the cache on other backends.
        cos, sin = cache.chunk(2, dim=-1)
        self.cos_cached_total = torch.cat((cos, cos), dim=-1)
        self.sin_cached_total = torch.cat((sin, sin), dim=-1)
    return cache

def _DeepseekScalingRotaryEmbedding___cos_sin_cache_inv_freq(self) -> torch.Tensor:
    return self._compute_inv_freq(self.scaling_factor)

def _DeepseekScalingRotaryEmbedding___cos_sin_cache_mscale(self) -> float:
    return self.mscale

def _DeepseekScalingRotaryEmbedding___cos_sin_cache_positions(self) -> torch.Tensor:
    return torch.arange(
        self.max_position_embeddings * self.scaling_factor,
        device=self._cos_sin_cache_work_device(self.device),
        dtype=torch.float32,
    )

def _DynamicNTKAlphaRotaryEmbedding___cos_sin_cache_inv_freq(self) -> torch.Tensor:
    base = self.base * self.scaling_alpha ** (
        self.rotary_dim / (self.rotary_dim - 2)
    )
    return self._compute_inv_freq(base)

def _DynamicNTKScalingRotaryEmbedding___cos_sin_cache_inv_freq(self) -> torch.Tensor:
    max_len = self.max_position_embeddings * self.scaling_factor
    base = self.base * (
        (self.scaling_factor * max_len / self.max_position_embeddings)
        - (self.scaling_factor - 1)
    ) ** (self.rotary_dim / (self.rotary_dim - 2))
    return self._compute_inv_freq(base)

def _DynamicNTKScalingRotaryEmbedding___cos_sin_cache_positions(self) -> torch.Tensor:
    return torch.arange(
        self.max_position_embeddings * self.scaling_factor, dtype=torch.float
    )

def _DeepseekScalingRotaryEmbedding___compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
    device = self._cos_sin_cache_work_device(self.device)
    pos_freqs = self.base ** (
        torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device)
        / self.rotary_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
    low, high = yarn_find_correction_range(
        self.beta_fast,
        self.beta_slow,
        self.rotary_dim,
        self.base,
        self.max_position_embeddings,
    )
    inv_freq_mask = (
        1
        - yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2, dtype=torch.float, device=device
        )
    ) * self.extrapolation_factor
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    return inv_freq


def __apply_patch__(mod):
    mod.DeepseekScalingRotaryEmbedding._build_cos_sin_cache = rebind(_DeepseekScalingRotaryEmbedding___build_cos_sin_cache, mod, name="_build_cos_sin_cache")
    mod.DeepseekScalingRotaryEmbedding._cos_sin_cache_inv_freq = rebind(_DeepseekScalingRotaryEmbedding___cos_sin_cache_inv_freq, mod, name="_cos_sin_cache_inv_freq")
    mod.DeepseekScalingRotaryEmbedding._cos_sin_cache_mscale = rebind(_DeepseekScalingRotaryEmbedding___cos_sin_cache_mscale, mod, name="_cos_sin_cache_mscale")
    mod.DeepseekScalingRotaryEmbedding._cos_sin_cache_positions = rebind(_DeepseekScalingRotaryEmbedding___cos_sin_cache_positions, mod, name="_cos_sin_cache_positions")
    mod.DynamicNTKAlphaRotaryEmbedding._cos_sin_cache_inv_freq = rebind(_DynamicNTKAlphaRotaryEmbedding___cos_sin_cache_inv_freq, mod, name="_cos_sin_cache_inv_freq")
    mod.DynamicNTKScalingRotaryEmbedding._cos_sin_cache_inv_freq = rebind(_DynamicNTKScalingRotaryEmbedding___cos_sin_cache_inv_freq, mod, name="_cos_sin_cache_inv_freq")
    mod.DynamicNTKScalingRotaryEmbedding._cos_sin_cache_positions = rebind(_DynamicNTKScalingRotaryEmbedding___cos_sin_cache_positions, mod, name="_cos_sin_cache_positions")
    mod.DeepseekScalingRotaryEmbedding._compute_inv_freq = rebind(_DeepseekScalingRotaryEmbedding___compute_inv_freq, mod, name="_compute_inv_freq")
    # Removed by the exact port; the base-class impl takes over.
    del mod.DeepseekScalingRotaryEmbedding._compute_cos_sin_cache
    # Removed by the exact port; the base-class impl takes over.
    del mod.DynamicNTKAlphaRotaryEmbedding._compute_cos_sin_cache
    # Removed by the exact port; the base-class impl takes over.
    del mod.DynamicNTKScalingRotaryEmbedding._compute_cos_sin_cache
