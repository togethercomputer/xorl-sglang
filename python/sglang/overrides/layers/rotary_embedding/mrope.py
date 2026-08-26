"""Override twin of ``sglang.srt.layers.rotary_embedding.mrope`` -- xorl exact serving (zero-srt port of PR #41).

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

def _YaRNScalingMRotaryEmbedding___cos_sin_cache_inv_freq(self) -> torch.Tensor:
    return self._compute_inv_freq(self.scaling_factor)

def _YaRNScalingMRotaryEmbedding___cos_sin_cache_mscale(self) -> float:
    return self.mscale

def _YaRNScalingMRotaryEmbedding___cos_sin_cache_positions(self) -> torch.Tensor:
    return torch.arange(
        self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
    )

def _MRotaryEmbedding__forward_native(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    fused_set_kv_buffer_arg=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        fused_set_kv_buffer_arg is None
    ), "save kv cache is not supported for MRotaryEmbedding."
    assert positions.ndim == 1 or positions.ndim == 2

    class_b_enabled = bool(
        getattr(
            get_exec().deterministic,
            "qwen35_rope_class_b",
            False,
        )
    )
    if class_b_enabled and positions.ndim == 2:
        torch._assert_async(
            (positions == positions[:1]).all(),
            "Qwen3.5-family Class-B RoPE does not support multimodal "
            "positions with distinct temporal/height/width axes",
        )
        positions = positions[0]

    cos_sin = self.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        assert self.mrope_section
        if self.mrope_interleaved:
            cos = apply_interleaved_rope(cos, self.mrope_section)
            sin = apply_interleaved_rope(sin, self.mrope_section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )

    seq_len_q = query.shape[0]
    query_shape = query.shape
    query = query.view(seq_len_q, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_pass = query[..., self.rotary_dim :]
    apply_rotary = (
        self._apply_rotary_emb_wrapped if class_b_enabled else apply_rotary_emb
    )
    query_rot = apply_rotary(query_rot, cos, sin, self.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    seq_len_k = key.shape[0]
    key_shape = key.shape
    key = key.view(seq_len_k, -1, self.head_size)
    key_rot = key[..., : self.rotary_dim]
    key_pass = key[..., self.rotary_dim :]
    key_rot = apply_rotary(key_rot, cos, sin, self.is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def __apply_patch__(mod):
    mod.YaRNScalingMRotaryEmbedding._cos_sin_cache_inv_freq = rebind(_YaRNScalingMRotaryEmbedding___cos_sin_cache_inv_freq, mod, name="_cos_sin_cache_inv_freq")
    mod.YaRNScalingMRotaryEmbedding._cos_sin_cache_mscale = rebind(_YaRNScalingMRotaryEmbedding___cos_sin_cache_mscale, mod, name="_cos_sin_cache_mscale")
    mod.YaRNScalingMRotaryEmbedding._cos_sin_cache_positions = rebind(_YaRNScalingMRotaryEmbedding___cos_sin_cache_positions, mod, name="_cos_sin_cache_positions")
    mod.MRotaryEmbedding.forward_native = rebind(_MRotaryEmbedding__forward_native, mod, name="forward_native")
    # Removed by the exact port; the base-class impl takes over.
    del mod.YaRNScalingMRotaryEmbedding._compute_cos_sin_cache
