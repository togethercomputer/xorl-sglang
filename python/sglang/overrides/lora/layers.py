"""Override twin of ``sglang.srt.lora.layers``.

Makes ``FusedMoEWithLoRA`` MoE LoRA slicing produce a full padded-width shard,
matching how the *base* MoE weight loader shards under a padded intermediate.

Why the padded stride (and not the unpadded one):

``FusedMoE.use_padded_loading`` is True for the flashinfer-TRT-LLM MoE family
(``fused_moe_triton/layer.py``), so the base loader goes through
``narrow_padded_param_and_loaded_weight(..., weight_start=shard_size * tp_rank,
shard_size=shard_size, ...)`` with ``shard_size`` = the **padded**
per-partition intermediate. For Qwen3-30B-A3B (768) at TP=4 with padding to 256
that gives rank0 ``[0:256]``, rank1 ``[256:512]``, rank2 ``[512:768]`` and rank3
an empty slice which ``get_actual_shard_size`` zero-fills. Unbalanced, but
self-consistent and correct: the four local shards still cover the global 768
exactly once.

LoRA must use the *same* convention or its per-expert deltas are applied against
mismatched base columns. Upstream already sliced at the padded stride; its only
bug was that the pool buffer was sized at the *unpadded* width (fixed in the
``lora.mem_pool`` twin) and that a short/empty tail slice was returned as-is
instead of being zero-filled to the buffer width.

This twin therefore keeps the padded stride and adds the clip + zero-fill that
``narrow_padded_param_and_loaded_weight`` performs on the base side. When no
padding is in effect (any non-TRT-LLM runner, or TP where the split is already a
multiple of 128) the slice is exactly the full width and this is byte-identical
to upstream.
"""

import torch


def _padded_shard(t: torch.Tensor, dim: int, start: int, width: int) -> torch.Tensor:
    """``t``'s ``[start:start+width)`` along ``dim``, zero-filled past the end.

    Mirrors ``narrow_padded_param_and_loaded_weight``: a shard that runs past the
    tensor (or starts beyond it) contributes zeros rather than a short tensor.
    """
    avail = max(0, min(width, t.shape[dim] - start))
    if avail == width:
        return t.narrow(dim, start, width).contiguous()

    shape = list(t.shape)
    shape[dim] = width
    out = t.new_zeros(shape)
    if avail > 0:
        out.narrow(dim, 0, avail).copy_(t.narrow(dim, start, avail))
    return out


def __apply_patch__(public_mod):
    moe_cls = public_mod.FusedMoEWithLoRA

    def _slice_moe_a(self, A: torch.Tensor, tp_rank: int, target_module: str):
        """LoRA A for ``down_proj_moe``: input dim is the sharded intermediate."""
        width = self.intermediate_size_per_partition
        return _padded_shard(A, A.dim() - 1, tp_rank * width, width)

    def _slice_moe_b_2d(self, B: torch.Tensor, tp_rank: int, target_module: str):
        """LoRA B for ``gate_up_proj_moe``: output matches the sharded base w13."""
        if target_module != "gate_up_proj_moe":
            return B

        width = self.intermediate_size_per_partition
        start = tp_rank * width

        is_gated = self.base_layer.moe_runner_config.is_gated
        if not is_gated:
            # Non-gated MoE (e.g. Nemotron-H): only w1, shard B directly.
            if self.tp_size <= 1:
                return B
            return _padded_shard(B, 0, start, width)

        full_inter = B.shape[0] // 2
        gate_b = _padded_shard(B[:full_inter], 0, start, width)
        up_b = _padded_shard(B[full_inter:], 0, start, width)

        if self._uses_interleaved_gate_up:
            return torch.stack([gate_b, up_b], dim=1).reshape(-1, B.shape[-1])
        # gate partition then up partition, each one padded width -- the same
        # layout _load_w13 writes into the base w13 buffer.
        return torch.cat([gate_b, up_b], dim=0).contiguous()

    moe_cls._slice_moe_a = _slice_moe_a
    moe_cls._slice_moe_b_2d = _slice_moe_b_2d
