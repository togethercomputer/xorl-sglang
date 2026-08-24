"""Override twin of ``sglang.srt.lora.mem_pool``.

Sizes the routed-MoE LoRA pool buffers with the same padded per-partition
intermediate the MoE layer allocates, so the buffer matches the sharded LoRA
weight. See ``_moe_padding`` for the full bug description.

Paired with the ``lora.layers`` twin, which fixes the slicing side; the two must
land together -- either alone still mismatches.
"""

from sglang.overrides.lora._moe_padding import padded_moe_inter

# LoRA A input dim is the sharded MoE intermediate for these (row-parallel).
_A_PADDED_MODULES = frozenset({"down_proj_moe"})
# LoRA B output dim is 2 x sharded MoE intermediate for these (column-parallel,
# gate and up stacked).
_B_PADDED_MODULES = frozenset({"gate_up_proj_moe"})


def __apply_patch__(public_mod):
    pool_cls = public_mod.LoRAMemoryPool
    orig_get_lora_A_shape = pool_cls.get_lora_A_shape
    orig_get_lora_B_shape = pool_cls.get_lora_B_shape

    def get_lora_A_shape(self, module_name, base_model, max_lora_dim, layer_idx):
        shape = orig_get_lora_A_shape(
            self, module_name, base_model, max_lora_dim, layer_idx
        )
        if module_name not in _A_PADDED_MODULES:
            return shape
        # trailing dim is the per-partition intermediate (input of down_proj)
        per_partition = shape[-1]
        padded = padded_moe_inter(per_partition)
        if padded == per_partition:
            return shape
        return (*shape[:-1], padded)

    def get_lora_B_shape(self, module_name, base_model, max_lora_dim, layer_idx):
        shape = orig_get_lora_B_shape(
            self, module_name, base_model, max_lora_dim, layer_idx
        )
        if module_name not in _B_PADDED_MODULES:
            return shape
        # ...[output_dim, rank]; output_dim is gate+up stacked, so each half is
        # one per-partition intermediate and each half is padded independently
        # (matching the base w13 buffer layout).
        output_dim = shape[-2]
        if output_dim % 2 != 0:
            return shape
        half = output_dim // 2
        padded_half = padded_moe_inter(half)
        if padded_half == half:
            return shape
        return (*shape[:-2], padded_half * 2, shape[-1])

    pool_cls.get_lora_A_shape = get_lora_A_shape
    pool_cls.get_lora_B_shape = get_lora_B_shape
