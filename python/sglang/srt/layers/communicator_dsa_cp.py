# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from enum import Enum
from functools import partial
from typing import Callable, Optional

import torch
import torch.distributed as dist

from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
)
from sglang.srt.layers.glm52_positions import (
    CanonicalMoEPositions,
)
from sglang.srt.layers.logical_row_ownership import LogicalRowOwnership
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_parallel


def dsa_enable_prefill_cp():
    # After using cp, the communication mode of this part changes.
    # The three parts of prepare_attn, prepare_mlp, and postprocess_layer
    # no longer require additional communication for reduce, scatter, etc.
    return is_dsa_enable_prefill_cp()


def maybe_prefetch_next_full_attention_kv(
    forward_batch: ForwardBatch,
    next_full_attention_layer_id: Optional[int],
) -> None:
    """Prefetch (owner-broadcast) the next layer's DSA KV under layer split.

    No-op unless the current batch runs DSA prefill-CP and the active KV pool is
    a layer-sharded pool exposing ``prefetch_kv_buffer`` (i.e.
    ``LayerSplitDSATokenToKVPool``). Kicking the broadcast off one layer ahead
    overlaps it with the current layer's attention compute.
    """
    if next_full_attention_layer_id is None or not dsa_use_prefill_cp(forward_batch):
        return

    prefetch_kv_buffer = getattr(get_token_to_kv_pool(), "prefetch_kv_buffer", None)
    if prefetch_kv_buffer is not None:
        prefetch_kv_buffer(next_full_attention_layer_id)


def dsa_cp_gather_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    del attn_dp_size
    assert attn_tp_size == 1
    hidden_states, local_hidden_states = (
        get_local_dp_buffer(get_parallel().attn_cp_group),
        hidden_states,
    )
    attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)
    return hidden_states


def _gather_glm52_cp_logical_rows(
    rows: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """Gather CP-v2 rows in the strategy's logical token order.

    CP-v2 pads each physical rank independently for collectives.  Those pad
    rows are an attention implementation detail: the canonical MLP layout is
    the unpadded logical request layout reconstructed by the active strategy.
    Older CP paths do not carry CP-v2 metadata and retain their rank-major
    gather behavior.
    """

    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    if metadata is None:
        return dsa_cp_gather_hidden_states(rows)

    from sglang.srt.layers.cp.base import get_cp_strategy

    strategy = get_cp_strategy()
    if strategy is None:
        raise RuntimeError(
            "GLM-5.2 CP-v2 rows require an active context-parallel strategy"
        )
    total_seq_lens = getattr(metadata, "total_seq_lens", None)
    if total_seq_lens is None or int(total_seq_lens) < 0:
        raise RuntimeError(
            "GLM-5.2 CP-v2 metadata requires a nonnegative logical row count"
        )
    gathered = strategy.gather_hidden_states(rows, forward_batch)
    if gathered.shape[0] != int(total_seq_lens):
        raise RuntimeError(
            "GLM-5.2 CP-v2 strategy returned the wrong logical row count: "
            f"gathered={gathered.shape[0]}, expected={int(total_seq_lens)}"
        )
    return gathered


def _glm52_row_ownership(context=None) -> LogicalRowOwnership:
    parallel = get_parallel() if context is None else context
    dp_size = getattr(parallel, "attn_dp_size", 1)
    cp_size = parallel.attn_cp_size
    dp_rank = getattr(parallel, "attn_dp_rank", None)
    if dp_rank is None:
        if dp_size == 1:
            dp_rank = 0
        else:
            dp_rank = parallel.tp_rank // (cp_size * parallel.attn_tp_size)
    contributor_count = getattr(
        parallel,
        "tp_size",
        dp_size * cp_size,
    )
    return LogicalRowOwnership(
        dp_size=dp_size,
        cp_size=cp_size,
        dp_rank=dp_rank,
        cp_rank=parallel.attn_cp_rank,
        contributor_count=contributor_count,
    )


def _gather_dp_owned_rows(
    dp_local_rows: torch.Tensor,
    *,
    output: torch.Tensor,
    ownership: LogicalRowOwnership,
) -> torch.Tensor:
    """Replicate DP-owned row blocks without duplicating their CP replicas."""

    segment_lengths = list(get_dp_global_num_tokens() or [])
    block = ownership.dp_block_slice(segment_lengths)
    if dp_local_rows.shape[0] != block.stop - block.start:
        raise RuntimeError(
            "GLM-5.2 DP-owned rows do not match the prepared global layout: "
            f"local={dp_local_rows.shape[0]}, expected={block.stop - block.start}"
        )
    if (
        output.shape[0] != sum(segment_lengths)
        or output.shape[1:] != dp_local_rows.shape[1:]
    ):
        raise RuntimeError(
            "GLM-5.2 global row buffer does not match the DP ownership layout"
        )
    output.zero_()
    if ownership.cp_rank == 0:
        output[block].copy_(dp_local_rows)
    dist.all_reduce(output, group=get_parallel().tp_group.device_group)
    return output


def gather_glm52_mlp_rows(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    context=None,
) -> torch.Tensor:
    """Compose CP source gathering with DP logical-owner replication."""

    ownership = _glm52_row_ownership(context)
    context_sharded = dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(
        forward_batch
    )
    if context_sharded:
        dp_local_rows = (
            _gather_glm52_cp_logical_rows(hidden_states, forward_batch)
            if ownership.dp_size > 1
            and getattr(forward_batch, "attn_cp_metadata", None) is not None
            else dsa_cp_gather_hidden_states(hidden_states)
        )
    else:
        dp_local_rows = hidden_states
    if ownership.dp_size == 1:
        return dp_local_rows
    global_rows = get_global_dp_buffer(get_parallel().tp_group)
    return _gather_dp_owned_rows(
        dp_local_rows,
        output=global_rows,
        ownership=ownership,
    )


def dsa_cp_reduce_scatter_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states


def align_glm52_moe_positions(
    positions: torch.Tensor,
    full_hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> CanonicalMoEPositions:
    """Align positions to the same DP-major/CP-minor FULL row layout."""

    cached = getattr(forward_batch, "_glm52_owned_moe_positions", None)
    if cached is not None:
        if cached.values.numel() != full_hidden_states.shape[0]:
            raise RuntimeError(
                "Cached GLM-5.2 positions do not match the gathered MLP rows"
            )
        return cached

    ownership = _glm52_row_ownership()
    segment_lengths = list(get_dp_global_num_tokens() or [])
    prefill_cp = dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch)
    cp_v2_metadata = getattr(forward_batch, "attn_cp_metadata", None)
    if prefill_cp and cp_v2_metadata is not None and ownership.dp_size == 1:
        # DP1 canonical-v3b keeps the equal padded CP buckets consumed by the
        # source-sharded MoE transport.  CP-v2's DP token counts are logical
        # (ragged), so they cannot describe this physical FULL-row capacity.
        segment_lengths = [full_hidden_states.shape[0]]
    elif not segment_lengths and ownership.dp_size == 1:
        segment_lengths = [full_hidden_states.shape[0]]
    block = ownership.dp_block_slice(segment_lengths)
    block_rows = block.stop - block.start
    local_positions = positions.reshape(-1).to(torch.int64)
    if prefill_cp and cp_v2_metadata is not None and ownership.dp_size > 1:
        local_valid = (local_positions >= 0).to(torch.int32)
        dp_positions = _gather_glm52_cp_logical_rows(local_positions, forward_batch)
        dp_valid = _gather_glm52_cp_logical_rows(local_valid, forward_batch)
        if dp_positions.numel() != block_rows or dp_valid.numel() != block_rows:
            raise RuntimeError(
                "GLM-5.2 CP-v2 logical positions do not match the DP row layout: "
                f"positions={dp_positions.numel()}, valid={dp_valid.numel()}, "
                f"expected={block_rows}"
            )
    elif prefill_cp:
        # Legacy CP exposes a rank-major physical layout without CP-v2
        # metadata. Preserve its fixed-width gather contract.
        if block_rows % ownership.cp_size:
            raise RuntimeError("GLM-5.2 DP row capacity must divide across CP owners")
        local_capacity = block_rows // ownership.cp_size
        if local_positions.numel() > local_capacity:
            raise RuntimeError("GLM-5.2 local positions exceed the CP source capacity")
        local_logical_rows = local_positions.numel()
        if cp_v2_metadata is not None:
            logical_counts = getattr(cp_v2_metadata, "per_rank_logical_token", None)
            physical_counts = getattr(cp_v2_metadata, "per_rank_actual_token", None)
            if (
                logical_counts is None
                or physical_counts is None
                or len(logical_counts) != ownership.cp_size
                or len(physical_counts) != ownership.cp_size
            ):
                raise RuntimeError(
                    "GLM-5.2 DP1 CP-v2 positions require complete per-rank row metadata"
                )
            logical_counts = [int(count) for count in logical_counts]
            physical_counts = [int(count) for count in physical_counts]
            total_logical_rows = getattr(cp_v2_metadata, "total_seq_lens", None)
            if (
                total_logical_rows is None
                or sum(logical_counts) != int(total_logical_rows)
                or sum(physical_counts) != block_rows
                or any(count != local_capacity for count in physical_counts)
                or any(
                    logical < 0 or logical > physical
                    for logical, physical in zip(logical_counts, physical_counts)
                )
            ):
                raise RuntimeError(
                    "GLM-5.2 DP1 CP-v2 row metadata does not describe the physical FULL-row layout"
                )
            local_logical_rows = logical_counts[ownership.cp_rank]
            local_physical_rows = physical_counts[ownership.cp_rank]
            if (
                local_physical_rows != local_capacity
                or local_positions.numel() != local_physical_rows
            ):
                raise RuntimeError(
                    "GLM-5.2 DP1 CP-v2 position metadata does not match the physical CP bucket"
                )
        padded_positions = local_positions.new_full((local_capacity,), -1)
        padded_positions[:local_logical_rows].copy_(
            local_positions[:local_logical_rows]
        )
        padded_valid = torch.zeros(
            (local_capacity,), dtype=torch.int32, device=local_positions.device
        )
        padded_valid[:local_logical_rows].copy_(
            (local_positions[:local_logical_rows] >= 0).to(torch.int32)
        )
        dp_positions = local_positions.new_empty((block_rows,))
        dp_valid = padded_valid.new_empty((block_rows,))
        attn_cp_all_gather_into_tensor(dp_positions, padded_positions)
        attn_cp_all_gather_into_tensor(dp_valid, padded_valid)
    else:
        if local_positions.numel() > block_rows:
            raise RuntimeError("GLM-5.2 local positions exceed the DP source capacity")
        dp_positions = local_positions.new_full((block_rows,), -1)
        dp_positions[: local_positions.numel()].copy_(local_positions)
        dp_valid = torch.zeros(
            (block_rows,), dtype=torch.int32, device=local_positions.device
        )
        dp_valid[: local_positions.numel()].copy_(
            (local_positions >= 0).to(torch.int32)
        )

    if ownership.dp_size > 1:
        full_positions = local_positions.new_empty((sum(segment_lengths),))
        full_valid = dp_valid.new_empty((sum(segment_lengths),))
        _gather_dp_owned_rows(dp_positions, output=full_positions, ownership=ownership)
        _gather_dp_owned_rows(dp_valid, output=full_valid, ownership=ownership)
    else:
        full_positions, full_valid = dp_positions, dp_valid
    if full_positions.numel() != full_hidden_states.shape[0]:
        raise RuntimeError(
            "GLM-5.2 positions and gathered MLP rows have different capacities"
        )
    aligned = CanonicalMoEPositions(full_positions, full_valid.to(torch.bool))
    forward_batch._glm52_owned_moe_positions = aligned
    return aligned


class DSAMLPOutputLayout(str, Enum):
    """Cross-rank reduction state of a DSA/MLA layer's MLP output."""

    LEGACY_PARTIAL = "legacy_partial"
    COMPLETE = "complete"


class DSACPLayerCommunicator(LayerCommunicator):
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        mlp_output_layout: DSAMLPOutputLayout = DSAMLPOutputLayout.LEGACY_PARTIAL,
    ):
        self.mlp_output_layout = mlp_output_layout
        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            is_last_layer,
            qkv_latent_func,
        )

    def _post_init_communicate(self):
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1
        if (
            self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED
            and self.mlp_output_layout is not DSAMLPOutputLayout.COMPLETE
        ):
            assert (
                self._context.attn_dp_size == 1
            ), f"dp_size should be 1 when moe_runner_backend is none"
        self._communicate_simple_fn = DSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = DSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode=ScatterMode.SCATTERED,
            residual_input_mode=ScatterMode.SCATTERED,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        if self.mlp_output_layout is DSAMLPOutputLayout.COMPLETE:
            if self.layer_scatter_modes.mlp_mode not in {
                ScatterMode.SCATTERED,
                ScatterMode.FULL,
            }:
                raise RuntimeError("Complete DSA MLP output has an unsupported layout")
            if not self._context.is_same_group_size(
                self.layer_scatter_modes.middle_residual_mode,
                ScatterMode.SCATTERED,
            ) or not self._context.is_same_group_size(
                self.layer_scatter_modes.layer_output_mode,
                ScatterMode.SCATTERED,
            ):
                raise RuntimeError(
                    "Complete DSA MLP output requires a rank-local CP "
                    "residual/output layout"
                )
            self._communicate_summable_tensor_pair_fn = (
                self._reject_legacy_complete_postprocess
            )
            return
        self._communicate_summable_tensor_pair_fn = (
            DSACPCommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=ScatterMode.SCATTERED,
                output_mode=ScatterMode.SCATTERED,
                context=self._context,
            )
        )

    @staticmethod
    def _reject_legacy_complete_postprocess(**_kwargs):
        raise RuntimeError(
            "Complete DSA MLP output cannot use the legacy summed reduce-scatter path"
        )

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        if self.mlp_output_layout is DSAMLPOutputLayout.COMPLETE:
            return False
        return super().should_use_reduce_scatter(forward_batch)

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.mlp_output_layout is DSAMLPOutputLayout.LEGACY_PARTIAL:
            return super().postprocess_layer(hidden_states, residual, forward_batch)
        if residual is None:
            raise RuntimeError("Complete DSA MLP output requires a residual layout")
        if self.layer_scatter_modes.mlp_mode is ScatterMode.SCATTERED:
            if hidden_states.shape != residual.shape:
                raise RuntimeError(
                    "Complete local DSA MLP output must match the residual layout"
                )
            return hidden_states, residual
        prefill_cp = dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(
            forward_batch
        )
        local_rows = residual.shape[0]
        if (
            getattr(self._context, "attn_dp_size", 1) == 1
            and hidden_states.shape == residual.shape
        ):
            # DP1 prefill uses the existing consumer-sharded canonical
            # transport, which already returns this rank's physical CP rows.
            return hidden_states, residual
        ownership = _glm52_row_ownership(self._context)
        segment_lengths = list(get_dp_global_num_tokens() or [])
        if not segment_lengths and ownership.dp_size == 1:
            metadata = getattr(forward_batch, "attn_cp_metadata", None)
            logical_rows = getattr(metadata, "total_seq_lens", None)
            segment_lengths = [
                (
                    int(logical_rows)
                    if prefill_cp and logical_rows is not None
                    else local_rows * ownership.cp_size if prefill_cp else local_rows
                )
            ]
        try:
            block = ownership.dp_block_slice(segment_lengths)
        except ValueError as error:
            raise RuntimeError(
                "Replicated complete DSA MLP rows have invalid DP ownership metadata"
            ) from error
        if (
            hidden_states.shape[0] != sum(segment_lengths)
            or hidden_states.shape[1:] != residual.shape[1:]
        ):
            raise RuntimeError(
                "Replicated complete DSA MLP rows do not match the rank-local "
                "CP residual capacity in the DP/CP layout"
            )
        cp_metadata = getattr(forward_batch, "attn_cp_metadata", None)
        if prefill_cp and cp_metadata is not None and ownership.dp_size > 1:
            logical_rows = getattr(cp_metadata, "total_seq_lens", None)
            block_rows = block.stop - block.start
            if logical_rows is None or block_rows != int(logical_rows):
                raise RuntimeError(
                    "Replicated complete DSA MLP rows do not match the CP-v2 "
                    "logical row metadata"
                )
            from sglang.srt.layers.cp.base import get_cp_strategy

            strategy = get_cp_strategy()
            if strategy is None:
                raise RuntimeError(
                    "GLM-5.2 CP-v2 source reassembly requires an active strategy"
                )
            local_hidden_states = strategy.shard_hidden_states(
                hidden_states[block], forward_batch
            )
            if local_hidden_states.shape != residual.shape:
                raise RuntimeError(
                    "GLM-5.2 CP-v2 source reassembly returned the wrong physical "
                    f"shape: local={tuple(local_hidden_states.shape)}, "
                    f"residual={tuple(residual.shape)}"
                )
            return local_hidden_states, residual
        try:
            source = ownership.local_source_slice(
                segment_lengths,
                local_rows=local_rows,
                context_sharded=prefill_cp,
            )
        except ValueError as error:
            raise RuntimeError(
                "Replicated complete DSA MLP rows do not match the logical source layout"
            ) from error
        return hidden_states[source], residual


class DSACPCommunicateSimpleFn(CommunicateSimpleFn):
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return DSACPCommunicateSimpleFn._trivial

        raise NotImplementedError(f"{input_mode=} {output_mode=}")


class DSACPCommunicateWithAllReduceAndLayerNormFn(
    CommunicateWithAllReduceAndLayerNormFn
):
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        assert hidden_states_input_mode == ScatterMode.SCATTERED
        assert residual_input_mode == ScatterMode.SCATTERED
        assert residual_output_mode == ScatterMode.SCATTERED
        if hidden_states_output_mode == ScatterMode.SCATTERED:
            return DSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                DSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        # for prefill: attn tp scattered -> full
        # for decode: attn tp full -> full
        if (
            dsa_use_prefill_cp(forward_batch)
            or mla_use_prefill_cp(forward_batch)
            or get_parallel().attn_dp_size > 1
        ):
            hidden_states = gather_glm52_mlp_rows(
                hidden_states,
                forward_batch,
                context=context,
            )
        return hidden_states, residual


class DSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        # Check exact enum match first: even if group sizes happen to be equal
        # (e.g. tp_size == attn_cp_size makes FULL and SCATTERED both size 1),
        # FULL and SCATTERED have different data layouts under CP and require
        # an explicit scatter operation.
        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return DSACPCommunicateSummableTensorPairFn._scatter_hidden_states

        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return DSACPCommunicateSummableTensorPairFn._trivial

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
    ):
        # for prefill: full -> attn tp scattered
        # for decode: full -> attn tp full
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):
            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
        return hidden_states, residual
