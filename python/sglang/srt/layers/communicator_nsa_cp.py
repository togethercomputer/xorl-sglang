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

from sglang.srt.layers.attention.nsa.utils import (
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
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
    get_attention_cp_size,
    get_local_dp_buffer,
)
from sglang.srt.layers.glm52_positions import (
    CanonicalMoEPositions,
    align_glm52_moe_positions as _align_glm52_moe_positions,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def nsa_enable_prefill_cp():
    # After using cp, the communication mode of this part changes.
    # The three parts of prepare_attn, prepare_mlp, and postprocess_layer
    # no longer require additional communication for reduce, scatter, etc.
    return is_nsa_enable_prefill_cp()


def align_glm52_moe_positions(
    positions: torch.Tensor,
    full_hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> CanonicalMoEPositions:
    """Apply the same rank-major CP gather and padding layout as FULL MoE rows."""

    prefill_cp = nsa_use_prefill_cp(forward_batch)
    return _align_glm52_moe_positions(
        positions,
        full_hidden_states,
        prefill_cp=prefill_cp,
        cp_size=get_attention_cp_size() if prefill_cp else 1,
        all_gather=attn_cp_all_gather_into_tensor if prefill_cp else None,
    )


class NSAMLPOutputLayout(str, Enum):
    """Cross-rank reduction state of an NSA layer's MLP output."""

    LEGACY_PARTIAL = "legacy_partial"
    COMPLETE = "complete"


class NSACPLayerCommunicator(LayerCommunicator):
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        mlp_output_layout: NSAMLPOutputLayout = NSAMLPOutputLayout.LEGACY_PARTIAL,
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
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            assert self._context.attn_dp_size == 1, (
                "dp_size should be 1 when moe_runner_backend is none"
            )
        self._communicate_simple_fn = NSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = NSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode=ScatterMode.SCATTERED,
            residual_input_mode=ScatterMode.SCATTERED,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        if self.mlp_output_layout is NSAMLPOutputLayout.COMPLETE:
            if self.layer_scatter_modes.mlp_mode not in {
                ScatterMode.SCATTERED,
                ScatterMode.FULL,
            }:
                raise RuntimeError("Complete NSA MLP output has an unsupported layout")
            if not self._context.is_same_group_size(
                self.layer_scatter_modes.middle_residual_mode,
                ScatterMode.SCATTERED,
            ) or not self._context.is_same_group_size(
                self.layer_scatter_modes.layer_output_mode,
                ScatterMode.SCATTERED,
            ):
                raise RuntimeError(
                    "Complete NSA MLP output requires a rank-local CP residual/output layout"
                )
            self._communicate_summable_tensor_pair_fn = (
                self._reject_legacy_complete_postprocess
            )
            return
        self._communicate_summable_tensor_pair_fn = NSACPCommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )

    @staticmethod
    def _reject_legacy_complete_postprocess(**_kwargs):
        raise RuntimeError(
            "Complete NSA MLP output cannot use the legacy summed reduce-scatter path"
        )

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        if self.mlp_output_layout is NSAMLPOutputLayout.COMPLETE:
            return False
        return super().should_use_reduce_scatter(forward_batch)

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.mlp_output_layout is NSAMLPOutputLayout.LEGACY_PARTIAL:
            return super().postprocess_layer(hidden_states, residual, forward_batch)
        if residual is None:
            raise RuntimeError("Complete NSA MLP output requires a residual layout")
        if self.layer_scatter_modes.mlp_mode is ScatterMode.SCATTERED:
            if hidden_states.shape != residual.shape:
                raise RuntimeError(
                    "Complete local NSA MLP output must match the residual layout"
                )
            return hidden_states, residual
        if not nsa_use_prefill_cp(forward_batch):
            if hidden_states.shape != residual.shape:
                raise RuntimeError(
                    "Complete NSA decode output must match the replicated residual layout"
                )
            return hidden_states, residual
        # Canonical v3 may deliver the complete folded rows directly to the CP
        # rank that consumes them. This is already the residual's local layout;
        # no slice or floating-point collective remains.
        if hidden_states.shape == residual.shape:
            return hidden_states, residual
        cp_size = self._context.attn_cp_size
        cp_rank = self._context.attn_cp_rank
        if cp_size <= 1 or not 0 <= cp_rank < cp_size:
            raise RuntimeError(
                f"Invalid canonical NSA CP coordinates rank={cp_rank}, size={cp_size}"
            )
        local_rows = residual.shape[0]
        if (
            hidden_states.shape[0] != local_rows * cp_size
            or hidden_states.shape[1:] != residual.shape[1:]
        ):
            raise RuntimeError(
                "Replicated complete NSA MLP rows do not match the rank-local CP residual capacity"
            )
        # The NSA FULL gather is rank-major. Canonical reduction preserves that
        # row order, so selecting this block changes layout without another sum.
        return hidden_states.narrow(0, cp_rank * local_rows, local_rows), residual


class NSACPCommunicateSimpleFn(CommunicateSimpleFn):
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return NSACPCommunicateSimpleFn._trivial

        raise NotImplementedError(f"{input_mode=} {output_mode=}")


class NSACPCommunicateWithAllReduceAndLayerNormFn(
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
            return NSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                NSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
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
        if nsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            hidden_states, local_hidden_states = (
                get_local_dp_buffer(),
                hidden_states,
            )
            attn_cp_all_gather_into_tensor(
                hidden_states,
                local_hidden_states,
            )
        return hidden_states, residual


class NSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return NSACPCommunicateSummableTensorPairFn._trivial

        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return NSACPCommunicateSummableTensorPairFn._scatter_hidden_states

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
        if nsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            input_hidden_states = hidden_states
            hidden_states = hidden_states.tensor_split(context.attn_cp_size)[
                context.attn_cp_rank
            ]
            attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
        return hidden_states, residual
