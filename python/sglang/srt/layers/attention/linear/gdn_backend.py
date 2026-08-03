from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

BI_GDN_PREFILL_ENABLED = False
BI_GDN_DECODE_ENABLED = False
BI_GDN_DECODE_GRAPH = False
BI_GDN_BS1_STATIC = False


def _use_bi_gdn_graph_workspace() -> bool:
    """Use slot zero only while recording the graph that replays from it.

    A graph-enabled server can still execute an eager fallback for a local DP
    batch.  That path has no replay prologue to gather the live request into
    slot zero, so it must retain the ordinary live-slot decode semantics.
    """
    return BI_GDN_DECODE_GRAPH and get_is_capture_mode()


if is_cuda():
    from sglang.srt.layers.attention.fla.bi_gdn_decode import (
        BI_GDN_BS1_STATIC,
        BI_GDN_DECODE_ENABLED,
        BI_GDN_DECODE_GRAPH,
        BIGDNDecodeCache,
        BIGDNDecodeStepMetadata,
    )
    from sglang.srt.layers.attention.fla.bi_gdn_prefill import (
        BI_GDN_PREFILL_ENABLED,
        bi_chunk_gated_delta_rule_prefill,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer if either decode or prefill selected it
        if decode_backend.is_flashinfer() or prefill_backend.is_flashinfer():
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__}"
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        if BI_GDN_PREFILL_ENABLED and not prefill_backend.is_triton():
            raise RuntimeError(
                "SGLANG_BI_GDN_PREFILL=1 contracts the triton chunked-prefill path only; "
                f"got --linear-attn-prefill-backend {prefill_backend}."
            )
        if BI_GDN_DECODE_ENABLED:
            if not BI_GDN_PREFILL_ENABLED:
                raise RuntimeError(
                    "SGLANG_BI_GDN_DECODE=1 requires SGLANG_BI_GDN_PREFILL=1."
                )
            if (
                not model_runner.server_args.disable_cuda_graph
                and not BI_GDN_DECODE_GRAPH
            ):
                raise RuntimeError(
                    "SGLANG_BI_GDN_DECODE=1 is eager-only (dynamic partial-chunk "
                    "rescan shapes); launch with --disable-cuda-graph or opt in with "
                    "SGLANG_BI_GDN_DECODE_GRAPH=1 (XORL-245)."
                )
            if BI_GDN_DECODE_GRAPH:
                if not BI_GDN_BS1_STATIC:
                    raise RuntimeError(
                        "SGLANG_BI_GDN_DECODE_GRAPH=1 requires SGLANG_BI_GDN_BS1_STATIC=1."
                    )
                if model_runner.server_args.disable_cuda_graph:
                    raise RuntimeError(
                        "SGLANG_BI_GDN_DECODE_GRAPH=1 requires CUDA graphs to be enabled."
                    )
                graph_bs = list(model_runner.server_args.cuda_graph_bs or [])
                if len(graph_bs) != 1 or graph_bs[0] < 1:
                    raise RuntimeError(
                        "SGLANG_BI_GDN_DECODE_GRAPH=1 currently requires one exact-size "
                        "--cuda-graph-bs bucket (XORL-245)."
                    )
        self._bi_decode_caches = {}  # layer_id -> BIGDNDecodeCache (XORL-245)
        self._bi_decode_step_metadata: Optional[BIGDNDecodeStepMetadata] = None
        self._bi_decode_slots: Optional[list[int]] = None
        self._bi_graph_bs: Optional[int] = None
        self._bi_graph_workspace_indices: Optional[torch.Tensor] = None
        self._bi_graph_seq_lens: Optional[torch.Tensor] = None
        self._bi_graph_step_metadata: Optional[BIGDNDecodeStepMetadata] = None
        if BI_GDN_DECODE_ENABLED:
            rank0_log("GDN decode contract ENGAGED: partial-chunk rescan")
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

    def _bi_decode_cache(self, layer: RadixLinearAttention, ssm_states: torch.Tensor):
        cache = self._bi_decode_caches.get(layer.layer_id)
        if cache is None:
            cache = BIGDNDecodeCache(
                num_slots=ssm_states.shape[0],
                qkv_dim=layer.q_dim + layer.k_dim + layer.v_dim,
                num_v_heads=layer.num_v_heads,
                head_k_dim=layer.head_k_dim,
                head_v_dim=layer.head_v_dim,
                device=ssm_states.device,
            )
            if BI_GDN_DECODE_GRAPH and self._bi_graph_bs is not None:
                cache.configure_graph_workspace(self._bi_graph_bs)
            self._bi_decode_caches[layer.layer_id] = cache
        return cache

    def _bi_decode_metadata(
        self,
        cache: BIGDNDecodeCache,
        slots: list[int],
        slot_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> BIGDNDecodeStepMetadata:
        metadata = self._bi_decode_step_metadata
        if metadata is None:
            metadata = cache.prepare_step_metadata(
                slots,
                slot_indices,
                seq_lens,
                graph_capture=BI_GDN_DECODE_GRAPH and get_is_capture_mode(),
            )
            self._bi_decode_step_metadata = metadata
            if (
                BI_GDN_DECODE_GRAPH
                and get_is_capture_mode()
                and self._bi_decode_slots is not None
                and slots == self._bi_decode_slots
            ):
                self._bi_graph_step_metadata = metadata
        elif metadata.slots != tuple(slots):
            raise RuntimeError(
                "SGLANG_BI_GDN_DECODE slot selection changed within one forward: "
                f"expected {metadata.slots}, got {tuple(slots)} (XORL-245)."
            )
        return metadata

    def _bi_gdn_decode_seed(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        pre_states: torch.Tensor,
    ) -> None:
        cache = self._bi_decode_cache(layer, ssm_states)
        starts = query_start_loc.tolist()
        prefix_lens = forward_batch.extend_prefix_lens.tolist()
        slots = cache_indices.tolist()
        g = g.reshape(-1, layer.num_v_heads)
        beta = beta.reshape(-1, layer.num_v_heads)
        for row, slot in enumerate(slots):
            lo, hi = starts[row], starts[row + 1]
            cache.seed_from_extend(
                slot=slot,
                pre_scan_state=pre_states[row],
                qkv_rows=mixed_qkv[lo:hi],
                g_rows=g[lo:hi],
                beta_rows=beta[lo:hi],
                prefix_len=int(prefix_lens[row]),
                ssm_states=ssm_states,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._bi_decode_step_metadata = None
        self._bi_decode_slots = None
        if BI_GDN_DECODE_ENABLED and forward_batch.forward_mode.is_decode_or_idle():
            # One device-to-host synchronization per model forward, rather than
            # one in every GDN layer. Exact decode is eager-only, so the host
            # slot list is safe to reuse for all layers in this forward.
            self._bi_decode_slots = self.forward_metadata.mamba_cache_indices.tolist()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        if BI_GDN_DECODE_GRAPH:
            if bs < 1 or num_tokens != bs:
                raise RuntimeError(
                    "BI GDN graph capture requires an exact single-token decode "
                    f"bucket (bs={bs}, num_tokens={num_tokens})"
                )
            self._bi_graph_bs = bs
            self._bi_decode_step_metadata = None
            self._bi_graph_step_metadata = None
            self._bi_decode_slots = list(range(bs))
            self._bi_graph_workspace_indices = torch.arange(
                bs, dtype=torch.int32, device=seq_lens.device
            )
            self._bi_graph_seq_lens = seq_lens[:bs]
            for cache in self._bi_decode_caches.values():
                cache.configure_graph_workspace(bs)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        seq_lens_sum,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )
        if BI_GDN_DECODE_GRAPH:
            self._bi_graph_bs = bs
            self._bi_graph_seq_lens = seq_lens[:bs]

    def begin_cuda_graph_capture(self):
        if not BI_GDN_DECODE_GRAPH:
            return None
        pool = self.req_to_token_pool.mamba_pool.mamba_cache
        graph_bs = self._bi_graph_bs or 1
        return {
            "temporal": pool.temporal[:, :graph_bs].clone(),
            "conv": [tensor[:, :graph_bs].clone() for tensor in pool.conv],
        }

    def end_cuda_graph_capture(self, snapshot) -> None:
        if snapshot is None:
            return
        pool = self.req_to_token_pool.mamba_pool.mamba_cache
        graph_bs = self._bi_graph_bs or 1
        pool.temporal[:, :graph_bs].copy_(snapshot["temporal"])
        for tensor, saved in zip(pool.conv, snapshot["conv"]):
            tensor[:, :graph_bs].copy_(saved)
        for cache in self._bi_decode_caches.values():
            cache.clear_graph_workspace()

    def prepare_cuda_graph_replay(self) -> None:
        if not BI_GDN_DECODE_GRAPH:
            return
        indices = self.forward_metadata.mamba_cache_indices
        first_cache = next(iter(self._bi_decode_caches.values()), None)
        if first_cache is not None:
            first_cache.refresh_graph_metadata(
                self._bi_graph_step_metadata, self._bi_graph_seq_lens
            )
        for cache in self._bi_decode_caches.values():
            cache.copy_to_graph_workspace(indices)

    def finish_cuda_graph_replay(self) -> None:
        if not BI_GDN_DECODE_GRAPH:
            return
        indices = self.forward_metadata.mamba_cache_indices
        for cache in self._bi_decode_caches.values():
            cache.copy_from_graph_workspace(indices)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        if BI_GDN_DECODE_ENABLED:
            num_tokens = mixed_qkv.shape[0]
            if num_tokens == 0:
                return mixed_qkv.new_empty(1, 0, layer.num_v_heads, layer.head_v_dim)
            cache = self._bi_decode_cache(layer, ssm_states)
            slots_list = self._bi_decode_slots
            if slots_list is None:
                slots_list = cache_indices.tolist()
            if len(set(slots_list)) != len(slots_list):
                raise RuntimeError(
                    "SGLANG_BI_GDN_DECODE received duplicate cache slots: "
                    f"{slots_list}"
                )
            if any(slot < 0 for slot in slots_list):
                raise RuntimeError(
                    "SGLANG_BI_GDN_DECODE does not support padded cache slots"
                )
            g_rows, beta_rows = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            metadata_indices = cache_indices
            metadata_seq_lens = forward_batch.seq_lens
            state_indices = None
            use_graph_workspace = _use_bi_gdn_graph_workspace()
            if use_graph_workspace:
                metadata_indices = self._bi_graph_workspace_indices
                metadata_seq_lens = self._bi_graph_seq_lens
                state_indices = cache_indices
            core_attn_out = cache.step(
                metadata=self._bi_decode_metadata(
                    cache, slots_list, metadata_indices, metadata_seq_lens
                ),
                qkv_rows=mixed_qkv,
                g_rows=g_rows.view(num_tokens, -1),
                beta_rows=beta_rows.view(num_tokens, -1),
                ssm_states=ssm_states,
                state_indices=state_indices,
            ).unsqueeze(0)
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            if BI_GDN_PREFILL_ENABLED:
                # GDN prefill contract: run the trainer's exact scan
                # composition; chunk checkpoints come from the fp32 final-state
                # chain instead of the bf16 h buffer.
                bi_decode_pre_states = (
                    ssm_states[cache_indices.long()].clone()
                    if BI_GDN_DECODE_ENABLED
                    else None
                )
                core_attn_out = bi_chunk_gated_delta_rule_prefill(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    cu_seqlens=query_start_loc,
                    scale=layer.head_k_dim**-0.5,
                    ckpt_batch_rows=forward_metadata.bi_ckpt_batch_rows,
                    ckpt_token_starts=forward_metadata.bi_ckpt_token_starts,
                    ckpt_lens=forward_metadata.bi_ckpt_lens,
                    ckpt_dst_indices=forward_metadata.bi_ckpt_dst_indices,
                )
                if (
                    forward_metadata.has_mamba_track_mask
                    and forward_metadata.track_ssm_final_src.numel() > 0
                ):
                    # aligned tracked requests: fp32 final-state slot copy,
                    # same as the default path.
                    ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                        forward_metadata.track_ssm_final_src
                    ]
                if BI_GDN_DECODE_ENABLED:
                    self._bi_gdn_decode_seed(
                        layer,
                        forward_batch,
                        mixed_qkv,
                        g,
                        beta,
                        ssm_states,
                        cache_indices,
                        query_start_loc,
                        bi_decode_pre_states,
                    )
                return core_attn_out

            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out
