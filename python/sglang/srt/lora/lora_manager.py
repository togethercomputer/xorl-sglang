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

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

import logging
import re
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from typing import Dict, Iterable, Iterator, List, Optional

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.backend.lora_registry import get_backend_from_name
from sglang.srt.lora.dsv4 import is_dsv4_flash_exact_adapter
from sglang.srt.lora.glm52 import is_glm52_xorl_shared_outer_adapter
from sglang.srt.lora.layers import BaseLayerWithLoRA, FusedMoEWithLoRA, get_lora_layer
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import (
    DSA_INDEXER_LORA_NAMES,
    EMBEDDING_NAMES,
    LoRABatchInfo,
    LoRAType,
    auto_detect_lora_target_modules,
    get_normalized_target_modules,
    get_target_module_name,
)
from sglang.srt.managers.io_struct import LoRAUpdateOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory, replace_submodule
from sglang.srt.utils.hf_transformers_utils import AutoConfig

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

logger = logging.getLogger(__name__)


class LoRAManager:
    def __init__(
        self,
        base_model: torch.nn.Module,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        server_args: ServerArgs,
        lora_backend: str = "triton",
        tp_size: int = 1,
        tp_rank: int = 0,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
    ):
        self.base_model: torch.nn.Module = base_model
        if hasattr(base_hf_config, "get_text_config"):
            self.base_hf_config: AutoConfig = base_hf_config.get_text_config()
        else:
            self.base_hf_config: AutoConfig = base_hf_config
        self.max_loras_per_batch: int = max_loras_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.dp_size: int = getattr(server_args, "dp_size", 1)
        self.ep_size: int = getattr(server_args, "ep_size", 1)
        self.pp_size: int = getattr(server_args, "pp_size", 1)
        self.attn_cp_size: int = getattr(server_args, "attn_cp_size", 1)
        self.cp_strategy: Optional[str] = getattr(server_args, "cp_strategy", None)
        # Attention projections shard on the attn-TP group; extracted once
        # here (parallel groups are frozen after init_torch_distributed).
        self.attn_tp_size: int = get_parallel().attn_tp_size
        self.lora_added_tokens_size: Optional[int] = None
        self.enable_lora_overlap_loading: Optional[bool] = (
            server_args.enable_lora_overlap_loading
        )
        self.pending_lora_load_events = {}

        self.eviction_policy = server_args.lora_eviction_policy
        self.enable_dp_attention: bool = server_args.enable_dp_attention
        self._experts_shared_outer_override: Optional[bool] = (
            server_args.experts_shared_outer_loras
        )
        self.lora_use_virtual_experts: bool = server_args.lora_use_virtual_experts
        self.lora_strict_loading: bool = getattr(
            server_args, "lora_strict_loading", False
        )

        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
        backend_type = get_backend_from_name(lora_backend)
        self.lora_backend: BaseLoRABackend = backend_type(
            max_loras_per_batch=max_loras_per_batch,
            device=self.device,
            server_args=server_args,
        )
        self.lora_backend._dsv4_flash_exact_mode = bool(
            getattr(self.base_hf_config, "_dsv4_flash_exact_mode", False)
        )
        self.lora_backend._dsv4_flash_exact_batch_certified = False

        # Initialize mutable internal state of the LoRAManager.
        self.init_state(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
            lora_paths=lora_paths,
        )

    def init_cuda_graph_batch_info(
        self, max_bs_in_cuda_graph: int, num_tokens_per_req: int
    ):
        """Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.

        Called during CudaGraphRunner.__init__(), after init_memory_pool().
        Phase 1 (MoE buffers) is handled earlier via init_cuda_graph_moe_buffers().
        """
        self.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        self.lora_backend.init_cuda_graph_batch_info(
            max_bs_in_cuda_graph=max_bs_in_cuda_graph,
            num_tokens_per_req=num_tokens_per_req,
        )

        # ===== TO BE REFACTORED ====
        # Pre-create the experimental LoRA two-stream side stream now (gated) so the
        # torch.cuda.Stream() call never lands inside a cuda-graph capture region.
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            from sglang.srt.lora.trtllm_lora_temp import (
                init_lora_two_stream_resources,
            )

            init_lora_two_stream_resources(self.device)
        # ===== END TO BE REFACTORED ====

    def init_prefill_cuda_graph_batch_info(self, max_num_tokens: int):
        """Allocate the static prefill-CUDA-graph LoRA metadata, sized by the
        largest captured token bucket. Called before capture."""
        self.lora_backend.init_prefill_cuda_graph_batch_info(
            max_num_tokens=max_num_tokens
        )

    @property
    def supports_prefill_cuda_graph(self) -> bool:
        """Whether LoRA kernels can be captured into the prefill CUDA graph;
        excludes MoE LoRA and DP attention."""
        return (
            self.lora_backend.supports_prefill_cuda_graph
            and not self.lora_backend.is_moe_lora
            and not self.enable_dp_attention
        )

    @property
    def prefill_cuda_graph_max_bs(self) -> Optional[int]:
        """Request-count cap for prefill-graph LoRA batches; None until
        init_prefill_cuda_graph_batch_info() ran."""
        return self.lora_backend.prefill_cuda_graph_max_bs

    def can_use_prefill_cuda_graph(self, forward_batch: ForwardBatch) -> bool:
        """Whether this batch can use the static prefill-graph LoRA metadata;
        shared by prepare_lora_batch and can_run_graph so they stay consistent."""
        max_bs = self.lora_backend.prefill_cuda_graph_max_bs
        max_tokens = self.lora_backend.prefill_cuda_graph_max_tokens
        if max_bs is None or max_tokens is None:
            return False
        # DP attention: per-rank eligibility could diverge across ranks and
        # desync collectives; keep LoRA prefill eager.
        if self.enable_dp_attention:
            return False
        # Decode-CUDA-graph extend modes (TARGET_VERIFY, DLLM_EXTEND) are
        # owned by the decode static batch info path.
        if (
            not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_cuda_graph()
        ):
            return False
        if forward_batch.extend_num_tokens is None:
            return False
        return (
            forward_batch.batch_size <= max_bs
            and forward_batch.extend_num_tokens <= max_tokens
        )

    def init_cuda_graph_moe_buffers(
        self, max_bs: int, max_loras: int, compute_dtype, moe_layer
    ):
        """Phase 1 of LoRA CUDA graph init: MoE intermediate buffers.

        Called before init_memory_pool() so memory profiling accounts for them.
        Phase 2 (dense batch metadata) is handled later via init_cuda_graph_batch_info().
        """
        self.lora_backend.init_cuda_graph_moe_buffers(
            max_bs=max_bs,
            max_loras=max_loras,
            compute_dtype=compute_dtype,
            moe_layer=moe_layer,
        )

    def create_lora_update_result(
        self, success: bool, error_message: str = ""
    ) -> LoRAUpdateOutput:
        return LoRAUpdateOutput(
            success=success,
            error_message=error_message,
            loaded_adapters={
                lora_ref.lora_name: lora_ref.lora_path
                for lora_ref in self.lora_refs.values()
            },
        )

    def load_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateOutput:
        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device.type, self.device.index):.2f} GB"
        )
        result = self._load_lora_adapter(lora_ref)
        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device.type, self.device.index):.2f} GB"
        )
        return result

    def _load_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateOutput:
        """
        Load a single LoRA adapter from the specified path.

        Args:
            lora_ref (LoRARef): The LoRARef object containing the LoRA name, path, and ID.
        """
        assert (
            lora_ref.lora_name is not None and lora_ref.lora_path is not None
        ), "LoRARef must have both lora_name and lora_path set for loading."
        assert (
            lora_ref.lora_id not in self.loras
        ), f"LoRA adapter with ID {lora_ref.lora_id} is already loaded. This should have been verified before request is sent to the backend."

        try:
            # load configs
            new_adapter = LoRAConfig(
                lora_ref.lora_path,
                base_vocab_size=self.base_hf_config.vocab_size,
            )
            self.validate_new_adapter(new_adapter, lora_ref)
            self.configs[lora_ref.lora_id] = new_adapter

            # load weights
            self.load_lora_weights(lora_ref)

            # keep metadata for displayed messages
            self.lora_refs[lora_ref.lora_id] = lora_ref
            self.num_pinned_loras += int(lora_ref.pinned)
        except Exception as e:
            rollback_result = self.rollback_lora_adapter(lora_ref.lora_id)
            error_message = str(e)
            if not rollback_result.success:
                error_message += (
                    "; local rollback failed: " + rollback_result.error_message
                )
            return self.create_lora_update_result(
                success=False,
                error_message=error_message,
            )

        return self.create_lora_update_result(success=True)

    def rollback_lora_adapter(self, lora_id: str) -> LoRAUpdateOutput:
        """Remove every local trace of an uncommitted adapter load.

        Dynamic loads are committed by the control plane only after all TP
        ranks agree.  This compensation path is deliberately idempotent so it
        can clean both a locally failed load and a locally successful load
        whose peer failed.
        """
        errors = []

        pending_event = self.pending_lora_load_events.pop(lora_id, None)
        if pending_event is not None:
            try:
                pending_event.synchronize()
            except Exception as e:
                errors.append(f"pending load synchronization failed: {e}")

        memory_pool = getattr(self, "memory_pool", None)
        if memory_pool is not None:
            try:
                removed_slot = memory_pool.remove_lora(lora_id)
                if removed_slot is not None:
                    self._notify_lora_slots_updated({removed_slot})
            except Exception as e:
                errors.append(f"memory-pool rollback failed: {e}")

        lora_ref = self.lora_refs.pop(lora_id, None)
        self.loras.pop(lora_id, None)
        self.configs.pop(lora_id, None)
        if lora_ref is not None:
            self.num_pinned_loras -= int(lora_ref.pinned)

        return self.create_lora_update_result(
            success=not errors,
            error_message="; ".join(errors),
        )

    def validate_new_adapter(self, lora_config: LoRAConfig, lora_ref: LoRARef):
        """
        Validate if an adapter can be loaded into the current LoRA memory pool and generate error if it is incompatible.
        """
        if lora_config.lora_added_tokens_size > 0:
            raise ValueError(
                f"Failed to load {lora_ref.lora_name} because LoRA serving currently doesn't support adapters that add tokens to the vocabulary"
            )

        if lora_config.use_dora:
            raise ValueError(
                f"Failed to load {lora_ref.lora_name} because LoRA serving currently doesn't support DoRA adapters"
            )

        # Check if this LoRA adapter is already loaded
        for existing_lora_ref in self.lora_refs.values():
            if lora_ref.lora_name == existing_lora_ref.lora_name:
                raise ValueError(
                    f"Failed to load LoRA adapter {lora_ref.lora_name} because it is already loaded"
                )

            if lora_ref.lora_path == existing_lora_ref.lora_path:
                logger.warning(
                    f"{lora_ref.lora_path} is already loaded with name: {existing_lora_ref.lora_name}, "
                    f"but another copy is being loaded with name: {lora_ref.lora_name}"
                )

        # Check if the LoRA adapter shape is compatible with the current LoRA memory pool configuration.
        memory_pool = getattr(self, "memory_pool", None)
        if memory_pool is not None:
            self._validate_glm52_runtime_layout(lora_config)
            self._validate_dsv4_flash_runtime_layout(lora_config)
        incompatible = memory_pool and not memory_pool.can_support(lora_config)
        if incompatible:
            raise ValueError(
                f"LoRA adapter {lora_ref.lora_name} with rank {lora_config.r} is incompatible with the current "
                "LoRA memory pool configuration. Please ensure that the LoRA adapter's rank is within the configured "
                "`--max-lora-rank` and that the target modules are included in `--lora-target-modules`."
            )

        # Ensure pinned LoRA adapters does not exceed maximal limit or cause starvation.
        if lora_ref.pinned and self.num_pinned_loras >= self.max_loras_per_batch - 1:
            raise ValueError(
                f"Failed to load LoRA adapter {lora_ref.lora_name} as a pinned adapter. It is not allowed to pin all slots "
                "in the LoRA memory pool to avoid starvation for unpinned adapters and base models. Please increase your "
                "`--max-loras-per-batch` or load it as unpinned LoRA adapters."
            )

    def _validate_glm52_runtime_layout(self, lora_config: LoRAConfig) -> None:
        shared_outer = is_glm52_xorl_shared_outer_adapter(
            self.base_hf_config, lora_config.hf_config
        )
        if (
            getattr(self.base_hf_config, "_glm52_exact_mode", False)
            and not shared_outer
        ):
            raise ValueError(
                "The exact GLM-5.2 XORL active-LoRA contract requires "
                "_sglang_lora_format='shared_outer'; ordinary or missing "
                "adapter formats are not admitted."
            )
        if not shared_outer:
            return
        if not getattr(self, "experts_shared_outer_loras", False):
            raise ValueError(
                "GLM-5.2 XoRL shared-outer adapters require a shared-outer "
                "LoRA memory pool. Start the server with "
                "--experts-shared-outer-loras before using POST "
                "/load_lora_adapter."
            )
        if getattr(self.base_model, "num_fused_shared_experts", 0) != 0:
            raise ValueError(
                "GLM-5.2 XoRL adapters require unfused shared-expert modules. "
                "Start the server with --disable-shared-experts-fusion."
            )

    def _validate_dsv4_flash_runtime_layout(self, lora_config: LoRAConfig) -> None:
        identified = is_dsv4_flash_exact_adapter(
            self.base_hf_config, lora_config.hf_config
        )
        if (
            getattr(self.base_hf_config, "_dsv4_flash_exact_mode", False)
            and not identified
        ):
            raise ValueError(
                "The exact DSV4-Flash active-LoRA contract requires "
                "_sglang_lora_format='dsv4_expert_banks'; ordinary or missing "
                "adapter formats are not admitted."
            )
        if not identified:
            return
        if getattr(self, "experts_shared_outer_loras", False):
            raise ValueError(
                "DSV4-Flash per-expert A/B banks require a non-shared-outer "
                "LoRA memory pool."
            )
        if getattr(self.base_model, "num_fused_shared_experts", 0) != 0:
            raise ValueError(
                "DSV4-Flash exact adapters require unfused shared-expert "
                "modules. Start with --disable-shared-experts-fusion."
            )

    def _validate_dsv4_flash_exact_batch(self, forward_batch: ForwardBatch) -> None:
        self.lora_backend._dsv4_flash_exact_batch_certified = False
        if not getattr(self.base_hf_config, "_dsv4_flash_exact_mode", False):
            return
        # ``ForwardMode.is_cuda_graph()`` means that a mode *can* be captured;
        # eager decode uses the same ``DECODE`` enum.  Reject only an actual
        # capture or a batch backed by initialized decode-graph metadata.
        uses_decode_cuda_graph = (
            hasattr(self, "max_bs_in_cuda_graph")
            and forward_batch.batch_size <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        )
        if get_is_capture_mode() or uses_decode_cuda_graph:
            raise RuntimeError(
                "The exact DSV4-Flash active-LoRA contract admits eager execution only."
            )
        lora_ids = list(forward_batch.lora_ids)
        if len(lora_ids) != 1:
            raise RuntimeError(
                "The exact DSV4-Flash active-LoRA contract admits exactly one "
                f"logical request, got {len(lora_ids)}."
            )
        uid = lora_ids[0]
        if uid is None:
            self.lora_backend._dsv4_flash_exact_batch_certified = True
            return
        self._validate_dsv4_flash_exact_uid(uid)
        self.lora_backend._dsv4_flash_exact_batch_certified = True

    def _validate_dsv4_flash_exact_uid(self, uid: str) -> None:
        """Validate one active adapter independently of its scheduler owner."""

        adapter = self.loras.get(uid)
        if adapter is None or uid not in self.memory_pool.uid_to_buffer_id:
            raise RuntimeError(
                "The exact DSV4-Flash active-LoRA request references a missing "
                f"or nonresident adapter UID: {uid!r}."
            )
        if not getattr(adapter, "_dsv4_flash_exact_adapter_certified", False):
            raise RuntimeError(
                "The exact DSV4-Flash active-LoRA request requires an adapter "
                "certified from the complete 948-factor inventory."
            )
        if adapter.config.r != 1 or adapter.scaling != 1:
            raise RuntimeError(
                "The exact DSV4-Flash active-LoRA request requires rank 1 and "
                f"unit scaling, got rank={adapter.config.r}, scaling={adapter.scaling}."
            )

    def _validate_glm52_active_adapters(self, forward_batch: ForwardBatch) -> None:
        """Reject stale adapter references without restricting normal batching."""

        if not getattr(self.base_hf_config, "_glm52_exact_mode", False):
            return

        lora_ids = list(forward_batch.lora_ids)
        if get_is_capture_mode():
            return

        for uid in set(lora_ids):
            if uid is None:
                continue
            if uid not in self.loras or uid not in self.memory_pool.uid_to_buffer_id:
                raise RuntimeError(
                    "The GLM-5.2 active-LoRA request references a missing or "
                    f"nonresident adapter UID: {uid!r}."
                )

    def unload_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateOutput:
        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device.type, self.device.index):.2f} GB"
        )
        result = self._unload_lora_adapter(lora_ref)
        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device.type, self.device.index):.2f} GB"
        )
        return result

    def _unload_lora_adapter(self, lora_ref: LoRARef) -> LoRAUpdateOutput:
        """
        Unload LoRA adapters by their names. This will remove the adapters from the memory pool and
        delete the corresponding LoRA modules.
        """

        adapter = self.configs.get(lora_ref.lora_id)
        lora_ref = self.lora_refs.get(lora_ref.lora_id)
        assert (
            adapter is not None and lora_ref is not None
        ), f"LoRA adapter with ID {lora_ref.lora_id} is not loaded. This should have been verified before request is sent to the backend."

        try:
            pending_events = getattr(self, "pending_lora_load_events", {})
            pending_event = pending_events.get(lora_ref.lora_id)
            if pending_event is not None:
                pending_event.synchronize()
                pending_events.pop(lora_ref.lora_id, None)

            removed_slot = self.memory_pool.remove_lora(lora_ref.lora_id)
            if removed_slot is not None:
                self._notify_lora_slots_updated({removed_slot})
            del self.configs[lora_ref.lora_id]
            del self.loras[lora_ref.lora_id]
            del self.lora_refs[lora_ref.lora_id]
            self.num_pinned_loras -= int(lora_ref.pinned)
        except Exception as e:
            return self.create_lora_update_result(
                success=False,
                error_message=str(e),
            )

        return self.create_lora_update_result(success=True)

    def validate_lora_batch(self, lora_ids: set[Optional[str]]) -> bool:
        """
        Validate if the LoRA IDs in the batch can be loaded into the current LoRA memory pool.
        """
        if len(lora_ids) > self.max_loras_per_batch:
            return False

        # skip pinned LoRA check if no pinned LoRA adapters are loaded.
        if self.num_pinned_loras == 0:
            return True

        # counting the number of pinned LoRA adapters in the batch.
        pinned_loras_in_batch = 0
        for lora_id in lora_ids:
            if lora_id is not None:
                lora_ref = self.lora_refs.get(lora_id)
                assert (
                    lora_ref is not None
                ), f"LoRA ID {lora_id} not found in lora_refs."
                pinned_loras_in_batch += int(lora_ref.pinned)

        assert pinned_loras_in_batch <= self.num_pinned_loras, (
            f"Number of pinned LoRA adapters in the batch ({pinned_loras_in_batch}) exceeds the total number of pinned adapters "
            f"({self.num_pinned_loras}). This indicates a bug in the LoRA loading logic."
        )

        required_slots = len(lora_ids) - pinned_loras_in_batch
        mem_pool_vacancy = self.memory_pool.max_loras_per_batch - self.num_pinned_loras

        return required_slots <= mem_pool_vacancy

    def fetch_new_loras(
        self, new_loras: set[Optional[str]], running_loras: set[Optional[str]] = set()
    ):
        # Load active loras into lora memory pool
        cur_uids = new_loras | running_loras

        assert len(cur_uids) <= self.max_loras_per_batch
        new_uids = {
            uid for uid in cur_uids if uid not in self.memory_pool.uid_to_buffer_id
        }
        self.memory_pool.prepare_lora_batch(
            cur_uids=cur_uids,
            lora_adapters=self.loras,
            lora_modules=self.lora_modules,
            lora_refs=self.lora_refs.copy(),  # copy snapshot of current lora_refs to avoid mutation during the batch preparation.
            lora_embed_tokens_module=self.embed_tokens_module,  # merge into embedding or lora module
            lora_lm_head_module=self.lm_head_module,  # merge into embedding or lora module
        )
        if new_uids:
            changed_slots = {self.memory_pool.uid_to_buffer_id[uid] for uid in new_uids}
            self._notify_lora_slots_updated(changed_slots)

    def _notify_lora_slots_updated(self, slot_ids: set[int]) -> None:
        for layer_modules in self.lora_modules:
            for module in layer_modules.values():
                notify = getattr(module, "on_lora_slots_updated", None)
                if callable(notify):
                    notify(slot_ids)

    def reset_lora_batch(self):
        """Clear per-batch LoRA state. Called instead of prepare_lora_batch()
        on DP-attention idle forwards (zero local tokens), so the LoRA layers
        take the base path instead of reading the previous batch's stale
        metadata."""
        self.lora_backend.reset_batch_state()
        if getattr(self.base_hf_config, "_dsv4_flash_exact_mode", False):
            self.lora_backend._dsv4_flash_exact_batch_certified = False

    def prepare_dsv4_flash_exact_dp_lora_batch(
        self, forward_batch: ForwardBatch
    ) -> None:
        """Build rank-major LoRA metadata for the DP-gathered DSV4 MLP rows."""

        if not getattr(self.base_hf_config, "_dsv4_flash_exact_mode", False):
            return
        if get_is_capture_mode():
            raise RuntimeError(
                "Gather-aware DSV4-Flash active LoRA admits eager execution only."
            )

        local_lora_ids = list(forward_batch.lora_ids or [])
        if len(local_lora_ids) > 1:
            raise RuntimeError(
                "The exact DSV4-Flash lane admits at most one logical request "
                f"per DP rank, got {len(local_lora_ids)}."
            )
        local_uid = local_lora_ids[0] if local_lora_ids else None

        tp_group = get_parallel().tp_group
        global_uids = tp_group.all_gather_object(local_uid)
        global_num_tokens = list(forward_batch.global_num_tokens_cpu or [])
        if len(global_uids) != len(global_num_tokens):
            raise RuntimeError(
                "DSV4-Flash gathered LoRA ownership does not match the DP row "
                f"layout: owners={len(global_uids)}, row_segments={len(global_num_tokens)}."
            )

        active_uids = {uid for uid in global_uids if uid is not None}
        active_adapter_limit = self.max_loras_per_batch - 1
        if len(active_uids) > active_adapter_limit:
            raise RuntimeError(
                "DSV4-Flash gathered batch exceeds the one-base-plus-adapter "
                f"resident layout: active={sorted(active_uids)}, "
                f"adapter_limit={active_adapter_limit}."
            )
        missing = sorted(
            uid
            for uid in active_uids
            if uid not in self.loras or uid not in self.lora_refs
        )
        if missing:
            raise RuntimeError(
                "DSV4-Flash gathered batch references adapters missing from this "
                f"EP rank: {missing}."
            )

        # Every EP rank must own the adapter factors for every gathered row,
        # including the base rows. This also makes an active-LoRA request safe
        # as the first request after startup.
        self.fetch_new_loras(active_uids | {None})
        if None not in self.memory_pool.uid_to_buffer_id:
            raise RuntimeError(
                "DSV4-Flash gather-aware LoRA requires a resident base-model slot."
            )
        for uid in active_uids:
            self._validate_dsv4_flash_exact_uid(uid)

        weight_indices = []
        lora_ranks = [0] * self.max_loras_per_batch
        scalings = [0.0] * self.max_loras_per_batch
        for uid in global_uids:
            effective_uid = uid
            if uid is not None and getattr(
                self.loras[uid], "_dsv4_flash_exact_all_zero", False
            ):
                effective_uid = None
            slot = self.memory_pool.get_buffer_id(effective_uid)
            weight_indices.append(slot)
            if effective_uid is not None:
                adapter = self.loras[effective_uid]
                lora_ranks[slot] = adapter.config.r
                scalings[slot] = adapter.scaling

        has_active_lora = any(rank > 0 for rank in lora_ranks)
        if not has_active_lora:
            self.lora_backend.context_parallel_mlp_batch_info = None
            return

        device = self.device
        segment_lens = torch.tensor(global_num_tokens, dtype=torch.int32, device=device)
        segment_indptr = torch.zeros(
            (len(global_num_tokens) + 1,), dtype=torch.int32, device=device
        )
        segment_indptr[1:] = torch.cumsum(segment_lens, dim=0)
        total_tokens = sum(global_num_tokens)
        global_batch_info = LoRABatchInfo(
            use_cuda_graph=False,
            bs=len(global_num_tokens),
            num_segments=len(global_num_tokens),
            seg_indptr=segment_indptr,
            weight_indices=torch.tensor(
                weight_indices, dtype=torch.int32, device=device
            ),
            lora_ranks=torch.tensor(lora_ranks, dtype=torch.int64, device=device),
            scalings=torch.tensor(scalings, dtype=torch.float32, device=device),
            max_len=max(global_num_tokens, default=0),
            seg_lens=segment_lens,
            permutation=None,
            expected_tokens=total_tokens,
            has_active_lora=True,
            req_seg_indptr=segment_indptr,
            req_weight_indices=torch.tensor(
                weight_indices, dtype=torch.int32, device=device
            ),
        )

        physical_batch = copy(forward_batch)
        physical_batch.batch_size = len(global_num_tokens)
        physical_batch.lora_ids = global_uids
        physical_batch.forward_mode = (
            ForwardMode.EXTEND
            if forward_batch.is_extend_in_batch
            else ForwardMode.DECODE
        )
        physical_batch.extend_num_tokens = total_tokens
        physical_batch.extend_seq_lens_cpu = global_num_tokens
        physical_batch.extend_seq_lens = segment_lens
        global_batch_info = self.lora_backend._add_moe_lora_info(
            physical_batch, global_batch_info
        )
        self.lora_backend.context_parallel_mlp_batch_info = global_batch_info

    @contextmanager
    def glm52_context_parallel_lora_batch(
        self, forward_batch: ForwardBatch, local_num_tokens: int
    ) -> Iterator[None]:
        """Use certified GLM-5.2 CP-local LoRA metadata in the sharded body.

        ``ForwardBatch.init_new`` prepares LoRA metadata before CP-v2 shards the
        model input.  Triton LoRA kernels size their grids and row masks from
        that metadata, so leaving the global sequence lengths installed while
        the body sees only rank-local rows can issue out-of-bounds accesses.

        Sparse MLPs gather those local rows in raw rank-major order before
        reducing/scattering them again. Build both rank-local body metadata and
        rank-major physical MLP metadata, then restore the full object before
        returning to logits processing. This initial path is deliberately
        limited to the qualified WORLD16 topology.

        This context is only for CP-v2 extend. WORLD16 decode does not shard
        rows through CP-v2: its CUDA graph uses the backend's fixed decode
        batch/SGEMM/MoE buffers, refreshed by ``prepare_lora_batch`` before
        replay. Decode graph metadata must therefore never enter this context.
        """
        if forward_batch.lora_ids is None:
            yield
            return

        backend = self.lora_backend
        architectures = getattr(self.base_hf_config, "architectures", None) or []
        architecture = architectures[0] if architectures else None
        strategy = None
        moe_a2a_backend = None
        if backend.name == "triton":
            from sglang.srt.layers.cp.base import get_cp_strategy
            from sglang.srt.layers.moe import get_moe_a2a_backend

            strategy = get_cp_strategy()
            moe_a2a_backend = get_moe_a2a_backend().value
        geometry = {
            "architecture": architecture,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "ep_size": self.ep_size,
            "pp_size": self.pp_size,
            "attn_cp_size": self.attn_cp_size,
            "cp_strategy": self.cp_strategy,
            "live_cp_strategy": getattr(strategy, "name", None),
            "live_cp_size": getattr(strategy, "cp_size", None),
            "experts_shared_outer_loras": self._experts_shared_outer_override,
            "enable_dp_attention": self.enable_dp_attention,
            "backend": backend.name,
            "moe_lora": backend.is_moe_lora,
            "moe_a2a_backend": moe_a2a_backend,
            "experimental_lora_opti": bool(_SGLANG_EXPERIMENTAL_LORA_OPTI),
        }
        certified_geometry = {
            "architecture": "GlmMoeDsaForCausalLM",
            "tp_size": 16,
            "dp_size": 1,
            "ep_size": 16,
            "pp_size": 1,
            "attn_cp_size": 16,
            "cp_strategy": "interleave",
            "live_cp_strategy": "interleave",
            "live_cp_size": 16,
            "experts_shared_outer_loras": True,
            "enable_dp_attention": True,
            "backend": "triton",
            "moe_lora": True,
            "moe_a2a_backend": "none",
            "experimental_lora_opti": False,
        }
        if geometry != certified_geometry:
            raise RuntimeError(
                "GLM-5.2 CP-v2 LoRA is only certified for the WORLD16 "
                f"shared-outer Triton geometry; got {geometry}."
            )
        if backend.batch_info is None:
            raise RuntimeError(
                "GLM-5.2 CP-v2 LoRA entered without prepared global batch metadata."
            )
        if (
            forward_batch.extend_seq_lens_cpu is None
            or forward_batch.extend_seq_lens is None
        ):
            raise RuntimeError(
                "GLM-5.2 CP-v2 LoRA requires host and device extend sequence lengths."
            )
        if len(forward_batch.lora_ids) != forward_batch.batch_size:
            raise RuntimeError(
                "GLM-5.2 CP-v2 LoRA request metadata is inconsistent: "
                f"batch_size={forward_batch.batch_size}, lora_ids={len(forward_batch.lora_ids)}."
            )

        full_batch_info = backend.batch_info
        if full_batch_info.use_cuda_graph or full_batch_info.permutation is not None:
            raise RuntimeError(
                "GLM-5.2 CP-v2 extend requires eager, unpermuted full-batch metadata; "
                "decode CUDA graphs use the separate fixed-buffer LoRA path."
            )
        if backend.sgemm_batch_info is not None:
            raise RuntimeError(
                "GLM-5.2 CP-v2 extend received decode-style SGEMM routing; "
                "decode CUDA graphs must not enter the CP-v2 extend context."
            )
        if backend.context_parallel_mlp_batch_info is not None:
            raise RuntimeError(
                "GLM-5.2 CP-v2 LoRA entered with stale gathered MLP metadata."
            )
        if (
            full_batch_info.bs != forward_batch.batch_size
            or full_batch_info.num_segments != forward_batch.batch_size
        ):
            raise RuntimeError(
                "GLM-5.2 CP-v2 LoRA full-batch metadata does not match the request batch: "
                f"batch_size={forward_batch.batch_size}, batch_info_bs={full_batch_info.bs}, "
                f"segments={full_batch_info.num_segments}."
            )
        global_lens_cpu = [int(length) for length in forward_batch.extend_seq_lens_cpu]
        if len(global_lens_cpu) != forward_batch.batch_size or any(
            length <= 0 for length in global_lens_cpu
        ):
            raise RuntimeError(
                "GLM-5.2 CP-v2 LoRA requires one positive extend length per request; "
                f"batch_size={forward_batch.batch_size}, lengths={global_lens_cpu}."
            )

        try:
            cp_size = int(strategy.cp_size)
            cp_rank = int(strategy.cp_rank)
            if not 0 <= cp_rank < cp_size:
                raise RuntimeError(
                    f"GLM-5.2 CP-v2 LoRA received invalid CP rank {cp_rank}."
                )

            local_num_tokens = int(local_num_tokens)
            cp_metadata = getattr(forward_batch, "attn_cp_metadata", None)
            physical_rank_tokens = getattr(cp_metadata, "per_rank_actual_token", None)
            if physical_rank_tokens is None:
                raise RuntimeError(
                    "GLM-5.2 CP-v2 LoRA requires physical per-rank CP metadata."
                )
            physical_rank_tokens = [int(tokens) for tokens in physical_rank_tokens]
            if (
                len(physical_rank_tokens) != cp_size
                or len(set(physical_rank_tokens)) != 1
                or physical_rank_tokens[cp_rank] != local_num_tokens
            ):
                raise RuntimeError(
                    "GLM-5.2 CP-v2 LoRA physical CP metadata does not match the sharded input: "
                    f"rank={cp_rank}, input_rows={local_num_tokens}, "
                    f"per_rank_rows={physical_rank_tokens}."
                )
            physical_num_tokens = physical_rank_tokens[0]

            def physical_segments(rank: int) -> tuple[list[int], list[int]]:
                # Match InterleaveCPStrategy.shard_per_request's host carry
                # rule without launching its device metadata kernel again.
                carry = 0
                all_lens = []
                for global_len in global_lens_cpu:
                    carried_len = global_len + carry
                    rank_len = carried_len // cp_size + int(
                        carried_len % cp_size > rank
                    )
                    all_lens.append(rank_len)
                    carry = carried_len - rank_len * cp_size
                request_indices = [
                    index for index, length in enumerate(all_lens) if length > 0
                ]
                rank_lens = [all_lens[index] for index in request_indices]
                if not rank_lens or any(length < 0 for length in all_lens):
                    raise RuntimeError(
                        "GLM-5.2 CP-v2 LoRA produced invalid local Interleave segments: "
                        f"rank={rank}, global={global_lens_cpu}, local={all_lens}."
                    )
                logical_num_tokens = sum(rank_lens)
                if logical_num_tokens > physical_num_tokens:
                    raise RuntimeError(
                        "GLM-5.2 CP-v2 LoRA local segments exceed the sharded input: "
                        f"rank={rank}, segments={logical_num_tokens}, "
                        f"input_rows={physical_num_tokens}."
                    )

                # pad_local_rows appends physical zeros after the logical
                # rank rows. Attribute them to the final active segment so
                # dense and MoE LoRA metadata cover the collective buffer.
                rank_lens[-1] += physical_num_tokens - logical_num_tokens
                return request_indices, rank_lens

            def build_batch_info(
                request_indices: list[int], segment_lens_cpu: list[int]
            ) -> LoRABatchInfo:
                physical_batch = copy(forward_batch)
                physical_batch.batch_size = len(request_indices)
                physical_batch.lora_ids = [
                    forward_batch.lora_ids[index] for index in request_indices
                ]
                physical_batch.extend_num_tokens = sum(segment_lens_cpu)
                physical_batch.extend_seq_lens_cpu = segment_lens_cpu
                segment_lens = torch.tensor(
                    segment_lens_cpu,
                    dtype=forward_batch.extend_seq_lens.dtype,
                    device=forward_batch.extend_seq_lens.device,
                )
                physical_batch.extend_seq_lens = segment_lens
                segment_indptr = torch.zeros(
                    (len(request_indices) + 1,),
                    dtype=segment_lens.dtype,
                    device=segment_lens.device,
                )
                segment_indptr[1:] = torch.cumsum(segment_lens, dim=0)
                weight_indices = full_batch_info.weight_indices[request_indices]
                batch_info = replace(
                    full_batch_info,
                    use_cuda_graph=False,
                    bs=len(request_indices),
                    num_segments=len(request_indices),
                    seg_lens=segment_lens,
                    seg_indptr=segment_indptr,
                    max_len=max(segment_lens_cpu),
                    weight_indices=weight_indices,
                    permutation=None,
                    expected_tokens=physical_batch.extend_num_tokens,
                    req_seg_indptr=segment_indptr,
                    req_weight_indices=weight_indices,
                    moe_lora_info=None,
                )
                batch_info = backend._add_moe_lora_info(physical_batch, batch_info)
                if batch_info.moe_lora_info is None:
                    raise RuntimeError(
                        "GLM-5.2 CP-v2 LoRA did not construct physical MoE routing metadata."
                    )
                return batch_info

            local_request_indices, local_lens_cpu = physical_segments(cp_rank)
            local_batch_info = build_batch_info(local_request_indices, local_lens_cpu)

            gathered_request_indices = []
            gathered_lens_cpu = []
            for rank in range(cp_size):
                rank_request_indices, rank_lens_cpu = physical_segments(rank)
                gathered_request_indices.extend(rank_request_indices)
                gathered_lens_cpu.extend(rank_lens_cpu)
            gathered_batch_info = build_batch_info(
                gathered_request_indices, gathered_lens_cpu
            )
            expected_gathered_tokens = sum(physical_rank_tokens)
            if gathered_batch_info.expected_tokens != expected_gathered_tokens:
                raise RuntimeError(
                    "GLM-5.2 CP-v2 LoRA gathered metadata has the wrong row count: "
                    f"metadata_rows={gathered_batch_info.expected_tokens}, "
                    f"collective_rows={expected_gathered_tokens}."
                )

            backend.batch_info = local_batch_info
            backend.context_parallel_mlp_batch_info = gathered_batch_info
            yield
        finally:
            backend.context_parallel_mlp_batch_info = None
            backend.batch_info = full_batch_info

    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # set up batch info shared by all lora modules
        self._validate_glm52_active_adapters(forward_batch)
        if getattr(self.base_hf_config, "_dsv4_flash_exact_mode", False):
            self.lora_backend._dsv4_flash_exact_batch_certified = False
        self._validate_dsv4_flash_exact_batch(forward_batch)
        bs = forward_batch.batch_size

        use_cuda_graph = (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        )
        # Eligible extend batches refresh the static prefill batch info in
        # place so captured kernels read current values at replay.
        use_prefill_cuda_graph = not use_cuda_graph and self.can_use_prefill_cuda_graph(
            forward_batch
        )

        weight_indices = [0] * len(forward_batch.lora_ids)
        lora_ranks = [0] * self.max_loras_per_batch
        scalings = [0] * self.max_loras_per_batch
        for i, uid in enumerate(forward_batch.lora_ids):
            if uid not in self.memory_pool.uid_to_buffer_id:
                continue
            weight_indices[i] = self.memory_pool.get_buffer_id(uid)
            if uid is not None:
                lora = self.loras[uid]
                # A DSV4-Flash adapter is marked all-zero only after every one
                # of the required 948 BF16 tensors has passed the exact
                # inventory validator.  Treat that certified identity adapter
                # as an inactive rank-zero slot.  This makes A1 execute the
                # literal base path instead of launching numerically pointless
                # LoRA kernels whose routed Marlin prefill reductions are not
                # bitwise stable for all route distributions.
                if getattr(
                    self.base_hf_config, "_dsv4_flash_exact_mode", False
                ) and getattr(lora, "_dsv4_flash_exact_all_zero", False):
                    if None not in self.memory_pool.uid_to_buffer_id:
                        raise RuntimeError(
                            "The exact DSV4-Flash all-zero adapter requires a "
                            "resident base-model LoRA slot."
                        )
                    weight_indices[i] = self.memory_pool.get_buffer_id(None)
                    continue
                lora_ranks[weight_indices[i]] = lora.config.r
                scalings[weight_indices[i]] = lora.scaling
        # Do in-place updates when CUDA graph is enabled and the batch forward mode
        # could use CUDA graph.
        self.lora_backend.prepare_lora_batch(
            forward_batch=forward_batch,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            use_cuda_graph=use_cuda_graph,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )
        self.lora_backend.batch_info.has_active_lora = any(
            lora_ranks[wi] > 0 for wi in weight_indices
        )
        if getattr(self.base_hf_config, "_dsv4_flash_exact_mode", False):
            self.lora_backend._dsv4_flash_exact_batch_certified = True

    def update_lora_info(self):
        """
        Update all LoRA modules to associate them with the latest memory buffer.
        """
        for layer_id, layer_modules in enumerate(self.lora_modules):
            for module_name, module in layer_modules.items():
                if (
                    isinstance(module, FusedMoEWithLoRA)
                    or getattr(module, "is_shared_fused_moe", False)
                ) and all(
                    x in self.target_modules for x in ["gate_up_proj", "down_proj"]
                ):
                    base_layer = getattr(module, "base_layer", module)
                    suffix = "_shared_moe" if base_layer.is_shared_fused_moe else "_moe"
                    gate_up_key = (
                        f"gate_up_proj{suffix}"
                        if f"gate_up_proj{suffix}" in self.memory_pool.A_buffer
                        else "gate_up_proj"
                    )
                    down_key = (
                        f"down_proj{suffix}"
                        if f"down_proj{suffix}" in self.memory_pool.A_buffer
                        else "down_proj"
                    )
                    gate_up_a = self.memory_pool.get_tensor(
                        target_module=gate_up_key,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_A,
                    )
                    gate_up_b = self.memory_pool.get_tensor(
                        target_module=gate_up_key,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_B,
                    )
                    down_a = self.memory_pool.get_tensor(
                        target_module=down_key,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_A,
                    )
                    down_b = self.memory_pool.get_tensor(
                        target_module=down_key,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_B,
                    )

                    module.set_lora_info(
                        gate_up_lora_a_weights=gate_up_a,
                        gate_up_lora_b_weights=gate_up_b,
                        down_lora_a_weights=down_a,
                        down_lora_b_weights=down_b,
                    )
                    continue

                target_module = get_target_module_name(
                    module_name, self.memory_pool.target_modules
                )

                module.set_lora_info(
                    self.memory_pool.get_tensor(
                        target_module=target_module,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_A,
                    ),
                    self.memory_pool.get_tensor(
                        target_module=target_module,
                        layer_id=layer_id,
                        lora_type=LoRAType.LORA_B,
                    ),
                )

        # Update embedding layer if present - gotta merge (refer to PR codebase)
        if self.embed_tokens_module is not None:
            self.embed_tokens_module.set_lora_info(
                self.memory_pool.get_embedding_tensor("added_tokens", LoRAType.LORA_A),
                self.memory_pool.get_embedding_tensor("embed_tokens", LoRAType.LORA_A),
                self.memory_pool.get_embedding_tensor("embed_tokens", LoRAType.LORA_B),
            )

        # Update lm_head layer if present
        if self.lm_head_module is not None:
            self.lm_head_module.set_lora_info(
                self.memory_pool.get_embedding_tensor("lm_head", LoRAType.LORA_A),
                self.memory_pool.get_embedding_tensor("lm_head", LoRAType.LORA_B),
            )

    def init_state(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
    ):
        """
        Initialize the internal (mutable) state of the LoRAManager.

        When `lora_paths` is provided and not empty, it might be used for inferring LoRA shape info such as
        the target modules and max_lora_rank.
        """

        assert lora_paths or (
            max_lora_rank is not None and target_modules is not None
        ), "When no initial --lora-paths is provided, you need to specify both --max-lora-rank and --lora-target-modules for LoRA initialization."

        self.init_lora_adapters(lora_paths)
        self.init_lora_shapes(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
        )

        if self._experts_shared_outer_override is not None:
            self.experts_shared_outer_loras = self._experts_shared_outer_override
        else:
            self.experts_shared_outer_loras = self._detect_shared_outer_loras()
        for config in self.configs.values():
            self._validate_glm52_runtime_layout(config)
        if self.experts_shared_outer_loras:
            logger.info(
                "Shared outer LoRA mode enabled: gate_up lora_A and "
                "down lora_B will be shared across experts (expert_dim=1)."
            )

        self.init_lora_modules()
        self.init_memory_pool()
        self.update_lora_info()

    def init_lora_adapters(self, lora_paths: Optional[List[LoRARef]] = None):
        # Configs of all active LoRA adapters, indexed by LoRA ID.
        self.configs: Dict[str, LoRAConfig] = {}

        # LoRA adapter weights cached in CPU memory, indexed by LoRA ID.
        self.loras: Dict[str, LoRAAdapter] = {}

        # Mapping from LoRA ID to LoRARef object.
        self.lora_refs: Dict[str, LoRARef] = {}

        # Count of pinned LoRA adapters.
        self.num_pinned_loras: int = 0

        if lora_paths:
            for lora_ref in lora_paths:
                result = self._load_lora_adapter(lora_ref)
                if not result.success:
                    raise RuntimeError(
                        f"Failed to load LoRA adapter {lora_ref.lora_name}: {result.error_message}"
                    )

    def _detect_shared_outer_loras(self) -> bool:
        """Auto-detect shared outer LoRA format from loaded adapter weights.

        MoE adapters with shared outer experts store 3D tensors where
        dim[0]=1 indicates weights shared across all experts, while
        dim[0]=num_experts indicates per-expert weights.
        Returns True if gate_up lora_A has expert_dim=1 (shared).

        All loaded adapters that expose a 3D gate_up lora_A must agree;
        mixed formats raise RuntimeError.
        """
        shared_outer: Optional[bool] = None
        for adapter_id, adapter in self.loras.items():
            for layer in adapter.layers:
                for name, weight in layer.weights.items():
                    if "gate_up_proj" not in name or "lora_A" not in name:
                        continue
                    if weight.dim() == 3:
                        is_shared = weight.shape[0] == 1
                    elif re.search(r"(?:shared_)?experts\.\d+\.", name):
                        # Per-expert adapters keep numbered 2D expert weights;
                        # they must count against the layout agreement too.
                        is_shared = False
                    else:
                        continue
                    if shared_outer is None:
                        shared_outer = is_shared
                    elif shared_outer != is_shared:
                        raise RuntimeError(
                            "Mixed shared-outer LoRA formats detected across "
                            f"loaded adapters (conflict in adapter '{adapter_id}'). "
                            "All MoE adapters must either all use shared outer "
                            "experts (expert_dim=1) or all use per-expert weights."
                        )
        return bool(shared_outer) if shared_outer is not None else False

    def init_lora_shapes(
        self,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
    ):
        """Infer LoRA target modules and max_lora_rank from loaded adapters if not provided."""

        if target_modules and target_modules == {"all"}:
            self.target_modules = auto_detect_lora_target_modules(self.base_model)
            self.target_modules.update(EMBEDDING_NAMES)
            logger.info(
                "CLI --lora-target-modules='all' resolved to %s "
                "by inspecting the base model.",
                sorted(self.target_modules),
            )
            target_modules = self.target_modules
        elif target_modules:
            self.target_modules = get_normalized_target_modules(target_modules)
        else:
            self.target_modules = set()

        for lora_id, config in self.configs.items():
            # Handle PEFT shorthand strings like "all-linear" or "all".
            if isinstance(config.target_modules, str):
                if config.target_modules in ("all-linear", "all"):
                    if target_modules is not None:
                        # CLI --lora-target-modules already provided; skip
                        # per-adapter inference for this adapter.
                        continue
                    else:
                        # Resolve by scanning the base model for all
                        # LoRA-compatible linear modules.
                        adapter_target_modules = auto_detect_lora_target_modules(
                            self.base_model
                        )
                        logger.info(
                            "LoRA adapter '%s' uses target_modules='%s'. "
                            "Resolved to %s by inspecting the base model.",
                            self.lora_refs[lora_id].lora_name,
                            config.target_modules,
                            sorted(adapter_target_modules),
                        )
                        self.target_modules.update(adapter_target_modules)
                        continue
                else:
                    raise ValueError(
                        f"SGLang does not recognize target_modules="
                        f"'{config.target_modules}'. Please use a list of module "
                        "name suffixes in the adapter's PEFT config, or explicitly "
                        "specify --lora-target-modules during server startup."
                    )

            if not isinstance(config.target_modules, list):
                raise ValueError(
                    f"SGLang currently only supports inferring LoRA target modules when a list of "
                    "suffixes is provided in `target_modules` field of PEFT config. Please explicitly "
                    "specify `--lora-target-modules` during server startup. You can specify `all` to "
                    "enable all support modules types. "
                )

            adapter_target_modules = get_normalized_target_modules(
                config.target_modules
            )

            if target_modules is not None:
                # When `--lora-target-modules` is provided, validate adapter target modules is a subset of the specified target modules.
                if not adapter_target_modules.issubset(self.target_modules):
                    unsupported_modules = adapter_target_modules - self.target_modules
                    lora_name = self.lora_refs[lora_id].lora_name
                    raise ValueError(
                        f"LoRA adapter '{lora_name}' contains target modules {sorted(unsupported_modules)} "
                        f"that are not included in the specified --lora-target-modules {sorted(self.target_modules)}. "
                        f"Please update --lora-target-modules to include all required modules: "
                        f"{sorted(self.target_modules | adapter_target_modules)}, or use 'all' to enable all supported modules."
                    )
            else:
                # Otherwise, infer target_modules from adapter configs.
                self.target_modules.update(adapter_target_modules)

        # Fusion folds wk + weights_proj into wk_weights_proj, so the modules
        # LoRA wraps are absent and an indexer-targeted adapter is silently dropped.
        indexer_targets = self.target_modules & DSA_INDEXER_LORA_NAMES
        if indexer_targets:
            from sglang.srt.layers.attention.dsa.dsa_indexer import (
                _use_dsa_indexer_fusion,
            )

            if _use_dsa_indexer_fusion:
                raise ValueError(
                    f"LoRA targets the DSA indexer ({sorted(indexer_targets)}), which is "
                    "incompatible with DSA indexer Q/K fusion. Set "
                    "SGLANG_DISABLE_DSA_INDEXER_FUSION=1 to disable fusion and use indexer LoRA."
                )

        if max_lora_rank is not None:
            self.max_lora_rank = max_lora_rank
        else:
            self.max_lora_rank = max(
                [x.r for x in self.configs.values()],
                default=0,
            )

        # Auto-infer self.lora_added_vocab_size from loaded LoRA configs
        # This happens automatically without requiring user input
        # if self.lora_added_vocab_size is None:
        if self.lora_added_tokens_size is None:
            inferred_extra_vocab_size = next(
                (
                    x.lora_added_tokens_size
                    for x in self.configs.values()
                    if x.lora_added_tokens_size > 0
                ),
                0,
            )
            if inferred_extra_vocab_size > 0:
                logger.info(
                    f"self.lora_added_tokens_size={inferred_extra_vocab_size} from LoRA adapters."
                )
            self.lora_added_tokens_size = inferred_extra_vocab_size

    def load_lora_weights(self, lora_ref: LoRARef):
        """
        Load the weights of a LoRA adapter to CPU memory and conducts post-loading validation.
        """
        lora_adapter = LoRAAdapter(
            lora_ref.lora_id,
            self.configs[lora_ref.lora_id],
            self.base_hf_config,
            self.load_config,
            self.lora_backend,
            base_model=self.base_model,
        )
        lora_adapter.initialize_weights()

        self.loras[lora_ref.lora_id] = lora_adapter

    def load_lora_weights_from_tensors(
        self, lora_ref: LoRARef, tensors: Dict[str, torch.Tensor]
    ):
        """
        Load the weights of a LoRA adapter from tensors to CPU memory.
        """
        lora_adapter = LoRAAdapter(
            lora_ref.lora_id,
            self.configs[lora_ref.lora_id],
            self.base_hf_config,
            self.load_config,
            self.lora_backend,
            base_model=self.base_model,
        )
        lora_adapter.initialize_weights_from_tensors(tensors)
        self.loras[lora_ref.lora_id] = lora_adapter

    def load_lora_adapter_from_tensors(
        self,
        lora_ref: LoRARef,
        tensors: Dict[str, torch.Tensor],
        config_dict: Dict,
        added_tokens_config: Optional[Dict] = None,
    ) -> LoRAUpdateOutput:
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self._load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    def _load_lora_adapter_from_tensors(
        self,
        lora_ref: LoRARef,
        tensors: Dict[str, torch.Tensor],
        config_dict: Dict,
        added_tokens_config: Optional[Dict] = None,
    ) -> LoRAUpdateOutput:
        """
        Load a single LoRA adapter from tensors and config dict.
        """
        assert (
            lora_ref.lora_name is not None and lora_ref.lora_path is not None
        ), "LoRARef must have both lora_name and lora_path set for loading."
        assert (
            lora_ref.lora_id not in self.loras
        ), f"LoRA adapter with ID {lora_ref.lora_id} is already loaded. This should have been verified before request is sent to the backend."

        try:
            new_adapter = LoRAConfig.from_dict(
                config_dict,
                added_tokens_config,
                base_vocab_size=self.base_hf_config.vocab_size,
            )
            self.validate_new_adapter(new_adapter, lora_ref)
            self.configs[lora_ref.lora_id] = new_adapter

            self.load_lora_weights_from_tensors(lora_ref, tensors)

            self.lora_refs[lora_ref.lora_id] = lora_ref
            self.num_pinned_loras += int(lora_ref.pinned)
        except Exception as e:
            rollback_result = self.rollback_lora_adapter(lora_ref.lora_id)
            error_message = str(e)
            if not rollback_result.success:
                error_message += (
                    "; local rollback failed: " + rollback_result.error_message
                )
            return self.create_lora_update_result(
                success=False,
                error_message=error_message,
            )

        return self.create_lora_update_result(success=True)

    def init_memory_pool(self):
        """(Re)initialize the LoRA memory pool based on the current configurations."""
        self.memory_pool = LoRAMemoryPool(
            base_hf_config=self.base_hf_config,
            max_loras_per_batch=self.max_loras_per_batch,
            dtype=self.dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            attn_tp_size=self.attn_tp_size,
            max_lora_rank=self.max_lora_rank,
            target_modules=self.target_modules,
            base_model=self.base_model,
            eviction_policy=self.eviction_policy,
            lora_added_tokens_size=self.lora_added_tokens_size,
            experts_shared_outer_loras=self.experts_shared_outer_loras,
            strict_loading=self.lora_strict_loading,
            enable_lora_overlap_loading=self.enable_lora_overlap_loading,
            lora_modules=self.lora_modules,
        )

        # Initializing memory pool with base model
        self.fetch_new_loras({None})

    def set_lora_module(self, module_name, module):
        """Wrap any module (standard or MoE) with LoRA support."""
        lora_module = get_lora_layer(module, self.lora_backend)
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def init_lora_modules(self):
        # Look-up table that essentially maps (layer_index, module_name) to the corresponding LoRA module.
        self.lora_modules: List[Dict[str, torch.nn.Module]] = [
            {} for _ in range(self.base_hf_config.num_hidden_layers)
        ]

        self.embed_tokens_module: Optional[BaseLayerWithLoRA] = None
        self.lm_head_module: Optional[BaseLayerWithLoRA] = None

        # When tie_word_embeddings=True, lm_head is the same Python object as
        # embed_tokens. PyTorch's named_modules() deduplicates by object identity,
        # so lm_head will not appear as a separate entry in the scan below,
        # preventing LoRA from wrapping it. To fix this, we create a new
        # ParallelLMHead that shares the same base weight tensor (no extra GPU
        # memory) so that named_modules() yields it as an independent module.
        if "lm_head" in self.target_modules:
            lm_head = getattr(self.base_model, "lm_head", None)
            embed_tokens = None
            for name, mod in self.base_model.named_modules():
                if name.endswith("embed_tokens"):
                    embed_tokens = mod
                    break
            if (
                lm_head is not None
                and embed_tokens is not None
                and lm_head is embed_tokens
            ):
                logger.info(
                    "lm_head is tied with embed_tokens. Creating a separate "
                    "ParallelLMHead that shares the base weight for LoRA support."
                )
                untied_lm_head = ParallelLMHead(
                    num_embeddings=embed_tokens.org_vocab_size,
                    embedding_dim=embed_tokens.embedding_dim,
                    params_dtype=embed_tokens.weight.dtype,
                    org_num_embeddings=embed_tokens.org_vocab_size,
                )
                # Share the base weight tensor — no additional GPU memory.
                untied_lm_head.weight = embed_tokens.weight
                # Replace the model attribute so named_modules() sees it
                # independently.
                self.base_model.lm_head = untied_lm_head

        from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP

        for module_name, module in self.base_model.named_modules():
            # Handle embed_tokens and lm_head before the should_apply_lora gate,
            # since VL models' should_apply_lora patterns only match language
            # model layers and would incorrectly skip these.
            # Handle embed_tokens
            if "embed_tokens" in module_name and "embed_tokens" in self.target_modules:
                if isinstance(module, VocabParallelEmbedding) and not isinstance(
                    module, BaseLayerWithLoRA
                ):
                    lora_module = self.set_lora_module(module_name, module)
                    self.embed_tokens_module = lora_module
                    continue
            # Handle lm_head
            if "lm_head" in module_name and "lm_head" in self.target_modules:
                if isinstance(module, ParallelLMHead) and not isinstance(
                    module, BaseLayerWithLoRA
                ):
                    lora_module = self.set_lora_module(module_name, module)
                    self.lm_head_module = lora_module
                    continue

            # Handle DeepSeek MLA fused projection: set the boundary
            # between q_a and kv_a output partitions so the LoRA layer
            # can apply separate B projections for each.
            if (
                "fused_qkv_a_proj_with_mqa" in self.target_modules
                and module_name.endswith("fused_qkv_a_proj_with_mqa")
            ):
                from sglang.srt.lora.layers import ReplicatedLinearWithLoRA

                layer_id = get_layer_id(module_name)
                if layer_id is None:
                    continue
                lora_module = self.set_lora_module(module_name, module)
                if isinstance(lora_module, ReplicatedLinearWithLoRA):
                    q_lora_rank = getattr(self.base_hf_config, "q_lora_rank", None) or 0
                    lora_module.first_output_dim = q_lora_rank
                self.lora_modules[layer_id][module_name] = lora_module
                continue

            # The module should be converted if it is included in target_names
            parts = module_name.split(".")
            if (
                parts[-1] in self.target_modules
                or ".".join(parts[-2:]) in self.target_modules
            ):
                layer_id = get_layer_id(module_name)
                if layer_id is None:
                    continue
                self.lora_modules[layer_id][module_name] = self.set_lora_module(
                    module_name, module
                )
                continue

            if isinstance(module, (FusedMoE, InklingBatchDenseMLP)) and all(
                x in self.target_modules for x in ["gate_up_proj", "down_proj"]
            ):
                layer_id = get_layer_id(module_name)
                if layer_id is None:
                    # FusedMoE submodules outside the decoder layer hierarchy
                    # (e.g. nested helpers under non-".layers." prefixes) have
                    # no resolvable layer id; skip them so we don't index
                    # `self.lora_modules` with `None`.
                    continue
                if isinstance(module, InklingBatchDenseMLP):
                    from sglang.srt.models.inkling_common.lora import (
                        InklingBatchDenseMLPWithLoRA,
                    )

                    module.__class__ = InklingBatchDenseMLPWithLoRA
                    module.initialize_lora(self.lora_backend)
                    lora_module = module
                else:
                    lora_module = self.set_lora_module(module_name, module)
                    lora_module.experts_shared_outer_loras = (
                        self.experts_shared_outer_loras
                    )
                    lora_module.lora_use_virtual_experts = self.lora_use_virtual_experts
                self.lora_modules[layer_id][module_name] = lora_module


def init_lora_cuda_graph_moe_buffers(
    *,
    server_args: ServerArgs,
    model: torch.nn.Module,
    lora_manager: LoRAManager,
    dtype: torch.dtype,
):
    """Phase 1 of LoRA CUDA graph init: pre-allocate MoE intermediate buffers.

    Must be called before init_memory_pool() so that memory profiling
    sees the reduced available memory and sizes KV cache correctly.
    All MoE LoRA layers share one set of buffers (managed by the
    lora_backend) since they execute sequentially during forward.

    Phase 2 (dense LoRA batch metadata) is handled later in
    CudaGraphRunner.__init__() via lora_manager.init_cuda_graph_batch_info(),
    because it needs capture-time parameters (max_bs, num_tokens_per_req)
    that are only available at that stage.
    """
    from sglang.srt.lora.layers import FusedMoEWithLoRA

    max_bs = server_args.cuda_graph_config.decode.max_bs
    max_loras = server_args.max_loras_per_batch
    for module in model.modules():
        if isinstance(module, FusedMoEWithLoRA):
            lora_manager.init_cuda_graph_moe_buffers(max_bs, max_loras, dtype, module)
            logger.info(
                f"Pre-allocated shared MoE LoRA CUDA graph buffers "
                f"(max_bs={max_bs}, max_loras={max_loras})"
            )
            break
