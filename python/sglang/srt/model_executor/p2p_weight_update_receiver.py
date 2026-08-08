from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_world_group
from sglang.srt.model_executor.p2p_weight_update import (
    annotate_p2p_locators_with_memory_handles,
    p2p_capped_block_registration_regions,
    p2p_locator_registration_regions,
    p2p_missing_locators_for_regions,
    p2p_qwen35_full_attention_hf_name,
    p2p_qwen35_linear_attn_conv1d_locators,
    p2p_qwen35_linear_attn_qkvz_locators,
    p2p_regions_from_memory_snapshot,
    p2p_register_regions,
    p2p_segment_regions_from_memory_snapshot,
)
from sglang.srt.model_loader.loader import device_loading_context

logger = logging.getLogger(__name__)


class P2PWeightUpdateReceiver:
    """Mooncake receiver retained across the upstream WeightUpdater refactor."""

    def __init__(self, get_model_runner: Callable[[], Any]):
        self._get_model_runner = get_model_runner
        self._pending_p2p_update_state: Dict[str, Dict[str, Any]] = {}
        self._cached_p2p_locators_world: Optional[List[Dict[str, Any]]] = None
        self._cached_p2p_registered_ptrs: Optional[List[int]] = None
        self._cached_p2p_session_id: Optional[str] = None
        self._p2p_pre_fingerprints: Dict[str, float] = {}
        self._p2p_pre_buf_fingerprints: Dict[str, float] = {}

    def __getattr__(self, name: str) -> Any:
        runner = self._get_model_runner()
        rank_aliases = {
            "tp_rank": "tp_rank",
            "tp_size": "tp_size",
            "dp_rank": "dp_rank",
            "moe_ep_rank": "moe_ep_rank",
        }
        if name in rank_aliases:
            return getattr(runner.ps, rank_aliases[name])
        return getattr(runner, name)

    def _clear_fp32_lm_head_cache(self) -> None:
            model = getattr(self, "model", None)
            if not isinstance(model, nn.Module):
                return

            cleared = 0
            for module in model.modules():
                clear_cache = getattr(module, "clear_fp32_lm_head_cache", None)
                if callable(clear_cache):
                    clear_cache()
                    cleared += 1
            if cleared:
                logger.debug("Cleared fp32 lm-head caches: count=%d", cleared)

    def _run_process_weights_after_loading(self):
            """Run quant_method.process_weights_after_loading() on every model module.

            Mirrors DefaultModelLoader.load_weights_and_postprocess() but invoked
            after a runtime weight update rather than initial load. Quantized
            modules (FP8, MXFP8, etc.) need this to recompute scales / repack
            tensors against the freshly-broadcast weights.
            """
            target_device = torch.device(self.device)
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

    def prepare_weights_update_p2p(
            self,
            group_name: str,
            return_tensor_map: bool = True,
            invalidate_cache: bool = False,
        ) -> Tuple[bool, str, Optional[List[Dict[str, Any]]], Optional[str]]:
            """Phase 1 of two-phase P2P (Mooncake) weight update protocol.

            Registers every named parameter's GPU memory with the local Mooncake
            TransferEngine and returns a per-HF-name locator list plus this
            rank's session id, so the trainer can issue one-sided RDMA writes
            directly into the right sub-region of each param.

            For fused params (qkv_proj, gate_up_proj, etc.) the locator list
            contains one entry *per HF logical name* (q_proj, k_proj, v_proj),
            each pointing at the correct sub-region of the fused tensor in
            memory. The trainer never sees fused names — it ships HF tensors
            and the receiver-side memory layout is opaque.

            Returns:
                (success, message, locators, session_id).
                locators is a list of dicts with keys, or None when the caller
                requested ``return_tensor_map=False`` and this receiver can reuse a
                warm cached tensor map:
                    hf_name, tp_rank, dp_rank, ep_rank, dtype,
                    full_shape (full HF shape, pre-TP),
                    slice (list of [start, stop] in full HF coords for this rank),
                    ptr (GPU address; may point inside a fused tensor),
                    nbytes.
                session_id is the Mooncake session string for this rank.
            """
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                get_mooncake_transfer_engine,
            )

            engine = get_mooncake_transfer_engine()
            if engine is None:
                return (
                    False,
                    "Mooncake TransferEngine is not initialized. Set "
                    "--enable-rdma-weight-updates so the engine is created at "
                    "startup.",
                    None,
                    None,
                )

            if group_name in self._pending_p2p_update_state:
                return (
                    False,
                    f"A P2P weight update for group '{group_name}' is already in "
                    "progress. Call complete_weights_update_p2p first.",
                    None,
                    None,
                )

            if invalidate_cache:
                logger.info(
                    f"[P2P tp_rank={self.tp_rank}] invalidating receiver warm cache "
                    "because prepare requested p2p_invalidate_cache=True"
                )
                self._invalidate_p2p_cache()

            # Pre-sync content sniff. Captures receiver state before any
            # bytes flow. Pair with the post-sync sniff in
            # complete_weights_update_p2p to see *what changed*. If a
            # canonical param's first-8 doesn't change post-sync, the
            # sync isn't writing to that address. If it changes but to
            # values that don't match the trainer's post-extract sniff,
            # the bytes are wrong on arrival.
            sniff_names = (
                "model.embed_tokens.weight",
                "model.layers.0.mlp.experts.w13_weight",
                "model.layers.0.mlp.experts.w2_weight",
                # Layer 2 is where the post-sync drift kicks in (layers 0
                # and 1 stay byte-identical post-sync, layers 2..47 diverge).
                "model.layers.2.mlp.experts.w13_weight",
                "model.layers.2.mlp.experts.w2_weight",
                "model.layers.5.mlp.experts.w13_weight",
                "lm_head.weight",
            )
            # Also fingerprint a broader set: every named_parameter and
            # every named_buffer's first-1024-element sum. Stash into
            # self._p2p_pre_fingerprints / _p2p_pre_buf_fingerprints so
            # complete_weights_update_p2p can diff PRE vs POST and tell us
            # *exactly which params/buffers changed*. This is the only way
            # to distinguish "sync wrote nothing here" from "sync wrote
            # identical bytes that were already there" — both look like
            # PRE==POST in the first-8 sniff. Buffers cover RoPE inv_freq,
            # attention scaling tables, RMSNorm running stats — anything
            # that affects forward but isn't a Parameter.
            # Fingerprint = sum over the *entire* tensor's flattened view.
            # Computes one full-pass per param via .float().sum() — costs
            # ~20-30 s for a 30B model. Off by default; set
            # XORL_P2P_FINGERPRINT_DIFF=1 to enable for sync-correctness
            # debugging.
            self._p2p_pre_fingerprints: Dict[str, float] = {}
            self._p2p_pre_buf_fingerprints: Dict[str, float] = {}
            if os.environ.get("XORL_P2P_FINGERPRINT_DIFF", "0") == "1":
                try:
                    for n, p in self.model.named_parameters():
                        d = p.data
                        if d.numel() == 0 or not d.is_floating_point():
                            continue
                        fp = float(d.float().sum().item())
                        self._p2p_pre_fingerprints[n] = fp
                    for n, b in self.model.named_buffers():
                        if b.numel() == 0 or not b.is_floating_point():
                            continue
                        fp = float(b.float().sum().item())
                        self._p2p_pre_buf_fingerprints[n] = fp
                    logger.info(
                        f"[P2P recv-fingerprint PRE tp_rank={self.tp_rank}] "
                        f"captured FULL fingerprints for {len(self._p2p_pre_fingerprints)} params, "
                        f"{len(self._p2p_pre_buf_fingerprints)} buffers"
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(f"[P2P recv-fingerprint PRE] failed: {e!r}")
            if os.environ.get("XORL_P2P_RECV_SNIFF", "0") == "1":
                try:
                    for sn in sniff_names:
                        p = None
                        for n, _p in self.model.named_parameters():
                            if n == sn:
                                p = _p
                                break
                        if p is None or p.data.numel() == 0:
                            continue
                        f8 = p.data.flatten()[:8].float().cpu().tolist()
                        dp = int(p.data.data_ptr())
                        logger.info(
                            f"[P2P recv-sniff PRE tp_rank={self.tp_rank}] {sn} "
                            f"shape={tuple(p.data.shape)} data_ptr=0x{dp:x} first8={f8}"
                        )
                except Exception as e:  # pragma: no cover
                    logger.warning(f"[P2P recv-sniff PRE] failed: {e!r}")

            # Warm-cache fast path: tensor_map + Mooncake registrations
            # built on the first sync are reused on every subsequent sync,
            # since param.data layout is frozen at model load time. The
            # cache is invalidated on model reload or post-process weights
            # (see _invalidate_p2p_cache).
            session_id = engine.get_session_id()
            if (
                self._cached_p2p_locators_world is not None
                and self._cached_p2p_session_id == session_id
            ):
                self._pending_p2p_update_state[group_name] = {"warm": True}
                maybe_locators = (
                    self._cached_p2p_locators_world if return_tensor_map else None
                )
                response_note = (
                    "with tensor_map" if return_tensor_map else "without tensor_map"
                )
                logger.info(
                    f"[P2P warm-cache hit tp_rank={self.tp_rank}] reusing tensor_map "
                    f"({len(self._cached_p2p_locators_world)} locators) and "
                    f"{len(self._cached_p2p_registered_ptrs or [])} cached "
                    f"registrations — skipped build, register, and all_gather; "
                    f"responding {response_note}."
                )
                return (
                    True,
                    f"Reused cached tensor_map ({len(self._cached_p2p_locators_world)} locators) {response_note}.",
                    maybe_locators,
                    session_id,
                )
            if (
                self._cached_p2p_locators_world is not None
                and self._cached_p2p_session_id != session_id
            ):
                # Engine restart between syncs invalidates session-stamped
                # locators. Drop the cache and rebuild.
                logger.info(
                    f"[P2P tp_rank={self.tp_rank}] Mooncake session changed "
                    f"(was {self._cached_p2p_session_id[:8] if self._cached_p2p_session_id else None}..., "
                    f"now {session_id[:8]}...); rebuilding tensor_map cache"
                )
                self._invalidate_p2p_cache()

            try:
                locators = self._build_hf_tensor_map_locators()
            except Exception as e:
                logger.error(f"Failed to build HF tensor map: {e}", exc_info=True)
                return False, f"Failed to build HF tensor map: {e}", None, None

            registration_regions, uncovered_locators = (
                self._p2p_registration_regions_for_locators(locators)
            )
            if uncovered_locators:
                sample = "; ".join(uncovered_locators[:8])
                return (
                    False,
                    f"Unable to cover {len(uncovered_locators)} P2P locators with "
                    f"registered CUDA memory regions; first missing: {sample}",
                    None,
                    None,
                )
            self._attach_p2p_memory_handles(locators, registration_regions)

            ret, ptrs = self._p2p_batch_register_regions(engine, registration_regions)
            if ret != 0:
                return (
                    False,
                    f"Mooncake receiver memory registration failed with code {ret} for "
                    f"{len(registration_regions)} memory regions.",
                    None,
                    None,
                )

            self._pending_p2p_update_state[group_name] = {"warm": False, "ptrs": ptrs}

            # Each locator carries its owner TP rank's session_id so the trainer
            # can write directly to that rank's Mooncake engine without needing
            # a separate session_id lookup. This also lets us aggregate across
            # TP ranks below.
            for loc in locators:
                loc["session_id"] = session_id

            # Locator summary: for the canonical sniffed names, log
            # (hf_name, ptr, nbytes, expert_idx, slice). Cross-reference this
            # against the receiver-side data_ptr in the PRE/POST sniff —
            # if loc["ptr"] doesn't fall in [param.data_ptr,
            # param.data_ptr + param.nbytes), the locator is pointing at
            # the wrong address and the sync writes to dead memory.
            if os.environ.get("XORL_P2P_RECV_SNIFF", "0") == "1":
                try:
                    sniff_targets = {
                        "model.embed_tokens.weight",
                        "model.layers.0.mlp.experts.0.gate_proj.weight",
                        "model.layers.0.mlp.experts.0.down_proj.weight",
                        "model.layers.0.mlp.experts.gate_up_proj",
                        "model.layers.0.mlp.experts.down_proj",
                        "model.layers.2.mlp.experts.0.gate_proj.weight",
                        "model.layers.2.mlp.experts.0.down_proj.weight",
                        "model.layers.2.mlp.experts.gate_up_proj",
                        "model.layers.2.mlp.experts.down_proj",
                        "model.layers.5.mlp.experts.0.gate_proj.weight",
                        "model.layers.5.mlp.experts.gate_up_proj",
                        "lm_head.weight",
                    }
                    for loc in locators:
                        hn = loc.get("hf_name", "")
                        if hn in sniff_targets:
                            logger.info(
                                f"[P2P recv-locator tp_rank={self.tp_rank}] "
                                f"hf_name={hn} "
                                f"ptr=0x{int(loc.get('ptr', 0)):x} "
                                f"nbytes={loc.get('nbytes')} "
                                f"expert_idx={loc.get('expert_idx', '-')} "
                                f"slice={loc.get('slice', '-')} "
                                f"full_shape={loc.get('full_shape', '-')}"
                            )
                except Exception as e:  # pragma: no cover
                    logger.warning(f"[P2P recv-locator] summary failed: {e!r}")

            # Each rank produces locators only for the slice of the model
            # that physically lives on it. The receiver mesh can span TP,
            # EP, DP, and PP, so we all-gather across the *full inference
            # world group* — TP-only would miss EP shards (per-rank expert
            # ranges), and EP-only would miss TP shards (per-rank
            # intermediate columns). The caller (TokenizerManager) only
            # consumes one scheduler response; whichever rank's response
            # wins must already carry the complete tensor map.
            world_group = get_world_group()
            world_size = world_group.world_size
            if world_size > 1:
                try:
                    gathered: List[Optional[List[Dict[str, Any]]]] = [None] * world_size
                    torch.distributed.all_gather_object(
                        gathered, locators, group=world_group.cpu_group
                    )
                    merged: List[Dict[str, Any]] = []
                    for rank_locators in gathered:
                        if rank_locators:
                            merged.extend(rank_locators)
                    locators = merged
                except Exception as e:
                    logger.error(
                        f"[P2P] all_gather_object across world group failed: {e}. "
                        "Falling back to local-only locators."
                    )

            # Populate the warm cache so subsequent prepares short-circuit
            # the build + register + all_gather entirely. Stays alive until
            # _invalidate_p2p_cache (model reload / post-process weights)
            # or process exit.
            self._cached_p2p_locators_world = locators
            self._cached_p2p_registered_ptrs = ptrs
            self._cached_p2p_session_id = session_id

            return (
                True,
                f"Registered {len(ptrs)} memory regions for P2P update.",
                locators,
                session_id,
            )

    def _invalidate_p2p_cache(self) -> None:
            """Clear the cached P2P tensor_map and deregister any cached
            Mooncake regions.

            Call this any time the set of named parameters or their
            ``data_ptr()`` could change: model reload, post-process weights
            that may quantize/reallocate, or explicit topology changes.
            """
            ptrs = self._cached_p2p_registered_ptrs or []
            if ptrs:
                try:
                    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                        get_mooncake_transfer_engine,
                    )

                    engine = get_mooncake_transfer_engine()
                    if engine is not None:
                        engine.batch_deregister(ptrs)
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        f"[P2P cache invalidate tp_rank={self.tp_rank}] "
                        f"batch_deregister failed: {e!r}"
                    )
            self._cached_p2p_locators_world = None
            self._cached_p2p_registered_ptrs = None
            self._cached_p2p_session_id = None

    def _p2p_batch_register_regions(
            self, engine: Any, regions: List[Tuple[int, int]]
        ) -> Tuple[int, List[int]]:
            chunk_size = max(
                1,
                int(os.environ.get("XORL_P2P_MOONCAKE_REGISTER_CHUNK", "4096")),
            )
            strict = os.environ.get("XORL_P2P_RECEIVER_STRICT_REGISTER", "1") != "0"
            location = self._p2p_receiver_memory_location()
            cuda_device = self._p2p_receiver_cuda_device_index()
            logger.info(
                f"[P2P] receiver registering {len(regions)} memory regions "
                f"strict={strict} chunk_size={chunk_size} "
                f"location={location or '<auto>'} cuda_device={cuda_device}"
            )
            try:
                if cuda_device is not None:
                    with torch.cuda.device(cuda_device):
                        return p2p_register_regions(
                            engine,
                            regions,
                            chunk_size=chunk_size,
                            strict=strict,
                            location=location,
                        )
                return p2p_register_regions(
                    engine,
                    regions,
                    chunk_size=chunk_size,
                    strict=strict,
                    location=location,
                )
            except Exception as e:
                logger.warning(
                    f"[P2P] receiver registration raised while strict={strict}: {e!r}"
                )
                return -1, []

    def _p2p_receiver_memory_location(self) -> Optional[str]:
            override = os.environ.get("XORL_P2P_RECEIVER_REGISTER_LOCATION")
            if override is not None:
                override = override.strip()
                return override or None
            if self.device == "cuda":
                try:
                    return f"cuda:{int(self.gpu_id)}"
                except (TypeError, ValueError):
                    return None
            return None

    def _p2p_receiver_cuda_device_index(self) -> Optional[int]:
            if self.device != "cuda" or not torch.cuda.is_available():
                return None
            try:
                return int(self.gpu_id)
            except (TypeError, ValueError):
                return None

    def _p2p_registration_regions_for_locators(
            self, locators: List[Dict[str, Any]]
        ) -> Tuple[List[Tuple[int, int]], List[str]]:
            mode = os.environ.get("XORL_P2P_RECEIVER_REGISTER_MODE", "allocator").lower()
            if mode in ("locator", "exact", "locator_exact"):
                regions = p2p_locator_registration_regions(locators)
                logger.info(
                    f"[P2P] receiver register mode={mode}: "
                    f"{len(regions)} exact locator regions"
                )
                return regions, p2p_missing_locators_for_regions(locators, regions)
            if mode not in (
                "allocator",
                "segment",
                "allocator_segment",
                "block",
                "allocator_block",
                "capped",
                "block_capped",
                "capped_block",
                "allocator_capped",
            ):
                logger.warning(
                    f"[P2P] unknown XORL_P2P_RECEIVER_REGISTER_MODE={mode!r}; "
                    "using allocator mode"
                )

            try:
                snapshot = torch.cuda.memory.memory_snapshot()
            except Exception as e:
                logger.warning(
                    f"[P2P] torch.cuda.memory.memory_snapshot failed: {e}; "
                    "falling back to raw parameter ranges for registration."
                )
                regions = self._p2p_parameter_memory_regions()
                return regions, p2p_missing_locators_for_regions(locators, regions)

            if mode in ("capped", "block_capped", "capped_block", "allocator_capped"):
                # Register only physically mapped ('active_allocated') blocks,
                # coalesced into contiguous runs capped at
                # XORL_P2P_MOONCAKE_MAX_REGION_BYTES. Avoids the -202 that
                # whole-segment registration hits when a segment's reserved
                # extent (total_size) runs past mapped memory (ibv_reg_mr EFAULT
                # "Bad address"), while collapsing the tens-of-thousands of
                # per-tensor regions of exact/block mode down to a few hundred so
                # the strict (fail-hard) serial registration path stays fast.
                max_region_bytes = max(
                    1,
                    int(
                        os.environ.get(
                            "XORL_P2P_MOONCAKE_MAX_REGION_BYTES",
                            str(512 * 1024 * 1024),
                        )
                    ),
                )
                regions, missing = p2p_capped_block_registration_regions(
                    locators, snapshot, max_region_bytes=max_region_bytes
                )
                region_label = f"capped(<= {max_region_bytes}B) mapped-block"
            elif mode in ("block", "allocator_block"):
                regions, missing = p2p_regions_from_memory_snapshot(locators, snapshot)
                region_label = "CUDA allocator block"
            else:
                regions, missing = p2p_segment_regions_from_memory_snapshot(
                    locators, snapshot
                )
                region_label = "CUDA allocator segment"
            logger.info(
                f"[P2P] receiver register mode={mode}: "
                f"{len(regions)} {region_label} regions"
            )
            if not regions and locators:
                logger.warning(
                    "[P2P] memory_snapshot returned no allocator regions covering "
                    "weight locators; falling back to raw parameter ranges."
                )
                regions = self._p2p_parameter_memory_regions()
                return regions, p2p_missing_locators_for_regions(locators, regions)
            return regions, missing

    def _p2p_parameter_memory_regions(self) -> List[Tuple[int, int]]:
            regions: List[Tuple[int, int]] = []
            for _, param in self.model.named_parameters():
                data = param.data
                nbytes = data.numel() * data.element_size()
                if nbytes <= 0:
                    continue
                base = int(data.data_ptr())
                regions.append((base, base + nbytes))
            return sorted(set(regions))

    def _build_hf_tensor_map_locators(self) -> List[Dict[str, Any]]:
            """Walk the model and emit one HF-keyed locator per logical HF name.

            Handles the standard non-MoE decoder cases:

            * QKVParallelLinear -> q_proj / k_proj / v_proj entries
            * MergedColumnParallelLinear (e.g. gate_up_proj) -> per-shard entries
            * ColumnParallelLinear (standalone) -> single entry, output_dim shard
            * RowParallelLinear (standalone) -> single entry, input_dim shard
            * VocabParallelEmbedding -> single entry, vocab-dim shard
            * Anything else with sharding attrs -> generic output_dim/input_dim
            * Fully replicated params (norms, biases) -> full == sharded

            MoE / linear-attention quirks fall through to the replicated path
            with a one-line warning per skipped fused module.
            """
            from sglang.srt.layers.linear import (
                ColumnParallelLinear,
                MergedColumnParallelLinear,
                MergedColumnParallelRepeatedLinear,
                QKVParallelLinear,
                ReplicatedLinear,
                RowParallelLinear,
            )
            from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

            # FusedMoE may not be importable in some inference-only configs;
            # leave it None and the per-expert branch will simply not fire.
            try:
                from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
            except Exception:  # pragma: no cover
                FusedMoE = None  # type: ignore[assignment]

            ep_rank = getattr(self, "moe_ep_rank", -1)
            dp_rank = (
                getattr(self, "dp_rank", 0) if self.server_args.enable_dp_attention else 0
            )
            tp_rank = self.tp_rank
            tp_size = self.tp_size
            layers_block_type = getattr(
                getattr(self.model, "config", None), "layers_block_type", None
            )

            def _module_tp(module: Any) -> Tuple[int, int]:
                # Modules built on a sub-TP group carry their own rank/size
                # (attention/GDN under DP attention run attn_tp=1 with FULL local
                # tensors). Slicing with the runner-global tp_rank emits
                # out-of-range slices on every rank > 0 for those modules.
                rank = getattr(module, "tp_rank", None)
                size = getattr(module, "tp_size", None)
                if rank is None:
                    rank = getattr(module, "attn_tp_rank", None)
                if size is None:
                    size = getattr(module, "attn_tp_size", None)
                return (
                    tp_rank if rank is None else int(rank),
                    tp_size if size is None else int(size),
                )

            param_group_tp: Dict[str, Tuple[int, int]] = {}
            out: List[Dict[str, Any]] = []
            consumed_param_names: set = set()

            def _hf_name(name: str) -> str:
                return p2p_qwen35_full_attention_hf_name(name, layers_block_type)

            def _ceil_div(value: int, divisor: int) -> int:
                return (int(value) + int(divisor) - 1) // int(divisor)

            def _block_scale_info(module: Any):
                scale = getattr(module, "weight_scale_inv", None)
                if scale is None or getattr(scale, "data", None) is None:
                    return None
                scale_data = scale.data
                if scale_data.ndim != 2:
                    return None
                quant_method = getattr(module, "quant_method", None)
                quant_config = getattr(quant_method, "quant_config", None)
                weight_block_size = getattr(quant_config, "weight_block_size", None)
                if not weight_block_size:
                    return None
                return scale_data, int(weight_block_size[0]), int(weight_block_size[1])

            def _append_block_scale_locator(
                module: Any,
                hf_weight_name: str,
                full_shape: List[int],
                slc: List[List[int]],
                *,
                local_row_block_offset: int = 0,
                local_col_block_offset: int = 0,
            ) -> bool:
                info = _block_scale_info(module)
                if info is None:
                    return False
                scale_data, block_n, block_k = info
                row_start, row_end = int(slc[0][0]), int(slc[0][1])
                col_start, col_end = int(slc[1][0]), int(slc[1][1])
                row_block_start = row_start // block_n
                row_block_end = _ceil_div(row_end, block_n)
                col_block_start = col_start // block_k
                col_block_end = _ceil_div(col_end, block_k)
                local_rows = row_block_end - row_block_start
                local_cols = col_block_end - col_block_start
                if (
                    local_rows <= 0
                    or local_cols <= 0
                    or local_row_block_offset + local_rows > scale_data.shape[0]
                    or local_col_block_offset + local_cols > scale_data.shape[1]
                ):
                    logger.warning(
                        f"[P2P tensor_map] skipping FP8 scale locator for {hf_weight_name!r}: "
                        f"scale_shape={tuple(scale_data.shape)}, local_offset="
                        f"({local_row_block_offset}, {local_col_block_offset}), "
                        f"local_blocks=({local_rows}, {local_cols})"
                    )
                    return False
                itemsize = scale_data.element_size()
                ptr = (
                    int(scale_data.data_ptr())
                    + (
                        local_row_block_offset * scale_data.stride(0)
                        + local_col_block_offset * scale_data.stride(1)
                    )
                    * itemsize
                )
                out.append(
                    {
                        "hf_name": hf_weight_name.replace(".weight", ".weight_scale_inv"),
                        "tp_rank": tp_rank,
                        "dp_rank": dp_rank,
                        "ep_rank": ep_rank,
                        "dtype": str(scale_data.dtype).removeprefix("torch."),
                        "full_shape": [
                            _ceil_div(int(full_shape[0]), block_n),
                            _ceil_div(int(full_shape[1]), block_k),
                        ],
                        "slice": [
                            [row_block_start, row_block_end],
                            [col_block_start, col_block_end],
                        ],
                        "ptr": ptr,
                        "nbytes": local_rows * local_cols * itemsize,
                    }
                )
                return True

            for module_name, module in self.model.named_modules():
                mod_tp_rank, mod_tp_size = _module_tp(module)
                param_group_tp[module_name] = (mod_tp_rank, mod_tp_size)
                if FusedMoE is not None and isinstance(module, FusedMoE):
                    # FusedMoE: w13_weight is [num_local_experts, 2*I_p, H]
                    # (gate concatenated with up along output dim) and
                    # w2_weight is [num_local_experts, H, I_p] where
                    # I_p = intermediate_size_per_partition = I / moe_tp_size.
                    # Per-expert HF entries: ".experts.{global_idx}.{gate,up,
                    # down}_proj.weight". The fused-MoE state machine uses
                    # *local* expert indices internally; we emit the global
                    # idx so the trainer's tensor map looks like a vanilla
                    # HF state dict.
                    self._emit_fused_moe_locators(
                        module=module,
                        module_name=module_name,
                        out=out,
                        consumed_param_names=consumed_param_names,
                        tp_rank=tp_rank,
                        dp_rank=dp_rank,
                    )
                    continue

                if isinstance(module, QKVParallelLinear):
                    weight = module.weight
                    qkv_ptr = int(weight.data.data_ptr())
                    itemsize = weight.data.element_size()
                    hidden = module.hidden_size
                    # Local row counts (already TP-divided)
                    q_rows = module.q_proj_shard_size
                    k_rows = module.kv_proj_shard_size
                    v_rows = module.v_proj_shard_size

                    # Full HF dimensions per name (pre-TP).
                    q_full_rows = module.total_num_heads * module.head_size
                    kv_full_rows = module.total_num_kv_heads * module.head_size
                    v_full_rows = module.total_num_kv_heads * module.v_head_size

                    qkv_prefix = f"{module_name}.weight"
                    consumed_param_names.add(qkv_prefix)
                    scale_prefix = f"{module_name}.weight_scale_inv"
                    scale_info = _block_scale_info(module)
                    if scale_info is not None:
                        consumed_param_names.add(scale_prefix)

                    offset_rows = 0
                    offset_row_blocks = 0
                    for sub_name, full_rows, local_rows in (
                        ("q_proj", q_full_rows, q_rows),
                        ("k_proj", kv_full_rows, k_rows),
                        ("v_proj", v_full_rows, v_rows),
                    ):
                        # QKVParallelLinear replicates K/V heads when TP exceeds
                        # the number of KV heads. Match its weight_loader shard
                        # selection so duplicated K/V ranks point back to the
                        # same HF slice instead of emitting out-of-range slices.
                        if sub_name == "q_proj":
                            shard_id = mod_tp_rank
                        else:
                            shard_id = mod_tp_rank // module.num_kv_head_replicas
                        hf_name = _hf_name(self._derive_hf_name(module_name, sub_name))
                        slc = [
                            [shard_id * local_rows, shard_id * local_rows + local_rows],
                            [0, hidden],
                        ]
                        out.append(
                            {
                                "hf_name": hf_name,
                                "tp_rank": tp_rank,
                                "dp_rank": dp_rank,
                                "ep_rank": ep_rank,
                                "dtype": str(weight.data.dtype).removeprefix("torch."),
                                "full_shape": [full_rows, hidden],
                                "slice": slc,
                                "ptr": qkv_ptr + offset_rows * hidden * itemsize,
                                "nbytes": local_rows * hidden * itemsize,
                            }
                        )
                        _append_block_scale_locator(
                            module,
                            hf_name,
                            [full_rows, hidden],
                            slc,
                            local_row_block_offset=offset_row_blocks,
                        )
                        offset_rows += local_rows
                        if scale_info is not None:
                            offset_row_blocks += _ceil_div(local_rows, scale_info[1])
                    continue

                if isinstance(module, MergedColumnParallelLinear) and not isinstance(
                    module, QKVParallelLinear
                ):
                    weight = module.weight
                    base_ptr = int(weight.data.data_ptr())
                    itemsize = weight.data.element_size()
                    in_features = module.input_size
                    output_sizes = module.output_sizes  # full per-shard sizes (pre-TP)

                    if (
                        module_name.endswith(".linear_attn.in_proj_qkvz")
                        and len(output_sizes) == 4
                    ):
                        consumed_param_names.add(f"{module_name}.weight")
                        scale_info = _block_scale_info(module)
                        if scale_info is not None:
                            consumed_param_names.add(f"{module_name}.weight_scale_inv")

                        try:
                            locators = p2p_qwen35_linear_attn_qkvz_locators(
                                module_name=module_name,
                                output_sizes=list(output_sizes),
                                input_size=in_features,
                                tp_rank=mod_tp_rank,
                                tp_size=mod_tp_size,
                                base_ptr=base_ptr,
                                itemsize=itemsize,
                                dtype=str(weight.data.dtype).removeprefix("torch."),
                                dp_rank=dp_rank,
                                ep_rank=ep_rank,
                            )
                        except ValueError as exc:
                            logger.warning(
                                f"[P2P tensor_map] skipping Qwen3.5 qkvz locator "
                                f"for {module_name!r}: {exc}"
                            )
                            continue

                        offset_row_blocks = 0
                        for loc in locators:
                            local_rows = int(loc.pop("_local_rows"))
                            out.append(loc)
                            _append_block_scale_locator(
                                module,
                                loc["hf_name"],
                                loc["full_shape"],
                                loc["slice"],
                                local_row_block_offset=offset_row_blocks,
                            )
                            if scale_info is not None:
                                offset_row_blocks += _ceil_div(local_rows, scale_info[1])
                        continue

                    # Default sub-name mapping covers the common case (gate_up_proj).
                    sub_names = self._guess_merged_subnames(module_name, len(output_sizes))
                    consumed_param_names.add(f"{module_name}.weight")
                    scale_info = _block_scale_info(module)
                    if scale_info is not None:
                        consumed_param_names.add(f"{module_name}.weight_scale_inv")

                    offset_rows_local = 0
                    offset_row_blocks = 0
                    for sub_name, full_size in zip(sub_names, output_sizes):
                        local_size = full_size // mod_tp_size
                        hf_name = _hf_name(self._derive_hf_name(module_name, sub_name))
                        slc = [
                            [
                                mod_tp_rank * local_size,
                                mod_tp_rank * local_size + local_size,
                            ],
                            [0, in_features],
                        ]
                        out.append(
                            {
                                "hf_name": hf_name,
                                "tp_rank": tp_rank,
                                "dp_rank": dp_rank,
                                "ep_rank": ep_rank,
                                "dtype": str(weight.data.dtype).removeprefix("torch."),
                                "full_shape": [full_size, in_features],
                                "slice": slc,
                                "ptr": base_ptr
                                + offset_rows_local * in_features * itemsize,
                                "nbytes": local_size * in_features * itemsize,
                            }
                        )
                        _append_block_scale_locator(
                            module,
                            hf_name,
                            [full_size, in_features],
                            slc,
                            local_row_block_offset=offset_row_blocks,
                        )
                        offset_rows_local += local_size
                        if scale_info is not None:
                            offset_row_blocks += _ceil_div(local_size, scale_info[1])
                    continue

                if isinstance(module, ColumnParallelLinear):
                    weight = module.weight
                    param_name = f"{module_name}.weight"
                    hf_name = _hf_name(param_name)
                    consumed_param_names.add(param_name)
                    if _block_scale_info(module) is not None:
                        consumed_param_names.add(f"{module_name}.weight_scale_inv")
                    local_out = module.output_size_per_partition
                    full_out = module.output_size
                    in_features = module.input_size

                    if module_name.endswith(".linear_attn.conv1d"):
                        config = getattr(self.model, "config", None)
                        linear_key_dim = int(
                            getattr(config, "linear_num_key_heads", 0) or 0
                        ) * int(getattr(config, "linear_key_head_dim", 0) or 0)
                        linear_value_dim = int(
                            getattr(config, "linear_num_value_heads", 0) or 0
                        ) * int(getattr(config, "linear_value_head_dim", 0) or 0)
                        output_sizes = [linear_key_dim, linear_key_dim, linear_value_dim]
                        if sum(output_sizes) != full_out:
                            logger.warning(
                                f"[P2P tensor_map] skipping Qwen3.5 conv1d locator "
                                f"for {module_name!r}: derived output_sizes={output_sizes} "
                                f"do not sum to module output_size={full_out}"
                            )
                            continue

                        try:
                            locators = p2p_qwen35_linear_attn_conv1d_locators(
                                module_name=module_name,
                                output_sizes=output_sizes,
                                input_size=in_features,
                                tp_rank=mod_tp_rank,
                                tp_size=mod_tp_size,
                                base_ptr=int(weight.data.data_ptr()),
                                itemsize=weight.data.element_size(),
                                dtype=str(weight.data.dtype).removeprefix("torch."),
                                dp_rank=dp_rank,
                                ep_rank=ep_rank,
                            )
                        except ValueError as exc:
                            logger.warning(
                                f"[P2P tensor_map] skipping Qwen3.5 conv1d locator "
                                f"for {module_name!r}: {exc}"
                            )
                            continue

                        for loc in locators:
                            loc.pop("_local_rows")
                            out.append(loc)
                        continue

                    slc = [
                        [mod_tp_rank * local_out, mod_tp_rank * local_out + local_out],
                        [0, in_features],
                    ]
                    out.append(
                        {
                            "hf_name": hf_name,
                            "tp_rank": tp_rank,
                            "dp_rank": dp_rank,
                            "ep_rank": ep_rank,
                            "dtype": str(weight.data.dtype).removeprefix("torch."),
                            "full_shape": [full_out, in_features],
                            "slice": slc,
                            "ptr": int(weight.data.data_ptr()),
                            "nbytes": weight.data.numel() * weight.data.element_size(),
                        }
                    )
                    _append_block_scale_locator(
                        module,
                        hf_name,
                        [full_out, in_features],
                        slc,
                    )
                    continue

                if isinstance(module, MergedColumnParallelRepeatedLinear):
                    # Mixed layer: first ``num_column_parallel`` outputs are
                    # column-parallel (TP-sharded along output dim), the rest
                    # are replicated. Each sub-output has its own HF name
                    # (q/k/v/beta/f_a/g_a-style for Kimi linear-attn fuse).
                    weight = module.weight
                    base_ptr = int(weight.data.data_ptr())
                    itemsize = weight.data.element_size()
                    in_features = module.input_size
                    num_cp = int(module.num_column_parallel)
                    partitions = list(module.output_partition_sizes)
                    consumed_param_names.add(f"{module_name}.weight")
                    scale_info = _block_scale_info(module)
                    if scale_info is not None:
                        consumed_param_names.add(f"{module_name}.weight_scale_inv")

                    sub_names = self._guess_merged_subnames(module_name, len(partitions))
                    offset_rows_local = 0
                    offset_row_blocks = 0
                    for i, (sub_name, local_rows) in enumerate(zip(sub_names, partitions)):
                        hf_name = _hf_name(self._derive_hf_name(module_name, sub_name))
                        if i < num_cp:
                            full_rows = local_rows * mod_tp_size
                            slc = [
                                [
                                    mod_tp_rank * local_rows,
                                    mod_tp_rank * local_rows + local_rows,
                                ],
                                [0, in_features],
                            ]
                        else:
                            full_rows = local_rows
                            slc = [[0, full_rows], [0, in_features]]
                        out.append(
                            {
                                "hf_name": hf_name,
                                "tp_rank": tp_rank,
                                "dp_rank": dp_rank,
                                "ep_rank": ep_rank,
                                "dtype": str(weight.data.dtype).removeprefix("torch."),
                                "full_shape": [full_rows, in_features],
                                "slice": slc,
                                "ptr": base_ptr
                                + offset_rows_local * in_features * itemsize,
                                "nbytes": local_rows * in_features * itemsize,
                            }
                        )
                        _append_block_scale_locator(
                            module,
                            hf_name,
                            [full_rows, in_features],
                            slc,
                            local_row_block_offset=offset_row_blocks,
                        )
                        offset_rows_local += local_rows
                        if scale_info is not None:
                            offset_row_blocks += _ceil_div(local_rows, scale_info[1])
                    continue

                if isinstance(module, ReplicatedLinear):
                    # Replicated weights are full-size on every TP rank; the
                    # fall-through path below would mis-multiply by tp_size
                    # because UnquantizedLinearMethod.create_weights stamps
                    # ``output_dim`` on every linear weight regardless of
                    # whether it's actually TP-sharded.
                    weight = module.weight
                    param_name = f"{module_name}.weight"
                    hf_name = _hf_name(param_name)
                    consumed_param_names.add(param_name)
                    if _block_scale_info(module) is not None:
                        consumed_param_names.add(f"{module_name}.weight_scale_inv")
                    shape = list(weight.data.shape)
                    slc = [[0, s] for s in shape]
                    out.append(
                        {
                            "hf_name": hf_name,
                            "tp_rank": tp_rank,
                            "dp_rank": dp_rank,
                            "ep_rank": ep_rank,
                            "dtype": str(weight.data.dtype).removeprefix("torch."),
                            "full_shape": shape,
                            "slice": slc,
                            "ptr": int(weight.data.data_ptr()),
                            "nbytes": weight.data.numel() * weight.data.element_size(),
                        }
                    )
                    _append_block_scale_locator(module, hf_name, shape, slc)
                    if getattr(module, "bias", None) is not None:
                        bias = module.bias
                        param_name = f"{module_name}.bias"
                        bias_name = _hf_name(param_name)
                        consumed_param_names.add(param_name)
                        bshape = list(bias.data.shape)
                        out.append(
                            {
                                "hf_name": bias_name,
                                "tp_rank": tp_rank,
                                "dp_rank": dp_rank,
                                "ep_rank": ep_rank,
                                "dtype": str(bias.data.dtype).removeprefix("torch."),
                                "full_shape": bshape,
                                "slice": [[0, s] for s in bshape],
                                "ptr": int(bias.data.data_ptr()),
                                "nbytes": bias.data.numel() * bias.data.element_size(),
                            }
                        )
                    continue

                if isinstance(module, RowParallelLinear):
                    weight = module.weight
                    param_name = f"{module_name}.weight"
                    hf_name = _hf_name(param_name)
                    consumed_param_names.add(param_name)
                    if _block_scale_info(module) is not None:
                        consumed_param_names.add(f"{module_name}.weight_scale_inv")
                    local_in = module.input_size_per_partition
                    full_in = module.input_size
                    out_features = module.output_size
                    slc = [
                        [0, out_features],
                        [mod_tp_rank * local_in, mod_tp_rank * local_in + local_in],
                    ]
                    out.append(
                        {
                            "hf_name": hf_name,
                            "tp_rank": tp_rank,
                            "dp_rank": dp_rank,
                            "ep_rank": ep_rank,
                            "dtype": str(weight.data.dtype).removeprefix("torch."),
                            "full_shape": [out_features, full_in],
                            "slice": slc,
                            "ptr": int(weight.data.data_ptr()),
                            "nbytes": weight.data.numel() * weight.data.element_size(),
                        }
                    )
                    _append_block_scale_locator(
                        module,
                        hf_name,
                        [out_features, full_in],
                        slc,
                    )
                    continue

                if isinstance(module, VocabParallelEmbedding):
                    weight = module.weight
                    param_name = f"{module_name}.weight"
                    hf_name = _hf_name(param_name)
                    consumed_param_names.add(param_name)
                    shard = module.shard_indices
                    full_vocab = module.org_vocab_size
                    hidden = weight.data.shape[1]
                    out.append(
                        {
                            "hf_name": hf_name,
                            "tp_rank": tp_rank,
                            "dp_rank": dp_rank,
                            "ep_rank": ep_rank,
                            "dtype": str(weight.data.dtype).removeprefix("torch."),
                            "full_shape": [full_vocab, hidden],
                            "slice": [
                                [shard.org_vocab_start_index, shard.org_vocab_end_index],
                                [0, hidden],
                            ],
                            "ptr": int(weight.data.data_ptr()),
                            "nbytes": weight.data.numel() * weight.data.element_size(),
                        }
                    )
                    continue

            # Replicated / non-sharded params: emit one entry per remaining param.
            for name, param in self.model.named_parameters():
                if name in consumed_param_names:
                    continue
                data = param.data
                owner = name.rsplit(".", 1)[0] if "." in name else ""
                p_tp_rank, p_tp_size = param_group_tp.get(owner, (tp_rank, tp_size))
                output_dim = getattr(param, "output_dim", None)
                input_dim = getattr(param, "input_dim", None)
                shape = list(data.shape)
                full_shape = list(shape)
                slc: List[List[int]] = [[0, s] for s in shape]
                if output_dim is not None:
                    full_shape[output_dim] = shape[output_dim] * p_tp_size
                    slc[output_dim] = [
                        p_tp_rank * shape[output_dim],
                        p_tp_rank * shape[output_dim] + shape[output_dim],
                    ]
                elif input_dim is not None:
                    full_shape[input_dim] = shape[input_dim] * p_tp_size
                    slc[input_dim] = [
                        p_tp_rank * shape[input_dim],
                        p_tp_rank * shape[input_dim] + shape[input_dim],
                    ]
                out.append(
                    {
                        "hf_name": _hf_name(name),
                        "tp_rank": tp_rank,
                        "dp_rank": dp_rank,
                        "ep_rank": ep_rank,
                        "dtype": str(data.dtype).removeprefix("torch."),
                        "full_shape": full_shape,
                        "slice": slc,
                        "ptr": int(data.data_ptr()),
                        "nbytes": data.numel() * data.element_size(),
                    }
                )
            self._attach_p2p_memory_handles(out)
            return out

    def _attach_p2p_memory_handles(
            self,
            locators: List[Dict[str, Any]],
            regions: Optional[List[Tuple[int, int]]] = None,
        ) -> None:
            """Annotate locators with the receiver registration base address."""

            regions = regions or self._p2p_parameter_memory_regions()
            missing = annotate_p2p_locators_with_memory_handles(locators, regions)
            for item in missing:
                logger.warning(
                    f"[P2P tensor_map] locator is outside registered memory: {item}"
                )

    def _emit_fused_moe_locators(
            self,
            module: Any,
            module_name: str,
            out: List[Dict[str, Any]],
            consumed_param_names: set,
            tp_rank: int,
            dp_rank: int,
        ) -> None:
            """Emit per-(global_expert, projection) HF locators for one FusedMoE.

            Each FusedMoE holds two on-device parameters:

            * ``w13_weight`` shape ``[E_local, 2 * I_p, H]`` — gate_proj and
              up_proj concatenated along the output (intermediate) axis.
            * ``w2_weight``  shape ``[E_local, H, I_p]`` — down_proj.

            ``I_p = intermediate_size / moe_tp_size`` and
            ``E_local = num_routed_experts / moe_ep_size`` (shared experts add
            an extra slot on every rank, after the routed experts; we skip
            those here — they're not addressed by per-expert HF names in the
            standard state dict layout).
            """
            ep_rank = int(getattr(module, "moe_ep_rank", 0))
            ep_size = int(getattr(module, "moe_ep_size", 1))
            moe_tp_rank = int(getattr(module, "moe_tp_rank", tp_rank))
            moe_tp_size = int(getattr(module, "moe_tp_size", self.tp_size))
            hidden_size = int(getattr(module, "hidden_size", 0))
            i_p = int(getattr(module, "intermediate_size_per_partition", 0))
            # FusedMoE doesn't store the unsharded intermediate_size directly;
            # reconstruct it from the per-partition value. moe_tp_size==1 means
            # i_p already equals the full intermediate_size.
            intermediate_size = i_p * moe_tp_size
            num_routed = int(getattr(module, "_num_global_routed", 0))
            num_local_routed = int(
                getattr(
                    module,
                    "_num_local_routed",
                    int(getattr(module, "num_local_experts", 0)),
                )
            )
            if hidden_size == 0 or i_p == 0 or num_local_routed == 0:
                logger.warning(
                    f"[P2P tensor_map] skipping FusedMoE {module_name!r}: "
                    f"hidden_size={hidden_size}, i_p={i_p}, "
                    f"num_local_routed={num_local_routed}"
                )
                return

            w13 = getattr(module, "w13_weight", None)
            w2 = getattr(module, "w2_weight", None)
            if w13 is None or w2 is None:
                logger.warning(
                    f"[P2P tensor_map] FusedMoE {module_name!r} missing "
                    "w13_weight / w2_weight; skipping (quantized fused-MoE "
                    "layouts are model-specific and not yet covered)."
                )
                return

            w13_scale = getattr(module, "w13_weight_scale_inv", None)
            w2_scale = getattr(module, "w2_weight_scale_inv", None)
            block_scale_locators = (
                w13.element_size() < 2
                and w2.element_size() < 2
                and w13_scale is not None
                and w2_scale is not None
                and getattr(w13_scale, "data", None) is not None
                and getattr(w2_scale, "data", None) is not None
                and w13_scale.data.ndim == 3
                and w2_scale.data.ndim == 3
            )
            if w13.element_size() < 2 and not block_scale_locators:
                scale_attrs = [
                    a
                    for a in dir(module)
                    if a.startswith("w13_weight_scale") or a.startswith("w2_weight_scale")
                ]
                logger.warning(
                    f"[P2P tensor_map] FusedMoE {module_name!r} is quantized "
                    f"(w13 element_size={w13.element_size()}, scale params "
                    f"{scale_attrs}) but does not expose block scale_inv tensors; "
                    "skipping quantized FusedMoE locators."
                )
                return

            # Mark these so the replicated/fall-through pass doesn't re-emit them.
            consumed_param_names.add(f"{module_name}.w13_weight")
            consumed_param_names.add(f"{module_name}.w2_weight")
            if block_scale_locators:
                consumed_param_names.add(f"{module_name}.w13_weight_scale_inv")
                consumed_param_names.add(f"{module_name}.w2_weight_scale_inv")

            w13_data = w13.data
            w2_data = w2.data
            item13 = w13_data.element_size()
            item2 = w2_data.element_size()
            dtype13 = str(w13_data.dtype).removeprefix("torch.")
            dtype2 = str(w2_data.dtype).removeprefix("torch.")
            w13_per_expert_stride = w13_data.stride(0) * item13
            w2_per_expert_stride = w2_data.stride(0) * item2
            w13_base = int(w13_data.data_ptr())
            w2_base = int(w2_data.data_ptr())

            if block_scale_locators:
                w13_scale_data = w13_scale.data
                w2_scale_data = w2_scale.data
                item13_scale = w13_scale_data.element_size()
                item2_scale = w2_scale_data.element_size()
                dtype13_scale = str(w13_scale_data.dtype).removeprefix("torch.")
                dtype2_scale = str(w2_scale_data.dtype).removeprefix("torch.")
                w13_scale_base = int(w13_scale_data.data_ptr())
                w2_scale_base = int(w2_scale_data.data_ptr())
                w13_scale_expert_stride = w13_scale_data.stride(0) * item13_scale
                w2_scale_expert_stride = w2_scale_data.stride(0) * item2_scale
                w13_scale_rows_per_proj = int(w13_scale_data.shape[1]) // 2
                w13_scale_cols = int(w13_scale_data.shape[2])
                w2_scale_rows = int(w2_scale_data.shape[1])
                w2_scale_cols_per_tp = int(w2_scale_data.shape[2])
                gate_up_scale_full_shape = [
                    w13_scale_rows_per_proj * moe_tp_size,
                    w13_scale_cols,
                ]
                down_scale_full_shape = [
                    w2_scale_rows,
                    w2_scale_cols_per_tp * moe_tp_size,
                ]

            # Two HF naming conventions exist for MoE expert weights:
            #
            # 1. Fused 3D tensors on the parent ``experts`` module (modern
            #    HF transformers >=4.46, used by AutoModelForCausalLM):
            #       <prefix>.gate_up_proj  [E, 2*I, H]
            #       <prefix>.down_proj     [E, H, I]
            #
            # 2. Per-expert nn.Linear modules (legacy HF + xorl handler's
            #    ``_direct_ep_transfer_experts`` which ships per-expert
            #    HF-named tensors out of ``ctx["local_experts"]``):
            #       <prefix>.experts.{global_idx}.gate_proj.weight  [I, H]
            #       <prefix>.experts.{global_idx}.up_proj.weight    [I, H]
            #       <prefix>.experts.{global_idx}.down_proj.weight  [H, I]
            #
            # Both reference the same physical receiver memory; just the
            # source-tensor view differs. We emit locators for both names
            # so the trainer matches whichever shape its state dict uses.
            # SGLang's FusedMoE is registered as ``self.experts = FusedMoE``
            # of e.g. Qwen3MoeSparseMoeBlock, so module_name ends in
            # ".experts" and:
            #   - fused-format hf_name = "{module_name}.gate_up_proj"
            #   - per-expert hf_name   = "{module_name}.{idx}.gate_proj.weight"
            gate_up_proj_hf = f"{module_name}.gate_up_proj"
            down_proj_hf = f"{module_name}.down_proj"

            E_total = num_routed if num_routed else num_local_routed * ep_size

            for local_idx in range(num_local_routed):
                global_idx = ep_rank * num_local_routed + local_idx
                if num_routed and global_idx >= num_routed:
                    break  # safety: don't emit beyond the routed-expert range
                w13_expert_ptr = w13_base + local_idx * w13_per_expert_stride
                w2_expert_ptr = w2_base + local_idx * w2_per_expert_stride

                common = {
                    "tp_rank": moe_tp_rank,
                    "dp_rank": dp_rank,
                    "ep_rank": ep_rank,
                    "expert_idx": global_idx,
                }

                # ----- fused (3D source) locators -----
                # gate part of gate_up_proj: trainer's [E, 2*I, H] sliced as
                # [global_idx:global_idx+1, moe_tp_rank*i_p:(moe_tp_rank+1)*i_p, :]
                # → receiver's w13_weight[local_idx, 0:i_p, :].
                out.append(
                    {
                        "hf_name": gate_up_proj_hf,
                        **common,
                        "dtype": dtype13,
                        "full_shape": [E_total, 2 * intermediate_size, hidden_size],
                        "slice": [
                            [global_idx, global_idx + 1],
                            [moe_tp_rank * i_p, (moe_tp_rank + 1) * i_p],
                            [0, hidden_size],
                        ],
                        "ptr": w13_expert_ptr,
                        "nbytes": i_p * hidden_size * item13,
                    }
                )
                # up part of gate_up_proj: [..., I+moe_tp_rank*i_p:I+(moe_tp_rank+1)*i_p, :]
                # → receiver's w13_weight[local_idx, i_p:2*i_p, :].
                out.append(
                    {
                        "hf_name": gate_up_proj_hf,
                        **common,
                        "dtype": dtype13,
                        "full_shape": [E_total, 2 * intermediate_size, hidden_size],
                        "slice": [
                            [global_idx, global_idx + 1],
                            [
                                intermediate_size + moe_tp_rank * i_p,
                                intermediate_size + (moe_tp_rank + 1) * i_p,
                            ],
                            [0, hidden_size],
                        ],
                        "ptr": w13_expert_ptr + i_p * hidden_size * item13,
                        "nbytes": i_p * hidden_size * item13,
                    }
                )
                # down_proj: trainer's [E, H, I] sliced as
                # [global_idx:global_idx+1, :, moe_tp_rank*i_p:(moe_tp_rank+1)*i_p]
                # → receiver's w2_weight[local_idx].
                out.append(
                    {
                        "hf_name": down_proj_hf,
                        **common,
                        "dtype": dtype2,
                        "full_shape": [E_total, hidden_size, intermediate_size],
                        "slice": [
                            [global_idx, global_idx + 1],
                            [0, hidden_size],
                            [moe_tp_rank * i_p, (moe_tp_rank + 1) * i_p],
                        ],
                        "ptr": w2_expert_ptr,
                        "nbytes": hidden_size * i_p * item2,
                    }
                )

                # ----- per-expert (2D source) locators -----
                # The xorl handler's _direct_ep_transfer_experts ships each
                # expert as a separate HF-named [I, H] / [H, I] tensor (one
                # per (expert, projection) call to backend.transfer_bucket).
                expert_prefix = f"{module_name}.{global_idx}"
                out.append(
                    {
                        "hf_name": f"{expert_prefix}.gate_proj.weight",
                        **common,
                        "dtype": dtype13,
                        "full_shape": [intermediate_size, hidden_size],
                        "slice": [
                            [moe_tp_rank * i_p, (moe_tp_rank + 1) * i_p],
                            [0, hidden_size],
                        ],
                        "ptr": w13_expert_ptr,
                        "nbytes": i_p * hidden_size * item13,
                    }
                )
                out.append(
                    {
                        "hf_name": f"{expert_prefix}.up_proj.weight",
                        **common,
                        "dtype": dtype13,
                        "full_shape": [intermediate_size, hidden_size],
                        "slice": [
                            [moe_tp_rank * i_p, (moe_tp_rank + 1) * i_p],
                            [0, hidden_size],
                        ],
                        "ptr": w13_expert_ptr + i_p * hidden_size * item13,
                        "nbytes": i_p * hidden_size * item13,
                    }
                )
                out.append(
                    {
                        "hf_name": f"{expert_prefix}.down_proj.weight",
                        **common,
                        "dtype": dtype2,
                        "full_shape": [hidden_size, intermediate_size],
                        "slice": [
                            [0, hidden_size],
                            [moe_tp_rank * i_p, (moe_tp_rank + 1) * i_p],
                        ],
                        "ptr": w2_expert_ptr,
                        "nbytes": hidden_size * i_p * item2,
                    }
                )

                if block_scale_locators:
                    w13_scale_expert_ptr = (
                        w13_scale_base + local_idx * w13_scale_expert_stride
                    )
                    w2_scale_expert_ptr = w2_scale_base + local_idx * w2_scale_expert_stride
                    gate_scale_nbytes = (
                        w13_scale_rows_per_proj * w13_scale_cols * item13_scale
                    )
                    down_scale_nbytes = w2_scale_rows * w2_scale_cols_per_tp * item2_scale
                    out.append(
                        {
                            "hf_name": f"{expert_prefix}.gate_proj.weight_scale_inv",
                            **common,
                            "dtype": dtype13_scale,
                            "full_shape": gate_up_scale_full_shape,
                            "slice": [
                                [
                                    moe_tp_rank * w13_scale_rows_per_proj,
                                    (moe_tp_rank + 1) * w13_scale_rows_per_proj,
                                ],
                                [0, w13_scale_cols],
                            ],
                            "ptr": w13_scale_expert_ptr,
                            "nbytes": gate_scale_nbytes,
                        }
                    )
                    out.append(
                        {
                            "hf_name": f"{expert_prefix}.up_proj.weight_scale_inv",
                            **common,
                            "dtype": dtype13_scale,
                            "full_shape": gate_up_scale_full_shape,
                            "slice": [
                                [
                                    moe_tp_rank * w13_scale_rows_per_proj,
                                    (moe_tp_rank + 1) * w13_scale_rows_per_proj,
                                ],
                                [0, w13_scale_cols],
                            ],
                            "ptr": w13_scale_expert_ptr
                            + w13_scale_rows_per_proj
                            * w13_scale_data.stride(1)
                            * item13_scale,
                            "nbytes": gate_scale_nbytes,
                        }
                    )
                    out.append(
                        {
                            "hf_name": f"{expert_prefix}.down_proj.weight_scale_inv",
                            **common,
                            "dtype": dtype2_scale,
                            "full_shape": down_scale_full_shape,
                            "slice": [
                                [0, w2_scale_rows],
                                [
                                    moe_tp_rank * w2_scale_cols_per_tp,
                                    (moe_tp_rank + 1) * w2_scale_cols_per_tp,
                                ],
                            ],
                            "ptr": w2_scale_expert_ptr,
                            "nbytes": down_scale_nbytes,
                        }
                    )

    @staticmethod
    def _derive_hf_name(module_name: str, sub_name: str) -> str:
        """Replace the trailing fused-module name with its HF sub-name.

        e.g. ("model.layers.0.self_attn.qkv_proj", "q_proj") ->
             "model.layers.0.self_attn.q_proj.weight"
        """
        parts = module_name.rsplit(".", 1)
        prefix = parts[0] + "." if len(parts) == 2 else ""
        return f"{prefix}{sub_name}.weight"

    @staticmethod
    def _guess_merged_subnames(module_name: str, n: int) -> List[str]:
        """Best-effort HF sub-names for a MergedColumnParallelLinear.

        Covers the gate_up_proj convention used by Llama/Qwen/Mistral. For
        anything else we fall back to indexed names and leave it to the
        caller to provide a model-specific override.
        """
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf == "gate_up_proj" and n == 2:
            return ["gate_proj", "up_proj"]
        if leaf == "in_proj_ba" and n == 2:
            return ["in_proj_b", "in_proj_a"]
        if leaf == "fused_qkvbfg_a_proj" and n == 6:
            return ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"]
        if leaf == "qkv_conv1d" and n == 3:
            return ["q_conv1d", "k_conv1d", "v_conv1d"]
        return [f"{leaf}_shard{i}" for i in range(n)]

    def complete_weights_update_p2p(
            self,
            group_name: str,
            run_post_process_weights: bool = False,
            tied_weight_aliases: Optional[Dict[str, str]] = None,
        ) -> Tuple[bool, str]:
            """Phase 2 of two-phase P2P weight update protocol.

            Clears the in-progress flag and (optionally) runs quant
            post-processing. The tensor data was already written in-place
            by the trainer via Mooncake, so there is no model.load_weights()
            call here.

            Mooncake registrations are kept alive in the warm cache (see
            ``_cached_p2p_registered_ptrs``) so subsequent syncs skip the
            ~1.4 s register tax. They are released by
            ``_invalidate_p2p_cache`` when post-process weights runs or
            when explicitly invalidated.
            """
            state = self._pending_p2p_update_state.pop(group_name, None)
            if state is None:
                return (
                    False,
                    f"No P2P weight update in progress for group '{group_name}'. "
                    "Call prepare_weights_update_p2p first.",
                )

            # No per-sync deregister: registrations live in the warm cache
            # and persist across syncs. They are released by
            # _invalidate_p2p_cache (post-process weights / explicit
            # invalidation) or at engine teardown.

            if tied_weight_aliases:
                success, message = self._copy_p2p_tied_weight_aliases(tied_weight_aliases)
                if not success:
                    return False, message

            if run_post_process_weights:
                # Post-process MAY reallocate param storage (quantization that
                # swaps the data tensor) — if so the cached locator ptrs dangle and
                # we must invalidate. But for an already-quantized receiver (e.g. a
                # block-FP8 checkpoint receiving block-FP8 weights), post-process
                # typically rewrites in-place and leaves every param.data_ptr()
                # unchanged, so the warm cache stays valid. Unconditionally
                # invalidating forced a full ~34s locator rebuild + region
                # re-register on EVERY FP8 sync (the dominant cost of the 60s warm
                # sync on the 8×8 Q3.6 layout, 2026-05-28). Snapshot data_ptrs
                # around post-process and only invalidate if storage actually moved
                # — self-protecting: if anything reallocates we still invalidate.
                # Set XORL_P2P_CONDITIONAL_POSTPROCESS_INVALIDATE=0 to force the old
                # always-invalidate behavior.
                conditional = (
                    os.environ.get("XORL_P2P_CONDITIONAL_POSTPROCESS_INVALIDATE", "1") != "0"
                )
                ptrs_before = (
                    {n: p.data_ptr() for n, p in self.model.named_parameters()}
                    if conditional
                    else None
                )
                try:
                    self._run_process_weights_after_loading()
                except Exception as e:
                    error_msg = f"process_weights_after_loading failed: {e}"
                    logger.error(error_msg)
                    return False, error_msg
                if not conditional:
                    self._invalidate_p2p_cache()
                else:
                    ptrs_after = {n: p.data_ptr() for n, p in self.model.named_parameters()}
                    moved = [
                        n
                        for n in ptrs_before
                        if ptrs_before[n] != ptrs_after.get(n)
                    ]
                    added_removed = set(ptrs_before) ^ set(ptrs_after)
                    if moved or added_removed:
                        logger.info(
                            f"[P2P tp_rank={self.tp_rank}] post-process moved "
                            f"{len(moved)} param storage(s) "
                            f"(+/- {len(added_removed)} names; e.g. {moved[:3]}); "
                            f"invalidating P2P warm cache"
                        )
                        self._invalidate_p2p_cache()
                    else:
                        logger.info(
                            f"[P2P tp_rank={self.tp_rank}] post-process kept all "
                            f"{len(ptrs_after)} param storages in-place; retaining "
                            f"warm P2P cache (skipped next-sync rebuild + re-register)"
                        )

            # Receiver-side content sniff. Dumps first-8 elements of
            # canonical params right after sync completes — pair this
            # against the trainer-side sniffs (entry/post-unshard/
            # post-extract in the xorl handler). If the trainer's
            # post-extract first-8 differs from this receiver post-sync
            # first-8 for the same logical weight, the bytes were
            # corrupted in transit / at the receiver's destination
            # address. If they match, the bytes are correct on the
            # receiver and the broken /generate output points elsewhere
            # (e.g., KV cache not flushed, post-quant skipped, etc.).
            if os.environ.get("XORL_P2P_RECV_SNIFF", "0") == "1":
                try:
                    sniff_names = [
                        "model.embed_tokens.weight",
                        "model.layers.0.mlp.experts.w13_weight",
                        "model.layers.0.mlp.experts.w2_weight",
                        "model.layers.2.mlp.experts.w13_weight",
                        "model.layers.2.mlp.experts.w2_weight",
                        "model.layers.5.mlp.experts.w13_weight",
                        "lm_head.weight",
                    ]
                    for sn in sniff_names:
                        p = None
                        for n, _p in self.model.named_parameters():
                            if n == sn:
                                p = _p
                                break
                        if p is None:
                            logger.info(
                                f"[P2P recv-sniff tp_rank={self.tp_rank}] {sn}: <not found>"
                            )
                            continue
                        d = p.data
                        if d.numel() == 0:
                            logger.info(
                                f"[P2P recv-sniff tp_rank={self.tp_rank}] {sn}: numel=0"
                            )
                            continue
                        f8 = d.flatten()[:8].float().cpu().tolist()
                        dp = int(d.data_ptr())
                        logger.info(
                            f"[P2P recv-sniff tp_rank={self.tp_rank}] {sn} "
                            f"shape={tuple(d.shape)} dtype={d.dtype} "
                            f"data_ptr=0x{dp:x} first8={f8}"
                        )
                except Exception as e:  # pragma: no cover
                    logger.warning(f"[P2P recv-sniff] failed: {e!r}")

            # PRE→POST fingerprint diff. Tells us which params actually
            # had bytes written to them by the sync. Crucial because
            # PRE==POST first-8 can mean either "no write" or "write
            # produced identical bytes" — only the diff lets us
            # distinguish.
            pre = getattr(self, "_p2p_pre_fingerprints", None)
            pre_buf = getattr(self, "_p2p_pre_buf_fingerprints", None)
            if pre:
                try:
                    post: Dict[str, float] = {}
                    changed: List[Tuple[str, float, float]] = []
                    unchanged_count = 0
                    missing: List[str] = []
                    for n, p in self.model.named_parameters():
                        d = p.data
                        if d.numel() == 0 or not d.is_floating_point():
                            continue
                        fp = float(d.float().sum().item())
                        post[n] = fp
                        pre_fp = pre.get(n)
                        if pre_fp is None:
                            missing.append(n)
                            continue
                        # Tolerate ~1e-3 relative diff to absorb FP32 reduction
                        # nondeterminism over very large tensors. Tighter than
                        # this risks false positives from sum-order on H100.
                        abs_diff = abs(fp - pre_fp)
                        rel_thresh = max(1e-3, abs(pre_fp) * 1e-5)
                        if abs_diff > rel_thresh or (fp != fp) != (pre_fp != pre_fp):
                            changed.append((n, pre_fp, fp))
                        else:
                            unchanged_count += 1
                    # Buffer diff
                    buf_changed: List[Tuple[str, float, float]] = []
                    buf_unchanged = 0
                    if pre_buf:
                        for n, b in self.model.named_buffers():
                            if b.numel() == 0 or not b.is_floating_point():
                                continue
                            fp = float(b.float().sum().item())
                            pre_fp = pre_buf.get(n)
                            if pre_fp is None:
                                continue
                            abs_diff = abs(fp - pre_fp)
                            rel_thresh = max(1e-3, abs(pre_fp) * 1e-5)
                            if abs_diff > rel_thresh or (fp != fp) != (pre_fp != pre_fp):
                                buf_changed.append((n, pre_fp, fp))
                            else:
                                buf_unchanged += 1
                    logger.info(
                        f"[P2P recv-fingerprint DIFF tp_rank={self.tp_rank}] "
                        f"params: changed={len(changed)} unchanged={unchanged_count} "
                        f"missing_in_pre={len(missing)} | "
                        f"buffers: changed={len(buf_changed)} unchanged={buf_unchanged}"
                    )
                    if buf_changed:
                        logger.info(
                            f"[P2P recv-fingerprint BUFFERS-CHANGED "
                            f"tp_rank={self.tp_rank}] first 30:"
                        )
                        for n, pre_fp, post_fp in buf_changed[:30]:
                            logger.info(f"  {n}: {pre_fp:.4f} -> {post_fp:.4f}")
                    # Log the first 30 changed names + a few representative ones
                    # by category to give visibility without flooding logs.
                    if changed:
                        logger.info(
                            f"[P2P recv-fingerprint CHANGED tp_rank={self.tp_rank}] "
                            f"first 30:"
                        )
                        for n, pre_fp, post_fp in changed[:30]:
                            logger.info(f"  {n}: {pre_fp:.4f} -> {post_fp:.4f}")
                        # Also surface any of the canonical sniffed names that
                        # changed.
                        canonical = {
                            "model.embed_tokens.weight",
                            "model.layers.0.mlp.experts.w13_weight",
                            "model.layers.0.mlp.experts.w2_weight",
                            "model.layers.2.mlp.experts.w13_weight",
                            "model.layers.2.mlp.experts.w2_weight",
                            "model.layers.5.mlp.experts.w13_weight",
                            "lm_head.weight",
                        }
                        canonical_changed = [c for c in changed if c[0] in canonical]
                        if canonical_changed:
                            logger.info(
                                f"[P2P recv-fingerprint canonical-changed "
                                f"tp_rank={self.tp_rank}]:"
                            )
                            for n, pre_fp, post_fp in canonical_changed:
                                logger.info(f"  {n}: {pre_fp:.4f} -> {post_fp:.4f}")
                        canonical_unchanged = [
                            cn
                            for cn in canonical
                            if cn in pre and cn in post and abs(pre[cn] - post[cn]) <= 1e-6
                        ]
                        if canonical_unchanged:
                            logger.info(
                                f"[P2P recv-fingerprint canonical-UNCHANGED "
                                f"tp_rank={self.tp_rank}]: {canonical_unchanged}"
                            )
                except Exception as e:  # pragma: no cover
                    logger.warning(f"[P2P recv-fingerprint DIFF] failed: {e!r}")
                self._p2p_pre_fingerprints = {}
                self._p2p_pre_buf_fingerprints = {}

            # P2P writes weights in place; refresh the fp32 lm-head buffer like every other update path.
            self._clear_fp32_lm_head_cache()

            return True, "Succeeded to complete P2P weight update."

    def _resolve_weight_tensor_for_p2p_alias(self, name: str) -> torch.Tensor:
            current = self.model
            for part in name.split("."):
                if part.isdigit() and hasattr(current, "__getitem__"):
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            if isinstance(current, torch.nn.Parameter):
                return current.data
            if isinstance(current, torch.Tensor):
                return current
            raise TypeError(f"{name!r} resolved to {type(current).__name__}, not a tensor")

    def _copy_p2p_tied_weight_aliases(
            self,
            tied_weight_aliases: Dict[str, str],
        ) -> Tuple[bool, str]:
            copied = 0
            skipped = 0
            for target_name, source_name in tied_weight_aliases.items():
                try:
                    source = self._resolve_weight_tensor_for_p2p_alias(source_name)
                    target = self._resolve_weight_tensor_for_p2p_alias(target_name)
                except Exception as e:
                    return (
                        False,
                        f"P2P tied-weight alias resolution failed for {target_name} <- {source_name}: {e}",
                    )

                if tuple(source.shape) != tuple(target.shape):
                    return (
                        False,
                        f"P2P tied-weight alias shape mismatch for {target_name} <- {source_name}: "
                        f"target={tuple(target.shape)} source={tuple(source.shape)}",
                    )
                if source.data_ptr() == target.data_ptr():
                    skipped += 1
                    continue
                target.copy_(source)
                copied += 1

            if torch.cuda.is_available() and copied:
                torch.cuda.synchronize()
            logger.info(
                f"[P2P tied-weight aliases tp_rank={self.tp_rank}] copied={copied} already_aliased={skipped}"
            )
            return True, "Succeeded to copy P2P tied-weight aliases."
