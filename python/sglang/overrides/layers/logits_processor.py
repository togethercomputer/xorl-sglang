"""Override twin of ``sglang.srt.layers.logits_processor`` -- xorl exact serving (zero-srt port of PR #41).

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


def _LogitsProcessor___bi_lm_head_decode_active(
    self, logits_metadata: LogitsMetadata
) -> bool:
    if not self.use_qwen35_bi_lm_head:
        return False
    mode = logits_metadata.forward_mode
    if mode in (
        ForwardMode.DECODE,
        ForwardMode.IDLE,
        ForwardMode.EXTEND,
        ForwardMode.MIXED,
    ):
        return True
    raise ValueError(
        "The exact-contract decode lm head does not support forward mode "
        f"{mode.name}"
    )


def _LogitsProcessor___bi_lm_head_input_token_logprobs(
    self,
    pruned_states: torch.Tensor,
    input_logprob_indices: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_metadata: LogitsMetadata,
) -> torch.Tensor:
    """Score input tokens with the same pinned lm-head contract as training."""
    from sglang.srt.batch_invariant_ops import bi_lm_head_selected_logprob

    if input_logprob_indices is None:
        raise ValueError(
            "The exact-contract input-token rescore requires input-logprob indices"
        )
    hidden = pruned_states[input_logprob_indices]
    weight = self._validate_bi_lm_head(hidden, lm_head, operation="input-token lm head")
    token_ids = logits_metadata.extend_input_logprob_token_ids_gpu
    if token_ids is None or token_ids.shape[0] != hidden.shape[0]:
        raise ValueError(
            "The exact-contract input-token rescore requires one token id per "
            f"scored row, got {None if token_ids is None else token_ids.shape[0]} "
            f"for {hidden.shape[0]} rows"
        )
    logprob, _, _ = bi_lm_head_selected_logprob(
        hidden.contiguous(), weight, token_ids, temperature=None
    )
    return logprob


def _LogitsProcessor___bi_lm_head_next_token_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    logits_metadata: LogitsMetadata,
) -> torch.Tensor:
    """Build every sampled distribution with the contract's fixed GEMM."""
    from sglang.srt.batch_invariant_ops import bi_lm_head_full_logits

    weight = self._validate_bi_lm_head(
        hidden_states, lm_head, operation="decode lm head"
    )
    return bi_lm_head_full_logits(hidden_states.contiguous(), weight)


def _LogitsProcessor___validate_bi_lm_head(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    *,
    operation: str,
) -> torch.Tensor:
    if not self.use_fp32_lm_head:
        raise ValueError(
            f"The exact-contract {operation} requires --enable-fp32-lm-head"
        )
    if self.do_tensor_parallel_all_gather or self.do_tensor_parallel_all_gather_dp_attn:
        raise ValueError(
            f"The exact-contract {operation} does not support vocab/DP-parallel "
            "lm-head gathers yet"
        )
    if self.logit_scale is not None:
        raise ValueError(f"The exact-contract {operation} does not support logit_scale")
    if self.final_logit_softcapping:
        raise ValueError(
            f"The exact-contract {operation} does not support "
            "final_logit_softcapping"
        )
    if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
        raise ValueError(
            f"The exact-contract {operation} does not support a LoRA-wrapped " "lm_head"
        )
    if not hasattr(lm_head, "weight"):
        raise ValueError(
            f"The exact-contract {operation} requires a plain lm_head weight"
        )
    weight = lm_head.weight.data
    if weight.dtype != torch.bfloat16 or hidden_states.dtype != torch.bfloat16:
        raise ValueError(
            f"The exact-contract {operation} requires bf16 hidden/weight, got "
            f"{hidden_states.dtype}/{weight.dtype}"
        )
    if weight.shape[0] != self.vocab_size:
        raise ValueError(
            f"The exact-contract {operation} requires the full-vocab weight, "
            f"got {weight.shape[0]} != {self.vocab_size}"
        )
    return weight


@classmethod
def _LogitsMetadata__from_forward_batch(cls, forward_batch: ForwardBatch):
    if (
        forward_batch.forward_mode.is_extend()
        and forward_batch.return_logprob
        and not forward_batch.forward_mode.is_target_verify()
    ):
        extend_return_top_logprob = any(x > 0 for x in forward_batch.top_logprobs_nums)
        extend_token_ids_logprob = any(
            x is not None for x in forward_batch.token_ids_logprobs
        )
        extend_return_logprob = False
        extend_logprob_pruned_lens_cpu = []
        for extend_len, start_len in zip(
            forward_batch.extend_seq_lens_cpu,
            forward_batch.extend_logprob_start_lens_cpu,
        ):
            if extend_len - start_len > 0:
                extend_return_logprob = True
            extend_logprob_pruned_lens_cpu.append(extend_len - start_len)
    else:
        extend_return_logprob = extend_return_top_logprob = extend_token_ids_logprob = (
            extend_logprob_pruned_lens_cpu
        ) = False

    if forward_batch.forward_mode.is_draft_extend_v2():
        draft_extend_select_index = forward_batch.spec_info.select_index
    else:
        draft_extend_select_index = None

    return cls(
        forward_mode=forward_batch.forward_mode,
        capture_hidden_mode=forward_batch.capture_hidden_mode,
        next_token_logits_buffer=forward_batch.next_token_logits_buffer,
        extend_return_logprob=extend_return_logprob,
        extend_return_top_logprob=extend_return_top_logprob,
        extend_token_ids_logprob=extend_token_ids_logprob,
        extend_seq_lens=forward_batch.extend_seq_lens,
        extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
        extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
        top_logprobs_nums=forward_batch.top_logprobs_nums,
        token_ids_logprobs=forward_batch.token_ids_logprobs,
        extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
        is_prefill_only=forward_batch.is_prefill_only,
        global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
        dp_local_start_pos=forward_batch.dp_local_start_pos,
        dp_local_num_tokens=forward_batch.dp_local_num_tokens,
        global_dp_buffer_len=forward_batch.global_dp_buffer_len,
        global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
        global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
        dp_padding_mode=DpPaddingMode.SUM_LEN,
        dsv4_exact_logits_rows_reconstructed=forward_batch.dsv4_exact_logits_rows_reconstructed,
        dsv4_exact_logits_owner_rows=forward_batch.dsv4_exact_logits_owner_rows,
        dsv4_exact_logits_dp_rank=forward_batch.dsv4_exact_logits_dp_rank,
        mm_input_embeds=forward_batch.mm_input_embeds,
        draft_extend_select_index=draft_extend_select_index,
    )


def _LogitsProcessor____init__(
    self,
    config,
    skip_all_gather: bool = False,
    logit_scale: Optional[float] = None,
    return_full_logits: bool = False,
):
    super(LogitsProcessor, self).__init__()
    self.config = config
    self.vocab_size = config.vocab_size
    self.logit_scale = logit_scale
    self._glm52_exact_mode = is_glm52_exact_mode(get_server_args())
    self._qwen3_dense_exact_mode = is_qwen3_dense_exact_mode(get_server_args())
    self._dsv4_exact_mode = bool(getattr(config, "_dsv4_flash_exact_mode", False))
    self.use_attn_tp_group = (
        get_parallel().enable_dp_lm_head and not self._dsv4_exact_mode
    )
    self.use_fp32_lm_head = get_exec().features.enable_fp32_lm_head
    # The Qwen3.5-family exact contract must own both sides of serving's
    # probability surface: input-token rescoring and every next-token
    # sampling distribution. Merely enabling the exact trunk kernels is
    # insufficient because the stock lm-head reduction is batch-shaped.
    self.use_qwen35_bi_lm_head = is_qwen35_gdn_exact_mode(get_server_args())
    if self.use_attn_tp_group:
        self.attn_tp_size = get_parallel().attn_tp_size
        self.do_tensor_parallel_all_gather = (
            not skip_all_gather and self.attn_tp_size > 1
        )
        self.do_tensor_parallel_all_gather_dp_attn = False
    else:
        self.do_tensor_parallel_all_gather = (
            not skip_all_gather and get_parallel().tp_size > 1
        )
        self.do_tensor_parallel_all_gather_dp_attn = (
            self.do_tensor_parallel_all_gather and get_parallel().attn_dp_size != 1
        )
    self.final_logit_softcapping = getattr(self.config, "final_logit_softcapping", None)
    if self.final_logit_softcapping is not None and self.final_logit_softcapping < 0:
        self.final_logit_softcapping = None
    if self._glm52_exact_mode or getattr(self, "_qwen3_dense_exact_mode", False):
        validate_xorl_bi_logit_transforms(
            self.logit_scale,
            self.final_logit_softcapping,
        )

    self.return_full_logits = return_full_logits
    self.enable_mis = get_exec().features.enable_mis
    self.rl_on_policy_target = get_exec().deterministic.rl_on_policy_target

    self._logits_gatherer = triton_symm_mem_ag.MultimemAllGatherer(
        max_tokens=triton_symm_mem_ag.recommended_max_tokens(
            include_prefill=False, floor=128
        ),
        enabled=self.do_tensor_parallel_all_gather and not self.use_attn_tp_group,
        skip_entry_sync=True,
    )

    self.input_logprob_processor = InputLogprobProcessor()


def _LogitsProcessor___compute_lm_head(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    embedding_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if self._glm52_exact_mode or getattr(self, "_qwen3_dense_exact_mode", False):
        return xorl_bi_lm_head(
            hidden_states,
            lm_head,
            use_fp32_lm_head=self.use_fp32_lm_head,
            embedding_bias=embedding_bias,
            family="v2",
        )

    quant_method = getattr(lm_head, "quant_method", None)
    if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
        # This is a LoRA-wrapped module, use its forward method
        logits = lm_head(hidden_states)
    elif should_apply_lm_head_quant_method(lm_head, quant_method):
        logits = quant_method.apply(lm_head, hidden_states, embedding_bias)
    elif hasattr(lm_head, "weight"):
        # Normal linear layer
        if self.use_fp32_lm_head:
            logits = torch.matmul(
                hidden_states.to(torch.float32), lm_head.weight.to(torch.float32).T
            )
        elif use_intel_amx_backend(lm_head):
            logits = torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states.to(lm_head.weight.dtype),
                lm_head.weight,
                None,  # bias
                True,  # is_vnni
            )
        elif self.rl_on_policy_target is not None:
            # Due to tie-weight, we may not be able to change lm_head's weight dtype
            logits = torch.matmul(hidden_states.bfloat16(), lm_head.weight.T.bfloat16())
        else:
            logits = torch.matmul(
                hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
            )
    else:
        # GGUF models
        # TODO: use weight_packed_linear for GGUF models
        if self.use_fp32_lm_head:
            with torch.cuda.amp.autocast(enabled=False):
                logits = lm_head.quant_method.apply(
                    lm_head, hidden_states.to(torch.float32), embedding_bias
                )
        else:
            logits = lm_head.quant_method.apply(lm_head, hidden_states, embedding_bias)
    return logits


def _LogitsProcessor___copy_logits_to_buffer(
    self,
    logits: torch.Tensor,
    logits_metadata: LogitsMetadata,
    use_buffer: bool = True,
) -> torch.Tensor:
    logits_buffer = logits_metadata.next_token_logits_buffer if use_buffer else None
    if logits.shape[-1] > self.vocab_size:
        logits = logits[:, : self.vocab_size]
    # The shared logits buffer is keyed by vocab width and rows; skip it
    # when this batch has a different logits shape than the graph buffer.
    if logits_buffer is not None and tuple(logits_buffer.shape) == tuple(logits.shape):
        assert logits_buffer.dtype == torch.float
        logits_buffer.copy_(logits)
        logits = logits_buffer
    else:
        logits = logits.float()
    return logits


def _LogitsProcessor___gather_dp_attn_hidden_states(
    self, hidden_states: torch.Tensor, logits_metadata: LogitsMetadata
) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.do_tensor_parallel_all_gather_dp_attn:
        logits_metadata.compute_dp_attention_metadata()
        local_hidden_states = hidden_states
        hidden_states = logits_metadata.gathered_buffer
        if self._dsv4_exact_mode:
            from sglang.srt.layers.dsv4_ownership import (
                gather_dsv4_owner_plane_rows,
                resolve_dsv4_owner_plane,
            )

            ownership = resolve_dsv4_owner_plane()
            if not logits_metadata.dsv4_exact_logits_rows_reconstructed:
                raise RuntimeError(
                    "Exact DSV4 TP8 logits require reconstructed DP-owner rows"
                )
            if logits_metadata.dsv4_exact_logits_dp_rank != ownership.dp_rank:
                raise RuntimeError(
                    "Exact DSV4 logits ownership changed between model and head"
                )
            counts = logits_metadata.global_num_tokens_for_logprob_cpu
            segment_lengths = [] if counts is None else list(counts)
            gather_dsv4_owner_plane_rows(
                local_hidden_states,
                ownership,
                segment_lengths,
                output=hidden_states,
                group=get_parallel().tp_group,
            )
        else:
            dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)
        return hidden_states, local_hidden_states
    return hidden_states, hidden_states


def _LogitsProcessor___get_pruned_states(
    self,
    hidden_states: torch.Tensor,
    hidden_states_before_norm: Optional[torch.Tensor],
    aux_hidden_states: Optional[AuxHiddenStates],
    logits_metadata: LogitsMetadata,
):
    if logits_metadata.dsv4_exact_logits_rows_reconstructed:
        owner_rows = logits_metadata.dsv4_exact_logits_owner_rows
        if owner_rows is None or hidden_states.shape[0] != int(owner_rows):
            raise RuntimeError(
                "Exact DSV4 logits require the complete reconstructed DP-owner "
                "rows before pruning: "
                f"rows={hidden_states.shape[0]}, expected={owner_rows}"
            )

    pruned_states_before_norm: Optional[torch.Tensor] = None
    aux_pruned_states = None
    token_to_seq_idx = []

    if (
        logits_metadata.forward_mode.is_decode_or_idle()
        or logits_metadata.forward_mode.is_target_verify()
        or logits_metadata.forward_mode.is_draft_extend_v2()
    ):
        if logits_metadata.draft_extend_select_index is not None:
            # Only next_token_logits narrows to [bs, vocab]; the
            # FULL-capture hidden stays unpruned.
            pruned_states = hidden_states[logits_metadata.draft_extend_select_index]
        else:
            pruned_states = hidden_states
        pruned_states_before_norm = hidden_states_before_norm
        if aux_hidden_states is not None:
            aux_pruned_states = (
                aux_hidden_states
                if isinstance(aux_hidden_states, torch.Tensor)
                else [hidden for hidden in aux_hidden_states]
            )
        sample_indices = None
        input_logprob_indices = None

    elif (
        logits_metadata.forward_mode.is_extend()
        and not logits_metadata.extend_return_logprob
    ):
        # Prefill without input logprobs.
        last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
        pruned_states = hidden_states[last_index]
        if hidden_states_before_norm is not None:
            pruned_states_before_norm = hidden_states_before_norm[last_index]
        if aux_hidden_states is not None:
            aux_pruned_states = (
                aux_hidden_states[last_index]
                if isinstance(aux_hidden_states, torch.Tensor)
                else [hidden[last_index] for hidden in aux_hidden_states]
            )
        sample_indices = None
        input_logprob_indices = None
    else:
        # Prefill with input logprobs.
        # Find 4 different indices.
        # 1. pruned_states: hidden states that we want logprobs from.
        # 2. sample_indices: Indices that have sampled tokens.
        # 3. input_logprob_indices: Indices that have input logprob tokens.
        # 4. token_to_seq_idx: map each token to its sequence index
        #
        # Example
        # -------
        # Suppose a batch (flattened by sequence):
        # [t00, t01, t02, t03, t10, t11, t12, t13, t14, t20, t21, t22, t23, t24, t25]
        # extend_seq_lens_cpu           = [4, 5, 6]
        # extend_logprob_start_lens_cpu = [0, 5, 3]
        #
        # Then, the indices are:
        # pruned_states         -> [t00, t01, t02, t03, t14, t23, t24, t25]
        # sample_indices        -> [3, 4, 7]
        # input_logprob_indices -> [0, 1, 2, 3, 5, 6, 7]
        # token_to_seq_idx      -> [0, 0, 0, 0, 1, 2, 2, 2]
        #
        # If chunk is enabled and chunk_size = 3, the chunks will be computed in a chunked manner:
        # [t00, t01, t02], [t03, t14, t23], [t24, t25]

        sample_index_pt = -1
        sample_indices = []
        input_logprob_indices_pt = 0
        input_logprob_indices = []
        pt, pruned_states_list, pruned_states_before_norm_list = 0, [], []
        is_packed_aux_hidden_states = isinstance(aux_hidden_states, torch.Tensor)
        aux_pruned_states_lists = None
        if aux_hidden_states is not None:
            aux_pruned_states_lists = (
                [] if is_packed_aux_hidden_states else [[] for _ in aux_hidden_states]
            )

        for idx, (extend_logprob_start_len, extend_len) in enumerate(
            zip(
                logits_metadata.extend_logprob_start_lens_cpu,
                logits_metadata.extend_seq_lens_cpu,
            )
        ):
            # It can happen in chunked prefill. We still need to sample 1 token,
            # But we don't want to include it in input logprob.
            if extend_len == extend_logprob_start_len:
                start_len = extend_logprob_start_len - 1
            else:
                start_len = extend_logprob_start_len

            # We always need at least 1 token to sample because that's required
            # by a caller.
            assert extend_len > start_len
            pruned_states_list.append(hidden_states[pt + start_len : pt + extend_len])
            if hidden_states_before_norm is not None:
                pruned_states_before_norm_list.append(
                    hidden_states_before_norm[pt + start_len : pt + extend_len]
                )
            if aux_pruned_states_lists is not None:
                if is_packed_aux_hidden_states:
                    aux_pruned_states_lists.append(
                        aux_hidden_states[pt + start_len : pt + extend_len]
                    )
                else:
                    for j, hidden in enumerate(aux_hidden_states):
                        aux_pruned_states_lists[j].append(
                            hidden[pt + start_len : pt + extend_len]
                        )
            # Map each token to its sequence index, for chunked computation
            # of input logprobs
            token_to_seq_idx.extend([idx] * (extend_len - start_len))
            pt += extend_len
            sample_index_pt += extend_len - start_len
            sample_indices.append(sample_index_pt)
            input_logprob_indices.extend(
                [
                    input_logprob_indices_pt + i
                    for i in range(extend_len - extend_logprob_start_len)
                ]
            )
            input_logprob_indices_pt += extend_len - start_len

        pruned_states = torch.cat(pruned_states_list)
        if hidden_states_before_norm is not None:
            pruned_states_before_norm = torch.cat(pruned_states_before_norm_list)
        if aux_pruned_states_lists is not None:
            aux_pruned_states = (
                torch.cat(aux_pruned_states_lists)
                if is_packed_aux_hidden_states
                else [torch.cat(lst) for lst in aux_pruned_states_lists]
            )

        # Build the index tensors via pinned host memory + non-blocking H2D
        # so the small copy doesn't drain the stream.
        sample_indices = torch.tensor(
            sample_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(),
        ).to(pruned_states.device, non_blocking=True)
        input_logprob_indices = torch.tensor(
            input_logprob_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(),
        ).to(pruned_states.device, non_blocking=True)

    if logits_metadata.dsv4_exact_logits_rows_reconstructed:
        segment_lengths = logits_metadata.global_num_tokens_for_logprob_cpu
        dp_rank = logits_metadata.dsv4_exact_logits_dp_rank
        if segment_lengths is None or dp_rank is None:
            raise RuntimeError(
                "Exact DSV4 logits require per-DP pruned-row ownership metadata"
            )
        if not 0 <= int(dp_rank) < len(segment_lengths):
            raise RuntimeError(
                "Exact DSV4 logits DP rank is outside the pruned-row metadata"
            )
        expected_pruned_rows = int(segment_lengths[int(dp_rank)])
        if pruned_states.shape[0] != expected_pruned_rows:
            raise RuntimeError(
                "Exact DSV4 logits pruning changed the logical DP-owner row "
                "count unexpectedly: "
                f"rows={pruned_states.shape[0]}, expected={expected_pruned_rows}, "
                f"dp_rank={dp_rank}"
            )

    return (
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        input_logprob_indices,
        token_to_seq_idx,
    )


def _LogitsProcessor__forward(
    self,
    input_ids,
    hidden_states,
    lm_head: VocabParallelEmbedding,
    logits_metadata: Union[LogitsMetadata, ForwardBatch],
    aux_hidden_states: Optional[AuxHiddenStates] = None,
    hidden_states_before_norm: Optional[torch.Tensor] = None,
) -> LogitsProcessorOutput:
    # Extract MIS indices before ForwardBatch → LogitsMetadata conversion
    multi_item_delimiter_indices = None
    if isinstance(logits_metadata, ForwardBatch):
        multi_item_delimiter_indices = logits_metadata.multi_item_delimiter_indices
        logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

    # Autotune dummy run discards this output; see _in_autotune_dummy_run.
    # Placed before the MIS / DLLM / common dispatch so all three LM-head
    # paths are skipped.
    if _in_autotune_dummy_run:
        return LogitsProcessorOutput(next_token_logits=None)

    # Multi-item scoring only for prefill-only requests with pre-computed indices.
    if multi_item_delimiter_indices is not None and logits_metadata.is_prefill_only:
        return self.compute_logprobs_for_multi_item_scoring(
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            multi_item_delimiter_indices,
        )

    # Diffusion LLM only.
    if logits_metadata.forward_mode.is_dllm_extend():
        return self._get_dllm_logits(hidden_states, lm_head, logits_metadata)

    # Get the last hidden states and last logits for the next token prediction
    (
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        input_logprob_indices,
        token_to_seq_idx,
    ) = self._get_pruned_states(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        logits_metadata,
    )

    hidden_states_to_store = self._get_hidden_states_to_store(
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        pruned_states,
        pruned_states_before_norm,
        aux_pruned_states,
        sample_indices,
        logits_metadata,
    )
    del hidden_states

    if not logits_metadata.extend_return_logprob:
        # Compute logits for both input and sampled tokens.
        if self._bi_lm_head_decode_active(logits_metadata):
            logits = self._bi_lm_head_next_token_logits(
                pruned_states, lm_head, logits_metadata
            )
        else:
            logits = self._get_logits(pruned_states, lm_head, logits_metadata)
        sampled_logits = (
            logits[sample_indices] if sample_indices is not None else logits
        )

        # Decode mode or extend mode without return_logprob.
        return LogitsProcessorOutput(
            next_token_logits=sampled_logits,
            hidden_states=hidden_states_to_store,
            # FIXME: These fields are not logits-related but are passed through here as a
            # workaround since ForwardBatch is local to forward_batch_generation().
            # They should be moved to GenerationBatchResult to keep this class clean.
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )

    # When callers request only the ordinary per-token logprobs, the exact
    # contract below owns both outputs. Avoid materializing the stock fp32
    # head and log-softmax only to overwrite them.
    bi_covers_all_outputs = (
        self.use_qwen35_bi_lm_head
        and not logits_metadata.extend_return_top_logprob
        and not logits_metadata.extend_token_ids_logprob
    )
    if bi_covers_all_outputs:
        logprobs_result = LogprobResult()
        sampled_logits = None
    else:
        logprobs_result, sampled_logits = self.input_logprob_processor.forward(
            pruned_states=pruned_states,
            sample_indices=sample_indices,
            input_logprob_indices=input_logprob_indices,
            token_to_seq_idx=token_to_seq_idx,
            lm_head=lm_head,
            get_logits_fn=self._get_logits,
            logits_metadata=logits_metadata,
            skip_chunking_for_dp_attn=self.do_tensor_parallel_all_gather_dp_attn,
        )

    if self.use_qwen35_bi_lm_head:
        logprobs_result.token_logprobs = self._bi_lm_head_input_token_logprobs(
            pruned_states,
            input_logprob_indices,
            lm_head,
            logits_metadata,
        )

    if self._bi_lm_head_decode_active(logits_metadata):
        # The first generated token is sampled during extend, so it must
        # use the same contract head as all subsequent decode tokens.
        sampled_states = (
            pruned_states[sample_indices]
            if sample_indices is not None
            else pruned_states
        )
        sampled_logits = self._bi_lm_head_next_token_logits(
            sampled_states, lm_head, logits_metadata
        )

    logits_output = LogitsProcessorOutput(
        next_token_logits=sampled_logits,
        hidden_states=hidden_states_to_store,
        mm_input_embeds=logits_metadata.mm_input_embeds,
    )
    logprobs_result.write_input_to(logits_output)
    return logits_output


def __apply_patch__(mod):
    # Deferred: the finder imports twins under bypass(), so sglang imports at
    # twin top level would cache modules UNPATCHED. Import here (bypass off)
    # and publish onto mod -- in-tree these were the file's module globals.
    from sglang.srt.layers.logprob_processor import (
        InputLogprobProcessor,
        LogprobResult,
        LogprobStage,
        get_token_ids_logprobs_raw,
        get_top_logprobs_raw,
    )
    from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args
    from sglang.srt.server_args import (
        is_glm52_exact_mode,
        is_qwen3_dense_exact_mode,
        is_qwen35_gdn_exact_mode,
    )
    from sglang.xorl.batch_invariant import (
        validate_xorl_bi_logit_transforms,
        xorl_bi_lm_head,
    )

    # Publish the deferred imports onto mod: in-tree they were the srt
    # file's own module globals, and rebound copies resolve via mod.
    mod.InputLogprobProcessor = InputLogprobProcessor
    mod.LogprobResult = LogprobResult
    mod.LogprobStage = LogprobStage
    mod.get_token_ids_logprobs_raw = get_token_ids_logprobs_raw
    mod.get_top_logprobs_raw = get_top_logprobs_raw
    mod.get_exec = get_exec
    mod.get_parallel = get_parallel
    mod.get_server_args = get_server_args
    mod.is_glm52_exact_mode = is_glm52_exact_mode
    mod.is_qwen3_dense_exact_mode = is_qwen3_dense_exact_mode
    mod.is_qwen35_gdn_exact_mode = is_qwen35_gdn_exact_mode
    mod.validate_xorl_bi_logit_transforms = validate_xorl_bi_logit_transforms
    mod.xorl_bi_lm_head = xorl_bi_lm_head
    mod.LogitsProcessor._bi_lm_head_decode_active = rebind(
        _LogitsProcessor___bi_lm_head_decode_active,
        mod,
        name="_bi_lm_head_decode_active",
    )
    mod.LogitsProcessor._bi_lm_head_input_token_logprobs = rebind(
        _LogitsProcessor___bi_lm_head_input_token_logprobs,
        mod,
        name="_bi_lm_head_input_token_logprobs",
    )
    mod.LogitsProcessor._bi_lm_head_next_token_logits = rebind(
        _LogitsProcessor___bi_lm_head_next_token_logits,
        mod,
        name="_bi_lm_head_next_token_logits",
    )
    mod.LogitsProcessor._validate_bi_lm_head = rebind(
        _LogitsProcessor___validate_bi_lm_head, mod, name="_validate_bi_lm_head"
    )
    mod.LogitsMetadata.from_forward_batch = rebind(
        _LogitsMetadata__from_forward_batch, mod, name="from_forward_batch"
    )
    mod.LogitsProcessor.__init__ = rebind(
        _LogitsProcessor____init__, mod, name="__init__"
    )
    mod.LogitsProcessor._compute_lm_head = rebind(
        _LogitsProcessor___compute_lm_head, mod, name="_compute_lm_head"
    )
    mod.LogitsProcessor._copy_logits_to_buffer = rebind(
        _LogitsProcessor___copy_logits_to_buffer, mod, name="_copy_logits_to_buffer"
    )
    mod.LogitsProcessor._gather_dp_attn_hidden_states = rebind(
        _LogitsProcessor___gather_dp_attn_hidden_states,
        mod,
        name="_gather_dp_attn_hidden_states",
    )
    mod.LogitsProcessor._get_pruned_states = rebind(
        _LogitsProcessor___get_pruned_states, mod, name="_get_pruned_states"
    )
    mod.LogitsProcessor.forward = rebind(_LogitsProcessor__forward, mod, name="forward")
