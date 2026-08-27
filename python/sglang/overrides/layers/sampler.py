"""Override twin of ``sglang.srt.layers.sampler`` -- xorl exact serving (zero-srt port of PR #41).

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

_bi_decode_rescore_logged = False


def _Sampler___attach_exact_sampling_mask_to_output(
    self,
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    batch_next_token_ids: torch.Tensor,
    transformed_logits: torch.Tensor,
    support: torch.Tensor,
) -> None:
    """Return the support and normalization used by exact sampling."""
    return_sampling_masks = sampling_info.return_sampling_masks or []
    if not return_sampling_masks:
        logits_output.next_token_sampling_mask_idx = []
        logits_output.next_token_sampling_logprobs = []
        return

    selected_logprobs, _, _ = exact_selected_logprob_from_support(
        transformed_logits,
        batch_next_token_ids.to(torch.int64),
        support,
    )

    # Preserve the established descending-score response order while using
    # the exact support as the sole source of membership. Stable sorting
    # makes token ID the tie breaker, matching exact_sampling_support.
    ordered_ids = torch.argsort(
        transformed_logits.detach(), dim=-1, descending=True, stable=True
    )
    ordered_support = support.gather(1, ordered_ids)
    flat_rows, flat_cols = ordered_support.nonzero(as_tuple=True)
    flat_ids = ordered_ids[flat_rows, flat_cols].to(torch.int32)
    mask_lengths = ordered_support.sum(dim=-1, dtype=torch.int32)

    flat_ids_cpu = flat_ids.cpu().tolist()
    mask_lengths_cpu = mask_lengths.cpu().tolist()
    selected_logprobs_cpu = selected_logprobs.cpu().tolist()

    masks = []
    logprobs = []
    cursor = 0
    for i, should_return in enumerate(return_sampling_masks):
        mask_len = int(mask_lengths_cpu[i])
        row_ids = flat_ids_cpu[cursor : cursor + mask_len]
        cursor += mask_len
        if should_return:
            masks.append(row_ids)
            logprobs.append(float(selected_logprobs_cpu[i]))
        else:
            masks.append(None)
            logprobs.append(None)

    logits_output.next_token_sampling_mask_idx = masks
    logits_output.next_token_sampling_logprobs = logprobs


def _Sampler___bi_contract_sampled_logprob(
    self,
    logits: torch.Tensor,
    batch_next_token_ids: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    *,
    temperature_applied: bool = False,
) -> torch.Tensor:
    """Rescore selected tokens through the pinned contract LSE reduction."""
    from sglang.srt.batch_invariant_ops import (
        bi_lm_head_selected_logprob_from_logits,
    )

    if self.use_ascend_backend:
        raise ValueError("The exact-contract decode rescore does not support Ascend")
    if self.return_original_logprob:
        raise ValueError(
            "The exact-contract decode rescore does not support "
            "SGLANG_RETURN_ORIGINAL_LOGPROB"
        )
    if (
        sampling_info.has_custom_logit_processor
        or sampling_info.logit_bias is not None
        or sampling_info.grammars
        or sampling_info.grammar_mask is not None
        or sampling_info.acc_additive_penalties is not None
        or sampling_info.acc_scaling_penalties is not None
        or (
            sampling_info.penalizer_orchestrator is not None
            and sampling_info.penalizer_orchestrator.is_required
        )
    ):
        raise ValueError(
            "The exact-contract decode rescore does not support logit bias, "
            "penalties, grammar masks, or custom logit processors"
        )

    temperature = None
    if not sampling_info.is_all_greedy and not temperature_applied:
        if not (
            self.use_log_softmax_logprob
            and self.enable_deterministic
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        ):
            raise ValueError(
                "The exact-contract decode rescore requires all-greedy or the "
                "RL on-policy deterministic lane without top-k/top-p/min-p"
            )
        temperature = sampling_info.temperatures.reshape(-1)

    if is_bi_head_fastpath_enabled():
        from sglang.xorl.bi.bi_head_fastpath import (
            bi_lm_head_selected_logprob_from_logits_fast,
        )

        logprob, _, _ = bi_lm_head_selected_logprob_from_logits_fast(
            logits, batch_next_token_ids, temperature=temperature
        )
        return logprob

    logprob, _, _ = bi_lm_head_selected_logprob_from_logits(
        logits, batch_next_token_ids, temperature=temperature
    )
    return logprob


def _Sampler___forward_exact_filtered(
    self,
    logits_output: LogitsProcessorOutput,
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    return_logprob: bool,
    top_logprobs_nums: Optional[List[int]],
    token_ids_logprobs: Optional[List[Optional[List[int]]]],
    positions: torch.Tensor,
    *,
    return_sampling_mask: bool,
) -> torch.Tensor:
    """Sample and score one exact, jointly filtered distribution."""

    if not self.enable_deterministic or sampling_info.sampling_seed is None:
        raise RuntimeError(
            "Exact filtered sampling requires deterministic inference and per-request seeds"
        )
    if self.return_original_logprob:
        raise RuntimeError(
            "Exact filtered sampling rejects SGLANG_RETURN_ORIGINAL_LOGPROB"
        )
    if (
        sampling_info.has_custom_logit_processor
        or sampling_info.logit_bias is not None
        or sampling_info.grammars
        or sampling_info.grammar_mask is not None
        or sampling_info.acc_additive_penalties is not None
        or sampling_info.acc_scaling_penalties is not None
        or (
            sampling_info.penalizer_orchestrator is not None
            and sampling_info.penalizer_orchestrator.is_required
        )
    ):
        raise RuntimeError(
            "Exact filtered sampling does not support penalties, grammar, logit bias, or custom processors"
        )

    if self.use_qwen35_bi_decode_rescore:
        transformed_logits = exact_temperature_scale_fp32_logits(
            logits,
            sampling_info.temperatures,
        )
    else:
        transformed_logits = exact_temperature_scale_bf16_logits(
            logits.bfloat16(),
            sampling_info.temperatures,
        )
    masked_logits, support = exact_masked_logits(
        transformed_logits,
        sampling_info.top_ks,
        sampling_info.top_ps,
        sampling_info.min_ps,
    )
    greedy_rows = sampling_info.top_ks <= 1
    if sampling_info.is_all_greedy:
        batch_next_token_ids = torch.argmax(transformed_logits, dim=-1)
    else:
        exact_sampler = getattr(self, "_sample_from_exact_logits", None)
        if exact_sampler is None:
            # Dependency-injection fallback for focused CPU unit tests.
            batch_next_token_ids = self._sample_from_logprobs(
                masked_logits,
                sampling_info,
                positions,
            )
        else:
            batch_next_token_ids = exact_sampler(
                transformed_logits,
                sampling_info,
                positions,
                support,
            )
        if sampling_info.is_any_greedy:
            batch_next_token_ids = torch.where(
                greedy_rows,
                torch.argmax(transformed_logits, dim=-1),
                batch_next_token_ids,
            )

    if return_logprob:
        identity_rows = exact_sampling_identity_rows(
            sampling_info.top_ks,
            sampling_info.top_ps,
            sampling_info.min_ps,
            vocab_size=transformed_logits.shape[1],
        )
        native_full_logprobs = None

        if self.use_qwen35_bi_decode_rescore:

            def _native_score(native_logits, native_ids):
                if is_bi_head_fastpath_enabled():
                    from sglang.xorl.bi.bi_head_fastpath import (  # noqa: PLC0415
                        bi_lm_head_selected_logprob_from_logits_fast,
                    )

                    score = bi_lm_head_selected_logprob_from_logits_fast
                else:
                    from sglang.srt.batch_invariant_ops import (  # noqa: PLC0415
                        bi_lm_head_selected_logprob_from_logits,
                    )

                    score = bi_lm_head_selected_logprob_from_logits
                return score(native_logits, native_ids, temperature=None)

        else:
            from sglang.srt.batch_invariant_ops.batch_invariant_ops import (  # noqa: PLC0415
                log_softmax as _bi_log_softmax,
            )

            def _native_score(native_logits, native_ids):
                nonlocal native_full_logprobs
                native_full_logprobs = _bi_log_softmax(native_logits, dim=-1)
                native_selected = native_logits.gather(
                    1, native_ids.unsqueeze(1)
                ).squeeze(1)
                native_logprob = native_full_logprobs.gather(
                    1, native_ids.unsqueeze(1)
                ).squeeze(1)
                return (
                    native_logprob,
                    native_selected - native_logprob,
                    native_selected,
                )

        selected_logprobs, lse, _ = exact_selected_logprob_partitioned_from_support(
            transformed_logits,
            batch_next_token_ids.to(torch.int64),
            support,
            identity_rows,
            _native_score,
        )
        if sampling_info.is_any_greedy:
            # top_k=1 is the normalized temperature=0 decision program:
            # the selected argmax has probability one and logprob +0.
            selected_logprobs = torch.where(
                greedy_rows,
                torch.zeros_like(selected_logprobs),
                selected_logprobs,
            )
        filtered_logprobs = masked_logits - lse.unsqueeze(1)
        if native_full_logprobs is not None:
            filtered_logprobs = torch.where(
                identity_rows.unsqueeze(1),
                native_full_logprobs,
                filtered_logprobs,
            )
        logprob_result = self.output_logprob_processor.compute_logprobs(
            filtered_logprobs,
            top_logprobs_nums,
            token_ids_logprobs,
            batch_next_token_ids,
        )
        logprob_result.write_output_to(logits_output)
        logits_output.next_token_logprobs = selected_logprobs

    if return_sampling_mask:
        self._attach_exact_sampling_mask_to_output(
            logits_output,
            sampling_info,
            batch_next_token_ids,
            transformed_logits,
            support,
        )

    self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
    return batch_next_token_ids


def _Sampler___sample_from_exact_logits(
    self,
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    positions: torch.Tensor,
    support: torch.Tensor | None,
) -> torch.Tensor:
    """Exact-only endpoint-safe seeded Gumbel-max; generic sampling is unchanged."""

    return exact_seeded_gumbel_sample(
        logits,
        sampling_info.sampling_seed,
        positions,
        support=support,
    )


def _Sampler____init__(self):
    super(Sampler, self).__init__()
    self._glm52_exact_mode = is_glm52_exact_mode(get_server_args())
    self._qwen3_dense_exact_mode = is_qwen3_dense_exact_mode(get_server_args())
    self._dsv4_flash_exact_mode = is_dsv4_flash_exact_mode(get_server_args())
    self.tp_sync_group = get_tp_group().device_group
    if is_dp_attention_enabled():
        self.tp_sync_group = get_parallel().attn_tp_group.device_group

    self.rl_on_policy_target = get_exec().deterministic.rl_on_policy_target
    self.use_qwen35_bi_decode_rescore = is_qwen35_gdn_exact_mode(get_server_args())
    self.return_original_logprob = (
        False if self.use_qwen35_bi_decode_rescore else SGLANG_RETURN_ORIGINAL_LOGPROB
    )
    # In RL on-policy mode, deterministic inference is automatically enabled.
    self.enable_deterministic = get_exec().deterministic.enable_deterministic_inference
    # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
    self.use_log_softmax_logprob = self.rl_on_policy_target is not None
    self.use_ascend_backend = get_exec().kernel.sampling_backend == "ascend"

    self.output_logprob_processor = OutputLogprobProcessor()


def _Sampler___forward_ascend_backend(
    self,
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    simple_sampling_case: bool,
    return_logprob: bool,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Handle the full Ascend backend sampling path.

    Ascend backend has fused kernels that handle softmax internally,
    so we sample directly from temperature-scaled logits.

    Returns:
        A tuple of (batch_next_token_ids, logprobs). logprobs is None
        when return_logprob is False or SGLANG_RETURN_ORIGINAL_LOGPROB is set.
    """
    logits.div_(sampling_info.temperatures)
    batch_next_token_ids = self._sample_from_logits(
        logits, sampling_info, simple_sampling_case, positions
    )
    logprobs = None
    if return_logprob and not self.return_original_logprob:
        logprobs = torch.log_softmax(logits, dim=-1)
    return batch_next_token_ids, logprobs


def _Sampler___sample_from_logprobs(
    self,
    logprobs: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Sample from log-probabilities using the Gumbel trick.

    Exact lanes may pass either their full transformed logits or logits
    masked by the shared top-k/top-p/min-p program. Requires a per-row
    ``sampling_seed`` and uses deterministic seeded Gumbel-max.
    """
    assert (
        sampling_info.sampling_seed is not None
    ), "sampling_seed is required for sampling from logprobs"
    sampled_index = multinomial_with_seed(
        logprobs, sampling_info.sampling_seed, positions
    )
    return sampled_index.view(-1).to(torch.int32)


def _Sampler__forward(
    self,
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    return_logprob: bool,
    top_logprobs_nums: Optional[List[int]],
    token_ids_logprobs: Optional[List[Optional[List[int]]]],
    positions: torch.Tensor,
):
    """Run a sampler & compute logprobs and update logits_output accordingly.

    Args:
        logits_output: The logits from the model forward
        sampling_info: Metadata for sampling
        return_logprob: If set, store the output logprob information to
            logits_output
        top_logprobs_nums: Number of top lobprobs per sequence in a batch
        token_ids_logprobs: Per-sequence list of specific token IDs to retrieve
            logprobs for. Each element is a list of token IDs (or None) for one
            sequence in the batch. This is used in speculative decoding.
        positions: The positions of the tokens in the sequence. Used for deterministic sampling
            to get the unique seed for each position.
    """
    if self._glm52_exact_mode or getattr(self, "_qwen3_dense_exact_mode", False):
        return xorl_bi_sample_and_score(
            logits_output,
            sampling_info,
            return_logprob=return_logprob,
            top_logprobs_nums=top_logprobs_nums,
            token_ids_logprobs=token_ids_logprobs,
            positions=positions,
            sample_from_logprobs=None,
            sync_token_ids=self._sync_token_ids_across_tp,
            enable_deterministic=self.enable_deterministic,
            return_original_logprob=SGLANG_RETURN_ORIGINAL_LOGPROB,
            family="v2",
        )

    logits = logits_output.next_token_logits

    # Preprocess logits (custom processors and NaN handling)
    logits = self._preprocess_logits(logits, sampling_info)
    return_sampling_mask = any(sampling_info.return_sampling_masks or [])
    exact_sampling_logits = None

    has_sampling_filter = (
        sampling_info.need_top_p_sampling
        or sampling_info.need_top_k_sampling
        or sampling_info.need_min_p_sampling
    )
    if has_sampling_filter and (
        self.use_qwen35_bi_decode_rescore or self._dsv4_flash_exact_mode
    ):
        return self._forward_exact_filtered(
            logits_output,
            logits,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
            return_sampling_mask=return_sampling_mask,
        )

    if sampling_info.is_all_greedy:
        if _use_aiter and not _disable_aiter_greedy_sample:
            batch_next_token_ids = torch.empty(
                logits.shape[0], device=logits.device, dtype=torch.int32
            )
            _aiter_greedy_sample(batch_next_token_ids, logits)
        else:
            batch_next_token_ids = torch.argmax(logits, -1)
        if return_sampling_mask:
            self._attach_greedy_sampling_mask_to_output(
                logits_output, sampling_info, batch_next_token_ids
            )
        if return_logprob:
            original_logprobs = logprobs = torch.nn.functional.log_softmax(
                logits, dim=-1
            )
    else:
        simple_sampling_case = (
            not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        )

        # If requested, cache original logprobs before temperature scaling.
        if return_logprob and self.return_original_logprob:
            original_logprobs = torch.log_softmax(logits, dim=-1)

        # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
        logprobs_via_logsoftmax_kernel = None
        if self.rl_on_policy_target is not None:
            if self.use_qwen35_bi_decode_rescore:
                exact_sampling_logits = exact_temperature_scale_fp32_logits(
                    logits,
                    sampling_info.temperatures,
                )
                logits_div_temperature = exact_sampling_logits
            else:
                # TODO: use more inplace ops to save memory
                logits_div_temperature = exact_temperature_scale_bf16_logits(
                    logits.bfloat16(),
                    sampling_info.temperatures,
                )
            logprobs_via_logsoftmax_kernel = torch.log_softmax(
                logits_div_temperature, dim=-1
            )
            if exact_sampling_logits is None:
                del logits_div_temperature

        if self.use_ascend_backend:
            # Ascend backend: sample from logits directly.
            batch_next_token_ids, logprobs = self._forward_ascend_backend(
                logits,
                sampling_info,
                simple_sampling_case,
                return_logprob,
                positions,
            )
        elif (
            self.use_log_softmax_logprob
            and self.enable_deterministic
            and simple_sampling_case
        ):
            # RL on-policy path: sample from logprobs to match the trainer.
            batch_next_token_ids = self._sample_from_logprobs(
                (
                    exact_sampling_logits
                    if exact_sampling_logits is not None
                    else logprobs_via_logsoftmax_kernel
                ),
                sampling_info,
                positions,
            )
            if return_logprob and not self.return_original_logprob:
                logprobs = logprobs_via_logsoftmax_kernel
        else:
            # Standard path: do softmax and sample from probs.
            logits.div_(sampling_info.temperatures)

            # Deterministic inference must derive the returned logprobs
            # from F.log_softmax — the same kernel prefill rescoring uses —
            # not log(softmax(x)) below: the two disagree at ~1e-6 despite
            # being mathematically equivalent, which breaks bitwise
            # prefill/decode logprob alignment.
            if (
                return_logprob
                and self.enable_deterministic
                and logprobs_via_logsoftmax_kernel is None
                and not self.return_original_logprob
            ):
                logprobs_via_logsoftmax_kernel = torch.nn.functional.log_softmax(
                    logits, dim=-1
                )

            # In-place op to save memory
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits

            batch_next_token_ids = self._sample_from_probs(
                probs, sampling_info, positions, simple_sampling_case
            )
            if return_sampling_mask:
                sampling_mask_data = self._compute_sampling_mask_from_probs(
                    probs, sampling_info
                )
                self._attach_sampling_mask_to_output(
                    logits_output,
                    sampling_info,
                    batch_next_token_ids,
                    sampling_mask_data,
                )
            if return_logprob and not self.return_original_logprob:
                logprobs = (
                    logprobs_via_logsoftmax_kernel
                    if logprobs_via_logsoftmax_kernel is not None
                    else torch.log(probs)
                )
            del probs

    if return_logprob:
        if self.return_original_logprob:
            logprobs = original_logprobs
        logprob_result = self.output_logprob_processor.compute_logprobs(
            logprobs,
            top_logprobs_nums,
            token_ids_logprobs,
            batch_next_token_ids,
        )
        logprob_result.write_output_to(logits_output)
        if self.use_qwen35_bi_decode_rescore:
            logits_output.next_token_logprobs = self._bi_contract_sampled_logprob(
                (
                    exact_sampling_logits
                    if exact_sampling_logits is not None
                    else logits_output.next_token_logits
                ),
                batch_next_token_ids,
                sampling_info,
                temperature_applied=exact_sampling_logits is not None,
            )
            global _bi_decode_rescore_logged
            if not _bi_decode_rescore_logged:
                _bi_decode_rescore_logged = True
                logger.info(
                    "BI decode rescore active: output_token_logprobs are "
                    "contract values"
                )

    self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)

    return batch_next_token_ids


def __apply_patch__(mod):
    # Deferred: the finder imports twins under bypass(), so sglang imports at
    # twin top level would cache modules UNPATCHED. Import here (bypass off)
    # and publish onto mod -- in-tree these were the file's module globals.
    from sglang.srt.server_args import (
        is_dsv4_flash_exact_mode,
        is_glm52_exact_mode,
        is_qwen3_dense_exact_mode,
        is_qwen35_gdn_exact_mode,
    )
    from sglang.xorl.batch_invariant import xorl_bi_sample_and_score
    from sglang.xorl.bi.bi_families_v2 import (
        exact_temperature_scale_bf16_logits,
        exact_temperature_scale_fp32_logits,
    )
    from sglang.xorl.bi.ops_ext import is_bi_head_fastpath_enabled
    from sglang.xorl.exact_sampling_transforms import (
        exact_masked_logits,
        exact_sampling_identity_rows,
        exact_seeded_gumbel_sample,
        exact_selected_logprob_from_support,
        exact_selected_logprob_partitioned_from_support,
    )

    # Publish the deferred imports onto mod: in-tree they were the srt
    # file's own module globals, and rebound copies resolve via mod.
    mod.is_dsv4_flash_exact_mode = is_dsv4_flash_exact_mode
    mod.is_glm52_exact_mode = is_glm52_exact_mode
    mod.is_qwen3_dense_exact_mode = is_qwen3_dense_exact_mode
    mod.is_qwen35_gdn_exact_mode = is_qwen35_gdn_exact_mode
    mod.xorl_bi_sample_and_score = xorl_bi_sample_and_score
    mod.exact_temperature_scale_bf16_logits = exact_temperature_scale_bf16_logits
    mod.exact_temperature_scale_fp32_logits = exact_temperature_scale_fp32_logits
    mod.is_bi_head_fastpath_enabled = is_bi_head_fastpath_enabled
    mod.exact_masked_logits = exact_masked_logits
    mod.exact_sampling_identity_rows = exact_sampling_identity_rows
    mod.exact_seeded_gumbel_sample = exact_seeded_gumbel_sample
    mod.exact_selected_logprob_from_support = exact_selected_logprob_from_support
    mod.exact_selected_logprob_partitioned_from_support = (
        exact_selected_logprob_partitioned_from_support
    )
    mod._bi_decode_rescore_logged = _bi_decode_rescore_logged
    mod.Sampler._attach_exact_sampling_mask_to_output = rebind(
        _Sampler___attach_exact_sampling_mask_to_output,
        mod,
        name="_attach_exact_sampling_mask_to_output",
    )
    mod.Sampler._bi_contract_sampled_logprob = rebind(
        _Sampler___bi_contract_sampled_logprob, mod, name="_bi_contract_sampled_logprob"
    )
    mod.Sampler._forward_exact_filtered = rebind(
        _Sampler___forward_exact_filtered, mod, name="_forward_exact_filtered"
    )
    mod.Sampler._sample_from_exact_logits = rebind(
        _Sampler___sample_from_exact_logits, mod, name="_sample_from_exact_logits"
    )
    mod.Sampler.__init__ = rebind(_Sampler____init__, mod, name="__init__")
    mod.Sampler._forward_ascend_backend = rebind(
        _Sampler___forward_ascend_backend, mod, name="_forward_ascend_backend"
    )
    mod.Sampler._sample_from_logprobs = rebind(
        _Sampler___sample_from_logprobs, mod, name="_sample_from_logprobs"
    )
    mod.Sampler.forward = rebind(_Sampler__forward, mod, name="forward")
