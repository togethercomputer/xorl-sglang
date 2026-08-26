"""Override twin of ``sglang.srt.managers.tokenizer_manager`` -- xorl exact serving (zero-srt port of PR #41).

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

import dataclasses

import math

def _build_raw_token_logprobs_b64_fields(
    input_values: List[Optional[float]],
    output_values: List[Optional[float]],
) -> Dict[str, Any]:
    """Encode selected-token logprobs without a JSON float round trip.

    Positions with no defined logprob (normally the first prompt token) are
    represented by the canonical NumPy float32 NaN.  Length fields preserve
    the position mapping and make the wire format independently auditable.
    """

    def _encode(values: List[Optional[float]]) -> str:
        arr = np.asarray(
            [np.nan if value is None else value for value in values], dtype="<f4"
        )
        return pybase64.b64encode(arr.tobytes()).decode("utf-8")

    return {
        "input_token_logprobs_raw_b64": _encode(input_values),
        "input_token_logprobs_raw_length": len(input_values),
        "output_token_logprobs_raw_b64": _encode(output_values),
        "output_token_logprobs_raw_length": len(output_values),
        "token_logprobs_raw_b64_dtype": "float32_le",
    }

def _get_bi_decode_strict_ingress_violations(
    obj: GenerateReqInput,
    preferred_sampling_params: Optional[Dict[str, Any]] = None,
    *,
    sampled_logprob_only: bool = False,
) -> List[str]:
    """Return request transforms that the exact decode path cannot rescore."""
    sampling_params = dict(preferred_sampling_params or {})
    sampling_params.update(obj.sampling_params or {})

    violations = []
    plain_sampling_defaults = {
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_new_tokens": 0,
    }
    for name, expected in plain_sampling_defaults.items():
        value = sampling_params.get(name, expected)
        if value != expected:
            violations.append(f"{name}={value!r}")

    temperature = sampling_params.get("temperature", 1.0)
    try:
        normalized_temperature = float(temperature)
    except (TypeError, ValueError):
        normalized_temperature = float("nan")
    if not math.isfinite(normalized_temperature) or normalized_temperature < 0.0:
        violations.append(f"temperature={temperature!r}")

    for name in ("json_schema", "regex", "ebnf", "structural_tag", "logit_bias"):
        if sampling_params.get(name) is not None:
            violations.append(f"{name}=set")

    if sampling_params.get("mtp_enabled", False):
        violations.append("mtp_enabled=True")
    if obj.custom_logit_processor is not None:
        violations.append("custom_logit_processor=set")
    if obj.session_params is not None:
        violations.append("session_params=set")
    if sampled_logprob_only:
        if obj.top_logprobs_num not in (None, 0):
            violations.append(f"top_logprobs_num={obj.top_logprobs_num!r}")
        if obj.token_ids_logprob:
            violations.append("token_ids_logprob=set")

    return violations

@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: Union[GenerateReqInput, EmbeddingReqInput]

    # For performance metrics
    time_stats: APIServerReqTimeStats
    abort_requested: bool = False
    lifecycle_id: object = dataclasses.field(default_factory=object)
    dispatched: bool = False
    sampling_temperature: Optional[float] = None
    sampling_top_k: Optional[int] = None
    sampling_top_p: Optional[float] = None
    sampling_min_p: Optional[float] = None
    last_completion_tokens: int = 1
    ttft_observed: bool = False

    # For streaming output
    last_output_offset: int = 0

    # Accumulate text lazily so incremental streaming can emit the incoming
    # delta directly without rebuilding the full output prefix.
    text: str = ""
    text_chunks: List[str] = dataclasses.field(default_factory=list)

    def append_text(self, chunk: str):
        if chunk:
            self.text_chunks.append(chunk)

    def get_text(self) -> str:
        if self.text_chunks:
            self.text += "".join(self.text_chunks)
            self.text_chunks.clear()
        return self.text

    def get_crash_dump_output(self) -> Dict[Any, Any]:
        out = {}
        if self.text or self.text_chunks:
            out["text"] = self.get_text()
        if self.output_ids:
            out["output_ids"] = self.output_ids.copy()
        return out

    # For incremental state update.
    # TODO(lianmin): do not initialize some lists if not needed.
    output_ids: List[int] = dataclasses.field(default_factory=list)
    input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    input_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    output_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    output_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    input_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    input_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    output_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    output_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)
    output_token_sampling_mask: List = dataclasses.field(default_factory=list)
    output_token_sampling_logprobs: List = dataclasses.field(default_factory=list)

    # Cached flat-format prompt top logprob fields; rebuilt only when more
    # prefill chunks arrive, so streaming decode chunks reuse the payload.
    input_top_logprobs_flat_fields: Optional[Dict[str, Any]] = None
    input_top_logprobs_flat_num_rows: int = -1
    # Scheduler-assembled flat arrays (val float32 [rows, k], idx int32
    # [rows, k], null_prefix), sent once at prefill completion. When present,
    # the nested input_top_logprobs_val/idx above stay empty.
    input_top_logprobs_scheduler_flat: Optional[Tuple[np.ndarray, np.ndarray, int]] = (
        None
    )

    # For detokenized logprobs
    input_token_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_token_logprobs: List[Any] = dataclasses.field(default_factory=list)
    input_top_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_top_logprobs: List[Any] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)
    output_token_ids_logprobs: List[Any] = dataclasses.field(default_factory=list)
    customized_info_accumulated: Dict[str, List[Any]] = dataclasses.field(
        default_factory=dict
    )

    # For return_prompt_token_ids: stores prompt token IDs captured after tokenization
    prompt_token_ids: Optional[List[int]] = None

def _TokenizerManager___create_tokenized_object(
    self,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    input_text: str,
    input_ids: Optional[List[int]],
    input_embeds: Optional[List[List[float]]] = None,
    mm_inputs=None,
    token_type_ids: Optional[List[int]] = None,
) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
    """Create a tokenized request object from common parameters."""
    input_ids_arr: Optional[array[int]] = (
        array("q", input_ids) if input_ids is not None else None
    )
    # Parse sampling parameters
    # Note: if there are preferred sampling params, we use them if they are not
    # explicitly passed in sampling_params
    if self.preferred_sampling_params:
        sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
    else:
        sampling_kwargs = obj.sampling_params
    if isinstance(obj, GenerateReqInput) and obj.max_thinking_tokens is not None:
        sampling_kwargs = dict(sampling_kwargs)
        custom_params = dict(sampling_kwargs.get("custom_params") or {})
        custom_params["thinking_budget"] = obj.max_thinking_tokens
        sampling_kwargs["custom_params"] = custom_params
    sampling_params = self.sampling_params_class(**sampling_kwargs)
    sampling_params.normalize(self.tokenizer)
    sampling_params.verify(self.model_config.vocab_size)

    # Build return object
    if isinstance(obj, GenerateReqInput):
        session_params = (
            SessionParams(**obj.session_params) if obj.session_params else None
        )

        bootstrap_room = obj.bootstrap_room
        if (
            bootstrap_room is None
            and self.server_args.disaggregation_transfer_backend == "fake"
        ):
            bootstrap_room = self.fake_bootstrap_room_counter
            self.fake_bootstrap_room_counter += 1

        tokenized_obj = TokenizedGenerateReqInput(
            input_text=input_text,
            input_ids=input_ids_arr,
            mm_inputs=mm_inputs,
            sampling_params=sampling_params,
            return_logprob=obj.return_logprob,
            logprob_start_len=obj.logprob_start_len,
            top_logprobs_num=obj.top_logprobs_num,
            token_ids_logprob=obj.token_ids_logprob,
            return_sampling_mask=obj.return_sampling_mask,
            return_flat_raw_top_logprobs=obj.return_flat_raw_top_logprobs,
            stream=obj.stream,
            rid=obj.rid,
            http_worker_ipc=obj.http_worker_ipc,
            bootstrap_host=obj.bootstrap_host,
            bootstrap_port=obj.bootstrap_port,
            bootstrap_room=bootstrap_room,
            lora_id=obj.lora_id,
            input_embeds=input_embeds,
            positional_embed_overrides=obj.positional_embed_overrides,
            session_id=obj.session_id,
            session_params=session_params,
            custom_logit_processor=obj.custom_logit_processor,
            require_reasoning=obj.require_reasoning,
            return_hidden_states=obj.return_hidden_states,
            return_routed_experts=obj.return_routed_experts,
            routed_experts_start_len=obj.routed_experts_start_len,
            return_indexer_topk=obj.return_indexer_topk,
            routed_dp_rank=obj.routed_dp_rank,
            disagg_prefill_dp_rank=obj.disagg_prefill_dp_rank,
            priority=obj.priority,
            extra_key=obj.extra_key,
            routing_key=obj.routing_key,
            token_type_ids=token_type_ids,
            need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
            num_items_assigned=obj.num_items_assigned,
            multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
            mm_data_mooncake=obj.mm_data_mooncake,
            encoder_urls=obj.encoder_urls,
        )
    elif isinstance(obj, EmbeddingReqInput):
        # Resolve unresolved embed overrides now that input_ids are available
        positional_embed_overrides = obj.positional_embed_overrides
        if (
            positional_embed_overrides is None
            and obj.embed_overrides is not None
            and obj.embed_override_token_id is not None
        ):
            positional_embed_overrides = self._resolve_embed_overrides(
                input_ids_arr, obj.embed_override_token_id, obj.embed_overrides
            )

        tokenized_obj = TokenizedEmbeddingReqInput(
            input_text=input_text,
            input_ids=input_ids_arr,
            mm_inputs=mm_inputs,
            token_type_ids=token_type_ids,
            sampling_params=sampling_params,
            positional_embed_overrides=positional_embed_overrides,
            rid=obj.rid,
            priority=obj.priority,
            dimensions=obj.dimensions,
            lora_id=obj.lora_id,
            http_worker_ipc=obj.http_worker_ipc,
            return_pooled_hidden_states=obj.return_pooled_hidden_states,
            multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
        )

    state = self.rid_to_state[obj.rid]
    if isinstance(obj, GenerateReqInput):
        # Relay normalized values that the sampler actually consumes,
        # including preferred parameters and identity defaults.  Exact
        # trainer replay recomputes support from these parameters and
        # current logits; it never reuses a behavior-time support mask.
        state.sampling_temperature = float(sampling_params.temperature)
        state.sampling_top_k = int(sampling_params.top_k)
        state.sampling_top_p = float(sampling_params.top_p)
        state.sampling_min_p = float(sampling_params.min_p)
    tokenized_obj.time_stats = state.time_stats
    state.time_stats.set_tokenize_finish_time()

    return tokenized_obj

async def _TokenizerManager___handle_batch_output(
    self,
    recv_obj: Union[
        BatchStrOutput,
        BatchEmbeddingOutput,
        BatchTokenIDOutput,
    ],
):
    recv_obj.time_stats = unwrap_from_pickle(recv_obj.time_stats)
    if isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput)):
        customized_info = unwrap_from_pickle(recv_obj.customized_info)
    else:
        customized_info = None
    pending_notify: dict[str, ReqState] = {}
    batch_notify_size = self.server_args.batch_notify_size
    for i, rid in enumerate(recv_obj.rids):
        state = self.rid_to_state.get(rid, None)
        if state is None:
            # Known race: /health_generate pops its rid as soon as ANY message bumps last_receive_tstamp.
            if rid.startswith(HEALTH_CHECK_RID_PREFIX):
                continue
            logger.error(
                f"Received output for {rid=} but the state was deleted in TokenizerManager."
            )
            continue

        # Build meta_info and return value
        meta_info = {
            "id": rid,
            "finish_reason": recv_obj.finished_reasons[i],
            "prompt_tokens": recv_obj.prompt_tokens[i],
            "weight_version": self.config_value("weight_version"),
            "num_retractions": recv_obj.retraction_counts[i],
        }
        if state.sampling_temperature is not None:
            meta_info["sampling_temperature"] = state.sampling_temperature
        if state.sampling_top_k is not None:
            meta_info["sampling_top_k"] = state.sampling_top_k
        if state.sampling_top_p is not None:
            meta_info["sampling_top_p"] = state.sampling_top_p
        if state.sampling_min_p is not None:
            meta_info["sampling_min_p"] = state.sampling_min_p

        if self.enable_metrics:
            if recv_obj.time_stats is not None:
                scheduler_time_stats = recv_obj.time_stats[i]
                meta_info.update(scheduler_time_stats.convert_to_output_meta_info())

        if getattr(state.obj, "return_logprob", False):
            self.convert_logprob_style(
                meta_info,
                state,
                state.obj.top_logprobs_num,
                state.obj.token_ids_logprob,
                state.obj.return_text_in_logprobs and not self.skip_tokenizer_init,
                recv_obj,
                i,
            )
        if (
            isinstance(state.obj, GenerateReqInput)
            and state.obj.return_sampling_mask
        ):
            output_sampling_mask = recv_obj.output_token_sampling_mask
            if output_sampling_mask is not None:
                state.output_token_sampling_mask.extend(output_sampling_mask[i])
                output_sampling_logprobs = recv_obj.output_token_sampling_logprobs
                if output_sampling_logprobs is not None:
                    state.output_token_sampling_logprobs.extend(
                        output_sampling_logprobs[i]
                    )
                meta_info["output_token_sampling_mask"] = (
                    state.output_token_sampling_mask
                )
                meta_info["output_token_sampling_logprobs"] = (
                    state.output_token_sampling_logprobs
                )
                meta_info["output_token_sampling_mask_length"] = len(
                    state.output_token_sampling_mask
                )

        if not isinstance(recv_obj, BatchEmbeddingOutput):
            meta_info.update(
                {
                    "reasoning_tokens": recv_obj.reasoning_tokens[i],
                    "completion_tokens": recv_obj.completion_tokens[i],
                    "cached_tokens": recv_obj.cached_tokens[i],
                }
            )
            # Add detailed cache breakdown if available
            if (
                hasattr(recv_obj, "cached_tokens_details")
                and recv_obj.cached_tokens_details
            ):
                meta_info["cached_tokens_details"] = recv_obj.cached_tokens_details[
                    i
                ]
            if customized_info is not None:
                for k, v in customized_info.items():
                    if k not in state.customized_info_accumulated:
                        state.customized_info_accumulated[k] = []
                    state.customized_info_accumulated[k].extend(v[i])
                    meta_info[k] = state.customized_info_accumulated[k]

            # Add multimodal prompt token counts only for requests that
            # actually consumed them, so plain-text meta_info stays unchanged.
            image_tokens_list = getattr(recv_obj, "image_tokens", None)
            audio_tokens_list = getattr(recv_obj, "audio_tokens", None)
            video_tokens_list = getattr(recv_obj, "video_tokens", None)
            if image_tokens_list and image_tokens_list[i]:
                meta_info["image_tokens"] = image_tokens_list[i]
            if audio_tokens_list and audio_tokens_list[i]:
                meta_info["audio_tokens"] = audio_tokens_list[i]
            if video_tokens_list and video_tokens_list[i]:
                meta_info["video_tokens"] = video_tokens_list[i]

        if getattr(recv_obj, "output_hidden_states", None):
            hidden_states = recv_obj.output_hidden_states[i]
            if hidden_states is not None:
                meta_info["hidden_states"] = hidden_states
        if getattr(recv_obj, "routed_experts", None):
            val = recv_obj.routed_experts[i]
            if val is not None:
                # BatchStrOutput is pre-encoded by the detokenizer;
                # BatchTokenIDOutput (skip_tokenizer_init) bypasses it.
                if isinstance(val, torch.Tensor):
                    val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                meta_info["routed_experts"] = val
        if getattr(recv_obj, "expert_logits", None):
            val = recv_obj.expert_logits[i]
            if val is not None:
                if isinstance(val, torch.Tensor):
                    val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                meta_info["expert_logits"] = val
        if getattr(recv_obj, "indexer_topk", None):
            val = recv_obj.indexer_topk[i]
            if val is not None:
                if isinstance(val, torch.Tensor):
                    val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                meta_info["indexer_topk"] = val
        if getattr(recv_obj, "dp_ranks", None):
            meta_info["dp_rank"] = recv_obj.dp_ranks[i]

        state.finished = recv_obj.finished_reasons[i] is not None
        if isinstance(recv_obj, BatchStrOutput):
            # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
            is_stream = getattr(state.obj, "stream", False)
            incremental = is_stream and self.incremental_streaming_output
            delta_text = recv_obj.output_strs[i]
            delta_output_ids = list(recv_obj.output_ids[i])
            output_offset = state.last_output_offset
            state.append_text(delta_text)
            state.output_ids.extend(delta_output_ids)

            if is_stream:
                if incremental:
                    output_token_ids = delta_output_ids
                    _slice_streaming_output_meta_info(
                        meta_info,
                        output_offset,
                        state.customized_info_accumulated.keys(),
                    )
                    state.last_output_offset = len(state.output_ids)
                    out_dict = {
                        "text": delta_text,
                        "output_ids": output_token_ids,
                        "meta_info": meta_info,
                    }
                elif state.finished:
                    out_dict = {
                        "text": state.get_text(),
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    # Non-incremental intermediate: pass reference (no
                    # copy) and defer text to _wait_one_response to avoid
                    # O(n) per-step cost that compounds to O(n^2).
                    out_dict = {
                        "text": None,
                        "output_ids": state.output_ids,
                        "meta_info": meta_info,
                    }
            elif state.finished:
                out_dict = {
                    "text": state.get_text(),
                    "output_ids": state.output_ids.copy(),
                    "meta_info": meta_info,
                }
            else:
                out_dict = None
            if out_dict is not None and state.prompt_token_ids is not None:
                out_dict["prompt_token_ids"] = state.prompt_token_ids
        elif isinstance(recv_obj, BatchTokenIDOutput):
            is_stream = getattr(state.obj, "stream", False)
            incremental = is_stream and self.incremental_streaming_output
            delta_output_ids = list(recv_obj.output_ids[i])
            output_offset = state.last_output_offset
            state.output_ids.extend(delta_output_ids)

            if is_stream:
                if incremental:
                    output_token_ids = delta_output_ids
                    _slice_streaming_output_meta_info(
                        meta_info,
                        output_offset,
                        state.customized_info_accumulated.keys(),
                    )
                    state.last_output_offset = len(state.output_ids)
                    out_dict = {
                        "output_ids": output_token_ids,
                        "meta_info": meta_info,
                    }
                elif state.finished:
                    out_dict = {
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = {
                        "output_ids": state.output_ids,
                        "meta_info": meta_info,
                    }
            elif state.finished:
                out_dict = {
                    "output_ids": state.output_ids.copy(),
                    "meta_info": meta_info,
                }
            else:
                out_dict = None
            if out_dict is not None and state.prompt_token_ids is not None:
                out_dict["prompt_token_ids"] = state.prompt_token_ids
        else:
            assert isinstance(recv_obj, BatchEmbeddingOutput)
            out_dict = {
                "embedding": recv_obj.embeddings[i],
                "meta_info": meta_info,
            }
            # Unpack pooled hidden states (PHS).
            # See paired sender logic in output_streamer.py.
            #   Stacked:     len == 1 and N > 1 → unwrap the tensor
            #   Non-stacked: len == N → index directly
            pooled_hidden_states = recv_obj.pooled_hidden_states
            if pooled_hidden_states is not None:
                if len(pooled_hidden_states) == 1 and len(recv_obj.rids) > 1:
                    pooled_hidden_states = pooled_hidden_states[0]
                if pooled_hidden_states[i] is not None:
                    out_dict["pooled_hidden_state"] = pooled_hidden_states[i]

        # Set first_token_time on the first output batch.
        # This is the single write point for first_token_time.
        if state.time_stats.first_token_time == 0.0:
            state.time_stats.set_first_token_time()

        if state.finished:
            if state.time_stats.trace_ctx.tracing_enable:
                state.time_stats.trace_ctx.trace_set_root_attrs(
                    self.convert_to_span_attrs(state, recv_obj, i)
                )
            state.time_stats.set_finished_time()
            meta_info["e2e_latency"] = state.time_stats.get_e2e_latency()

            if self.server_args.speculative_algorithm:
                self._calculate_spec_decoding_metrics(meta_info, recv_obj, i)
            if self.enable_metrics:
                scheduler_time_stats = (
                    recv_obj.time_stats[i]
                    if recv_obj.time_stats is not None
                    else None
                )
                completion_tokens = (
                    recv_obj.completion_tokens[i]
                    if not isinstance(recv_obj, BatchEmbeddingOutput)
                    else 0
                )
                meta_info.update(
                    state.time_stats.convert_to_output_meta_info(
                        scheduler_time_stats, completion_tokens
                    )
                )

            self._remove_req_state(rid)

            # Mark ongoing LoRA request as finished.
            if self.enable_lora and state.obj.lora_path:
                asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

        if out_dict is not None:
            state.out_list.append(out_dict)
            pending_notify[rid] = state

            if len(pending_notify) >= batch_notify_size:
                for s in pending_notify.values():
                    s.event.set()
                pending_notify = {}
                await asyncio.sleep(0)

        if self.enable_metrics and state.obj.log_metrics:
            self.collect_metrics(state, recv_obj, i)
        if self.dump_requests_folder and state.finished and state.obj.log_metrics:
            self.dump_requests(state, out_dict)
        if self.crash_dump_folder and state.finished and state.obj.log_metrics:
            self.record_request_for_crash_dump(state, out_dict)

    # handle_loop awaits next recv immediately
    for s in pending_notify.values():
        s.event.set()

def _TokenizerManager___validate_one_request(
    self, obj: Union[GenerateReqInput, EmbeddingReqInput], input_ids: List[int]
) -> None:
    """Validates that the input token count and the requested token count doesn't exceed the model's context length."""
    # FIXME: unify the length validation logic with the one in the scheduler.
    _max_req_len = self.context_len
    input_token_num = len(input_ids) if input_ids is not None else 0
    input_token_num += self.num_reserved_tokens

    # Validate input length
    if input_token_num >= self.context_len:
        if self.allow_auto_truncate:
            logger.warning(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens). "
                "Truncating the input."
            )
            del input_ids[_max_req_len:]
            input_token_num = len(input_ids)
        else:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

    # Validate total tokens (input + max_new_tokens)
    max_new_tokens = obj.sampling_params.get("max_new_tokens")
    if (
        self.validate_total_tokens
        and max_new_tokens is not None
        and (max_new_tokens + input_token_num) > _max_req_len
    ):
        if self.allow_auto_truncate:
            logger.warning(
                f"Requested token count ({input_token_num} input + {max_new_tokens} new) "
                f"exceeds the model's context length ({self.context_len} tokens). "
                "Truncating max_new_tokens."
            )
            obj.sampling_params["max_new_tokens"] = max(
                0, _max_req_len - input_token_num
            )
        else:
            total_tokens = max_new_tokens + input_token_num
            error_msg = (
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{max_new_tokens} tokens for the completion. Please reduce the number "
                f"of tokens in the input messages or the completion to fit within the limit."
            )
            raise ValueError(error_msg)

    # Reject exact-lane request transforms before they can trip the
    # batch-level, scheduler-fatal sampler assertions on the GPU.
    exact_qwen35 = is_qwen35_gdn_exact_mode(self.server_args)
    exact_qwen3 = is_qwen3_dense_exact_mode(self.server_args)
    exact_glm = is_glm52_exact_mode(self.server_args)
    exact_dsv4 = is_dsv4_flash_exact_mode(self.server_args)
    if (exact_qwen35 or exact_qwen3 or exact_glm or exact_dsv4) and isinstance(
        obj, GenerateReqInput
    ):
        violations = _get_bi_decode_strict_ingress_violations(
            obj,
            self.preferred_sampling_params,
            sampled_logprob_only=exact_glm or exact_qwen3,
        )
        if violations:
            contract_name = (
                "GLM-5.2"
                if exact_glm
                else "dense Qwen3" if exact_qwen3 else "Qwen3.5-family"
            )
            if exact_dsv4:
                contract_name = "DSV4-Flash"
            raise ValueError(
                f"This server runs the exact {contract_name} RL on-policy "
                "decode contract; requests must use a finite non-negative "
                "temperature (zero selects greedy decoding) "
                "sampling without penalties, grammar, "
                "logit bias, custom processors, or MTP (incompatible "
                "fields: " + ", ".join(violations) + ")"
            )

    # Validate embedding requests
    if isinstance(obj, EmbeddingReqInput) and self.is_generation:
        raise ValueError(
            "This model does not appear to be an embedding model by default. "
            "Please add `--is-embedding` when launching the server or try another model."
        )

    # Validate Matryoshka embeddings
    if isinstance(obj, EmbeddingReqInput):
        self._validate_for_matryoshka_dim(obj)

    # Validate generation-specific fields
    if isinstance(obj, GenerateReqInput):
        self._validate_token_ids_logprob(obj)
        requested_hidden_mode = get_request_return_hidden_states_mode(
            obj.return_hidden_states
        )
        server_hidden_mode = get_server_return_hidden_states_mode(self.server_args)
        if requested_hidden_mode > server_hidden_mode:
            if server_hidden_mode.need_capture():
                raise ValueError(
                    "The requested return_hidden_states mode exceeds the "
                    f"server maximum `{self.server_args.return_hidden_states_mode}`. "
                    "Please launch with `--return-hidden-states-mode full` "
                    "to allow return_hidden_states=True."
                )
            raise ValueError(
                "The server is not configured to return hidden states. "
                "Please set `--return-hidden-states-mode last`, "
                "`--return-hidden-states-mode full`, or the legacy "
                "`--enable-return-hidden-states` flag."
            )
        if (
            obj.custom_logit_processor
            and not self.server_args.enable_custom_logit_processor
        ):
            raise ValueError(
                "The server is not configured to enable custom logit processor. "
                "Please set `--enable-custom-logit-processor` to enable this feature."
            )

def _TokenizerManager__add_logprob_to_meta_info(
    self,
    meta_info: dict,
    state: ReqState,
    top_logprobs_num: int,
    token_ids_logprob: List[int],
    return_text_in_logprobs: bool,
):
    # 1. Handle regular logprobs
    if len(state.input_token_logprobs_val) > len(state.input_token_logprobs):
        state.input_token_logprobs.extend(
            self.detokenize_logprob_tokens(
                state.input_token_logprobs_val[len(state.input_token_logprobs) :],
                state.input_token_logprobs_idx[len(state.input_token_logprobs) :],
                return_text_in_logprobs,
            )
        )

    if len(state.output_token_logprobs_val) > len(state.output_token_logprobs):
        state.output_token_logprobs.extend(
            self.detokenize_logprob_tokens(
                state.output_token_logprobs_val[len(state.output_token_logprobs) :],
                state.output_token_logprobs_idx[len(state.output_token_logprobs) :],
                return_text_in_logprobs,
            )
        )

    meta_info["input_token_logprobs"] = state.input_token_logprobs
    meta_info["output_token_logprobs"] = state.output_token_logprobs
    meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs)
    if state.obj.return_raw_token_logprobs_b64:
        meta_info.update(
            _build_raw_token_logprobs_b64_fields(
                state.input_token_logprobs_val,
                state.output_token_logprobs_val,
            )
        )

    # 2. Handle top logprobs
    if top_logprobs_num > 0:
        # Guarded by the caller's return_logprob check, so obj is a
        # GenerateReqInput here.
        use_flat = state.obj.return_flat_raw_top_logprobs
        if use_flat and state.input_top_logprobs_scheduler_flat is not None:
            # The scheduler already assembled the flat arrays (sent once
            # at prefill completion); encode them directly.
            if state.input_top_logprobs_flat_fields is None:
                val_arr, idx_arr, null_prefix = (
                    state.input_top_logprobs_scheduler_flat
                )
                state.input_top_logprobs_flat_fields = (
                    _build_flat_input_top_logprobs_fields_from_arrays(
                        val_arr,
                        idx_arr,
                        null_prefix,
                        return_b64=state.obj.return_flat_raw_top_logprobs_b64,
                    )
                )
            meta_info.update(state.input_top_logprobs_flat_fields)
        elif use_flat:
            # Flat replaces nested for the input side only.
            if state.input_top_logprobs_flat_num_rows != len(
                state.input_top_logprobs_val
            ):
                try:
                    state.input_top_logprobs_flat_fields = (
                        _build_flat_input_top_logprobs_fields(
                            state.input_top_logprobs_val,
                            state.input_top_logprobs_idx,
                            top_logprobs_num,
                            return_b64=state.obj.return_flat_raw_top_logprobs_b64,
                        )
                    )
                except ValueError as e:
                    # A raise here would disrupt unrelated requests in the
                    # shared batch-output loop; degrade to nested instead.
                    state.input_top_logprobs_flat_fields = None
                    logger.error(
                        "Falling back to nested input top logprobs for "
                        "rid=%s: %s",
                        meta_info.get("id"),
                        e,
                    )
                state.input_top_logprobs_flat_num_rows = len(
                    state.input_top_logprobs_val
                )
            if state.input_top_logprobs_flat_fields is not None:
                meta_info.update(state.input_top_logprobs_flat_fields)
            else:
                use_flat = False
        if not use_flat:
            if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):
                state.input_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.input_top_logprobs_val[
                            len(state.input_top_logprobs) :
                        ],
                        state.input_top_logprobs_idx[
                            len(state.input_top_logprobs) :
                        ],
                        return_text_in_logprobs,
                    )
                )
            meta_info["input_top_logprobs"] = state.input_top_logprobs
        if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
            state.output_top_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    state.output_top_logprobs_val[len(state.output_top_logprobs) :],
                    state.output_top_logprobs_idx[len(state.output_top_logprobs) :],
                    return_text_in_logprobs,
                )
            )
        meta_info["output_top_logprobs"] = state.output_top_logprobs

    # 3. Handle token_ids_logprob
    if token_ids_logprob is not None:
        if len(state.input_token_ids_logprobs_val) > len(
            state.input_token_ids_logprobs
        ):
            state.input_token_ids_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    state.input_token_ids_logprobs_val[
                        len(state.input_token_ids_logprobs) :
                    ],
                    state.input_token_ids_logprobs_idx[
                        len(state.input_token_ids_logprobs) :
                    ],
                    return_text_in_logprobs,
                )
            )
        if len(state.output_token_ids_logprobs_val) > len(
            state.output_token_ids_logprobs
        ):
            state.output_token_ids_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    state.output_token_ids_logprobs_val[
                        len(state.output_token_ids_logprobs) :
                    ],
                    state.output_token_ids_logprobs_idx[
                        len(state.output_token_ids_logprobs) :
                    ],
                    return_text_in_logprobs,
                )
            )

        meta_info["input_token_ids_logprobs"] = state.input_token_ids_logprobs
        meta_info["output_token_ids_logprobs"] = state.output_token_ids_logprobs


def __apply_patch__(mod):
    # Publish the twin's top-level imports onto mod: in-tree they were the
    # srt file's own module globals, and rebound copies resolve via mod.
    mod.dataclasses = dataclasses
    mod.math = math
    # Deferred: the finder imports twins under bypass(), so sglang imports at
    # twin top level would cache modules UNPATCHED. Import here (bypass off)
    # and publish onto mod -- in-tree these were the file's module globals.
    from sglang.srt.server_args import (
        PortArgs,
        ServerArgs,
        is_dsv4_flash_exact_mode,
        is_glm52_exact_mode,
        is_qwen3_dense_exact_mode,
        is_qwen35_gdn_exact_mode,
        set_global_server_args_for_tokenizer,
    )
    for _n, _v in list(locals().items()):
        if _n != "mod":
            setattr(mod, _n, _v)
    mod._build_raw_token_logprobs_b64_fields = rebind(_build_raw_token_logprobs_b64_fields, mod)
    mod._get_bi_decode_strict_ingress_violations = rebind(_get_bi_decode_strict_ingress_violations, mod)
    mod.ReqState = rebind(ReqState, mod)
    mod.TokenizerManager._create_tokenized_object = rebind(_TokenizerManager___create_tokenized_object, mod, name="_create_tokenized_object")
    mod.TokenizerManager._handle_batch_output = rebind(_TokenizerManager___handle_batch_output, mod, name="_handle_batch_output")
    mod.TokenizerManager._validate_one_request = rebind(_TokenizerManager___validate_one_request, mod, name="_validate_one_request")
    mod.TokenizerManager.add_logprob_to_meta_info = rebind(_TokenizerManager__add_logprob_to_meta_info, mod, name="add_logprob_to_meta_info")
