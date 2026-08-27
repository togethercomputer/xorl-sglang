"""Override twin of ``sglang.srt.managers.io_struct`` -- xorl exact serving.

Zero-srt port of PR #41: adds ``return_raw_token_logprobs_b64`` to
``GenerateReqInput``. Dataclass fields are fixed at class creation, so the
twin replaces the class with a subclass that appends the field (kwargs-only
construction everywhere makes the position change harmless) and carries the
two methods the port edits. The class is defined at module level so pickle
across the tokenizer/scheduler ZMQ boundary resolves it by import path --
every process activates the overlay via ``import sglang``.

Replaced upstream methods are pinned in ``sglang.overrides._twin_pins``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.managers.io_struct import GenerateReqInput as _UpstreamGenerateReqInput


@dataclass
class GenerateReqInput(_UpstreamGenerateReqInput):
    # Return the selected-token input/output logprob buffers as contiguous
    # little-endian FP32 bytes.  This is an evidence surface for exact replay:
    # unlike top-k output it always includes the sampled/teacher-forced token.
    return_raw_token_logprobs_b64: bool = False

    def _validate_inputs(self):
        """Validate that the input configuration is valid."""
        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )
        if (
            self.return_flat_raw_top_logprobs
            and self.multi_item_delimiter_indices is not None
        ):
            raise ValueError(
                "return_flat_raw_top_logprobs does not support multi-item "
                "scoring: delimiter-sparse top logprob rows have no contiguous "
                "position mapping."
            )
        if (
            self.return_flat_raw_top_logprobs_b64
            and not self.return_flat_raw_top_logprobs
        ):
            raise ValueError(
                "return_flat_raw_top_logprobs_b64 requires return_flat_raw_top_logprobs."
            )
        if self.return_raw_token_logprobs_b64:
            return_logprob_enabled = self.return_logprob is True or (
                isinstance(self.return_logprob, list)
                and bool(self.return_logprob)
                and all(value is True for value in self.return_logprob)
            )
            if not return_logprob_enabled:
                raise ValueError(
                    "return_raw_token_logprobs_b64 requires return_logprob=true "
                    "for every request row."
                )

    def __getitem__(self, i):
        # Cache sub-objects so that repeated obj[i] calls return the same instance.
        # This avoids subtle bugs where different call sites get divergent objects.
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]
        logical_index = i % self.batch_size
        sub = GenerateReqInput(
            rid=self.rid[logical_index],
            session_id=self.session_id,
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
            audio_data=self.audio_data[i],
            sampling_params=self.sampling_params[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_sampling_mask=self.return_sampling_mask[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            return_raw_token_logprobs_b64=self.return_raw_token_logprobs_b64,
            return_flat_raw_top_logprobs=self.return_flat_raw_top_logprobs,
            return_flat_raw_top_logprobs_b64=self.return_flat_raw_top_logprobs_b64,
            stream=self.stream,
            log_metrics=self.log_metrics,
            return_hidden_states=(
                self.return_hidden_states[i]
                if isinstance(self.return_hidden_states, list)
                else self.return_hidden_states
            ),
            return_routed_experts=self.return_routed_experts,
            routed_experts_start_len=self.routed_experts_start_len,
            return_indexer_topk=self.return_indexer_topk,
            modalities=self.modalities[i] if self.modalities else None,
            session_params=self.session_params,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            positional_embed_overrides=self._get_positional_embed_overrides_item(i),
            # If `__getitem__` is called, these bootstrap fields must be lists.
            bootstrap_host=(
                self.bootstrap_host[i] if self.bootstrap_host is not None else None
            ),
            bootstrap_port=(
                self.bootstrap_port[i] if self.bootstrap_port is not None else None
            ),
            bootstrap_room=(
                self.bootstrap_room[i] if self.bootstrap_room is not None else None
            ),
            bootstrap_pair_key=(
                self.bootstrap_pair_key[i]
                if self.bootstrap_pair_key is not None
                else None
            ),
            decode_tp_size=(
                self.decode_tp_size[i] if self.decode_tp_size is not None else None
            ),
            routed_dp_rank=self.routed_dp_rank,
            disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,
            conversation_id=self.conversation_id,
            http_worker_ipc=self.http_worker_ipc,
            require_reasoning=self.require_reasoning,
            max_thinking_tokens=self.max_thinking_tokens,
            priority=self.priority,
            extra_key=self.extra_key[i] if self.extra_key is not None else None,
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            return_prompt_token_ids=self.return_prompt_token_ids,
            external_trace_header=self.external_trace_header,
            received_time=self.received_time,
            multi_item_delimiter_indices=(
                self.multi_item_delimiter_indices[i]
                if self.multi_item_delimiter_indices is not None
                else None
            ),
        )
        cache[i] = sub
        return sub


def __apply_patch__(mod):
    g = globals()
    for _k, _v in vars(mod).items():
        g.setdefault(_k, _v)
    mod.GenerateReqInput = GenerateReqInput
