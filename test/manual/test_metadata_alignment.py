import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import BatchStrOutput
from sglang.srt.managers.multi_tokenizer_mixin import _extract_field_by_index
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestTokenizerManagerMetadataAlignment(unittest.TestCase):
    def setUp(self):
        with patch("sglang.srt.utils.get_device", return_value="cpu"):
            self.server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
            self.port_args = PortArgs.init_new(self.server_args)

        with patch("zmq.asyncio.Context"), patch(
            "sglang.srt.utils.get_zmq_socket"
        ), patch(
            "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
        ) as mock_tokenizer:
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def _make_state(self, rid: str, return_routed_experts: bool) -> ReqState:
        obj = SimpleNamespace(
            rid=rid,
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=[],
            return_text_in_logprobs=False,
            stream=False,
            return_hidden_states=False,
            return_routed_experts=return_routed_experts,
            return_expert_logits=False,
            log_metrics=False,
            lora_path=None,
        )
        return ReqState(
            out_list=[],
            finished=False,
            event=Mock(),
            obj=obj,
            created_time=0.0,
        )

    def test_handle_batch_output_tolerates_short_optional_metadata(self):
        rid_without_routing = "rid-without-routing"
        rid_with_routing = "rid-with-routing"
        self.tokenizer_manager.rid_to_state[rid_without_routing] = self._make_state(
            rid_without_routing, return_routed_experts=False
        )
        self.tokenizer_manager.rid_to_state[rid_with_routing] = self._make_state(
            rid_with_routing, return_routed_experts=True
        )

        recv_obj = BatchStrOutput(
            rids=[rid_without_routing, rid_with_routing],
            http_worker_ipcs=[None, None],
            queue_time=[0.0, 0.0],
            forward_entry_time=[0.0, 0.0],
            prefill_launch_delay=[0.0, 0.0],
            prefill_launch_latency=[0.0, 0.0],
            prefill_finished_ts=[0.0, 0.0],
            spec_verify_ct=[0, 0],
            spec_accepted_tokens=[0, 0],
            finished_reasons=[None, None],
            output_strs=["a", "b"],
            output_ids=[[1], [2]],
            prompt_tokens=[1, 1],
            completion_tokens=[1, 1],
            cached_tokens=[0, 0],
            input_token_logprobs_val=[[], []],
            input_token_logprobs_idx=[[], []],
            output_token_logprobs_val=[[], []],
            output_token_logprobs_idx=[[], []],
            input_top_logprobs_val=[[], []],
            input_top_logprobs_idx=[[], []],
            output_top_logprobs_val=[[], []],
            output_top_logprobs_idx=[[], []],
            input_token_ids_logprobs_val=[[], []],
            input_token_ids_logprobs_idx=[[], []],
            output_token_ids_logprobs_val=[[], []],
            output_token_ids_logprobs_idx=[[], []],
            output_token_entropy_val=[None, None],
            output_hidden_states=None,
            output_routed_experts=["only-first-item"],
            output_expert_logits=None,
            placeholder_tokens_idx=[None, None],
            placeholder_tokens_val=[None, None],
            retraction_counts=[0, 0],
            customized_info=None,
        )

        # Should not raise even when output_routed_experts is shorter than rids.
        self.tokenizer_manager._handle_batch_output(recv_obj)

        state_without_routing = self.tokenizer_manager.rid_to_state[rid_without_routing]
        state_with_routing = self.tokenizer_manager.rid_to_state[rid_with_routing]

        self.assertEqual(len(state_without_routing.out_list), 1)
        self.assertEqual(len(state_with_routing.out_list), 1)

        meta_without_routing = state_without_routing.out_list[0]["meta_info"]
        meta_with_routing = state_with_routing.out_list[0]["meta_info"]
        self.assertNotIn("routed_experts", meta_without_routing)
        self.assertNotIn("routed_experts", meta_with_routing)


class TestExtractFieldByIndex(unittest.TestCase):
    def test_dict_field_returns_none_for_missing_index(self):
        output = SimpleNamespace(customized_info={"a": [1]})
        self.assertEqual(
            _extract_field_by_index(output, "customized_info", 5, check_length=False),
            {"a": None},
        )

    def test_list_field_returns_none_placeholder_when_index_missing(self):
        output = SimpleNamespace(output_routed_experts=["x"])
        self.assertEqual(
            _extract_field_by_index(
                output, "output_routed_experts", 3, check_length=False
            ),
            [None],
        )

    def test_list_field_returns_none_with_check_length(self):
        output = SimpleNamespace(output_routed_experts=["x"])
        self.assertIsNone(
            _extract_field_by_index(
                output, "output_routed_experts", 3, check_length=True
            )
        )


if __name__ == "__main__":
    unittest.main()
