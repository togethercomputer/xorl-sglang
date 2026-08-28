"""Unit tests for the `return_*_expert_ids` request flags and their fail-closed
validation.

The flags are opt-in and default to false, and the server must refuse them
outright on any path where it cannot produce exact rows.  Returning missing,
zero, partial or stale routes instead would be worse than an error: a replay
consumer cannot tell a wrong route from a right one.
"""

import unittest
from array import array
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_VALIDATE = "sglang.srt.managers.scheduler.routed_experts_capture_enabled"
_MEMORY = "sglang.srt.managers.scheduler.get_memory"


class _StubParallelState:
    def __init__(self, pp_size=1):
        self.pp_size = pp_size


class _StubScheduler:
    """Just enough Scheduler surface for `_validate_expert_id_request`."""

    def __init__(self, disaggregation_mode=DisaggregationMode.NULL, pp_size=1):
        self.disaggregation_mode = disaggregation_mode
        self.ps = _StubParallelState(pp_size=pp_size)

    _validate_expert_id_request = Scheduler._validate_expert_id_request


class _StubMemory:
    def __init__(self, enable_hierarchical_cache=False):
        self.enable_hierarchical_cache = enable_hierarchical_cache


def _tokenized(**kwargs):
    """A real TokenizedGenerateReqInput -- the type the validator is handed."""
    defaults = dict(
        rid="r",
        input_text="hi",
        input_ids=array("i", [1, 2, 3]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )
    defaults.update(kwargs)
    return TokenizedGenerateReqInput(**defaults)


class TestExpertIdFlagNormalization(CustomTestCase):
    def test_flags_default_to_false(self):
        """Opt-in is the whole contract: a request that says nothing must get
        no expert-ID work and no expert-ID field."""
        req = GenerateReqInput(text="hi")
        req.normalize_batch_and_arguments()
        self.assertFalse(req.return_input_expert_ids)
        self.assertFalse(req.return_output_expert_ids)

    def test_flags_are_independent(self):
        req = GenerateReqInput(text="hi", return_output_expert_ids=True)
        req.normalize_batch_and_arguments()
        self.assertFalse(req.return_input_expert_ids)
        self.assertTrue(req.return_output_expert_ids)

    def test_flags_propagate_to_every_item_of_a_batch(self):
        """Batch splitting is where a per-request flag silently goes missing;
        each split item must carry the batch's setting."""
        req = GenerateReqInput(
            text=["a", "b", "c"],
            return_input_expert_ids=True,
            return_output_expert_ids=True,
        )
        req.normalize_batch_and_arguments()
        for i in range(3):
            self.assertTrue(req[i].return_input_expert_ids)
            self.assertTrue(req[i].return_output_expert_ids)

    def test_flags_survive_parallel_sampling_expansion(self):
        """Parallel sampling re-expands the batch after normalization; every
        expanded item must still carry the flag."""
        req = GenerateReqInput(
            text=["a", "b"],
            sampling_params=[{"n": 2}, {"n": 2}],
            return_input_expert_ids=True,
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.parallel_sample_num, 2)
        for i in range(req.batch_size * req.parallel_sample_num):
            self.assertTrue(req[i].return_input_expert_ids)
            self.assertFalse(req[i].return_output_expert_ids)


class TestExpertIdRequestValidation(CustomTestCase):
    def _validate(
        self,
        recv_req,
        capture_enabled=True,
        disaggregation_mode=DisaggregationMode.NULL,
        hierarchical_cache=False,
        pp_size=1,
    ):
        scheduler = _StubScheduler(
            disaggregation_mode=disaggregation_mode, pp_size=pp_size
        )
        with patch(_VALIDATE, return_value=capture_enabled), patch(
            _MEMORY, return_value=_StubMemory(hierarchical_cache)
        ):
            return scheduler._validate_expert_id_request(recv_req)

    def test_default_request_is_not_validated_at_all(self):
        """A request that opted out must pass even on a server with no capture
        capability -- the flags gate extraction, never ordinary serving."""
        self.assertIsNone(self._validate(_tokenized(), capture_enabled=False))

    def test_missing_capture_capability_is_rejected(self):
        """Without --enable-return-routed-experts nothing is captured, so the
        response would carry no rows at all. That must be an error, not an
        absent field the caller may not notice."""
        err = self._validate(
            _tokenized(return_input_expert_ids=True), capture_enabled=False
        )
        self.assertIsNotNone(err)
        self.assertIn("enable-return-routed-experts", err)

    def test_capture_capability_present_is_accepted(self):
        self.assertIsNone(
            self._validate(
                _tokenized(return_output_expert_ids=True), capture_enabled=True
            )
        )

    def test_disaggregation_is_rejected(self):
        """Under P/D the decode engine never runs the prompt forwards and the
        sidecar rows are not shipped with the KV cache, so neither side holds a
        complete causal history."""
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                err = self._validate(
                    _tokenized(return_input_expert_ids=True),
                    disaggregation_mode=mode,
                )
                self.assertIsNotNone(err)
                self.assertIn("disaggregation", err)

    def test_pipeline_parallelism_is_rejected(self):
        """Each pipeline stage owns only its slice of the decoder, so a rank's
        sidecar covers its own layers and every other plane stays zero --
        indistinguishable from expert id 0. The response would silently
        describe a fraction of the model."""
        err = self._validate(_tokenized(return_input_expert_ids=True), pp_size=2)
        self.assertIsNotNone(err)
        self.assertIn("pipeline parallelism", err)

    def test_single_stage_pipeline_is_accepted(self):
        """pp_size == 1 is the ordinary case and must not be caught by the
        pipeline guard."""
        self.assertIsNone(
            self._validate(_tokenized(return_input_expert_ids=True), pp_size=1)
        )

    def test_hierarchical_cache_is_rejected(self):
        """A host->device page restore fills fresh KV slots without running a
        forward, leaving the sidecar rows at those slots owned by the slot's
        previous occupant -- exactly the stale read this must prevent."""
        err = self._validate(
            _tokenized(return_output_expert_ids=True), hierarchical_cache=True
        )
        self.assertIsNotNone(err)
        self.assertIn("hierarchical-cache", err)

    def test_start_len_combination_is_rejected(self):
        """`routed_experts_start_len` crops the legacy single blob; the
        partitioned payloads carry their own absolute start positions. Honoring
        both at once would make the row-to-position mapping ambiguous."""
        err = self._validate(
            _tokenized(return_input_expert_ids=True, routed_experts_start_len=4)
        )
        self.assertIsNotNone(err)
        self.assertIn("routed_experts_start_len", err)

    def test_legacy_flag_alone_is_unaffected_by_the_new_validation(self):
        """Existing `return_routed_experts` consumers must not start failing on
        paths they use today (P/D, hierarchical cache, start_len)."""
        self.assertIsNone(
            self._validate(
                _tokenized(return_routed_experts=True, routed_experts_start_len=4),
                disaggregation_mode=DisaggregationMode.DECODE,
                hierarchical_cache=True,
                pp_size=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
