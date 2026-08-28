"""Unit tests for srt/state_capturer/expert_route_selection -- partition math.

The causal boundary is the whole point of this module: a route row belongs to
the forward position that predicts the *next* token, so the prompt/output split
sits at ``prompt_len - 1``, not ``prompt_len``.  An "obviously equivalent"
rewrite that moves the boundary by one silently hands replay consumers rows
misaligned by a whole token, which no shape check would catch.
"""

import unittest
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.srt.state_capturer.expert_route_selection import (
    EXPERT_ID_SPACE_LOGICAL_GLOBAL,
    EXPERT_ID_WIRE_DTYPE,
    EXPERT_ID_WIRE_LAYOUT,
    ExpertRouteResult,
    ExpertRouteSchema,
    causal_partition,
    validate_result_shapes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCausalPartition(CustomTestCase):
    def test_partition_tiles_the_legacy_row_range(self):
        """The two halves must be disjoint and exactly cover ``[0, seqlen - 1)``.

        That identity is what makes ``concat(input, output)`` reproduce the
        legacy full-history tensor; a gap or an overlap here would show up
        downstream as dropped or duplicated tokens.
        """
        for prompt_len in range(1, 12):
            for output_len in range(0, 12):
                seqlen = prompt_len + output_len
                with self.subTest(prompt_len=prompt_len, output_len=output_len):
                    inp, out = causal_partition(prompt_len=prompt_len, seqlen=seqlen)
                    self.assertEqual(inp.start, 0)
                    self.assertEqual(inp.end, out.start, "halves must not gap/overlap")
                    self.assertEqual(out.end, seqlen - 1)
                    self.assertEqual(
                        inp.num_rows + out.num_rows,
                        seqlen - 1,
                        "concat must reproduce the legacy row count",
                    )

    def test_boundary_row_belongs_to_the_output_half(self):
        """The forward at ``prompt_len - 1`` predicted the *first output token*.

        Placing it in the input half is the natural-looking off-by-one this
        module exists to prevent: it would leave the output half one row short
        of the generated tokens and misalign every output row by one.
        """
        inp, out = causal_partition(prompt_len=5, seqlen=8)
        self.assertEqual(inp.num_rows, 4, "input half is prompt_len - 1 rows")
        self.assertEqual(out.start, 4, "boundary row starts the output half")
        self.assertEqual(out.num_rows, 3, "output half is one row per output token")

    def test_output_half_has_exactly_one_row_per_output_token(self):
        """Row-per-output-token is the contract replay consumers index against."""
        for prompt_len, output_len in ((1, 0), (1, 1), (1, 5), (7, 0), (7, 4)):
            with self.subTest(prompt_len=prompt_len, output_len=output_len):
                _, out = causal_partition(
                    prompt_len=prompt_len, seqlen=prompt_len + output_len
                )
                self.assertEqual(out.num_rows, output_len)

    def test_single_token_prompt_has_no_input_rows(self):
        """A 1-token prompt has no predecessor to predict it, so 0 input rows.

        This is the length-0/length-1 boundary: "requested and legitimately
        empty" must stay distinguishable from "not requested", which is why the
        schema carries a row count of 0 rather than None.
        """
        inp, out = causal_partition(prompt_len=1, seqlen=1)
        self.assertEqual((inp.num_rows, out.num_rows), (0, 0))

        inp, out = causal_partition(prompt_len=1, seqlen=2)
        self.assertEqual((inp.num_rows, out.num_rows), (0, 1))

    def test_rejects_degenerate_lengths(self):
        """Negative-branch contract: a caller must not be able to ask for a
        partition of a sequence that cannot exist."""
        with self.assertRaises(ValueError):
            causal_partition(prompt_len=0, seqlen=4)
        with self.assertRaises(ValueError):
            causal_partition(prompt_len=5, seqlen=4)


class TestExpertRouteSchema(CustomTestCase):
    def test_wire_contract_constants(self):
        """These three strings are the published wire contract, not internal
        naming: consumers decode base64 -> int32 -> reshape, and treat the IDs
        as model-global. Changing a value here silently breaks every existing
        replay consumer, so the literals are pinned."""
        self.assertEqual(EXPERT_ID_WIRE_DTYPE, "int32")
        self.assertEqual(EXPERT_ID_WIRE_LAYOUT, "row_major")
        self.assertEqual(EXPERT_ID_SPACE_LOGICAL_GLOBAL, "logical_global")

    def test_schema_states_the_contract_even_at_default_values(self):
        """The metadata must be self-describing on the wire. If these fields
        were omitted when they hold their defaults, a consumer reading only the
        response could not tell the dtype or the ID space."""
        import msgspec

        encoded = msgspec.to_builtins(
            ExpertRouteSchema(num_layers=4, top_k=2, moe_layer_ids=[2, 3])
        )
        self.assertEqual(encoded["dtype"], "int32")
        self.assertEqual(encoded["layout"], "row_major")
        self.assertEqual(encoded["id_space"], "logical_global")

    def test_moe_layer_ids_disambiguates_dense_layers(self):
        """Dense layers occupy zero-filled planes indistinguishable from expert
        id 0. ``moe_layer_ids`` is the only thing that maps a plane index back
        to a model layer, so a model with leading dense layers must still be
        reconstructable."""
        schema = ExpertRouteSchema(num_layers=5, top_k=2, moe_layer_ids=[1, 2, 3, 4])
        self.assertEqual(schema.num_layers, 5)
        self.assertNotIn(0, schema.moe_layer_ids)
        self.assertEqual(len(schema.moe_layer_ids), 4)


class TestValidateResultShapes(CustomTestCase):
    @staticmethod
    def _rows(n, layers=4, top_k=2):
        return torch.zeros((n, layers, top_k), dtype=torch.int32)

    def test_accepts_matching_shapes(self):
        """Negative branch: a well-formed result must pass.

        The guard here is the *absence* of an exception -- it catches a
        predicate that degrades to always-raise, which every other case in this
        class (all of which expect a raise) would happily pass.
        """
        result = ExpertRouteResult(
            schema=ExpertRouteSchema(
                num_layers=4,
                top_k=2,
                moe_layer_ids=[0, 1, 2, 3],
                input_num_rows=3,
                output_num_rows=2,
            ),
            input_rows=self._rows(3),
            output_rows=self._rows(2),
        )
        try:
            validate_result_shapes(result=result)
        except ValueError as exc:  # pragma: no cover - only on regression
            self.fail(f"validator rejected a well-formed result: {exc}")

    def test_rejects_row_count_drift(self):
        """A gathered tensor that disagrees with its advertised row count still
        reshapes cleanly on the client, so it must fail here or not at all."""
        with self.assertRaises(ValueError):
            validate_result_shapes(
                result=ExpertRouteResult(
                    schema=ExpertRouteSchema(
                        num_layers=4,
                        top_k=2,
                        moe_layer_ids=[0],
                        output_num_rows=2,
                    ),
                    output_rows=self._rows(3),
                )
            )

    def test_rejects_layer_or_topk_drift(self):
        with self.assertRaises(ValueError):
            validate_result_shapes(
                result=ExpertRouteResult(
                    schema=ExpertRouteSchema(
                        num_layers=4,
                        top_k=2,
                        moe_layer_ids=[0],
                        input_num_rows=3,
                    ),
                    input_rows=self._rows(3, layers=8),
                )
            )

    def test_rejects_schema_payload_disagreement(self):
        """Both negative branches: advertising a partition without a tensor,
        and carrying a tensor the schema does not advertise. Either would leave
        a consumer unable to tell an empty partition from a missing one."""
        with self.assertRaises(ValueError):
            validate_result_shapes(
                result=ExpertRouteResult(
                    schema=ExpertRouteSchema(
                        num_layers=4, top_k=2, moe_layer_ids=[0], input_num_rows=3
                    ),
                    input_rows=None,
                )
            )
        with self.assertRaises(ValueError):
            validate_result_shapes(
                result=ExpertRouteResult(
                    schema=ExpertRouteSchema(num_layers=4, top_k=2, moe_layer_ids=[0]),
                    input_rows=self._rows(3),
                )
            )


class TestExpertIdTransport(CustomTestCase):
    """Serialization bookkeeping: the new output fields must survive both IPC
    codecs and the base64 hop.

    The pre-existing `routed_experts` field is typed `Any`, which only works
    because pickle IPC is the default -- under msgpack it would decode to a
    plain list and lose its tensor-ness. The new fields are typed precisely to
    avoid that, and this pins the property so a later "simplify to Any" cannot
    regress it silently.
    """

    @staticmethod
    def _schema():
        return ExpertRouteSchema(
            num_layers=4,
            top_k=2,
            moe_layer_ids=[1, 2, 3],
            input_num_rows=3,
            input_start_position=0,
            output_num_rows=2,
            output_start_position=3,
        )

    def test_msgpack_ipc_round_trip(self):
        import msgspec

        import sglang.srt.managers.io_struct as io_struct

        class _Frame(msgspec.Struct, kw_only=True):
            """Mirrors the field types added to BatchTokenIDOutput."""

            input_expert_ids: Optional[List[Optional[torch.Tensor]]] = None
            expert_ids_schema: Optional[List[Optional[ExpertRouteSchema]]] = None

        rows = torch.arange(3 * 4 * 2, dtype=torch.int32).reshape(3, 4, 2)
        frame = _Frame(
            input_expert_ids=[rows, None], expert_ids_schema=[self._schema(), None]
        )
        encoder = msgspec.msgpack.Encoder(enc_hook=io_struct.enc_hook)
        decoder = msgspec.msgpack.Decoder(_Frame, dec_hook=io_struct.dec_hook)
        back = decoder.decode(encoder.encode(frame))

        self.assertEqual(back.expert_ids_schema, [self._schema(), None])
        self.assertTrue(torch.equal(back.input_expert_ids[0], rows))
        self.assertIsNone(
            back.input_expert_ids[1], "an opted-out request must stay None"
        )

    def test_base64_transport_distinguishes_absent_from_empty(self):
        """`None` (not requested) and `""` (requested, zero rows) must not
        collapse into each other -- a 1-token prompt legitimately has zero input
        rows, and a consumer has to tell that apart from a missing partition."""
        from sglang.srt.managers.detokenizer_manager import DetokenizerManager

        rows = torch.arange(3 * 4 * 2, dtype=torch.int32).reshape(3, 4, 2)
        encoded = DetokenizerManager._b64_encode_per_request(
            [rows, None, torch.empty((0, 4, 2), dtype=torch.int32)]
        )
        self.assertIsNone(encoded[1])
        self.assertEqual(encoded[2], "")

        import numpy as np
        import pybase64

        decoded = np.frombuffer(pybase64.b64decode(encoded[0]), dtype=np.int32).reshape(
            3, 4, 2
        )
        np.testing.assert_array_equal(decoded, rows.numpy())


class TestAttachExpertIdsToMetaInfo(CustomTestCase):
    """The response-path assembly that turns scheduler columns into meta_info.

    Two branches matter and neither is exercised elsewhere: the
    skip_tokenizer_init path hands this a raw tensor that must be encoded here
    (a tensor left in meta_info is not JSON-serializable), and an opted-out
    request must produce no key at all rather than a null one.
    """

    @staticmethod
    def _recv(**kwargs):
        """A stand-in for the batch-output columns the helper reads.

        The isinstance narrowing lives at the call site, so the helper only
        needs the three parallel columns.
        """
        columns = {
            "input_expert_ids": None,
            "output_expert_ids": None,
            "expert_ids_schema": None,
        }
        unknown = set(kwargs) - set(columns)
        assert not unknown, f"unexpected column(s): {unknown}"
        columns.update(kwargs)
        return SimpleNamespace(**columns)

    def _attach(self, recv, i=0):
        from sglang.srt.managers.tokenizer_manager import (
            _attach_expert_ids_to_meta_info,
        )

        meta = {}
        _attach_expert_ids_to_meta_info(meta, recv, i)
        return meta

    def test_opted_out_request_gets_no_keys(self):
        self.assertEqual(self._attach(self._recv()), {})

    def test_mixed_batch_column_leaves_opted_out_entries_absent(self):
        """Request 0 opted in, request 1 did not. Request 1 must get no key --
        not a null one -- so a client cannot mistake it for an empty result."""
        rows = torch.zeros((2, 4, 2), dtype=torch.int32)
        recv = self._recv(
            output_expert_ids=[rows, None],
            expert_ids_schema=[
                ExpertRouteSchema(
                    num_layers=4, top_k=2, moe_layer_ids=[0], output_num_rows=2
                ),
                None,
            ],
        )
        self.assertIn("output_expert_ids", self._attach(recv, i=0))
        self.assertEqual(self._attach(recv, i=1), {})

    def test_tensor_column_is_base64_encoded(self):
        """skip_tokenizer_init bypasses the detokenizer, so the tensor arrives
        here unencoded and must not be left in meta_info as a tensor."""
        import numpy as np
        import pybase64

        rows = torch.arange(2 * 4 * 2, dtype=torch.int32).reshape(2, 4, 2)
        meta = self._attach(self._recv(input_expert_ids=[rows]))
        self.assertIsInstance(meta["input_expert_ids"], str)
        decoded = np.frombuffer(
            pybase64.b64decode(meta["input_expert_ids"]), dtype=np.int32
        ).reshape(2, 4, 2)
        np.testing.assert_array_equal(decoded, rows.numpy())

    def test_pre_encoded_column_passes_through(self):
        """BatchStrOutput is already base64 from the detokenizer; re-encoding it
        would corrupt the payload."""
        meta = self._attach(self._recv(output_expert_ids=["AAAA"]))
        self.assertEqual(meta["output_expert_ids"], "AAAA")

    def test_empty_partition_is_reported_as_zero_rows_not_omitted(self):
        """A 1-token prompt legitimately has zero input rows. The key must be
        present with an empty payload and a row count of 0, so it stays
        distinguishable from a partition that was never requested."""
        schema = ExpertRouteSchema(
            num_layers=4, top_k=2, moe_layer_ids=[0], input_num_rows=0
        )
        meta = self._attach(
            self._recv(
                input_expert_ids=[torch.empty((0, 4, 2), dtype=torch.int32)],
                expert_ids_schema=[schema],
            )
        )
        self.assertEqual(meta["input_expert_ids"], "")
        self.assertEqual(meta["expert_ids_schema"]["input_num_rows"], 0)
        self.assertIsNone(meta["expert_ids_schema"]["output_num_rows"])

    def test_schema_is_serialized_to_builtins(self):
        """meta_info goes out as JSON, so the struct must be converted here."""
        schema = ExpertRouteSchema(
            num_layers=4, top_k=2, moe_layer_ids=[1, 2], output_num_rows=3
        )
        meta = self._attach(
            self._recv(output_expert_ids=["AAAA"], expert_ids_schema=[schema])
        )
        self.assertIsInstance(meta["expert_ids_schema"], dict)
        self.assertEqual(meta["expert_ids_schema"]["moe_layer_ids"], [1, 2])


if __name__ == "__main__":
    unittest.main()
