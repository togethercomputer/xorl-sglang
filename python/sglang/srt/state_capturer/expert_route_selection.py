"""Causal partitioning of cached expert-route rows into input / output halves.

Row semantics
-------------
The routed-expert sidecar (:mod:`sglang.srt.state_capturer.base`) stores one
route row per KV token slot: the row at ``req_to_token[req_pool_idx][p]`` holds
the expert IDs chosen by the forward pass that *consumed* token ``p``.  That
forward produced the logits which predicted token ``p + 1`` -- so a row belongs
to the forward position that *predicts the next token*, never to the token it
predicts.

The forward at ``p == seqlen - 1`` never runs: generation had already stopped,
so the final token was never fed back in.  Valid rows are therefore
``p in [0, seqlen - 1)`` -- ``seqlen - 1`` rows.  That matches the legacy
full-history payload (``base.get_topk`` slices ``[start_len : seqlen - 1]``) and
the R3 consumer contract, which asserts ``len(routes) == len(tokens) - 1``.

Partition
---------
With ``prompt_len`` prompt tokens and ``seqlen == prompt_len + output_len``:

* input  rows: ``p in [0, prompt_len - 1)``          -> ``max(0, prompt_len - 1)``
* output rows: ``p in [prompt_len - 1, seqlen - 1)``  -> ``output_len``

The boundary sits at ``prompt_len - 1``, not ``prompt_len``.  That is the split
that provably aligns with the token IDs and logprobs the same response returns:

* Input logprobs carry one ``None`` at position 0 plus ``prompt_len - 1`` real
  values.  The logprob of prompt token ``i`` (for ``i in [1, prompt_len)``) comes
  from the forward at ``p = i - 1``, i.e. from forwards ``[0, prompt_len - 1)``.
* Output logprobs carry ``output_len`` values.  The logprob of output token ``j``
  comes from the forward at ``p = prompt_len - 1 + j``, i.e. from forwards
  ``[prompt_len - 1, seqlen - 1)``.

The two ranges are disjoint and exactly tile ``[0, seqlen - 1)``, so
``concat(input_rows, output_rows)`` reproduces the legacy full-history tensor
row for row, with no duplicated boundary row.

Layer axis
----------
The row's layer axis spans *every* decoder layer, not just the MoE ones: the
host buffer is allocated with ``num_hidden_layers`` planes and dense layers
never call ``capture()``, so their planes stay zero -- indistinguishable on the
wire from a genuine expert id 0.  :class:`ExpertRouteSchema` therefore carries
``moe_layer_ids``, the layer indices that actually wrote rows, so a consumer can
map a plane index to a model layer instead of guessing.

Expert-ID space
---------------
Capture runs at ``layers/moe/topk.py:capture_routed_experts_if_allowed``, which
is invoked *before* ``topk_ids_logical_to_physical``.  The cached IDs are
therefore canonical logical/global expert IDs, stable across expert-parallel
placements and EPLB rebalances -- which is what replay needs.  ``id_space``
records that contract explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.state_capturer.base import BaseTopkCapturer

# Wire contract shared with the legacy `routed_experts` payload. Consumers
# decode base64 -> int32 -> reshape(num_rows, num_layers, top_k).
EXPERT_ID_WIRE_DTYPE = "int32"
EXPERT_ID_WIRE_LAYOUT = "row_major"
# Capture precedes the logical->physical remap, so IDs are model-global.
EXPERT_ID_SPACE_LOGICAL_GLOBAL = "logical_global"


class ExpertRouteRange(msgspec.Struct, frozen=True):
    """Half-open range of absolute forward positions ``[start, end)``."""

    start: int
    end: int

    @property
    def num_rows(self) -> int:
        return max(0, self.end - self.start)


class ExpertRouteSchema(msgspec.Struct, frozen=True):
    """Everything a consumer needs to reshape and interpret the payloads.

    One schema covers both partitions: the layer/top_k/dtype axes are identical
    for input and output rows, while the per-partition row count and absolute
    start position differ.  ``*_num_rows`` is ``None`` for a partition that was
    not requested, which is what distinguishes "not asked for" from "asked for
    and legitimately empty" (a 1-token prompt has zero input rows).
    """

    num_layers: int
    top_k: int
    moe_layer_ids: List[int]
    dtype: str = EXPERT_ID_WIRE_DTYPE
    layout: str = EXPERT_ID_WIRE_LAYOUT
    id_space: str = EXPERT_ID_SPACE_LOGICAL_GLOBAL
    input_num_rows: Optional[int] = None
    input_start_position: Optional[int] = None
    output_num_rows: Optional[int] = None
    output_start_position: Optional[int] = None


class ExpertRouteResult(msgspec.Struct, omit_defaults=True):
    """Typed per-request selection result.

    Holds only the partitions that were requested; the large row tensors are
    never duplicated into a second internal object.
    """

    schema: ExpertRouteSchema
    input_rows: Optional[torch.Tensor] = None
    output_rows: Optional[torch.Tensor] = None


def causal_partition(
    *, prompt_len: int, seqlen: int
) -> Tuple[ExpertRouteRange, ExpertRouteRange]:
    """Split the valid route rows ``[0, seqlen - 1)`` into input / output halves.

    See the module docstring for the derivation. Returns ``(input, output)``
    ranges in absolute forward-position space; either may be empty.
    """
    if prompt_len < 1:
        raise ValueError(f"{prompt_len=} must be at least 1")
    if seqlen < prompt_len:
        raise ValueError(f"{seqlen=} must be at least {prompt_len=}")

    # The last valid row is the forward at seqlen - 2; the forward at seqlen - 1
    # never ran because generation had already stopped.
    last_row_end = seqlen - 1
    # Boundary: the forward at prompt_len - 1 predicted the *first output token*,
    # so it belongs to the output partition.
    boundary = prompt_len - 1
    return (
        ExpertRouteRange(start=0, end=boundary),
        ExpertRouteRange(start=boundary, end=last_row_end),
    )


def select_expert_routes(
    *,
    capturer: BaseTopkCapturer,
    req_to_token_pool: ReqToTokenPool,
    req_pool_idx: int,
    prompt_len: int,
    seqlen: int,
    want_input: bool,
    want_output: bool,
) -> ExpertRouteResult:
    """Gather the requested causal partitions for one finished request.

    Each partition is gathered independently, so an input-only or output-only
    request touches only its own slice of ``req_to_token`` and only its own host
    rows -- no full-history copy is materialized to be thrown away.

    Must be called after the forward's D2H capture has been committed
    (``copy_done.synchronize()`` then ``TopkCaptureOutput.finalize()``) and
    before the request's KV slots are released, so every row read belongs to
    this request's own tokens.
    """
    if not (want_input or want_output):
        raise ValueError("select_expert_routes called with neither partition requested")

    input_range, output_range = causal_partition(prompt_len=prompt_len, seqlen=seqlen)

    input_rows = None
    output_rows = None
    if want_input:
        input_rows = capturer.get_rows(
            req_pool_idx=req_pool_idx,
            start=input_range.start,
            end=input_range.end,
            req_to_token_pool=req_to_token_pool,
        )
    if want_output:
        output_rows = capturer.get_rows(
            req_pool_idx=req_pool_idx,
            start=output_range.start,
            end=output_range.end,
            req_to_token_pool=req_to_token_pool,
        )

    schema = ExpertRouteSchema(
        num_layers=capturer.num_layers,
        top_k=capturer.topk_size,
        moe_layer_ids=capturer.captured_layer_ids,
        input_num_rows=input_range.num_rows if want_input else None,
        input_start_position=input_range.start if want_input else None,
        output_num_rows=output_range.num_rows if want_output else None,
        output_start_position=output_range.start if want_output else None,
    )
    result = ExpertRouteResult(
        schema=schema, input_rows=input_rows, output_rows=output_rows
    )
    validate_result_shapes(result=result)
    return result


def validate_result_shapes(*, result: ExpertRouteResult) -> None:
    """Fail loudly if a gathered tensor disagrees with its advertised schema.

    A silent shape drift here would hand a replay consumer rows that reshape
    without error but are misaligned by whole tokens, so this is an assertion
    on the response path rather than a warning.
    """
    schema = result.schema
    for name, rows, expected_rows in (
        ("input", result.input_rows, schema.input_num_rows),
        ("output", result.output_rows, schema.output_num_rows),
    ):
        if rows is None:
            if expected_rows is not None:
                raise ValueError(
                    f"{name}_expert_ids: schema advertises {expected_rows} rows "
                    "but no tensor was gathered"
                )
            continue
        if expected_rows is None:
            raise ValueError(
                f"{name}_expert_ids: gathered a tensor but the schema does not "
                "advertise this partition"
            )
        actual = tuple(rows.shape)
        expected = (expected_rows, schema.num_layers, schema.top_k)
        if actual != expected:
            raise ValueError(f"{name}_expert_ids shape {actual} != expected {expected}")
