"""Override twin of ``sglang.srt.model_executor.forward_batch_info`` -- xorl exact serving.

Zero-srt port of PR #41: the three DSV4 exact-logits fields on
``ForwardBatch``. Dataclass fields are fixed at class creation, so the twin
replaces the class with a subclass appending them (all construction is
kwargs-based; dataclass copy/replace machinery sees them as real fields).

Note the upstream comment nuance the port carried: live batches carry
request-derived ``global_num_tokens_for_logprob_*`` counts; synthetic
decode/dummy buckets may carry fixed pruned-row counts; prefill capture must
leave them unset unless it separately computes post-pruning row counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch as _UpstreamForwardBatch,
)


@dataclass
class ForwardBatch(_UpstreamForwardBatch):
    # Exact DSV4 reconstructs CP-prefill rows into the logical DP-owner order
    # before logits pruning.  Carry that boundary explicitly so the TP8 head
    # never mistakes a physical CP shard for a complete owner block.
    #
    # Twin-added fields MUST default to None: upstream enumerations that
    # whitelist fields (TboForwardBatchPreparer.filter_batch) skip only
    # None-valued fields and hard-error on anything else it does not know.
    # None is falsy at every read site, so the tri-state is behaviorally
    # identical to the original False default.
    dsv4_exact_logits_rows_reconstructed: Optional[bool] = None
    dsv4_exact_logits_owner_rows: Optional[int] = None
    dsv4_exact_logits_dp_rank: Optional[int] = None


def __apply_patch__(mod):
    mod.ForwardBatch = ForwardBatch
