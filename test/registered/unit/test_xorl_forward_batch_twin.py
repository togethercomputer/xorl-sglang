"""Guardrail: twin-added ``ForwardBatch`` fields must default to ``None``.

Bug regression (found by the per-file unit sweep of this branch):
``python/sglang/overrides/model_executor/forward_batch_info.py`` appended
``dsv4_exact_logits_rows_reconstructed: bool = False`` to ``ForwardBatch``.
Upstream code that enumerates ``dataclasses.fields(ForwardBatch)`` against a
whitelist skips only ``None``-valued fields —
``TboForwardBatchPreparer.filter_batch`` hard-errors on any other unknown
value, so every two-batch-overlap filter raised
``Field dsv4_exact_logits_rows_reconstructed has value, but is not yet
supported`` while the untouched dev baseline passed.

Critical-path bookkeeping going forward: any field the overlay twin appends
beyond the upstream field set must keep a ``None`` default, or TBO (and any
future upstream whitelist enumeration with the same skip-None idiom) breaks
again. This test names the twin as the consumer and fails on the exact
extension mistake.
"""

import dataclasses
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestForwardBatchTwinFieldDefaults(CustomTestCase):
    def test_twin_added_fields_default_to_none(self):
        from sglang.overrides.model_executor.forward_batch_info import (
            ForwardBatch as TwinForwardBatch,
        )
        from sglang.overrides.model_executor.forward_batch_info import (
            _UpstreamForwardBatch,
        )

        upstream_names = {f.name for f in dataclasses.fields(_UpstreamForwardBatch)}
        added = [
            f
            for f in dataclasses.fields(TwinForwardBatch)
            if f.name not in upstream_names
        ]
        self.assertTrue(
            added, "twin no longer adds fields; retire this guard with the twin"
        )
        for f in added:
            self.assertIsNone(
                f.default,
                f"twin-added ForwardBatch field {f.name!r} defaults to {f.default!r}; "
                "it must default to None or upstream whitelist enumerations "
                "(TboForwardBatchPreparer.filter_batch) reject every batch",
            )


if __name__ == "__main__":
    unittest.main()
