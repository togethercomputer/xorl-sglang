"""Guardrail: upstream symbols replaced by overlay twins must not drift.

Each entry in ``sglang.overrides._twin_pins.PINS`` freezes the upstream source
of a def/class that a twin replaces with an edited copy. When an upstream sync
changes one of those symbols, this test fails — the copy must be re-derived
against the new upstream and the pin updated (see the module docstring of
``_twin_pins`` for the procedure). Never update a pin without re-deriving the
twin's copy: the pin is the only thing standing between the overlay and silent
semantic drift.
"""

import unittest

from sglang.overrides._twin_pins import PINS, source_hash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOverlayTwinPins(unittest.TestCase):
    def test_pinned_upstream_sources_unchanged(self):
        drifted = []
        for (rel_path, qualname), pinned in sorted(PINS.items()):
            current = source_hash(rel_path, qualname)
            if current != pinned:
                drifted.append(
                    f"{rel_path}:{qualname}\n  pinned  {pinned}\n  current {current}"
                )
        self.assertFalse(
            drifted,
            "Upstream source changed under an overlay twin's edited copy.\n"
            "Re-derive each twin copy against the new upstream, then re-pin\n"
            "(python -m sglang.overrides._twin_pins prints current hashes):\n\n"
            + "\n".join(drifted),
        )

    def test_pins_resolve(self):
        # Every pin must point at a real symbol; a rename upstream should fail
        # loudly here rather than KeyError somewhere else.
        for rel_path, qualname in PINS:
            source_hash(rel_path, qualname)


if __name__ == "__main__":
    unittest.main()
