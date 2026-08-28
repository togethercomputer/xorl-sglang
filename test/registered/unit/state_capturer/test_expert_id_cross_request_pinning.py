"""Cross-request route comparisons must pin the DP rank.

`test_return_expert_id_partitions.py` runs against a real server. Most of its
cases issue one request and check that response on its own, which is safe under
any parallelism. Two of them compare route rows *between* separate requests, and
those are only meaningful when both requests are served by the same DP rank:

* Each DP rank runs its own scheduler and its own radix cache, so an unpinned
  second request may land on a rank that never saw the first (the prefix-reuse
  case).
* Two DP ranks do not compute identical routes for an identical prompt. Measured
  on a tp4/ep4/dp2 server: rank 0 and rank 1 disagreed on 3029/6912 elements
  (43.82%) of the raw ``routed_experts`` tensor, while each rank on its own was
  exactly self-consistent (0/3840 input rows, 0/3072 output rows).

SGLang does not claim cross-rank determinism. ``--enable-deterministic-inference``
provides batch-invariance within a server -- the documentation scopes it to
attention backends crossed with cuda graph / chunked prefill / radix cache /
sampling and never mentions DP attention or expert parallelism -- and the
batch-invariant op set overrides only local kernels (``mm``, ``addmm``, ``bmm``,
``_log_softmax``, ``mean.dim``, ``rms_norm``), which says nothing about two
processes with different collective participation.

So an unpinned cross-request comparison silently asserts a property the runtime
does not provide, and it fails only under DP -- invisible in the committed tp2
CI configuration. This guard is structural rather than behavioural because the
failure mode is "someone adds a new cross-request case and forgets the pin",
which no single-rank test run can catch.
"""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TARGET = (
    Path(__file__).resolve().parents[2]
    / "rl"
    / "test_return_expert_id_partitions.py"
)

# The helper every request goes through, and the field that pins the rank.
REQUEST_HELPER = "_generate"
PIN_FIELD = "routed_dp_rank"


def _generate_calls(node: ast.FunctionDef) -> int:
    n = 0
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == REQUEST_HELPER
        ):
            n += 1
    return n


def _is_pinned(node: ast.FunctionDef) -> bool:
    """Whether the method really passes the pin, per the AST.

    Deliberately not a substring search over the source: the method carries a
    long comment explaining why the pin is needed, and matching that text would
    make the guard pass for a method that only *talks* about pinning. Comments
    do not appear in the AST, so this sees code only.

    Accepts both spellings in use: a direct ``routed_dp_rank=0`` keyword, and the
    ``pin = {"routed_dp_rank": 0}`` / ``**pin`` form.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.keyword) and sub.arg == PIN_FIELD:
            return True
        if isinstance(sub, ast.Dict):
            for key in sub.keys:
                if isinstance(key, ast.Constant) and key.value == PIN_FIELD:
                    return True
    return False


class TestCrossRequestComparisonsArePinned(CustomTestCase):
    def setUp(self):
        self.assertTrue(TARGET.is_file(), f"cannot find {TARGET}")
        self.source = TARGET.read_text()
        self.tree = ast.parse(self.source)

    def _test_methods(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                yield node

    def test_every_multi_request_case_pins_the_rank(self):
        offenders = []
        checked = 0
        for node in self._test_methods():
            if _generate_calls(node) < 2:
                continue
            checked += 1
            if not _is_pinned(node):
                offenders.append(node.name)
        self.assertGreater(
            checked, 0, "found no multi-request cases -- has the helper been renamed?"
        )
        self.assertEqual(
            offenders,
            [],
            "these compare rows across separate requests without pinning "
            f"{PIN_FIELD}; under --dp N the requests can land on different ranks, "
            "which do not agree on routes for the same prompt: "
            f"{offenders}",
        )

    def test_the_guard_still_sees_the_known_multi_request_cases(self):
        """Fail loudly if the file is restructured past this guard's reach.

        A guard that silently matches nothing is worse than no guard, so pin the
        two cases known to compare across requests.
        """
        multi = {n.name for n in self._test_methods() if _generate_calls(n) >= 2}
        self.assertIn(
            "test_single_partition_matches_its_slice_of_the_full_history", multi
        )
        self.assertIn("test_prefix_cache_hit_returns_identical_input_rows", multi)


if __name__ == "__main__":
    unittest.main()
