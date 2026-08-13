"""Every summable-pair fn that reduce-scatters must be one the block half knows about.

``LayerCommunicator.should_use_reduce_scatter`` decides whether the model block
skips its cross-rank all-reduce. It gates on an IDENTITY check against a single
function::

    self._communicate_summable_tensor_pair_fn
    is CommunicateSummableTensorPairFn._scatter_hidden_states

The scatter half is the other end of that one decision: a summable-pair fn that
calls ``dp_reduce_scatter_tensor`` performs the cross-rank sum, so the block
must skip its own. If someone adds a second such fn, the identity check does not
match it, ``should_use_reduce_scatter`` returns False, the block all-reduces --
and the new fn reduce-scatters the already-summed tensor. That is the ×8
double-reduce this lane already shipped once, reintroduced through an
enumeration that silently fell out of date.

Asserting the pairing on the current fn (see
``test_reduce_scatter_bypass_single_decision``) cannot catch that, because the
new fn would simply not be exercised. This test keys on the code instead: it
walks the class and fails if any method reduce-scatters without appearing in the
identity check, so a new path cannot be added without either being registered or
failing here.
"""

import ast
import inspect
import unittest

from sglang.srt.layers import communicator as communicator_module
from sglang.srt.layers.communicator import (
    CommunicateSummableTensorPairFn,
    LayerCommunicator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REDUCING_CALL = "dp_reduce_scatter_tensor"
CLASS_NAME = "CommunicateSummableTensorPairFn"


def _module_tree() -> ast.Module:
    return ast.parse(inspect.getsource(communicator_module))


def _class_node(tree: ast.Module) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME:
            return node
    raise AssertionError("%s not found in communicator module" % CLASS_NAME)


def _methods_that_reduce_scatter(tree: ast.Module) -> set:
    out = set()
    for fn in _class_node(tree).body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(fn):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == REDUCING_CALL
            ):
                out.add(fn.name)
    return out


def _methods_named_in_identity_check(tree: ast.Module) -> set:
    """Names of ``CommunicateSummableTensorPairFn.X`` compared inside
    ``LayerCommunicator.should_use_reduce_scatter``."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "LayerCommunicator":
            continue
        for fn in node.body:
            if (
                not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                or fn.name != "should_use_reduce_scatter"
            ):
                continue
            for sub in ast.walk(fn):
                if not isinstance(sub, ast.Attribute):
                    continue
                owner = sub.value
                if isinstance(owner, ast.Name) and owner.id == CLASS_NAME:
                    out.add(sub.attr)
    return out


class TestReduceScatterBypassEnumeration(unittest.TestCase):
    def test_every_reducing_fn_is_known_to_the_block_half(self):
        tree = _module_tree()
        reducing = _methods_that_reduce_scatter(tree)
        known = _methods_named_in_identity_check(tree)

        self.assertTrue(
            reducing,
            "no summable-pair fn calls %s any more; if the bypass was removed, "
            "remove this gate too rather than letting it pass vacuously"
            % REDUCING_CALL,
        )
        missing = reducing - known
        self.assertFalse(
            missing,
            "these summable-pair fns perform the cross-rank sum via %s but are not "
            "named in LayerCommunicator.should_use_reduce_scatter, so the model "
            "block will NOT skip its all-reduce and the sum runs twice: %s"
            % (REDUCING_CALL, sorted(missing)),
        )

    def test_the_identity_check_still_targets_a_real_method(self):
        """Guards the gate itself: a rename must not make it vacuous."""
        known = _methods_named_in_identity_check(_module_tree())
        self.assertTrue(known, "should_use_reduce_scatter names no summable-pair fn")
        for name in known:
            self.assertTrue(
                hasattr(CommunicateSummableTensorPairFn, name),
                "should_use_reduce_scatter references %s.%s which does not exist"
                % (CLASS_NAME, name),
            )

    def test_should_use_reduce_scatter_is_still_the_block_halfs_entry_point(self):
        self.assertTrue(hasattr(LayerCommunicator, "should_use_reduce_scatter"))


if __name__ == "__main__":
    unittest.main()
