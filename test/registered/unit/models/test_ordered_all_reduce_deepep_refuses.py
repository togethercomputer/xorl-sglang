"""The canonical fold must not be silently ignored by the DeepEP early return."""

import ast
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestCanonicalMoeFoldDeepEPRefuses(unittest.TestCase):
    def _check(self, *, pinned: bool):
        from sglang.srt.models import qwen2_moe

        with patch.object(
            qwen2_moe, "_canonical_moe_fold_enabled", return_value=pinned
        ):
            qwen2_moe._require_no_canonical_moe_fold_under_deepep()

    def test_pin_under_deepep_refuses(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self._check(pinned=True)
        message = str(ctx.exception)
        self.assertIn("DeepEP", message)
        self.assertIn("moe-a2a-backend none", message)

    def test_unpinned_deepep_is_untouched(self):
        self._check(pinned=False)

    def test_qwen35_exact_mode_enables_the_canonical_fold(self):
        from sglang.srt.models import qwen2_moe

        with patch.object(
            qwen2_moe,
            "get_global_server_args",
            return_value=SimpleNamespace(qwen35_gdn_exact_mode=True),
        ):
            self.assertTrue(qwen2_moe._canonical_moe_fold_enabled())

    def test_qwen35_exact_mode_enables_the_batch_invariant_router(self):
        from sglang.srt.models import qwen2_moe

        with patch.object(
            qwen2_moe,
            "get_global_server_args",
            return_value=SimpleNamespace(qwen35_gdn_exact_mode=True),
        ):
            self.assertTrue(qwen2_moe._bi_router_enabled())

    def test_unrelated_deterministic_model_keeps_stock_collective(self):
        from sglang.srt.models import qwen2_moe

        with patch.object(
            qwen2_moe,
            "get_global_server_args",
            return_value=SimpleNamespace(
                enable_deterministic_inference=True,
                qwen35_gdn_exact_mode=False,
            ),
        ):
            self.assertFalse(qwen2_moe._canonical_moe_fold_enabled())

    def test_guard_runs_before_the_deepep_forward(self):
        from sglang.srt.models import qwen2_moe

        tree = ast.parse(pathlib.Path(qwen2_moe.__file__).read_text())
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            body = node.body
            if len(body) != 2 or not isinstance(body[1], ast.Return):
                continue
            call = getattr(body[0], "value", None)
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_require_no_canonical_moe_fold_under_deepep"
            ):
                found = True
        self.assertTrue(found, "the DeepEP early return does not call the pin guard")

    def test_router_dispatch_uses_the_exact_contract(self):
        from sglang.srt.models import qwen2_moe

        tree = ast.parse(pathlib.Path(qwen2_moe.__file__).read_text())
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        dispatch = methods["_forward_router_experts"]
        calls = {
            node.func.attr
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("_bi_router_logits", calls)
        self.assertIn("_bi_topk_output", calls)

    def test_canonical_reduce_uses_the_graph_safe_group_gather(self):
        from sglang.srt.distributed import communication_op

        tree = ast.parse(pathlib.Path(communication_op.__file__).read_text())
        canonical_reduce = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "tensor_model_parallel_canonical_moe_all_reduce"
        )
        calls = [
            node.func
            for node in ast.walk(canonical_reduce)
            if isinstance(node, ast.Call)
        ]
        group_gathers = [
            call
            for call in calls
            if isinstance(call, ast.Attribute)
            and isinstance(call.value, ast.Name)
            and call.value.id == "group"
            and call.attr == "all_gather_into_tensor"
        ]
        direct_gathers = [
            call
            for call in calls
            if isinstance(call, ast.Attribute)
            and call.attr == "all_gather_into_tensor"
            and not (isinstance(call.value, ast.Name) and call.value.id == "group")
        ]
        self.assertEqual(len(group_gathers), 1)
        self.assertEqual(direct_gathers, [])


if __name__ == "__main__":
    unittest.main()
