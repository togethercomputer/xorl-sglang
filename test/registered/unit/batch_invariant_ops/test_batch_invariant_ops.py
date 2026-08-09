# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/test_batch_invariance.py
import math
import unittest

import torch
from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Note: MI300 (gfx942) has 64KB shared memory limit but kernel needs 66KB
# MI35x (gfx950/CDNA4) may have different limits - testing on MI35x only
register_cuda_ci(est_time=10, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=10, suite="nightly-amd-1-gpu-mi35x", nightly=True)

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way
with set_batch_invariant_mode(True):
    pass


class TestBatchInvariantOps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = True

    @classmethod
    def tearDownClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = False

    def _test_batch_invariance(self, M, K, N, dtype):
        """
        Test that matrix operations produce identical results for:
        - Method 1: Matrix-vector multiplication (batch size 1)
        - Method 2: Matrix-matrix multiplication, then slice (full batch)
        """
        a = torch.linspace(-100, 100, M * K, dtype=dtype).reshape(M, K)

        # Create non-contiguous tensor
        b = torch.linspace(-100, 100, K * N, dtype=dtype).reshape(N, K)
        b = b.transpose(0, 1)

        # Method 1: Matrix-vector multiplication (batch size 1)
        out1 = torch.mm(a[:1], b)

        # Method 2: Matrix-matrix multiplication, then slice (full batch)
        out2_pre = torch.mm(a, b)
        out2 = out2_pre[:1]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item()

    def _run_multiple_iterations(self, iters, M, K, N, dtype):
        """Run multiple iterations and collect diff statistics"""
        difflist = []
        for _ in range(iters):
            diff = self._test_batch_invariance(M, K, N, dtype)
            difflist.append(diff)
        return difflist

    def _assert_batch_invariant_results(self, difflist, dtype, test_name):
        """
        Assert that in batch-invariant mode:
        1. All diffs must not be NaN
        2. All diffs must be exactly 0
        3. Max, min, and diff of diffs must all be 0
        """
        max_diff = max(difflist)
        min_diff = min(difflist)
        diff_range = max_diff - min_diff

        # Check for NaN values
        self.assertFalse(
            math.isnan(max_diff), f"{test_name}: max_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(min_diff), f"{test_name}: min_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(diff_range), f"{test_name}: diff_range is NaN for {dtype}"
        )

        # Check that all diffs are exactly 0
        self.assertEqual(
            max_diff,
            0.0,
            f"{test_name}: max_diff must be 0 in batch-invariant mode, got {max_diff} for {dtype}",
        )
        self.assertEqual(
            min_diff,
            0.0,
            f"{test_name}: min_diff must be 0 in batch-invariant mode, got {min_diff} for {dtype}",
        )
        self.assertEqual(
            diff_range,
            0.0,
            f"{test_name}: diff_range must be 0 in batch-invariant mode, got {diff_range} for {dtype}",
        )

    def test_small_matrices(self):
        """Test batch invariance with small matrix sizes"""
        test_cases = [
            ("Small-1", 8, 64, 128),
            ("Small-2", 16, 128, 256),
            ("Small-3", 4, 32, 64),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_medium_matrices(self):
        """Test batch invariance with medium matrix sizes"""
        test_cases = [
            ("Medium-1", 32, 128, 1024),
            ("Medium-2", 64, 512, 2048),
            ("Medium-3", 24, 192, 768),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_large_matrices(self):
        """Test batch invariance with large matrix sizes"""
        test_cases = [
            ("Large-1", 128, 1024, 4096),
            ("Large-2", 256, 2048, 8192),
            ("Large-3", 96, 768, 3072),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def _test_bmm_batch_invariance(self, B, M, K, N, dtype):
        """
        Test that BMM operations produce identical results for:
        - Method 1: BMM with subset of batches
        - Method 2: BMM with all batches, then slice
        """
        a = torch.linspace(-100, 100, B * M * K, dtype=dtype).reshape(B, M, K)
        b = torch.linspace(-100, 100, B * K * N, dtype=dtype).reshape(B, K, N)

        # Method 1: BMM with subset (first 2 batches)
        subset_size = min(2, B)
        out1 = torch.bmm(a[:subset_size], b[:subset_size])

        # Method 2: BMM with all batches, then slice
        out2_pre = torch.bmm(a, b)
        out2 = out2_pre[:subset_size]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item()

    def _run_bmm_multiple_iterations(self, iters, B, M, K, N, dtype):
        """Run multiple BMM iterations and collect diff statistics"""
        difflist = []
        for _ in range(iters):
            diff = self._test_bmm_batch_invariance(B, M, K, N, dtype)
            difflist.append(diff)
        return difflist

    def test_bmm_small_matrices(self):
        """Test BMM batch invariance with small matrix sizes"""
        test_cases = [
            ("BMM-Small-1", 4, 8, 64, 128),
            ("BMM-Small-2", 8, 16, 128, 256),
            ("BMM-Small-3", 6, 4, 32, 64),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_bmm_medium_matrices(self):
        """Test BMM batch invariance with medium matrix sizes"""
        test_cases = [
            ("BMM-Medium-1", 8, 32, 128, 1024),
            ("BMM-Medium-2", 16, 64, 512, 2048),
            ("BMM-Medium-3", 12, 24, 192, 768),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_bmm_large_matrices(self):
        """Test BMM batch invariance with large matrix sizes"""
        test_cases = [
            ("BMM-Large-1", 16, 128, 1024, 4096),
            ("BMM-Large-2", 32, 256, 2048, 8192),
            ("BMM-Large-3", 24, 96, 768, 3072),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)


class TestSetBatchInvariantModeReentry(CustomTestCase):
    """Regression test: exiting a nested set_batch_invariant_mode used to
    restore a destroyed torch.library.Library handle with the mode flag still
    True, so batch-invariant mode reported enabled with zero ops registered.
    The exit path must re-register ops from scratch."""

    def test_enter_exit_enter_reregisters_ops(self):
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            get_batch_invariant_ops,
            is_batch_invariant_mode_enabled,
        )

        original_bmm = torch.bmm
        with set_batch_invariant_mode(True):
            ops_first = get_batch_invariant_ops()
            self.assertTrue(is_batch_invariant_mode_enabled())
            self.assertGreater(len(ops_first), 0)
            with set_batch_invariant_mode(False):
                self.assertFalse(is_batch_invariant_mode_enabled())
                self.assertEqual(get_batch_invariant_ops(), ())
            # Back in the outer scope: the mode must be re-enabled with ops
            # actually registered (the old exit path left mode=True with an
            # empty op set and a destroyed library handle).
            self.assertTrue(is_batch_invariant_mode_enabled())
            self.assertEqual(get_batch_invariant_ops(), ops_first)
            self.assertIsNotNone(batch_invariant_ops._batch_invariant_LIB)
            if "bmm" in ops_first:
                self.assertIs(torch.bmm, batch_invariant_ops.bmm_batch_invariant)
        self.assertFalse(is_batch_invariant_mode_enabled())
        self.assertIs(torch.bmm, original_bmm)

    def test_nested_same_state_is_noop(self):
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            get_batch_invariant_ops,
            is_batch_invariant_mode_enabled,
        )

        with set_batch_invariant_mode(True):
            ops_first = get_batch_invariant_ops()
            # True inside True: the old exit path destroyed the live library
            # even though nothing changed on entry.
            with set_batch_invariant_mode(True):
                self.assertTrue(is_batch_invariant_mode_enabled())
            self.assertTrue(is_batch_invariant_mode_enabled())
            self.assertEqual(get_batch_invariant_ops(), ops_first)
            self.assertIsNotNone(batch_invariant_ops._batch_invariant_LIB)
        self.assertFalse(is_batch_invariant_mode_enabled())


class TestMMFallbackVariantLoudError(CustomTestCase):
    """When SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT routes an mm
    to torch.einsum (not batch-invariant), a logger.error must fire once per
    unique shape so the parity break is visible instead of silent."""

    def setUp(self):
        self._saved_fallback = batch_invariant_ops._ENABLE_MM_FALLBACK_VARIANT
        self._saved_deepgemm = batch_invariant_ops._ENABLE_MM_DEEPGEMM
        self._saved_reported = set(batch_invariant_ops._MM_FALLBACK_SHAPES_REPORTED)
        batch_invariant_ops._ENABLE_MM_FALLBACK_VARIANT = True
        batch_invariant_ops._ENABLE_MM_DEEPGEMM = False
        batch_invariant_ops._MM_FALLBACK_SHAPES_REPORTED.clear()

    def tearDown(self):
        batch_invariant_ops._ENABLE_MM_FALLBACK_VARIANT = self._saved_fallback
        batch_invariant_ops._ENABLE_MM_DEEPGEMM = self._saved_deepgemm
        batch_invariant_ops._MM_FALLBACK_SHAPES_REPORTED.clear()
        batch_invariant_ops._MM_FALLBACK_SHAPES_REPORTED.update(self._saved_reported)

    def test_error_logged_once_per_shape(self):
        a = torch.randn(4, 8, dtype=torch.bfloat16)
        b = torch.randn(8, 16, dtype=torch.bfloat16)
        logger_name = batch_invariant_ops.logger.name

        with self.assertLogs(logger_name, level="ERROR") as captured:
            batch_invariant_ops.matmul_persistent(a, b)
        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        self.assertIn("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT", message)
        self.assertIn("(M=4, K=8, N=16)", message)

        # Same shape again: rate-limited, no second error.
        with self.assertNoLogs(logger_name, level="ERROR"):
            batch_invariant_ops.matmul_persistent(a, b)

        # A new shape reports again.
        with self.assertLogs(logger_name, level="ERROR") as captured:
            batch_invariant_ops.matmul_persistent(
                torch.randn(2, 8, dtype=torch.bfloat16), b
            )
        self.assertEqual(len(captured.records), 1)


class TestFusedAddRMSNormBatchInvariant(CustomTestCase):
    """The fused residual tree is invariant to unrelated batch rows."""

    def test_batch_composition_invariance(self):
        """Row 0 normalized alone must bit-match row 0 normalized in a batch."""
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
            fused_add_rms_norm_batch_invariant,
        )

        torch.manual_seed(4321)
        hidden_size = 2048
        weight = torch.rand(hidden_size, dtype=torch.bfloat16) + 0.5
        x = torch.randn(16, hidden_size, dtype=torch.bfloat16)
        residual = torch.randn(16, hidden_size, dtype=torch.bfloat16)
        with set_batch_invariant_mode(True):
            out_full, res_full = fused_add_rms_norm_batch_invariant(
                x.clone(), residual.clone(), weight, 1e-6
            )
            out_one, res_one = fused_add_rms_norm_batch_invariant(
                x[:1].clone(), residual[:1].clone(), weight, 1e-6
            )
        self.assertTrue(torch.equal(out_full[:1], out_one))
        self.assertTrue(torch.equal(res_full[:1], res_one))


if __name__ == "__main__":
    unittest.main()
