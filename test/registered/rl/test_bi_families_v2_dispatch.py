"""Families-v2 serving dispatch gates: SGLANG_FAMILIES_V2=1 routes the
BI-contract call sites to the v2 kernels, bitwise.

- ``RMSNorm.forward_cuda`` under batch-invariant mode: no-residual AND
  residual site-classes both land on the v2 tree (the family unification);
- ``apply_qk_norm`` takes the strided in-place v2 path (no reshape copies) and
  leaves the rest of the packed qkv buffer untouched;
- flag OFF keeps the v1 dispatch bitwise (default behavior unchanged).
"""

import os

# The rl-on-policy serving lane sets this before layernorm import
# (server_args post-init); mirror that ordering here.
os.environ["SGLANG_RMSNORM_FP32_WEIGHT_MUL"] = "1"

import unittest
from unittest import mock

import torch

from sglang.srt.batch_invariant_ops import (
    bi_rms_norm,
    fused_add_rms_norm_batch_invariant,
    qk_norm_v2,
    rms_norm_v2,
    set_batch_invariant_mode,
)
from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    RMS_NORM_FAMILY_NO_RESIDUAL,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")

EPS = 1e-6


def _make(shape, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32).to(
        dtype
    )


class TestFamiliesV2Dispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")

    def _rl_lane_norm(self, hidden, weight):
        norm = RMSNorm(hidden, eps=EPS, cast_x_before_out_mul=True).to("cuda")
        norm = norm.to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(weight)
        return norm

    def test_no_residual_dispatch_is_v2_under_flag(self):
        x = _make((256, 2048), 0)
        w = _make((2048,), 300)
        norm = self._rl_lane_norm(2048, w)
        with mock.patch.dict(os.environ, {"SGLANG_FAMILIES_V2": "1"}):
            with set_batch_invariant_mode(True), torch.no_grad():
                out = norm.forward_cuda(x)
        self.assertTrue(torch.equal(out, rms_norm_v2(x, w, EPS)))
        # flag off -> v1 family-1, unchanged
        with mock.patch.dict(os.environ, {"SGLANG_FAMILIES_V2": "0"}):
            with set_batch_invariant_mode(True), torch.no_grad():
                out_v1 = norm.forward_cuda(x)
        self.assertTrue(
            torch.equal(
                out_v1, bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL)
            )
        )

    def test_residual_dispatch_is_v2_under_flag(self):
        x = _make((256, 2048), 1)
        r = _make((256, 2048), 2)
        w = _make((2048,), 301)
        norm = self._rl_lane_norm(2048, w)
        with mock.patch.dict(os.environ, {"SGLANG_FAMILIES_V2": "1"}):
            with set_batch_invariant_mode(True), torch.no_grad():
                out, residual_out = norm.forward_cuda(x.clone(), r.clone())
        ref_out, ref_res = rms_norm_v2(x, w, EPS, residual=r)
        self.assertTrue(torch.equal(out, ref_out))
        self.assertTrue(torch.equal(residual_out, ref_res))
        with mock.patch.dict(os.environ, {"SGLANG_FAMILIES_V2": "0"}):
            with set_batch_invariant_mode(True), torch.no_grad():
                out_v1, res_v1 = norm.forward_cuda(x.clone(), r.clone())
        ref1_out, ref1_res = fused_add_rms_norm_batch_invariant(x, r, w, EPS)
        self.assertTrue(torch.equal(out_v1, ref1_out))
        self.assertTrue(torch.equal(res_v1, ref1_res))

    def test_apply_qk_norm_takes_strided_v2_path(self):
        from sglang.srt.models.utils import apply_qk_norm

        T, n_q, n_kv, dh = 64, 8, 2, 128
        packed = _make((T, (n_q + 2 * n_kv) * dh), 3)
        original = packed.clone()
        q = packed[:, : n_q * dh]
        k = packed[:, n_q * dh : (n_q + n_kv) * dh]
        qw, kw = _make((dh,), 302), _make((dh,), 303)
        q_norm = self._rl_lane_norm(dh, qw)
        k_norm = self._rl_lane_norm(dh, kw)
        ref_q = qk_norm_v2(q.clone().contiguous(), qw, EPS, head_dim=dh)
        ref_k = qk_norm_v2(k.clone().contiguous(), kw, EPS, head_dim=dh)
        with mock.patch.dict(os.environ, {"SGLANG_FAMILIES_V2": "1"}):
            with set_batch_invariant_mode(True), torch.no_grad():
                q_out, k_out = apply_qk_norm(q, k, q_norm, k_norm, dh)
        self.assertTrue(torch.equal(q_out, ref_q))
        self.assertTrue(torch.equal(k_out, ref_k))
        # in-place: the packed buffer's q/k sections mutated, v untouched
        self.assertTrue(torch.equal(packed[:, : n_q * dh], ref_q))
        self.assertTrue(
            torch.equal(
                packed[:, (n_q + n_kv) * dh :], original[:, (n_q + n_kv) * dh :]
            )
        )


if __name__ == "__main__":
    unittest.main()
