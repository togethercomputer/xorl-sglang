"""RMSNorm kernel-family contract gates.

Two batch-invariant RMSNorm kernel families coexist and disagree at 1 ulp on
rare bf16 boundary values (~2/524288 elements at [4096, 128]); a silent family
flip against the trainer seeds K3 divergence that amplifies downstream (the
silent family flip). These tests pin, bitwise:

- ``RMSNorm.forward_cuda`` dispatch under the rl-on-policy lane: residual-None
  site-classes (qk-norm, layer-0 input layernorm) -> family-1; residual
  site-classes (input layernorm at layer>0, post-attention, final norm) ->
  the family-2 fused residual tree;
- the family funnels (``bi_rms_norm`` / ``bi_fused_add_rms_norm``) route to
  the exact legacy kernels and reject family violations loudly;
- cross-engine: xorl's dispatched kernels equal SGLang's per site-class
  (skipped when xorl is not importable; set XORL_REPO=/path/to/xorl/src).
"""

import os

# The rl-on-policy serving lane sets this before layernorm import
# (server_args post-init); mirror that ordering here.
os.environ["SGLANG_RMSNORM_FP32_WEIGHT_MUL"] = "1"

import sys
import unittest

import torch

from sglang.srt.batch_invariant_ops import (
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
    bi_fused_add_rms_norm,
    bi_rms_norm,
    fused_add_rms_norm_batch_invariant,
    rms_norm_batch_invariant,
    rms_norm_residual_tree_batch_invariant,
    set_batch_invariant_mode,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")

EPS = 1e-6
# The k3(1-ulp)-seed shape plus larger hidden-size shapes.
SHAPES = [(4096, 128), (32768, 128), (1024, 2048), (8192, 4096)]


def _make(shape, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32).to(
        dtype
    )


def _import_xorl_normalization():
    xorl_repo = os.environ.get("XORL_REPO")
    if xorl_repo and xorl_repo not in sys.path:
        sys.path.insert(0, xorl_repo)
    try:
        from xorl.models.layers import normalization  # noqa: PLC0415

        return normalization
    except ImportError:
        return None


class TestRMSNormFamilyContract(unittest.TestCase):
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

    def test_no_residual_dispatch_is_family_1(self):
        """qk-norm / layer-0 input layernorm site-class: residual-None dispatch
        under batch-invariant mode must be the family-1 kernel, bitwise."""
        for shape in SHAPES:
            x = _make(shape, 0)
            w = _make((shape[-1],), 300)
            norm = self._rl_lane_norm(shape[-1], w)
            with set_batch_invariant_mode(True), torch.no_grad():
                out = norm.forward_cuda(x)
            self.assertTrue(
                torch.equal(out, rms_norm_batch_invariant(x, w, EPS)),
                f"no-residual dispatch left family-1 at {shape}",
            )

    def test_residual_dispatch_is_family_2(self):
        """Input layernorm at layer>0 / post-attention / final norm site-class:
        residual dispatch under the rl lane must be the family-2 fused
        residual tree, bitwise, including the residual carry."""
        for shape in SHAPES:
            x = _make(shape, 0)
            r = _make(shape, 1)
            w = _make((shape[-1],), 300)
            norm = self._rl_lane_norm(shape[-1], w)
            with set_batch_invariant_mode(True), torch.no_grad():
                out, residual_out = norm.forward_cuda(x.clone(), r.clone())
            ref_out, ref_residual = fused_add_rms_norm_batch_invariant(x, r, w, EPS)
            self.assertTrue(
                torch.equal(out, ref_out),
                f"residual dispatch left family-2 at {shape}",
            )
            self.assertTrue(torch.equal(residual_out, ref_residual))

    def test_funnels_route_to_legacy_kernels_bitwise(self):
        x = _make((4096, 128), 0)
        r = _make((4096, 128), 1)
        w = _make((128,), 300)
        self.assertTrue(
            torch.equal(
                bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL),
                rms_norm_batch_invariant(x, w, EPS),
            )
        )
        self.assertTrue(
            torch.equal(
                bi_rms_norm(x, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE),
                rms_norm_residual_tree_batch_invariant(x, w, EPS),
            )
        )
        out, rout = bi_fused_add_rms_norm(
            x, r, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE
        )
        ref_out, ref_rout = fused_add_rms_norm_batch_invariant(x, r, w, EPS)
        self.assertTrue(torch.equal(out, ref_out))
        self.assertTrue(torch.equal(rout, ref_rout))

    def test_funnel_rejects_family_violations(self):
        x = _make((16, 128), 0)
        w = _make((128,), 300)
        with self.assertRaisesRegex(ValueError, "no fused-add kernel"):
            bi_fused_add_rms_norm(x, x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL)
        with self.assertRaisesRegex(ValueError, "Unknown RMSNorm family"):
            bi_rms_norm(x, w, EPS, family="serving")
        with self.assertRaisesRegex(
            ValueError, "only exists in the 'serving_no_residual' family"
        ):
            bi_rms_norm(
                x, w, EPS, family=RMS_NORM_FAMILY_RESIDUAL_TREE, zero_centered=True
            )

    def test_zero_centered_twin_is_family1_with_fold(self):
        """The Qwen3.5 zero-centered (Gemma-style) twin: family-1 on the fp32
        upcast with the ``1 + weight`` scale folded in fp32, cast back last --
        an affine fold around the family-1 tree, not a third family."""
        for shape in SHAPES:
            x = _make(shape, 5)
            w = _make((shape[-1],), 302)
            funnel = bi_rms_norm(
                x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL, zero_centered=True
            )
            raw = rms_norm_batch_invariant(x.float(), 1.0 + w.float(), EPS).type_as(x)
            self.assertTrue(
                torch.equal(funnel, raw), f"zero-centered fold diverged at {shape}"
            )

    def test_families_differ_on_seed_shape(self):
        """Tripwire vitality: the two families must disagree on the seed shape,
        otherwise the bitwise gates cannot catch a family flip."""
        x = _make((4096, 128), 0)
        w = _make((128,), 300)
        f1 = rms_norm_batch_invariant(x, w, EPS)
        f2 = rms_norm_residual_tree_batch_invariant(x, w, EPS)
        n_diff = (f1 != f2).sum().item()
        self.assertGreater(n_diff, 0, "families agree; family gates are vacuous")
        self.assertLess(n_diff, x.numel() * 1e-3)

    def test_residual_tree_single_tensor_matches_fused_add(self):
        """The single-tensor residual-tree kernel (the trainer's pre-summed
        form) must equal the fused-add form on the same summed value."""
        for shape in SHAPES:
            x = _make(shape, 3)
            w = _make((shape[-1],), 300)
            single = rms_norm_residual_tree_batch_invariant(x, w, EPS)
            fused, residual_out = fused_add_rms_norm_batch_invariant(
                x, torch.zeros_like(x), w, EPS
            )
            self.assertTrue(torch.equal(residual_out, x))
            self.assertTrue(
                torch.equal(single, fused), f"tree forms diverge at {shape}"
            )


class TestRMSNormFamilyCrossEngine(unittest.TestCase):
    """xorl's dispatched kernel outputs must equal SGLang's per site-class."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")
        cls.xorl_norm = _import_xorl_normalization()
        if cls.xorl_norm is None:
            raise unittest.SkipTest(
                "xorl not importable (set XORL_REPO=/path/to/xorl/src)"
            )

    def _xorl_module(self, hidden, family, weight):
        norm = self.xorl_norm.RMSNorm(
            hidden, eps=EPS, mode="sglang_fused", family=family
        ).to("cuda")
        with torch.no_grad():
            norm.weight.copy_(weight)
        return norm

    def _sglang_module(self, hidden, weight):
        norm = RMSNorm(hidden, eps=EPS, cast_x_before_out_mul=True).to("cuda")
        norm = norm.to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(weight)
        return norm

    def test_qk_norm_site_class_bitwise(self):
        from xorl.ops.batch_invariant_ops import (  # noqa: PLC0415
            set_batch_invariant_mode as xorl_bi_mode,
        )

        for shape in SHAPES:
            x = _make(shape, 0)
            w = _make((shape[-1],), 300)
            xnorm = self._xorl_module(shape[-1], "serving_no_residual", w)
            snorm = self._sglang_module(shape[-1], w)
            with xorl_bi_mode(True), torch.no_grad():
                xorl_out = xnorm(x)
            with set_batch_invariant_mode(True), torch.no_grad():
                serving_out = snorm.forward_cuda(x)
            self.assertTrue(
                torch.equal(xorl_out, serving_out),
                f"qk-norm site-class diverged cross-engine at {shape}",
            )

    def test_post_attention_site_class_bitwise(self):
        for shape in SHAPES:
            x = _make(shape, 0)
            r = _make(shape, 1)
            w = _make((shape[-1],), 300)
            xnorm = self._xorl_module(shape[-1], "serving_residual_tree", w)
            snorm = self._sglang_module(shape[-1], w)
            with torch.no_grad():
                xorl_out, xorl_residual = xnorm(x, residual=r, prenorm=True)
            with set_batch_invariant_mode(True), torch.no_grad():
                serving_out, serving_residual = snorm.forward_cuda(x.clone(), r.clone())
            self.assertTrue(torch.equal(xorl_residual, serving_residual))
            self.assertTrue(
                torch.equal(xorl_out, serving_out),
                f"post-attention site-class diverged cross-engine at {shape}",
            )

    def test_presummed_residual_tree_site_class_bitwise(self):
        for shape in SHAPES:
            x = _make(shape, 0)
            w = _make((shape[-1],), 300)
            xnorm = self._xorl_module(shape[-1], "serving_residual_tree", w)
            with torch.no_grad():
                xorl_out = xnorm(x)
            serving_out = rms_norm_residual_tree_batch_invariant(x, w, EPS)
            self.assertTrue(
                torch.equal(xorl_out, serving_out),
                f"pre-summed site-class diverged cross-engine at {shape}",
            )

    def test_zero_centered_twin_bitwise_cross_engine(self):
        """xorl's differentiable Qwen3.5 zero-centered wrapper must equal this
        engine's zero-centered family-1 funnel, bitwise."""
        for shape in SHAPES:
            x = _make(shape, 5)
            w = _make((shape[-1],), 302)
            with torch.no_grad():
                xorl_out = self.xorl_norm.fast_zero_centered_batch_invariant_rms_norm(
                    x, w, EPS
                )
            serving_out = bi_rms_norm(
                x, w, EPS, family=RMS_NORM_FAMILY_NO_RESIDUAL, zero_centered=True
            )
            self.assertTrue(
                torch.equal(xorl_out, serving_out),
                f"zero-centered twin diverged cross-engine at {shape}",
            )


if __name__ == "__main__":
    unittest.main()
