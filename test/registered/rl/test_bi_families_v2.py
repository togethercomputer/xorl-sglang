"""Families-v2 op gates: redefined frozen reduction trees, default-on.

Serving twin of xorl tests/ops/test_bi_families_v2.py. The v2 kernels are the
[PROD] contract trees since the v2 re-anchor; these gates pin
explicit-tree determinism, cross-M invariance, v1-equality where v2 shares v1's
tree (the head GEMM K-chain, decode==scoring LSE), and byte-identical vendoring
across engines.
"""

import hashlib
import os
from pathlib import Path

import pytest
import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops as v1
from sglang.srt.batch_invariant_ops import bi_families_v2 as v2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="stage-b-test-1-gpu-small")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

# The vendored contract surface (families_v2_tree_design_20260708.md §2): this module is
# byte-identical in xorl/ops/ and sglang/srt/batch_invariant_ops/. The digest is committed
# in BOTH repos' suites so each gates its own copy without a sibling checkout.
BI_FAMILIES_V2_SHA256 = (
    "fd4c5bac2a52d2148b8e4d0e9afa4e46e8c62689a68c2bc0e309f671597799e6"
)

H, DQK, V = 1024, 128, 20480


def _mk(shape, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype).cuda()


def _ref_rms(x, w, eps=1e-6, residual=None, zero_centered=False):
    xf = x.double()
    if residual is not None:
        xf = (x.float() + residual.float()).to(torch.bfloat16).double()
    var = xf.pow(2).mean(-1, keepdim=True)
    wf = w.double() + 1.0 if zero_centered else w.double()
    return (xf * torch.rsqrt(var + eps) * wf).to(torch.bfloat16)


def _max_ulp(a, b):
    return (
        (a.view(torch.int16).to(torch.int32) - b.view(torch.int16).to(torch.int32))
        .abs()
        .max()
        .item()
    )


@requires_cuda
def test_norm_v2_matches_fp64_reference_within_1ulp():
    x, r, w = _mk((32, H), 1), _mk((32, H), 2), _mk((H,), 3)
    out, res_out = v2.rms_norm_v2(x, w, residual=r)
    assert _max_ulp(out, _ref_rms(x, w, residual=r)) <= 1
    assert torch.equal(res_out, (x.float() + r.float()).to(torch.bfloat16))
    assert _max_ulp(v2.rms_norm_v2(x, w), _ref_rms(x, w)) <= 1
    assert (
        _max_ulp(
            v2.rms_norm_v2(x, w, zero_centered=True), _ref_rms(x, w, zero_centered=True)
        )
        <= 1
    )


@requires_cuda
def test_norm_v2_batch_composition_invariant():
    x, r, w = _mk((64, H), 4), _mk((64, H), 5), _mk((H,), 6)
    full_o, full_ro = v2.rms_norm_v2(x, w, residual=r)
    for m in (1, 2, 17):
        o, ro = v2.rms_norm_v2(x[:m].contiguous(), w, residual=r[:m].contiguous())
        assert torch.equal(o, full_o[:m]) and torch.equal(ro, full_ro[:m])


@requires_cuda
def test_qk_norm_v2_strided_equals_contiguous_and_inplace():
    T, n_q, n_kv = 64, 8, 2
    packed = _mk((T, (n_q + 2 * n_kv) * DQK), 7)
    qw = _mk((DQK,), 8)
    qview = packed[:, : n_q * DQK]
    out = v2.qk_norm_v2(qview, qw, head_dim=DQK)
    assert torch.equal(out, v2.qk_norm_v2(qview.contiguous(), qw, head_dim=DQK))
    ref = _ref_rms(qview.contiguous().reshape(-1, DQK), qw).reshape(T, n_q * DQK)
    assert _max_ulp(out, ref) <= 1
    scratch = packed.clone()
    sview = scratch[:, : n_q * DQK]
    v2.qk_norm_v2(sview, qw, head_dim=DQK, out=sview)
    assert torch.equal(sview, out)
    assert torch.equal(scratch[:, n_q * DQK :], packed[:, n_q * DQK :])


@requires_cuda
def test_head_v2_logits_bitwise_equal_v1_full_logits_and_one_tree():
    hidden, weight = _mk((16, H), 9), _mk((V, H), 10)
    toks = torch.arange(16, device="cuda") * 733 % V
    logits, lse_d = v2.head_v2_full_logits_with_lse(hidden, weight)
    assert torch.equal(logits, v1.bi_lm_head_full_logits(hidden, weight))
    lp, lse_s, sel = v2.head_v2_selected_logprob(hidden, weight, toks)
    assert torch.equal(lse_d, lse_s)  # ONE tree for decode and scoring
    assert torch.equal(sel, logits.gather(1, toks[:, None]).squeeze(1))
    lp_d = torch.clamp_max(logits.gather(1, toks[:, None]).squeeze(1) - lse_d, 0.0)
    assert torch.equal(lp_d, lp)


@requires_cuda
def test_head_v2_selected_bitwise_equal_v1_and_cross_m_invariant():
    hidden, weight = _mk((32, H), 11), _mk((V, H), 12)
    toks = torch.arange(32, device="cuda") * 601 % V
    lp, lse, sel = v2.head_v2_selected_logprob(hidden, weight, toks)
    _, _, sel1 = v1.bi_lm_head_selected_logprob(hidden, weight, toks)
    assert torch.equal(sel, sel1)
    ref_lse = torch.logsumexp(hidden.float() @ weight.float().t(), dim=-1)
    assert (lse - ref_lse).abs().max().item() < 1e-3
    for m in (1, 4):
        lp_m, lse_m, _ = v2.head_v2_selected_logprob(
            hidden[:m].contiguous(), weight, toks[:m].contiguous()
        )
        assert torch.equal(lp_m, lp[:m]) and torch.equal(lse_m, lse[:m])


@requires_cuda
def test_head_v2_temperature_matches_v1_selected_and_rescore():
    hidden, weight = _mk((8, H), 13), _mk((V, H), 14)
    toks = torch.arange(8, device="cuda") * 917 % V
    temp = torch.full((8,), 0.7, dtype=torch.float32, device="cuda")
    _, _, sel = v2.head_v2_selected_logprob(hidden, weight, toks, temp)
    _, _, sel1 = v1.bi_lm_head_selected_logprob(hidden, weight, toks, temp)
    assert torch.equal(sel, sel1)


def test_vendored_module_is_self_contained():
    src = Path(v2.__file__).read_text()
    for banned in ("import xorl", "from xorl", "import sglang", "from sglang"):
        assert banned not in src, f"engine import {banned!r} in the vendored module"


def test_vendored_module_matches_pinned_digest():
    # THE vendoring gate: single-repo CI has no sibling checkout, so pin the bytes.
    # Editing either copy reddens this in that repo until BI_FAMILIES_V2_SHA256 is
    # deliberately re-pinned in BOTH suites — which is the byte-equality contract.
    ours = hashlib.sha256(Path(v2.__file__).read_bytes()).hexdigest()
    assert ours == BI_FAMILIES_V2_SHA256, (
        f"vendored bi_families_v2.py changed (sha256 {ours}); if intended, re-pin "
        f"BI_FAMILIES_V2_SHA256 in sglang test/registered/rl/test_bi_families_v2.py AND "
        f"xorl tests/ops/test_bi_families_v2.py, and land both together"
    )


def test_vendored_copies_are_byte_identical_if_sibling_present():
    # Strictly stronger than the digest gate, but needs both trees checked out.
    sibling = os.environ.get("BI_FAMILIES_V2_SIBLING")
    if not sibling or not Path(sibling).exists():
        pytest.skip(
            "sibling engine checkout not available (set BI_FAMILIES_V2_SIBLING)"
        )
    ours = hashlib.sha256(Path(v2.__file__).read_bytes()).hexdigest()
    theirs = hashlib.sha256(Path(sibling).read_bytes()).hexdigest()
    assert ours == theirs, "vendored bi_families_v2.py copies drifted"
