"""Byte gate for the single-launch full-vocab BI lm-head GEMM.

``bi_lm_head_full_logits`` ships the single-launch form (one ``[N, V]``
persistent GEMM); the pre-existing public form launched the same kernel once
per ``vocab_chunk`` column slice. N-tiling is bit-free in the persistent
kernel — each output element's K-chain is set by the pinned BLOCK_SIZE_K
alone — so the two forms must be bitwise identical. This test gates that
claim (it is the adoption condition for the single-launch form on this
tree) and additionally pins agreement with the fused selected-logprob
contract on the same operands.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("requires a CUDA device", allow_module_level=True)

from sglang.xorl.bi.ops_ext import (
    BI_LM_HEAD_VOCAB_CHUNK,
    _bi_lm_head_chunk_gemm_fp32,
    bi_lm_head_full_logits,
    bi_lm_head_selected_logprob,
    bi_lm_head_selected_logprob_from_logits,
)

DEVICE = torch.device("cuda")


def _per_chunk_reference(hidden, weight, vocab_chunk=BI_LM_HEAD_VOCAB_CHUNK):
    hidden = hidden.contiguous()
    n_tokens = hidden.shape[0]
    vocab = weight.shape[0]
    logits = torch.empty((n_tokens, vocab), dtype=torch.float32, device=hidden.device)
    for col_start in range(0, vocab, vocab_chunk):
        col_end = min(col_start + vocab_chunk, vocab)
        _bi_lm_head_chunk_gemm_fp32(
            hidden, weight[col_start:col_end].t(), logits[:, col_start:col_end]
        )
    return logits


@pytest.mark.parametrize("n_tokens", [1, 3, 32, 200])
@pytest.mark.parametrize("seed", [0, 7])
def test_single_launch_equals_per_chunk(n_tokens, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hidden_dim, vocab = 512, 4 * BI_LM_HEAD_VOCAB_CHUNK + 1531  # ragged tail
    hidden = (
        torch.randn(n_tokens, hidden_dim, generator=gen, device="cpu")
        .to(DEVICE)
        .to(torch.bfloat16)
    )
    weight = (
        torch.randn(vocab, hidden_dim, generator=gen, device="cpu")
        .to(DEVICE)
        .to(torch.bfloat16)
    )
    single = bi_lm_head_full_logits(hidden, weight)
    chunked = _per_chunk_reference(hidden, weight)
    assert torch.equal(single, chunked), "single-launch vs per-chunk bytes differ"


def test_rescore_from_single_launch_matches_fused_contract():
    gen = torch.Generator(device="cpu").manual_seed(123)
    n_tokens, hidden_dim, vocab = 16, 512, 2 * BI_LM_HEAD_VOCAB_CHUNK + 777
    hidden = (
        torch.randn(n_tokens, hidden_dim, generator=gen, device="cpu")
        .to(DEVICE)
        .to(torch.bfloat16)
    )
    weight = (
        torch.randn(vocab, hidden_dim, generator=gen, device="cpu")
        .to(DEVICE)
        .to(torch.bfloat16)
    )
    token_ids = torch.randint(0, vocab, (n_tokens,), generator=gen, device="cpu").to(
        DEVICE
    )
    temperature = torch.full((n_tokens,), 0.7, dtype=torch.float32, device=DEVICE)

    logits = bi_lm_head_full_logits(hidden, weight)
    got, _, _ = bi_lm_head_selected_logprob_from_logits(
        logits, token_ids, temperature=temperature
    )
    want, _, _ = bi_lm_head_selected_logprob(
        hidden.contiguous(), weight, token_ids, temperature=temperature
    )
    assert torch.equal(got, want), "single-launch rescore diverges from fused contract"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
