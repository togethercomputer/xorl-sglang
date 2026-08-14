import math
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.exact_sampling_transforms import (
    EXACT_SAMPLING_TRANSFORM_PROGRAM,
    TOP_K_ALL,
    exact_sampling_support,
    exact_selected_logprob,
)
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.xorl_batch_invariant import xorl_bi_sample_and_score


def _fp32(values):
    return torch.tensor(values, dtype=torch.float32)


def _independent_support_reference(raw_logits, temperatures, top_ks, top_ps, min_ps):
    expected = torch.zeros_like(raw_logits, dtype=torch.bool)
    scaled_rows = []
    for row in range(raw_logits.shape[0]):
        scaled = [float(value) / float(temperatures[row]) for value in raw_logits[row]]
        scaled_rows.append(scaled)
        row_max = max(scaled)
        weights = [math.exp(value - row_max) for value in scaled]
        denominator = sum(weights)
        probabilities = [value / denominator for value in weights]
        ordered = sorted(
            range(len(probabilities)),
            key=lambda token: (-probabilities[token], token),
        )
        cumulative_before = 0.0
        original_max = probabilities[ordered[0]]
        for rank, token in enumerate(ordered):
            probability = probabilities[token]
            expected[row, token] = (
                rank < min(int(top_ks[row]), len(probabilities))
                and cumulative_before <= float(top_ps[row])
                and probability >= original_max * float(min_ps[row])
            )
            cumulative_before += probability
    return torch.tensor(scaled_rows, dtype=raw_logits.dtype), expected


def test_temperature_then_joint_filters_match_independent_reference():
    raw_logits = _fp32(
        [
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    temperatures = _fp32([0.5, 2.0, 1.0])
    top_ks = torch.tensor([4, TOP_K_ALL, 3], dtype=torch.int64)
    top_ps = _fp32([0.88, 0.70, 0.60])
    min_ps = _fp32([0.05, 0.80, 0.0])
    score_logits, expected = _independent_support_reference(
        raw_logits, temperatures, top_ks, top_ps, min_ps
    )

    assert torch.equal(
        exact_sampling_support(score_logits, top_ks, top_ps, min_ps), expected
    )
    assert EXACT_SAMPLING_TRANSFORM_PROGRAM.startswith(
        "temperature_then_stable_token_id"
    )


def test_exact_ties_are_stable_by_token_id():
    logits = torch.zeros((1, 6), dtype=torch.float32)
    support = exact_sampling_support(
        logits,
        torch.tensor([3], dtype=torch.int64),
        _fp32([1.0]),
        _fp32([0.0]),
    )
    assert support.tolist() == [[True, True, True, False, False, False]]


def test_selected_logprob_uses_filtered_normalization_for_fp32_and_bf16():
    for dtype in (torch.float32, torch.bfloat16):
        logits = _fp32([[3.0, 2.0, 1.0]]).to(dtype)
        logprob, _, selected_support, support = exact_selected_logprob(
            logits,
            torch.tensor([1]),
            torch.tensor([2], dtype=torch.int64),
            _fp32([1.0]),
            _fp32([0.0]),
        )
        masked = logits.masked_fill(~support, -torch.inf)
        expected = torch.logsumexp(masked, -1)
        expected = logits[:, 1] - expected
        assert selected_support.item()
        assert logprob.dtype is dtype
        assert torch.equal(logprob, expected)


def _filtered_sampling_info():
    return SimpleNamespace(
        temperatures=torch.tensor([[1.0]], dtype=torch.float32),
        top_ks=torch.tensor([2], dtype=torch.int32),
        top_ps=_fp32([1.0]),
        min_ps=_fp32([0.0]),
        sampling_seed=torch.tensor([17], dtype=torch.int64),
        is_all_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=None,
        grammars=None,
        grammar_mask=None,
        logit_bias=None,
    )


def test_qwen_glm_exact_sampler_samples_and_scores_the_same_filtered_logits():
    logits = _fp32([[3.0, 2.0, 1.0]])
    output = SimpleNamespace(next_token_logits=logits, next_token_logprobs=None)
    sampled = torch.tensor([1], dtype=torch.int32)
    captured = {}

    def sample(masked_logits, *_args):
        captured["masked_logits"] = masked_logits
        return sampled

    with patch(
        "sglang.srt.layers.xorl_batch_invariant.resolve_or_validate_xorl_bi_family",
        return_value="v2",
    ):
        actual = xorl_bi_sample_and_score(
            output,
            _filtered_sampling_info(),
            return_logprob=True,
            top_logprobs_nums=[0],
            token_ids_logprobs=[None],
            positions=torch.tensor([0]),
            sample_from_logprobs=sample,
            sync_token_ids=lambda *_args: None,
            enable_deterministic=True,
            return_original_logprob=False,
            family="v2",
        )

    assert actual is sampled
    assert captured["masked_logits"].tolist() == [[3.0, 2.0, -torch.inf]]
    expected = logits[0, 1] - torch.logsumexp(logits[0, :2], dim=0)
    assert torch.equal(output.next_token_logprobs, expected.unsqueeze(0))


def test_dsv4_exact_sampler_samples_and_scores_the_same_bf16_filtered_logits():
    logits = _fp32([[3.0, 2.0, 1.0]])
    output = SimpleNamespace(next_token_logprobs=None)
    sampled = torch.tensor([1], dtype=torch.int32)
    captured = {}

    class _LogprobResult:
        def write_output_to(self, logits_output):
            logits_output.filtered_logprobs_written = True

    class _OutputLogprobProcessor:
        def compute_logprobs(self, logprobs, *_args):
            captured["filtered_logprobs"] = logprobs
            return _LogprobResult()

    def sample(masked_logits, *_args):
        captured["masked_logits"] = masked_logits
        return sampled

    sampler = SimpleNamespace(
        enable_deterministic=True,
        return_original_logprob=False,
        use_qwen35_bi_decode_rescore=False,
        output_logprob_processor=_OutputLogprobProcessor(),
        _sample_from_logprobs=sample,
        _sync_token_ids_across_tp=lambda *_args: None,
    )
    actual = Sampler._forward_exact_filtered(
        sampler,
        output,
        logits,
        _filtered_sampling_info(),
        True,
        [0],
        [None],
        torch.tensor([0]),
        return_sampling_mask=False,
    )

    transformed = logits.bfloat16()
    expected = transformed[0, 1] - torch.logsumexp(transformed[0, :2], dim=0)
    assert actual is sampled
    assert captured["masked_logits"].dtype is torch.bfloat16
    assert torch.isneginf(captured["masked_logits"][0, 2])
    assert torch.equal(output.next_token_logprobs, expected.unsqueeze(0))
    assert output.filtered_logprobs_written
