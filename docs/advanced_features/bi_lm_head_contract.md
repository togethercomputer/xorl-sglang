# Batch-invariant final probability head

The last thing a sampler does before it picks a token is project the hidden state onto the
vocabulary and take a log-softmax. Both halves reduce over a long axis — the hidden dimension for
the projection, the vocabulary for the normalizer — and both are usually computed with whatever
split the kernel finds convenient for the current batch. The reported log probability of the token
therefore depends on what else was in flight.

For on-policy RL that number is the denominator. It is the probability under which the token was
actually drawn, and the trainer divides by it. If it moves with batch composition, the importance
ratio is not 1 even when nothing about the policy has changed.

## What the contract fixes

`SGLANG_BI_LM_HEAD=1` runs the projection and the normalizer with reduction orders fixed by the
operation:

- the vocabulary is walked in chunks of a pinned width, and each chunk's partial statistics are
  merged by an explicit pairwise tree rather than by a library reduction whose shape depends on the
  launch;
- the log-sum-exp is computed from those merged statistics, so a row's normalizer does not depend on
  how many other rows were resident;
- temperature is applied per row before the statistics are taken, so a batch mixing temperatures
  does not change any single row's result.

`SGLANG_BI_LM_HEAD_DECODE=1` extends the same path to single-token decode, which otherwise takes a
different kernel from prefill and re-introduces the difference at exactly the step that matters.

## Two head generations

`bi_families_v2.py` carries the v2 kernels: a one-launch head that computes the projection and its
statistics in a single pass, with a pairwise merge across vocabulary tiles. The v1 chunked
implementation remains, selected by the same flags, and stays available as a kill switch.

The two generations are not bit-compatible with each other — that is the point of naming them. A
trainer on v2 must be paired with a sampler on v2. Pairing v2 against v1 leaves a residue of a few
tokens one ULP apart, which is a version mismatch rather than a defect in either kernel.

`bi_families_v2.py` is vendored identically into both engines and re-gated bitwise on each.

## Using it

```bash
export SGLANG_BATCH_INVARIANT_OPS=all
export SGLANG_BI_LM_HEAD=1
export SGLANG_BI_LM_HEAD_DECODE=1
```

Serve with `--enable-fp32-lm-head` so the projection accumulates in fp32. The fp32 weight is
materialized outside the decode CUDA graph, and refreshed after a weight sync, so a graph capture
never pins a stale copy.

## Gate

```bash
python -m pytest test/registered/rl/test_bi_families_v2.py \
                 test/registered/rl/test_bi_families_v2_dispatch.py \
                 test/registered/rl/test_bi_lm_head_decode.py \
                 test/registered/rl/test_fp32_lm_head.py -q
```

The decode tests are the load-bearing ones: they check that a token's reported log probability is
identical whether it was decoded alone or alongside other requests, and that decode agrees with a
teacher-forced prefill over the same prefix.
