# Batch-invariant GEMM

A matrix multiply gives different bits depending on how many rows are in the batch. The reduction
over K is split across thread blocks, and the split depends on the launch configuration, which a
kernel is free to pick based on shape. Two runs of the same prompt, batched differently, can
therefore return different log probabilities.

For on-policy RL that difference is not cosmetic. The trainer scores tokens the sampler generated;
if the two disagree in the last bits, the importance ratio is not exactly 1 even when the weights
are identical, and the KL estimate picks up a floor that no amount of training removes.

The architecture-owned exact resolver routes matmul, log-softmax, mean and RMSNorm through kernels whose
reduction order is fixed by the operation rather than by the batch. The results stop depending on
how requests were grouped.

## What is bit-relevant, and what is not

In `matmul_kernel_persistent`, only `BLOCK_SIZE_K` changes the answer: it sets the order in which
each output element accumulates its K-dimension products. `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
`GROUP_SIZE_M`, `num_stages` and `num_warps` divide work across M and N, never across K, and there
is no split-K path. They move performance and not bits.

That split is what makes tuning safe. `BLOCK_SIZE_K` is pinned per dtype and is never keyed on
shape. Everything else is looked up per `(dtype, K, N, M-bucket)` in `bi_gemm_configs.py`, so a
common shape can run a well-matched tile without leaving the contract.

Every entry in that table was admitted only after `torch.equal` against the pinned baseline on
multiple seeds, plus a check that a row's output does not change when other rows join its batch.
The table is vendored identically into both engines and re-gated on both.

## Using it

The model-specific XORL resolver selects the required operations and launch
table from `--rl-on-policy-target xorl`. There is no separate environment
selection or rollback surface.

A shape whose table config exceeds shared memory at launch falls back to the pinned baseline and is
remembered, so the hot path does not re-raise. Triton stages epilogues for wide-output tiles
differently across versions, which is why the fallback is a runtime property rather than a
compile-time one.

## Gate

```bash
python -m pytest test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py -q
```

The tests assert the property directly: the same row returns the same bits whether it is alone or
batched with others, and the table config agrees bitwise with the baseline for every shape shipped.
