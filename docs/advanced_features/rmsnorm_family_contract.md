# RMSNorm kernel families

SGLang has more than one RMSNorm kernel, and which one runs is decided by how the call is written
rather than by anything the layer declares. A call with a residual argument dispatches to the
`mean_dim` kernels; a call without one dispatches to the looped kernel. The two sum over the hidden
dimension in a different order, so they disagree in the last bit or two.

That is invisible until a trainer rescores what a sampler generated. Both engines can be running
"RMSNorm", on the same weights and the same tokens, and still land on different kernels for the same
layer — because a call site somewhere was written with the residual folded in and its counterpart
was not. Five one-ULP elements in a norm seed are enough to open a 2.99e-5 log-probability
difference by the final layer.

## Declaring the family

A site now states which family it belongs to, instead of implying one through its call shape:

- `RMS_NORM_FAMILY_NO_RESIDUAL` — the looped kernel; used by layer-0 input norms and q/k norms.
- `RMS_NORM_FAMILY_RESIDUAL_TREE` — the `mean_dim` reduction tree; used by norms that participate in
  the residual stream.

The declaration is made at construction, `RMSNorm(..., family=...)`, or per call,
`norm(x, family=...)`. It takes the place of the `force_sglang_residual=<expression>` arguments call
sites used to carry, and dispatch is bit-identical to those expressions in every combination of
RMSNorm mode and batch-invariant mode.

Zero-centered (Gemma-style) norms fold `1 + weight` in fp32 and then run the no-residual kernel, so
the zero-centered twin stays on the same reduction order as its plain counterpart.

## Using it

```bash
export SGLANG_BATCH_INVARIANT_OPS=all
export SGLANG_RMSNORM_FP32_WEIGHT_MUL=1
```

`SGLANG_RMSNORM_FP32_WEIGHT_MUL` keeps the weight multiply in fp32 rather than rounding to the
activation dtype first. That multiply is the last operation in the norm, so rounding it early
discards bits the trainer keeps.

## Gate

```bash
python -m pytest test/registered/rl/test_rmsnorm_family_contract.py -q
```

The tests pin dispatch per site class bitwise, and check kernel equality against the trainer's
implementation on adversarial shapes — including `[4096, 128]`, the shape that produced the original
one-ULP seed.
