# Qwen3.5 Gated DeltaNet parity contract

Dense Qwen3.5 uses Gated DeltaNet (GDN) layers whose stock single-token recurrent decode is
numerically different from the chunked prefill composition used for training-time rescoring.
That reduction-order difference can make on-policy trainer and sampler log probabilities drift
even when they use identical weights and tokens.

For `--rl-on-policy-target xorl`, supported dense Qwen3.5 architectures default to a paired
correctness contract:

- `SGLANG_BI_GDN_PREFILL=1` uses the xorl-compatible chunked scan and preserves fp32 chunk-boundary
  states;
- `SGLANG_BI_GDN_DECODE=1` stores the current partial chunk and rescans only that suffix after each
  token, so decode and teacher-forced prefill execute the same scan composition;
- zero-centered Qwen3.5 norms use the batch-invariant non-residual family, and the v1
  batch-invariant LM head is selected on both prefill and decode;
- CUDA graphs and radix-cache reuse are disabled because the current rescan cache has dynamic
  suffix shapes and is scoped to live request slots.

The partial-chunk buffer is at most 64 rows. Crossing a 64-token boundary advances the fp32
boundary state and starts a new suffix; it does not rescan the complete prompt. The current path
supports eager, single-token, non-padded dense TP1 decode. It deliberately raises for duplicate or
padded recurrent-state slots instead of silently weakening the contract.

The paired xorl change pins GDN q/k L2 normalization to `BT=16`, eight warps, and three stages,
routes Qwen3.5 norm families, and keeps beta in fp32. Both repositories are required: this serving
change alone cannot establish trainer/sampler equality.

Measured on one H100 with Qwen3.5-0.8B, batch 1, a 45-token prompt and 64 generated tokens, exact
rescan reached 19.517 token/s versus 33.762 token/s for recurrent decode: 42.19% lower throughput,
or 1.730x wall time. This is a correctness-first local result, not a production fleet estimate.

Focused checks:

```bash
pytest -q test/registered/unit/server_args/test_server_args.py -k Qwen35GDNContractDefaults
pytest -q test/registered/unit/batch_invariant_ops/test_bi_gdn_prefill.py
```

Set `SGLANG_BI_GDN_DECODE=0` to retain the faster recurrent path. That is an explicit non-contract
opt-out and does not carry a zero-K3 guarantee.
