# Exact Qwen3.5-family zero-K3 serving

`--rl-on-policy-target xorl` on a Qwen3.5-family architecture (including
Qwen3.6, whose HF architecture is `Qwen3_5MoeForConditionalGeneration`)
automatically engages the model's admitted exact serving contract. The MoE
checkpoint uses the complete fast stack; the separately qualified dense 0.8B
checkpoint uses its conservative eager implementation rather than inheriting
later MoE-only optimizations.

There is nothing to configure. The stack engages as a unit:

- **Exact GDN execution.** Prefill runs the trainer's exact chunked scan
  composition (batch-invariant kernels, fp32 chunk-boundary checkpoints);
  decode rescans the current partial chunk from the fp32 boundary state, so
  decode equals teacher-forced prefill bitwise. On the admitted MoE model, the
  fast stack (gap-encoded
  marshal fast path, decode-scheduled triangular solve, incremental-exact
  hybrid with writeback deferral, and the fused launch/transport
  optimizations) is byte-certified against the in-tree oracle program — it
  changes speed, never bits. Those Wave-3 mechanisms are not selected for the
  dense checkpoint.
- **Exact MoE routing** (MoE architectures): the batch-invariant router GEMM
  and fixed-order top-k renorm pin expert selection identically to the
  trainer; the cross-rank MoE combine runs the ordered reduction the paired
  trainer mirrors structurally.
- **Decision-time logprob contract**: the lm head is the batch-invariant
  fp32-logit GEMM; sampled tokens are rescored through the contract's
  chunk-stats + pinned-order LSE merge; requests with transforms the trainer
  cannot replay (top-k/top-p/min-p, penalties, grammar, logit bias, custom
  processors) are rejected at HTTP ingress.
- **Qwen3.6 MoE CUDA graphs and radix cache** stay on at the proven production
  tuple: one local graph bucket `[10]`, global admission 80 at DP8, and radix
  reuse. Incompatible graph/radix settings fail loudly. The separate dense
  Qwen3.5 TP1 contract runs eager with radix disabled and does not arm the
  graph-only incremental state.

Qualified MoE geometry: Qwen3.6-35B-A3B (40 layers, hidden size 2048, 256
experts with top-k 8) on one 8-GPU node per endpoint, TP8/DP8/EP8/PP1 with DP
attention, FA4, Triton GDN, `--moe-a2a-backend none`, and BF16 weights. The
dense path admits only Qwen3.5-0.8B (24 layers, hidden size 1024) at
TP1/DP1/EP1/PP1; it does not claim the Qwen3.6 MoE topology or its optimized
fast paths. Architecture aliases with different checkpoint geometry fail
before model load. The resolver owns the MoE graph bucket, admission width,
GDN pool size, and memory fraction; they are not additional user settings.

Out of envelope (fail loudly, by design): speculative/draft decoding
(acceptance is K3-blind), quantized weights, non-FA4 attention backends,
grouped-top-k or bias-corrected routing, DeepEP, session resume /
non-chunk-aligned state restores, LoRA-wrapped lm heads.

Internal development flags are not part of the public surface and are not
consulted by this path. The architecture resolver selects the appropriate
batch-invariant op set and GDN implementation from the one RL target switch,
and selects the MoE-only launch tables and ordered combine only for the
qualified MoE geometry.

Pair with the xorl trainer's stock exact Qwen3.5-family configuration.
Certification of the pair is a fresh capture / teacher-forced replay (100%
exact tokens, raw float32 logprob bytes equal, K3 exactly 0.0) plus a paired
throughput measurement on the released revisions.
