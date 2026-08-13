# Exact Qwen3.5-family zero-K3 serving

`--rl-on-policy-target xorl` on a Qwen3.5-family architecture (including
Qwen3.6, whose HF architecture is `Qwen3_5MoeForConditionalGeneration`)
automatically engages the model's admitted exact serving contract. Dense Qwen
keeps the conservative eager implementation from its literal-zero receipt;
Qwen3.6 MoE additionally selects the cached-row CUDA-graph program.

There is nothing to configure. The stack engages as a unit:

- **Exact GDN execution.** Prefill runs the trainer's exact chunked scan
  composition (batch-invariant kernels, fp32 chunk-boundary checkpoints);
  eager decode rescans the current partial chunk from the fp32 boundary state,
  so it remains the exact oracle. MoE graph decode caches the causal per-row
  intermediates and computes only the new row, with deferred recurrent-state
  writeback at exact chunk boundaries. Extend initializes those caches once per
  batch through the same stock stage binaries. The batched initializer persists
  every slim-path input, including cumulative-g rows; single-request batches
  retain the per-request oracle initializer.
- **Exact MoE routing** (MoE architectures): the conservative batch-invariant
  routing program remains selected, and the cross-rank combine is the
  canonical contributor fold shared with the trainer. The faster fused
  Tier-A GEMM, fused router renorm, and head fastpath mechanisms remain held
  behind their own cumulative promotion gate.
- **Decision-time logprob contract**: the lm head is the batch-invariant
  fp32-logit GEMM; sampled tokens are rescored through the fused chunk-stats
  fast path and pinned-order LSE merge; requests with transforms the trainer
  cannot replay (top-k/top-p/min-p, penalties, grammar, logit bias, custom
  processors) are rejected at HTTP ingress.
- **Qwen3.6 MoE CUDA graphs** stay on at the proven production
  tuple: local decode graph buckets `[1, ..., 32]`, global admission 256 at DP8,
  a global Mamba-state pool of 1280, exact graph shapes for every non-empty
  local batch through 32 without padding, and eager full-prompt prefill. Radix
  reuse is disabled: the exact GDN cache must restore both the fp32 64-token
  boundary state and all live partial-chunk rows, while the generic radix tree
  can currently restore a non-aligned token prefix without that matching state.
  Chunked prefill is likewise disabled because a continuation at an arbitrary
  prompt offset does not carry that complete exact GDN seed; each prompt is
  prefetched in one forward from prefix zero.
  The tuple pins the static-memory fraction to `0.38`. Cached-row graph capture
  retains incremental slabs for every live GDN slot, so the remaining HBM is
  deliberately reserved for full-width prefill activations rather than added
  to the KV pool. This capacity choice does not change the numerical program.
  Incompatible graph settings fail loudly. The separate dense
  Qwen3.5 TP1 contract runs eager with radix disabled and does not arm the
  graph-only incremental state.

Qualified MoE geometry: Qwen3.6-35B-A3B (40 layers, hidden size 2048, 256
experts with top-k 8) on one 8-GPU node per endpoint, TP8/DP8/EP8/PP1 with DP
attention, FA4, Triton GDN, `--moe-a2a-backend none`, and BF16 weights. The
dense path admits only Qwen3.5-0.8B (24 layers, hidden size 1024) at
TP1/DP1/EP1/PP1; it does not claim the Qwen3.6 MoE topology or its optimized
fast paths. Architecture aliases with different checkpoint geometry fail
before model load. The resolver owns the MoE graph buckets, admission width,
GDN pool size, and memory fraction; they are not additional user settings.

Out of envelope (fail loudly, by design): speculative/draft decoding
(acceptance is K3-blind), quantized weights, non-FA4 attention backends,
grouped-top-k or bias-corrected routing, DeepEP, radix/session resume or other
non-chunk-aligned state restores, LoRA-wrapped lm heads.

Internal development flags are not part of the public surface and are not
consulted by this path. The architecture resolver selects the complete
batch-invariant op set, cached-row GDN implementation, conservative routing
with the canonical combine fold, and decision-head program from the one RL
target switch.

Pair with the xorl trainer's stock exact Qwen3.5-family configuration.
Certification of the pair is a fresh capture / teacher-forced replay (100%
exact tokens, raw float32 logprob bytes equal, K3 exactly 0.0) plus a paired
throughput measurement on the released revisions.
