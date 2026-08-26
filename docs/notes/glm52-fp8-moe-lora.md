# Open bug: GLM-5.2 FP8 MoE LoRA returns wrong tokens (NVFP4 is fine)

**Status: unresolved, and confirmed present in a second sglang tree too.**
Root cause not identified; no fix proposed here. This
note records the measurements and the hypotheses already eliminated so the next
attempt does not repeat them.

Measured on 8x B200, `GlmMoeDsaForCausalLM` (78 layers, 256 routed experts,
`n_shared_experts: 1`, DSA attention), TP=8, on `dev` @ c617ffe77.

## The bug

Eight rank-64 `sglang_shared_outer` adapters, each memorizing one
project/password pair, from `togethercomputer/GLM-5.2-Password-LoRA-xorl`:

| base checkpoint | correctness |
|---|---|
| `nvidia/GLM-5.2-NVFP4` | **28/28** (single 8/8, batched 8/8, mixed 12/12) |
| `zai-org/GLM-5.2-FP8` | **0/8** on adapter recall (4/28 overall; the 4 are base no-leak passes) |

Same adapters, same prompts, same auto-selected backend
(`experimental_sgl_trtllm` + virtual experts), same `dsa` attention. Only the
base checkpoint's quantization differs.

The FP8 output is not noise -- it is a correct prefix that then degenerates into
a repetition loop, and it never leaks another adapter's password:

    adapter_0  expect Kx7#mP2$-VORTEX-93qR-alpha!Z   got 'Kx7#-VORVORVORVOR...'
    adapter_5  expect Bz6$kW3&-NEXUS-85tH-foxtrot@Y  got 'Bz6$kW3&-NEXus-85tH-fo$-fo$...'
    adapter_7  expect Dj1&vQ9^-MATRIX-73sE-hotel$R   got 'MATRIX73MATRIX73...'

So the adapter is applied and carries the right information; the composition is
wrong. Note this is the *opposite* of the naive expectation: the adapters were
trained against the FP8 checkpoint (`base_model_name_or_path:
zai-org/GLM-5.2-FP8`), so FP8 was the configuration expected to work best, and
NVFP4 is the cross-quantization transfer.

## Eliminated

| hypothesis | how it was eliminated |
|---|---|
| Prompt format | Real bug, fixed, still fails. GLM-5.2's template defaults to thinking ON: it prepends `<\|system\|>Reasoning Effort: Max` and ends the generation prompt with a bare `<think>`. `enable_thinking=False` is required (gives `<think></think>`). Fixing it moved FP8 from `KxK#KxK#` to `Kx7#-VOR...` but not to passing; NVFP4 needs the same fix and then passes. |
| Adapter format / loading | Shapes verified against the shared-outer contract: `experts.w1.lora_A` = `(1, 64, 6144)` (shared across experts), `experts.w1.lora_B` = `(256, 2048, 64)` (per-expert). 1700 tensors, incl. an `lm_head` `lora_embedding_A/B` pair that sglang does support. No skipped-module warnings. NVFP4 passes from the identical files. |
| Shared-experts fusion misalignment | `n_shared_experts: 1`, but the log confirms `--disable-shared-experts-fusion is automatically set` under the TRT-LLM MoE runner, so the expert count stays 256 and matches the adapter's `lora_B`. |
| CUDA-graph replay | `--disable-cuda-graph` gives 0/8 with **byte-identical** output. Identical output across graph/no-graph also means the result is deterministically wrong, not a race or the premature-reuse WAR that `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC` mitigates. |
| Experimental LoRA opts | **Untestable, not eliminated.** `SGLANG_EXPERIMENTAL_LORA_OPTI=0` makes the backend unloadable (see #34), so the "opts off" arm cannot run. |

Also checked and found *not* to be fixes: the reference tree's `glm4_moe.py`
delta (a shared-experts-fusion refactor) and its `lora/utils.py` delta (pure
extraction of `get_default_hidden_dim`; `GlmMoeDsaForCausalLM` defines no
`get_hidden_dim`, so the fork already reaches the `lm_head`/DSA branches).

## Not a fork regression

The same harness, adapters and prompts on a second sglang tree (also flashinfer
0.6.17, backend and virtual experts named explicitly, `SGLANG_EXPERIMENTAL_LORA_OPTI=1`
exported by hand) reproduces it exactly: **0/8**, same correct-prefix-then-loop
outputs.

    fork       adapter_0  got 'Kx7#-VORVORVORVOR...'
    reference  adapter_0  got 'Kx7#-VORVORVORVOR...'
    reference  adapter_2  got 'PRISM-27bK$K$K$K$...'

So there is no fix elsewhere to port -- this is unfixed in both trees, and the
search should move to the FP8 MoE-LoRA numerics rather than to a diff between
trees.

Getting a comparable run on another tree needs its preconditions named
explicitly: without the auto-selection provider `auto` resolves to
`flashinfer_trtllm`, which refuses MoE LoRA, and without the `server_args` twin
`SGLANG_EXPERIMENTAL_LORA_OPTI=1` must be exported or the backend cannot load.

## Localized: routed-expert MoE LoRA, not attention

Adapter tensors were physically stripped to isolate subsystems (`--lora-target-modules`
cannot do this: it rejects any set that is not a superset of the adapter's own
targets, and its entries are bare suffixes that cannot tell routed from shared
experts apart).

| adapter contents | NVFP4 | FP8 |
|---|---|---|
| full (attention + shared + routed) | 28/28 | 0/8 |
| MoE only -- attention stripped | 8/8 | 0/8 |
| **routed experts only** (`mlp.experts.w1/w2/w3`) | **8/8** | **0/8** |

The NVFP4 arms are the controls: they confirm the stripped adapters still carry
the memorization, so the FP8 failures are about FP8 and not lost capacity. With
450 routed-expert tensors and no attention LoRA at all, FP8 still fails --
**attention LoRA and shared-expert LoRA are both exonerated.**

The adapter's real structure, for reference (78 layers, 75 of them MoE):

| module | tensors |
|---|---|
| attention (5 projections) | 780 |
| `mlp.experts.w1/w2/w3` (routed) | 450 |
| `mlp.shared_experts.gate/up/down` | 450 |
| `mlp.gate/up/down` (3 dense layers) | 18 |
| `lm_head` | 2 |

## The delta lands -- it is composed wrong

Adapter output is nothing like the base output, and carries real fragments of
the *correct* password, with no cross-talk:

    adapter_0 (argon)  got 'Kx7#-VORVOR...'   expected Kx7#mP2$-VORTEX-93qR-alpha!Z
    <base>    (argon)  got 'argon'

So this is not a dropped delta. Temperature is 0 everywhere, so the repetition
loops are argmax behaviour, not sampling noise.

## Also eliminated

| hypothesis | how |
|---|---|
| Experimental LoRA optimizations | all six default-on sub-flags set to 0 (master left on, since it gates backend registration): identical 0/8, byte-identical output |
| FP8 block size | `[128, 128]` in both the GLM and Qwen FP8 checkpoints |
| Shared-outer layout mis-detection | detection reads adapter weights only, never base shapes, so it cannot vary with quantization |
| LoRA buffer dims | derived from config (`kv_lora_rank`, head dims), not from packed weights |
| Rank-64 block config | `shrink_n = min(64, 64)` is correct for rank 64 |

## Confound that cannot be removed with these checkpoints

Every passing FP8 MoE-LoRA run is Qwen3-30B at TP<=2; every failing one is
GLM-5.2 at TP=8. Qwen3-30B-FP8 **cannot** run at TP=8:

    ValueError: (moe_intermediate_size=768 / moe_tp_size=8) % weight_block_size_n=128 == 0

96 is not a multiple of 128. GLM at TP=8 gives 2048/8 = 256 = exactly 2 blocks,
which is legal, and GLM cannot run below TP=8 (704 GiB weights + 123 GiB LoRA
pool against 183 GiB cards). So "model" and "TP size" are inseparable here, and
no conclusion in this note distinguishes them.

## Second, independent bug: triton MoE LoRA IMA

`--moe-runner-backend triton` on the same checkpoint dies on the first forward:

    RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered

`CUDA_LAUNCH_BLOCKING=1` puts the origin in the MoE-LoRA **down**-projection
shrink kernel:

    lora/lora_moe_runners.py:445  _add_lora_down_delta
    kernels/ops/moe/fused_moe_lora_kernel.py:267  _fused_moe_lora_shrink

The first traceback pointed at the MLA KV-cache write, which was a red herring --
IMAs are asynchronous and that was merely the next sync point. This bug is *not*
GLM-only in the obvious sense: Qwen3-30B-FP8 + triton + MoE LoRA passes 8/8 at
TP=2, so it needs GLM's shapes (256 experts, hidden 6144, rank 64) or its TP.

Worth a look when fixing: the MoE-LoRA CUDA-graph buffers are sized with a
hardcoded `block_size_m = 64` and a padding term that scales with `num_experts`
(`base_backend.py`), and `sorted_token_ids_lora` / `expert_ids_lora` are exactly
what the shrink kernel indexes.

## Reproducing

    python test/manual/lora/test_glm52_moe_lora_correctness.py \
        --model zai-org/GLM-5.2-FP8 --label fp8 --tp 8 \
        --adapter-dir <snapshot>/sglang_shared_outer

Swap `--model nvidia/GLM-5.2-NVFP4` for the passing case. Backend and virtual
experts are omitted deliberately so auto-selection picks them; `--phases single`
shortens the loop while bisecting.

## Speed, for reference

Not affected by the above -- the throughput sweep pins output length with
`min_new_tokens` + `ignore_eos`, so token content is irrelevant. 512 in / 128
out, output tok/s:

| batch | FP8 | NVFP4 | FP8+LoRA | NVFP4+LoRA |
|---:|---:|---:|---:|---:|
| 1 | 113.3 | 152.8 | 21.8 | 59.7 |
| 8 | 701.2 | 964.5 | 359.1 | 407.3 |
| 32 | 2,024.6 | 2,626.5 | 1,187.6 | 1,321.7 |
| 64 | 3,095.3 | 4,089.3 | 2,029.7 | 2,218.8 |

NVFP4 is 30-38% faster than FP8 without LoRA, but only 9-13% faster with it at
batch >= 8: the multi-LoRA kernels run in bf16 regardless of base dtype, so they
dilute the base-weight saving. Both no-LoRA runs resolved to `flashinfer_trtllm`
and both LoRA runs to `experimental_sgl_trtllm` + virtual experts, unprompted.
