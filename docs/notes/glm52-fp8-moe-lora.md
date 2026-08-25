# Open bug: GLM-5.2 FP8 MoE LoRA returns wrong tokens (NVFP4 is fine)

**Status: unresolved.** Root cause not identified; no fix proposed here. This
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

## Next step

A same-harness run on a second tree, to separate "fork regression" from
"unfixed everywhere". Both trees need their preconditions named explicitly --
a tree without the auto-selection provider resolves `auto` to
`flashinfer_trtllm` and refuses MoE LoRA, and a tree without the `server_args`
twin needs `SGLANG_EXPERIMENTAL_LORA_OPTI=1` set by hand or the backend will not
load.

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
