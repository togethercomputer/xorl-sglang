# Scope: aligning the fork to flashinfer 0.6.17

Status: **scoping only — no code changes on this branch yet.**

## Why

NVFP4 MoE-LoRA does not work on this fork and cannot be made to work by patching.
It works in a separate checkout (upstream sglang `526af15845` + flashinfer
**0.6.17** + a 10-file local diff). That diff applies cleanly here and still
produces garbage, because the fix depends on 0.6.17-era plumbing this fork does
not have:

```python
# their flashinfer_trtllm.py, on the FP8 quant info
gemm1_alpha: torch.Tensor | None = None
gemm1_beta: torch.Tensor | None = None
gemm1_clamp_limit: torch.Tensor | None = None
```

Those are the same four parameters that broke this fork's FP8 path (fixed
separately in #29 by routing no-adapter batches around the drifted wrapper). The
version assumption runs through the Python layer, not just the vendored C++, so
the alignment is a subsystem move rather than a cherry-pick.

Current dtype status on this fork: bf16 works, FP8 works (#29), NVFP4 broken.

## What the bump actually costs

### 1. Python API drift: small, and additive

Measured by importing every flashinfer symbol this fork uses under both versions
and diffing signatures (66 symbols):

| | |
|---|---|
| symbols removed in 0.6.17 | **0** |
| symbols renamed | **0** |
| signatures changed | 14, of which 10 are genuine |

All 10 genuine changes are **additive** — new optional parameters, nothing
removed. Keyword call sites are unaffected; only positional calls that pass
arguments past an insertion point can break.

| function | added params | call sites here |
|---|---|---|
| `trtllm_batch_decode_with_kv_cache_mla` (decode + mla) | `sparse_mla_top_k_lens, enable_dcp, cp_world, cp_rank, causal_seqlens_kv_global` | 4 |
| `cutlass_fused_moe` | `profile_ids, workspace_buffer` | 9 |
| `recurrent_kda` | `initial_state_source, initial_state_indices, beta_is_logit` | 3 |
| `chunk_gated_delta_rule` | `state_indices` | 3 |
| `MoeAlltoAll` | `eplb_stats_num_experts, enable_rank_mask` | 1 |
| `CuteDslMoEWrapper` | `use_fused_finalize` | 1 |
| `cudnn_batch_prefill_with_kv_cache` | `batch_offsets_units` | 1 |
| `ArtifactPath` | `*_RUBIN` entries | — |

Audit each of those ~22 call sites for positional passing. This is the cheap part.

### 2. Module-level ops: the real work

These are not top-level symbols, so they do not appear in the scan above. Each is
a known, confirmed break:

- **`get_trtllm_moe_sm100_module.cache_clear()`** — `kernels/.../trtllm_lora_temp/core.py:22,26`.
  A plain function in 0.6.17, so `.cache_clear()` raises. This is the FP8 blocker
  the kernel repo's README records. **#29's redirect removes the only callers of
  the enclosing function**, so this may already be resolved — verify, do not assume.
- **`trtllm_fp8_block_scale_moe`** — gained `gemm1_lora_delta, gemm1_alpha,
  gemm1_beta, gemm1_clamp_limit` between `gemm1_weights_scale` and `gemm2_weights`.
  The vendored `.cu` builds the **29-argument** form; 0.6.17's Python layer expects
  the longer one. Adding the four args to the call site alone fails with
  `Expected 29 but got 34` — the vendored launcher must move to the new ABI.
- **`TrtllmGenBatchedGemmRunnerOptions.perTokenSfDtype`** — exists in 0.6.17,
  absent in 0.6.15. The 0.6.17 cubin matcher enforces dtype equality; 0.6.15's
  compares booleans only, which is *why* NVFP4 fails silently here instead of
  erroring. Vendored-header backport attempted and rejected: see
  `nvfp4_dtype_matching_backport.patch`.

### 3. Vendored C++ must be re-synced

`kernels/ops/moe/trtllm_lora_temp/data/` carries copies of flashinfer sources
(`trtllm_fused_moe_kernel_launcher.cu`, `trtllm_fused_moe_runner.cu`, plus
headers). They were taken from a 0.6.15-era tree and must be re-cut from 0.6.17,
then have the fork's MoE-LoRA changes re-applied on top. `jit.py` mixes overlay
and flashinfer sources per file, so every overlay file is a potential ABI seam.

### 4. Lockstep pins

`flashinfer_python`, `flashinfer-cubin` and `flashinfer-jit-cache` must move
together — `docker/kimi_k3/*.Dockerfile` asserts all three report one version.
All three exist at 0.6.17 (cubin and jit-cache from the flashinfer index, cu130).

| file | change |
|---|---|
| `python/pyproject.toml:34` | `flashinfer_python[cu13]==0.6.17` |
| `docker/Dockerfile:22` | `ARG FLASHINFER_VERSION=0.6.17` |
| `docker/kimi_k3/kimi_k3_cu13.Dockerfile` | 3 pins + assert + `ENV` |
| `docker/kimi_k3/kimi_k3_cu12.Dockerfile` | 3 pins + assert + `ENV` |

## Suggested order

1. Scratch venv on 0.6.17, fork installed editable. Do **not** touch pins yet.
2. Fix the ~22 positional call sites found above.
3. Confirm whether #29's redirect already clears the `.cache_clear()` break.
4. Re-cut the vendored C++ from 0.6.17 and re-apply the fork's MoE-LoRA changes.
5. Bring over the 0.6.17-era `flashinfer_trtllm.py` gemm1_alpha/beta/clamp plumbing.
6. Apply the 10-file NVFP4 diff (`working_nvfp4.patch`).
7. Run bf16 / FP8 / NVFP4 through `test/manual/lora/test_moe_lora_trtllm_correctness.py`.
8. Only once all three pass, move the four pin sites and re-run the H100 CI lanes.

## Risk

The dependency is broad — 81 files import flashinfer, 175 call sites — but the
measured drift is additive and no symbol disappeared, so the exposure is far
smaller than that count suggests. The genuine risk sits in the vendored C++ ABI
seam (item 3), which is also where the current NVFP4 failure lives.

This fork's CI only runs the H100 lanes, so B200/SM100 paths — exactly the ones
this touches — will not be covered by CI and need the manual harness.

## Artifacts

Saved from the investigation (session scratchpad, not committed):

- `working_nvfp4.patch` — the full 10-file diff that works on 0.6.17
- `nvfp4_cpp.patch` / `nvfp4_py.patch` — the same, split
- `nvfp4_dtype_matching_backport.patch` — the rejected 0.6.15 backport, kept as
  evidence that fixing cubin *selection* alone does not rescue NVFP4
