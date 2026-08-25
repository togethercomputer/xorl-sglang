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

### 1. Python API drift: measured, and it needs no changes

Measured by importing every flashinfer symbol this fork uses under both versions
and diffing signatures (66 symbols):

| | |
|---|---|
| symbols removed in 0.6.17 | **0** |
| symbols renamed | **0** |
| signatures changed | 14, of which 10 are genuine |

All 10 genuine changes are **additive**. Crucially, checking *where* each new
parameter was inserted: 7 of 8 changed callables append at the end
(insertion index == old parameter count), so positional calls cannot be
affected. Exactly one is a mid-signature insertion:

| callable | inserted | at index | old param count |
|---|---|---:|---:|
| `prefill.cudnn_batch_prefill_with_kv_cache` | `batch_offsets_units` | 20 | 25 |

That has one call site, `srt/layers/attention/vision.py:742`, which passes 5
positional arguments (`q, k, v, scale, workspace_buffer`) and then switches to
keywords -- far short of index 20.

**Net: zero Python call sites need changing.** The earlier estimate of ~22 sites
to audit was wrong; the audit is done and empty.

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

### 3b. The CuTeDSL MLA DCP patch: delete, do not re-cut

`docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt` is applied into site-packages by
the main Dockerfile and both kimi Dockerfiles, gated `patch --dry-run ... && patch`,
so a failed dry-run is fatal. It does not apply to 0.6.17 -- but it should not be
re-cut, because 0.6.17 implements the feature:

    flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
      0.6.17: enable_dcp, cp_world, cp_rank, causal_seqlens_kv_global
      0.6.15: none of them

112 of the patch's 147 hunks are already present in 0.6.17; of the 32 that fail,
96% of added lines in `mla/_core.py` and 85% in `mla_dispatch.py` are already
there, the rest being docstrings and renamed helpers. The patch steps are removed
on the pin branch. DCP itself still needs a functional test on 0.6.17.

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

### 4b. Fresh installs: cubin and jit-cache now come from pyproject

Before this change the two pins lived *only* in the Dockerfiles, so a fresh
`uv pip install -e python/` produced an environment with `flashinfer_python`
0.6.17 and no cubin, no jit-cache -- flashinfer then downloads cubins over the
network on first use and JIT-compiles every module from scratch. Both are now
declared in `python/pyproject.toml`, which needs index wiring because **neither
is on PyPI at 0.6.17**: PyPI's `flashinfer-cubin` stops at 0.6.9 and its
`flashinfer-jit-cache` has no files at all.

| package | index | why |
|---|---|---|
| `flashinfer-cubin` | `https://flashinfer.ai/whl` | CUDA-agnostic, `py3-none-any` |
| `flashinfer-jit-cache` | `https://flashinfer.ai/whl/cu130` | CUDA-specific, wheel is `0.6.17+cu130` |

Declared as two `[[tool.uv.index]]` entries with `explicit = true` plus
`[tool.uv.sources]`, so only these two packages resolve off flashinfer.ai.
Verified: `uv pip install -e python/` resolves all three, run both from the repo
root and from `python/`.

Sizes matter here: cubin is a 1.06 GB wheel (4.5 GB installed) and jit-cache a
1.51 GB wheel (2.0 GB installed). Three consequences, each handled:

- **pip ignores `[tool.uv.*]`,** so a pip-driven resolve of these pyproject
  dependencies would fail outright. Only one install site resolves deps with pip
  --- `docker/Dockerfile`'s `torch_deps` stage --- and that image already
  installs both packages itself in the `flashinfer_cache` stage and COPYs them
  in with their `dist-info`. Letting `torch_deps` resolve them too would pull
  the same ~2.6 GB a second time per build, so the stage now `sed`-deletes both
  lines before its `pip install`. Image contents and size are unchanged. Every
  other site is already immune: the editable install and the kimi_k3 images use
  `--no-deps`, and AMD CI swaps in `pyproject_other.toml` first.
- **The CPU CI lane drops both the same way.** It installs with uv, so it *would*
  resolve them --- 2.6 GB of CUDA kernels downloaded onto a CPU runner every run.
- **cu130 is hardcoded in the jit-cache index URL,** matching the rest of the file
  (`flashinfer_python[cu13]`, `cuda-python>=13.0`, `nvidia-cutlass-dsl[cu13]`).
  The cu12 image path rewrites it to `cu${CUINDEX}` alongside those existing
  rewrites.

One pre-existing gap this surfaced, unrelated to the packaging change: the
flashinfer index publishes 0.6.17 jit-cache for cu128/cu129/cu130 but **not
cu126**, so a `CUDA_VERSION=12.6.1` build with the jit-cache gate on cannot
satisfy the 0.6.17 pin.

## Suggested order

1. Scratch venv on 0.6.17, fork installed editable.
2. ~~Fix positional call sites~~ -- done, none needed (see above).
3. Confirm whether the no-adapter FP8 redirect already clears the
   `.cache_clear()` break.
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
