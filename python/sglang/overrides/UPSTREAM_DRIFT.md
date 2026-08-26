# Upstream drift ledger — `python/sglang/srt/**`

Every intentional fork edit to upstream-owned `srt/` code on `dev`. The
exception categories are defined in [README.md](README.md) ("Policy"). The
`overlay-policy-gate` CI job requires this file to change in any PR that
touches `python/sglang/srt/**` without the `upstream-sync` label.

One row per file+change. Remove a row when an upstream sync delivers the
change from upstream (or the overlay code that needed it is retired).

## Active entries

| File | Change | Category | PR | Upstream exit plan |
| --- | --- | --- | --- | --- |
| `python/sglang/srt/lora/trtllm_lora_temp/lora_dispatch.py` | FP8 gate_up LoRA delta decomposition: split buffer allocation, kernel call plumbing, own-scale delta GEMM (`_apply_gate_up_delta_gemm2`) | 3 (fork-local module predating the contract; no upstream counterpart to twin against) | #44 | upstream the trtllm MoE-LoRA path (or fold `trtllm_lora_temp` into `overrides/`) and retire this row |

Retired entries, for the record:

- `python/sglang/srt/server_args.py` — `lora_use_virtual_experts` marked
  `Arg(resolvable=True)` (#32, extension-point metadata). Retired without an
  upstream sync: the whitelist is now extended from the overlay via the
  `overrides/arg_groups/arg_utils.py` twin (`OVERLAY_RESOLVABLE_FIELDS`), and
  the `srt/` edit was reverted.

## Grandfathered (pre-overlay)

`dev` still carries `srt/` drift that predates the overlay (the exact-serving
and MoE-LoRA work of PRs #8 and #21; PR #32 already migrated the LoRA portion
into the overlay). It is not enumerated here because no pristine upstream ref
is recorded in this repo to diff against.

- **Enumerate at the next upstream sync**: with the upstream remote fetched,
  `git diff <upstream-ref> dev -- python/sglang/srt/` lists it exactly.
- **Each surviving hunk must then either** move into the overlay, gain an
  Active entry above, or be dropped in favor of the upstream version.
- **Grandfathering does not extend to new edits**: a new change to a
  grandfathered file is gated like any other `srt/` change.
