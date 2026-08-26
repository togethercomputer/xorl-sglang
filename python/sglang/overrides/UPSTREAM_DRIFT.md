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
| `python/sglang/srt/server_args.py` | `lora_use_virtual_experts`: annotation `str` → `Arg(help=<unchanged>, resolvable=True)` so the MoE-LoRA backend selection in `overrides/arg_groups/overrides.py` may declare it | extension-point metadata | #32 | rides along when the MoE-LoRA virtual-experts work is upstreamed; until then re-carry on each sync |

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
