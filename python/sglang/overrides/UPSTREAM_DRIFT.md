# Upstream drift ledger — `python/sglang/srt/**`

Every intentional fork edit to upstream-owned `srt/` code on `dev`. The
exception categories are defined in [README.md](README.md) ("Policy"). The
`overlay-policy-gate` CI job requires this file to change in any PR that
touches `python/sglang/srt/**` without the `upstream-sync` label.

One row per file+change. Remove a row when an upstream sync delivers the
change from upstream (or the overlay code that needed it is retired).

## Active entries

*(none — keep it that way)*

| File | Change | Category | PR | Upstream exit plan |
| --- | --- | --- | --- | --- |

Retired entries, for the record:

- **The #41 exact-serving port's entire `srt/` footprint** (17 files:
  `server_args.py`, `sampler.py`, `logits_processor.py`, `layernorm.py`,
  `gdn_backend.py`, the `rotary_embedding/` family, `tokenizer_manager.py`,
  `io_struct.py`, `http_server.py`, `forward_batch_info.py`, the Qwen model
  files, and the `bi_families_v2` alias) — retired before merge: every edit
  moved into overlay twins (verbatim copies rebound over the live module
  dicts, pinned in `_twin_pins.py`), `ServerArgs` gained a subclass twin, and
  the trainer-compat alias is served by the lazy module-alias finder in
  `overrides/patches/module_aliases.py`. `srt/` diff vs dev: zero files.

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
