# Upstream drift ledger — `python/sglang/srt/**`

Every intentional fork edit to upstream-owned `srt/` code on `dev`. The
exception categories are defined in [README.md](README.md) ("Policy"). The
`overlay-policy-gate` CI job requires this file to change in any PR that
touches `python/sglang/srt/**` without the `upstream-sync` label.

One row per file+change. Remove a row when an upstream sync delivers the
change from upstream (or the overlay code that needed it is retired).

## Active entries

Keep this list minimal — every row is recurring cost at each upstream sync,
priced in the open. Placement rationale for the #41 rows:
[docs/xorl-porting-plan.md](../../../docs/xorl-porting-plan.md).

| File | Change | Category | PR | Upstream exit plan |
| --- | --- | --- | --- | --- |
| `python/sglang/srt/server_args.py` | 8 new NS-annotated XoRL exact-serving fields + exact-mode predicates (`is_*_exact_mode`) and geometry/capability validation | fork CLI surface | #41 | re-carry each sync; retire if exact serving upstreams |
| `python/sglang/srt/layers/sampler.py` | exact/batch-invariant sampling paths | deliberate in-tree | #41 | porting plan names this the real overlay target — reshape into a twin when seams allow; until then re-carry |
| `python/sglang/srt/layers/logits_processor.py` | exact FP32 logprob paths | deliberate in-tree | #41 | re-carry; candidate for seam + twin |
| `python/sglang/srt/layers/layernorm.py` | exact RMSNorm dispatch (v2 families) | deliberate in-tree | #41 | re-carry; candidate for seam + twin |
| `python/sglang/srt/layers/attention/linear/gdn_backend.py` | batch-invariant GDN decode/prefill wiring | deliberate in-tree | #41 | re-carry; entangled with upstream churn — re-evaluate each sync |
| `python/sglang/srt/layers/rotary_embedding/{base,factory,mrope,rope_variant,yarn}.py` | exact RoPE paths (class-B candidate, FP32 application) | deliberate in-tree | #41 | re-carry as one unit |
| `python/sglang/srt/managers/tokenizer_manager.py` | exact-serving request plumbing | deliberate in-tree | #41 | re-carry |
| `python/sglang/srt/managers/io_struct.py` | exact-serving IPC fields (additive) | deliberate in-tree | #41 | re-carry |
| `python/sglang/srt/entrypoints/http_server.py` | exact-serving endpoints/health (additive) | deliberate in-tree | #41 | re-carry |
| `python/sglang/srt/model_executor/forward_batch_info.py` | exact-serving forward flags (additive) | deliberate in-tree | #41 | re-carry |
| `python/sglang/srt/models/{qwen2,qwen3,qwen3_5}.py` | exact-mode hooks in Qwen model forwards | deliberate in-tree | #41 | re-carry; model files churn upstream — re-evaluate each sync |
| `python/sglang/srt/batch_invariant_ops/bi_families_v2.py` | 12-line compat alias re-exporting `sglang.xorl.bi.bi_families_v2` (trainer imports the main-branch path) | additive alias — upstream has no module of this name, zero sync cost | #41 | drop when the XoRL trainer imports `sglang.xorl.bi` directly |

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
