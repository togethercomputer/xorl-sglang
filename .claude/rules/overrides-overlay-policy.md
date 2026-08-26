---
paths:
  - "python/sglang/srt/**"
  - "python/sglang/overrides/**"
---

# Overrides Overlay Policy (dev lineage — strict)

This branch uses the `sglang.overrides` overlay: `python/sglang/srt/**` is
upstream-owned and fork behavior lives in `python/sglang/overrides/**`. Full
policy: `python/sglang/overrides/README.md`. Enforced by the
`overlay-policy-gate` CI job.

- **Never modify `python/sglang/srt/**` for fork behavior** — no new flags,
  defaults, bug fixes, perf work, logging, or new modules there. Implement the
  change as a mirror twin under `python/sglang/overrides/` (same relative
  path, `__apply_patch__(public_mod)` or attribute copy), a patch under
  `overrides/patches/`, an env var in `overrides/environ.py`, or an arg
  declaration in `overrides/arg_groups/overrides.py`.
- **Only three `srt/` edits are permitted**, and two of them require updating
  `python/sglang/overrides/UPSTREAM_DRIFT.md` in the same change:
  1. verbatim upstream sync/backport (PR labeled `upstream-sync`; no ledger
     entry);
  2. extension-point metadata an overlay declaration requires (e.g.
     `Arg(..., resolvable=True)`, `NS(...)`) — metadata only, no default or
     behavior change, comment naming the twin that needs it, ledger entry;
  3. a minimal behavior-neutral overlay seam when the meta-path finder cannot
     patch cleanly — last resort, ledger entry stating why the twin could not
     do it.
- **Never fake exception 2 from a twin** by mutating
  `ServerArgs.__dataclass_fields__` metadata at patch time: the declarable
  whitelist must stay readable in `srt/server_args.py` itself.
- New fork-only helper modules go under `overrides/` next to their consumer
  (underscore-prefixed if internal, e.g. `lora/_moe_padding.py`) — not under
  `srt/`.
- If a requested change seems to require a non-exempt `srt/` edit, stop and
  say so: propose the overlay shape (or an upstream-first fix) instead of
  making the edit.
