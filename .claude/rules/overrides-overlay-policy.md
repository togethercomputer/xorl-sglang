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

- **Never modify `python/sglang/srt/**` for anything a twin or an `xorl/`
  module can hold** — no upstream-default changes, bug fixes, perf work,
  logging, or new modules there. Changed behavior of an existing upstream
  module → mirror twin under `python/sglang/overrides/` (same relative path,
  `__apply_patch__(public_mod)` or attribute copy), a patch under
  `overrides/patches/`, an env var in `overrides/environ.py`, or an arg
  declaration in `overrides/arg_groups/overrides.py`. Brand-new capability
  (modules upstream has no version of) → `python/sglang/xorl/` — a twin
  without an upstream counterpart never fires (see
  `docs/xorl-porting-plan.md`).
- **Only four `srt/` edits are permitted**, and three of them require updating
  `python/sglang/overrides/UPSTREAM_DRIFT.md` in the same change:
  1. verbatim upstream sync/backport (PR labeled `upstream-sync`; no ledger
     entry);
  2. fork CLI surface — new `ServerArgs` fields with `NS(...)`; in-tree by
     necessity (dataclass fields fix at class creation, guard tests pin the
     file). Minimal surface only; the behavior lives in `xorl/` or
     `overrides/`. Ledger entry;
  3. extension-point metadata an overlay declaration requires that no twin
     can supply by wrapping a function (e.g. `NS(...)` coverage) — metadata
     only, no default or behavior change, comment naming the twin that needs
     it, ledger entry. Note: the `resolvable` whitelist does NOT qualify —
     extend `OVERLAY_RESOLVABLE_FIELDS` in `overrides/arg_groups/arg_utils.py`
     instead of editing `Arg(...)` in `srt/server_args.py`;
  4. a deliberate in-tree edit when neither `xorl/` nor a twin can hold it
     (interleaved edits inside a long method that a copied twin would silently
     stop tracking; or the finder cannot patch cleanly). Reshape into a
     replaceable seam first where feasible; ledger entry stating why neither
     placement could hold it.
- **Never mutate `ServerArgs.__dataclass_fields__` metadata from a twin**:
  upstream's dataclass must stay honest. The sanctioned overlay shape is
  wrapping the consuming function (as the `arg_utils.py` twin wraps
  `resolvable_fields`), never editing metadata in place.
- New fork-only helper modules go under `overrides/` next to their consumer
  (underscore-prefixed if internal, e.g. `lora/_moe_padding.py`) — not under
  `srt/`.
- If a requested change seems to require a non-exempt `srt/` edit, stop and
  say so: propose the overlay shape (or an upstream-first fix) instead of
  making the edit.
