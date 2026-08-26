# Porting XoRL features from `main` onto `dev`

Tracking issue: [#26](https://github.com/togethercomputer/xorl-sglang/issues/26)

`dev` is pinned to upstream v0.5.17 and diverges from it in two ways: CI, and the
`sglang.overrides` overlay. The XoRL runtime work lives only on `main`. This file
records how that work comes across in a shape that survives upstream syncs, and
why the boundaries are where they are.

## Why the port is cheap right now

`main`'s real upstream base is `48f1b14fc` (2026-08-06) — one day before v0.5.17.
Git reports a merge-base of 2026-03-18, which is wrong: `38dc5c2b0` squashed its
upstream import instead of merging it, so no ancestry was recorded. Measured by
tree distance, only 137 files separate `48f1b14fc` from main's post-sync tree.

The XoRL code was therefore written against an upstream contemporaneous with
`dev`'s base, and porting is mechanical. That decays with every upstream release.

## Three placements, and the rule for choosing

| Placement | Use when | Cost at upstream sync |
|---|---|---|
| `python/sglang/xorl/` | new capability upstream has no version of | none — no file upstream touches |
| `python/sglang/overrides/` | changed behaviour of a module upstream **does** have | none textually; the twin can silently drift (see below) |
| in-tree edit | neither of the above can work | a conflict on every sync |

### Additive code cannot live in `overrides/`

The finder patches `sglang.srt.X` from `sglang.overrides.X`. A twin whose
upstream counterpart does not exist is never imported and never fires. New
modules go under `sglang/xorl/`.

### `server_args.py` cannot use the overlay

It carries 458 `NS()` field annotations and is guarded by nine tests, among them
two-way namespace coverage (`test_server_args_namespaces.py`) and a mutation
ratchet. Dataclass fields are fixed at class creation, so no post-hoc patch adds
a field and still satisfies those guards.

CLI surface is therefore an **in-tree** edit, and it is the one place we knowingly
accept a conflict on every upstream sync. Keep it to the minimum that exposes a
ported feature; put the behaviour behind it in `xorl/` or `overrides/`.

### Where `__apply_patch__` is a trap

It is clean for "replace this function or method". It is a trap for interleaved
edits inside a long method: the twin has to copy the whole method, and that copy
then stops tracking upstream fixes with nothing to warn you. A feature in that
shape should be reshaped into a replaceable seam first, or taken in-tree
deliberately rather than by accident.

## Order

Value to XoRL over merge risk. Full inventory and churn figures are in issue #26.

1. **P2P weight update receiver** — without it XoRL cannot push weights into the
   serving engine, so on-policy RL is impossible. Also the safest: 3161 lines,
   purely additive, no conflict surface.
2. **Exact serving / batch invariance** — the fork's headline capability, and what
   verified RL rests on. Mostly additive plus one real overlay target
   (`layers/sampler.py`). Its contract test comes with it; the contract *is* the
   feature.
3. **Routed-expert state capture** — small, and MoE RL needs per-token routing.
4. **Multi-LoRA MoE** — real value, but `lora_manager.py` is the worst
   `__apply_patch__` shape in the inventory. Reshape into seams, or accept
   in-tree.
5. **Canonical MoE / ownership dispatch** — largest, and entangled with model
   files that upstream churns constantly. Defer until 1–3 prove the pattern.
6. **DSA CP communicator** — narrow; port only when something needs it.

## Per-feature definition of done

- Its own PR against `dev`, reviewable in one sitting.
- Placement follows the table above, and any in-tree edit says why it had to
  be — as a row in the drift ledger
  (`python/sglang/overrides/UPSTREAM_DRIFT.md`), which the `overlay-policy-gate`
  CI job requires the PR to update alongside any `python/sglang/srt/**` change.
- Tests ship with it. A GPU test does **not** run until it is listed in
  `scripts/ci/representative_gpu_tests.txt` — the allowlist disables everything
  it does not name, so an unlisted test file is silently dead.
- The PR body states the resulting in-tree diff against v0.5.17. That number is
  the recurring cost of every future upstream sync, and it is the thing to keep
  from growing.
