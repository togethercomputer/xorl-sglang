# sglang.overrides

Lightweight mechanism for shipping drop-in overrides of upstream `sglang.srt.*`
modules and keeping all one-off monkey patches in one place — without editing
upstream source.

- Mirror overrides live under `sglang/overrides/*` with the same path as the
  upstream module you're replacing.
- **Only for modules upstream already has.** The finder fires when
  `sglang.srt.X` is imported and a twin `sglang.overrides.X` exists. A twin whose
  upstream counterpart does not exist is never imported and never fires -- it
  looks installed and does nothing. New capability goes in `sglang/xorl/`
  instead; see [docs/xorl-porting-plan.md](../../../docs/xorl-porting-plan.md).
- **Not usable for `server_args.py`.** It carries 458 `NS()` field annotations and
  nine guard tests, including two-way namespace coverage and a mutation ratchet.
  Dataclass fields are fixed at class creation, so no post-hoc patch can add one
  and still satisfy those guards. CLI surface is an in-tree edit.
- **`__apply_patch__` suits replacement, not interleaving.** Replacing a whole
  function or method is clean. Interleaved edits inside a long method force the
  twin to copy that method, and the copy then stops tracking upstream fixes with
  nothing to warn you. Reshape into a seam first, or take the change in-tree
  deliberately.
- No symbols or package names are exposed to users beyond the public
  `sglang.srt.*` API.

## Policy: what may be modified, and where (strict)

`python/sglang/srt/**` is upstream-owned. On `dev` — and every branch that
targets it — fork behavior never lands there; it lands in this package.
"Behavior" means anything observable: defaults, flags, kernels, bug fixes,
perf work, logging, new modules.

### Fork-owned — modify freely

| Tree | What goes there |
| --- | --- |
| `python/sglang/overrides/**` | all fork behavior: mirror twins, `__apply_patch__` patches, fork env vars (`environ.py`), fork arg declarations (`arg_groups/overrides.py`), and new supporting modules (underscore-prefixed if internal, e.g. `lora/_moe_padding.py`) |
| `test/**` (new files) | fork tests, registered per the `write-sglang-test` skill |
| `.github/**`, `docker/**`, `docs/**` | fork CI, images, docs |

### Upstream-owned (`python/sglang/srt/**`) — do not modify, with three narrow exceptions

Any PR to `dev` that touches `python/sglang/srt/**` fails the
`overlay-policy-gate` CI job unless it is one of:

1. **Upstream sync / backport** — a verbatim upstream commit (merge or
   cherry-pick, upstream hash in the message). Label the PR `upstream-sync`.
   This is not drift; no ledger entry.
2. **Extension-point metadata** — a metadata-only edit that an overlay
   declaration requires, upstream's own machinery defines, and that cannot be
   supplied by wrapping a function in a twin. Before reaching for this, check
   the overlay already covers it: the `resolvable` whitelist is extended
   overlay-side via `overrides/arg_groups/arg_utils.py`
   (`OVERLAY_RESOLVABLE_FIELDS`), no `srt/` edit needed. What remains here is
   metadata read structurally rather than through a wrappable function (e.g.
   `NS(...)` coverage, which two-way lints pin). No default change, no
   behavior change, a comment naming the twin that needs it, and a ledger
   entry (below) in the same PR.
3. **Overlay seam** — a minimal, behavior-neutral hook for the rare case the
   meta-path finder cannot patch cleanly (a symbol captured before patch
   time, callers that bypass the module object, …). Last resort: restructure
   the twin first. Requires a ledger entry that says why the twin could not
   do it.

Everything else goes through the overlay — or lands upstream first and
arrives here by sync. In particular, these are **not** exceptions: changing a
default, adding a flag, fixing a bug "just this once", perf tweaks, editing a
docstring or comment.

A hard line inside exception 2: never fake metadata by mutating
`ServerArgs.__dataclass_fields__` from a twin — that keeps `srt/` byte-identical
while making the dataclass lie to everyone who reads it. The sanctioned
overlay shape is wrapping the *function* that consumes the metadata (the
`arg_utils.py` twin wraps `resolvable_fields`), which leaves upstream's
dataclass honest and keeps the overlay's additions auditable in one named
set. Where no such function exists, honest, commented, ledgered `srt/` drift
beats a metadata mutation.

### The drift ledger

[`UPSTREAM_DRIFT.md`](UPSTREAM_DRIFT.md) (this directory) records every
intentional `srt/` edit: file, change, exception category, PR, and the
upstream exit plan. The CI gate requires that file to change in any PR that
touches `srt/` without the `upstream-sync` label — acknowledging the drift is
part of making it. Remove an entry when an upstream sync delivers the change
from upstream.

### Choosing the mechanism (cheat sheet)

| You want to… | Do this |
| --- | --- |
| change or extend an upstream module's behavior | mirror twin + `__apply_patch__` |
| add a fork env var | `overrides/environ.py` |
| set a fork default for a CLI arg | declaration in `overrides/arg_groups/overrides.py`; if the field is not yet `resolvable`, add it to `OVERLAY_RESOLVABLE_FIELDS` in `overrides/arg_groups/arg_utils.py` |
| add a brand-new fork module | put it in `overrides/` next to its consumer |
| fix an upstream bug | fix it upstream and sync it back; if urgent, patch via a twin and note the upstream PR in the twin's docstring |

## Quick start

1. Enable the override finder early, before anything imports `sglang.srt`:

   ```python
   # e.g. python/sglang/launch_server.py
   import sglang.overrides.patches
   ```

2. Any `sglang.srt.X` that has a twin at `sglang.overrides.X` is patched
   automatically. Everything else comes from upstream unchanged.

## How it works

`patches/auto_override.py` installs a `sys.meta_path` finder (at the front) when
imported. When Python imports a `sglang.srt.*` module that has an override twin,
the finder loads the **upstream module unchanged**, then patches the live module
object using the twin:

- If the twin defines `__apply_patch__(public_mod)`, it is called.
- Otherwise the twin's public attributes are copied onto the upstream module
  (honoring `__patch_include__` / `__patch_exclude__`).

The finder never replaces packages; it only patches module objects after load.

## Layout

```
sglang/
  srt/                       # upstream (public) modules — never edited (see Policy)
    ...
  overrides/
    patches/                 # put ALL monkey patches here
      __init__.py
      auto_override.py       # the meta-path finder
    environ.py               # override-only env-var registry (survives OSS copy)
    server_args.py           # twin: fork defaults for experimental LoRA opts
    arg_groups/
      overrides.py           # twin: fork model-override declarations
      arg_utils.py           # twin: overlay additions to the resolvable whitelist
    lora/                    # twins + fork-only supporting modules
      lora_manager.py
      layers.py
      mem_pool.py
      _moe_padding.py        # fork-only helper (not a twin)
      trtllm_lora_temp/
    UPSTREAM_DRIFT.md        # ledger of intentional srt/ edits
```
