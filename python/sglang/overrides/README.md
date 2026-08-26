# sglang.overrides

Lightweight mechanism for shipping drop-in overrides of upstream `sglang.srt.*`
modules and keeping all one-off monkey patches in one place — without editing
upstream source.

- Mirror overrides live under `sglang/overrides/*` with the same path as the
  upstream module you're replacing.
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
   declaration requires and that upstream's own machinery defines, e.g.
   `Arg(..., resolvable=True)` or `NS(...)` on a `ServerArgs` field. No
   default change, no behavior change, and a comment naming the twin that
   needs it. Requires a ledger entry (below) in the same PR.
3. **Overlay seam** — a minimal, behavior-neutral hook for the rare case the
   meta-path finder cannot patch cleanly (a symbol captured before patch
   time, callers that bypass the module object, …). Last resort: restructure
   the twin first. Requires a ledger entry that says why the twin could not
   do it.

Everything else goes through the overlay — or lands upstream first and
arrives here by sync. In particular, these are **not** exceptions: changing a
default, adding a flag, fixing a bug "just this once", perf tweaks, editing a
docstring or comment.

Why exception 2 exists at all: the declarable-fields whitelist is
deliberately dataclass metadata in `srt/server_args.py`, so the set of fields
model overrides may write stays auditable in one file. Faking it from a twin
(mutating `ServerArgs.__dataclass_fields__`) would keep the file
byte-identical while making the whitelist lie — worse than one line of honest,
commented drift. The same reasoning bounds the exception: it covers metadata
that upstream's machinery already understands, never new semantics.

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
| set a fork default for a CLI arg | declaration in `overrides/arg_groups/overrides.py` (plus an exception-2 metadata edit if the field is not yet `resolvable`) |
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
    lora/                    # twins + fork-only supporting modules
      lora_manager.py
      layers.py
      mem_pool.py
      _moe_padding.py        # fork-only helper (not a twin)
      trtllm_lora_temp/
    UPSTREAM_DRIFT.md        # ledger of intentional srt/ edits
```
