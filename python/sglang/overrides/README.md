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
  srt/                       # upstream (public) modules — never edited
    ...
  overrides/
    patches/                 # put ALL monkey patches here
      __init__.py
      auto_override.py       # the meta-path finder
    environ.py               # override-only env-var registry (survives OSS copy)
    layers/
      linear.py              # example mirror override (replaces srt.layers.linear)
    entrypoints/
      http_server.py         # example mirror override (replaces srt.entrypoints.http_server)
```
