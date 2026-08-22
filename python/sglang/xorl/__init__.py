"""XoRL-only code that has no upstream counterpart.

Additive modules live here, NOT under ``sglang/overrides/``. The override finder
only fires when an upstream ``sglang.srt.X`` is imported and a twin
``sglang.overrides.X`` exists, so a twin for a module upstream does not have
would never be triggered -- it would sit there looking installed and do nothing.

The split, in one line each:

    sglang/xorl/       new capability upstream has no version of
    sglang/overrides/  changed behaviour of a module upstream does have
    in-tree edits      only where neither can work (see server_args below)

Nothing here is imported automatically. Importing this package must stay free of
side effects and of ``sglang.srt`` imports: ``sglang/__init__.py`` installs the
override finder before the first ``sglang.srt`` import, and anything that drags
``sglang.srt`` in earlier would permanently un-override whatever it touched.
"""
