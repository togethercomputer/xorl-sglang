"""Rebind twin-copied functions onto the patched module's live globals.

A verbatim copy defined inside a twin resolves module-level names through the
*twin's* globals — a bridged snapshot. The in-tree original resolved them
through the srt module's live dict, which also means runtime rebinding
(``global`` writes, tests that ``monkeypatch.setattr(mod, name, ...)``)
affected it. ``rebind`` reconstructs the copied function over
``mod.__dict__`` so the copy behaves *identically* to the in-tree original:
same live-name resolution, same ``global`` write target, same
monkeypatchability.

Names the copy needs that upstream does not define (the port's added imports,
helpers, constants) must therefore be published onto ``mod`` — the twins do
that in their ``__apply_patch__``.
"""

import types


def rebind(fn, mod, name=None):
    """Return ``fn`` rebound over ``mod.__dict__`` (handles method wrappers).

    ``name`` restores the real attribute name when the twin had to define the
    copy under a collision-proof module-level name (e.g. two classes both
    defining ``forward_cuda``).
    """
    if isinstance(fn, type):
        # Whole-class copies keep their own (twin) globals; their body already
        # executed and their methods carry no upstream-name dependence that the
        # twin's imports don't satisfy.
        return fn
    if isinstance(fn, classmethod):
        return classmethod(rebind(fn.__func__, mod, name))
    if isinstance(fn, staticmethod):
        return staticmethod(rebind(fn.__func__, mod, name))
    if isinstance(fn, property):
        return property(
            rebind(fn.fget, mod) if fn.fget else None,
            rebind(fn.fset, mod) if fn.fset else None,
            rebind(fn.fdel, mod) if fn.fdel else None,
            fn.__doc__,
        )
    new = types.FunctionType(
        fn.__code__, mod.__dict__, name or fn.__name__, fn.__defaults__, fn.__closure__
    )
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__qualname__ = fn.__qualname__ if name is None else name
    new.__doc__ = fn.__doc__
    new.__dict__.update(fn.__dict__)
    new.__wrapped_twin_copy__ = True
    return new
