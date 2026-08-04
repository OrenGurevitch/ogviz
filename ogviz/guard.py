"""Run the checks on EVERY figure that gets saved, whoever saved it.

`ogviz.save` has always audited what it writes. The gap is that a caller has to find `ogviz.save`
first, and the thing they reach for is `fig.savefig`, which is matplotlib's and knows nothing about
any of this. Three separate projects hit that: they found the panel they wanted, hand-rolled the
save, and shipped figures with defects the gate would have named.

`guard()` closes it by wrapping `Figure.savefig` itself, so the checks run on the way to disk no
matter which door the figure leaves by — including `plt.savefig`, a notebook, or a third-party
helper that saves for you.

    import ogviz
    ogviz.guard()          # from here on, every savefig is audited

or, with no code change at all, set `OGVIZ_GUARD=1` in the environment and importing ogviz turns it
on. That is the form for CI and for a paper build, where the point is that nobody has to remember.

WHY IT IS NOT ON FOR A BARE IMPORT. A library that silently changes matplotlib's behaviour the
moment it is imported is a bad neighbour: it would fail saves in code that never asked for it, break
a notebook mid-exploration, and make "why did my savefig raise" a puzzle whose answer is an import
three files away. Deliberately choosing it is one line; having it chosen for you is a footgun. The
environment variable exists so a project can make that choice once, in the place where such choices
belong.

MODES. `"raise"` refuses to write a figure that fails — the same contract `save` has. `"warn"`
writes it and reports, which is what an existing project wants on the day it turns this on, before
it has fixed anything. `"repair"` applies the fixes that have one obvious answer, re-checks, and
then behaves like `"warn"` about whatever is left.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Iterator

Mode = Literal["raise", "warn", "repair"]
ENV_VAR = "OGVIZ_GUARD"
_ORIGINAL = Figure.savefig
_AUDITING = False  # set while the guard is working, so a nested save does not re-enter
_INSTALLED = None  # the wrapper this module put on `Figure.savefig`, if any


class FigureQuality(UserWarning):
    """What the checks found on a figure that was written anyway."""


class FigureRejectedError(AssertionError):
    """A figure failed the checks on its way to disk."""


def _complaints(fig: Figure, *, mode: Mode, min_gap: float, advise: bool) -> list[str]:
    from ogviz.qc import audit
    from ogviz.qc.repair import repair

    if mode == "repair":
        repair(fig)
    found = audit(fig, min_gap=min_gap)
    if advise:
        from ogviz.layout.density import dead_space

        # Advisory, and kept separate on purpose: a deliberately airy figure is a real choice and a
        # panel holding room for a bracket is not wasting it. These never decide whether a figure is
        # written — the shipped gallery carries 76 of them and is correct.
        found = found + [f"(advisory) {note}" for note in dead_space(fig)]
    return found


def guard(
    *,
    mode: Mode = "raise",
    min_gap: float | None = None,
    advise: bool = False,
) -> None:
    """Audit every figure on its way to disk, whatever saved it. Idempotent.

    `min_gap` is the breathing room two labels on one row must have; leaving it out uses the house
    default, which catches labels that have run together rather than a figure that is merely tight.
    `advise=True` adds the dead-space notes, which never fail anything.
    """
    from ogviz.layout.overlap import DEFAULT_MIN_GAP

    floor = DEFAULT_MIN_GAP if min_gap is None else min_gap

    def savefig(self: Figure, *args: object, **kwargs: object) -> object:
        global _AUDITING
        if _AUDITING:
            return _ORIGINAL(self, *args, **kwargs)  # type: ignore[arg-type]
        _AUDITING = True
        try:
            from ogviz.qc.reading import ensure_rendered

            ensure_rendered(self)
            found = _complaints(self, mode=mode, min_gap=floor, advise=advise)
            hard = [note for note in found if not note.startswith("(advisory)")]
            if hard and mode == "raise":
                where = args[0] if args else kwargs.get("fname", "a figure")
                raise FigureRejectedError(
                    f"ogviz.guard refused to write {where}:\n  - " + "\n  - ".join(found)
                )
            if found:
                warnings.warn("ogviz.guard: " + "; ".join(found), FigureQuality, stacklevel=2)
            return _ORIGINAL(self, *args, **kwargs)  # type: ignore[arg-type]
        finally:
            _AUDITING = False

    global _INSTALLED
    savefig.__doc__ = _ORIGINAL.__doc__
    _INSTALLED = savefig
    Figure.savefig = savefig  # type: ignore[method-assign, assignment]


def unguard() -> None:
    """Put matplotlib's own `savefig` back."""
    global _INSTALLED
    _INSTALLED = None
    Figure.savefig = _ORIGINAL  # type: ignore[method-assign]


def is_guarded() -> bool:
    """Whether the `savefig` in place is the one this module installed.

    By identity rather than by an attribute on the function: `ogviz_` names belong to the tag
    vocabulary, which is about what a MARK means, and a wrapper is not a mark.
    """
    return _INSTALLED is not None and Figure.savefig is _INSTALLED


@contextmanager
def guarded(**settings: object) -> Iterator[None]:
    """`guard()` for the duration of a block, then put `savefig` back as it was.

    For a process that renders some figures under the house rules and others under a project's own,
    and for tests, which must not leave a patched `savefig` behind them.
    """
    was = Figure.savefig
    guard(**settings)  # type: ignore[arg-type]
    try:
        yield
    finally:
        Figure.savefig = was  # type: ignore[method-assign]


def guard_from_environment() -> bool:
    """Turn the guard on if `OGVIZ_GUARD` asks for it. Called once, at import.

    `OGVIZ_GUARD=1` (or `raise`/`warn`/`repair` to pick the mode, and `+advise` to add the
    dead-space notes) is how a project makes this choice once, in its environment, rather than in
    every script that draws something.
    """
    setting = os.environ.get(ENV_VAR, "").strip().lower()
    if not setting or setting in ("0", "false", "off", "no"):
        return False
    advise = "+advise" in setting
    mode = setting.replace("+advise", "").strip(" +") or "raise"
    if mode in ("1", "true", "on", "yes"):
        mode = "raise"
    if mode not in ("raise", "warn", "repair"):
        raise ValueError(
            f"{ENV_VAR}={setting!r} is not a mode; use raise, warn or repair, optionally +advise"
        )
    guard(mode=mode, advise=advise)  # type: ignore[arg-type]
    return True
