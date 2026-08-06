"""Refusals that survive `python -O`.

Every contract in this package was written as `assert`, and `assert` is the one statement Python is
allowed to delete: `python -O` removes it, message and all. That put the whole of this package's
claim — "a broken figure cannot reach disk" — behind an interpreter flag nobody sets deliberately
and any packaging step might. Measured on 2026-08-06: under `-O`, `assert_clean` passed a figure
whose two labels shared 524 px of ink, and `stars(-5.0)` returned `"***"`, which is precisely the
outcome its own docstring says it exists to prevent — a sign error upstream printing as the most
significant result on the figure.

So the refusals go through here instead. `AssertionError` is kept as the type, because that is what
the messages read as, what the tests already expect, and what a consumer that catches anything
catches. Only the removability changes.

WHAT BELONGS HERE: the build gates, and every check on a value a CALLER passed in — a p-value, an
interval that runs backwards, a series with a non-finite value in it, a shape that does not match.
Those are the ones whose absence produces a wrong figure rather than a crash.

WHAT DOES NOT: an internal invariant such as `assert figure is not None`, which says "this cannot
happen" about this package's own code rather than about anyone's data. A plain `assert` is right
there, and that is what `assert` is for.
"""

from __future__ import annotations


def require(condition: object, message: str) -> None:
    """Raise `AssertionError(message)` unless `condition` is truthy.

    The message is built by the caller before the call, so keep it cheap — these guard figure-build
    calls that happen once per panel, not inner loops. Where a check IS in a loop (`is_vertical`,
    asked once per mark), the branch is written out at the site instead.
    """
    if not condition:
        raise AssertionError(message)
