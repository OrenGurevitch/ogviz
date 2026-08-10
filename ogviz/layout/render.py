"""Getting a figure into a state where it can be measured, once rather than once per question.

Everything in this package that reads geometry — extents, ink, pixels — needs the figure rendered
first, and each such function has always done it itself, because each has to work when called on its
own. Run as a SET, which is what `audit` does, that meant one full canvas render per question asked.

MEASURED on a 2x3 bar grid: `audit` cost 22 renders and 690 ms, for an answer that is identical when
the renders after the first are suppressed. The checks read and do not write, so the other 21 bought
nothing.

`one_render` is the scope that says so. Inside it, `ensure_rendered` returns without drawing;
outside it — which is every direct call, and every builder — it draws exactly as before.

WHY THE CALLER SCOPES IT rather than this module deciding from matplotlib's `Figure.stale`: that
flag is wrong in both directions for this use. `Text.set_bbox`, which is precisely what the knockout
repair does, leaves `stale` False; and merely reading a spine's window extent sets it True. Measured
both. A flag that says "clean" after a mutation and "dirty" after a read cannot gate a render.

This lives under `layout` rather than under `qc` because the readers do: `qc.reading` imports from
`layout.collision` and `layout.overlap`, so the render scope has to sit below both of them or the
import graph turns into a cycle. `qc` re-exports it, which is where a check looks for it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure


# NOT `ogviz_`-prefixed, and `test_tags.py` is what says so: that prefix is the tag vocabulary, for
# what a MARK means, and this is a scope flag rather than anything written onto an artist.
#
# A `ContextVar` keyed by the figure rather than a module-level bool, so two threads auditing two
# figures cannot silence each other's renders. `guard._AUDITING` is the older shape of this and is
# the module global it warns about.
_SWEEPING: ContextVar[int | None] = ContextVar("qc_sweeping_figure", default=None)


def ensure_rendered(fig: Figure) -> None:
    """Give the figure a canvas that can measure, and current geometry to measure on it.

    A `Figure` built without pyplot carries a `FigureCanvasBase`, which cannot produce a renderer.
    The failure was `AttributeError: 'FigureCanvasBase' object has no attribute 'get_renderer'`,
    raised from inside a helper, naming an internal and saying nothing about what to do.

    Attaching Agg is the answer rather than a better error message: the caller wanted the figure
    measured, this is what measuring needs, and it is what `save` does for them anyway.

    The draw is skipped inside `one_render` and nowhere else.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    if _SWEEPING.get() == id(fig):
        return
    fig.canvas.draw()


@contextmanager
def one_render(fig: Figure) -> Iterator[None]:
    """Render `fig` once for a whole sweep of READ-ONLY questions, instead of once per question.

    Sound only because nothing inside writes to the figure. That is not a hope — it is the reason a
    check that starts mutating has to leave `CHECKS`, and it is why `repair` is deliberately NOT
    wrapped: it moves labels and adds knockout boxes, so each of its steps must see the figure as
    the previous step left it.
    """
    ensure_rendered(fig)
    token = _SWEEPING.set(id(fig))
    try:
        yield
    finally:
        _SWEEPING.reset(token)
