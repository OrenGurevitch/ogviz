"""The render scope: draw once for a whole sweep, and draw at all when asked alone.

`audit` cost 22 renders and 690 ms before this existed, for an answer identical to the one it gives
with the renders after the first suppressed. Everything here is about that scope being correct in
both directions — suppressing inside it, and never suppressing outside it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from ogviz.layout.render import ensure_rendered, one_render

pytestmark = pytest.mark.usefixtures("pinned_font")


def _counting(fig):
    """Wrap the canvas draw so the renders can be counted."""
    calls = []
    real = fig.canvas.draw

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    fig.canvas.draw = counted  # type: ignore[method-assign]
    return calls


def test_a_figure_built_without_pyplot_gets_a_canvas_that_can_measure() -> None:
    """The failure was `'FigureCanvasBase' object has no attribute 'get_renderer'`, from inside a
    helper, naming an internal and saying nothing about what to do.

    Attaching Agg is the answer rather than a better message: the caller wanted the figure
    measured, this is what measuring needs, and it is what `save` does for them anyway.
    """
    fig = Figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ensure_rendered(fig)
    assert hasattr(fig.canvas, "get_renderer")
    assert ax.get_window_extent().width > 0


def test_outside_the_scope_every_call_draws() -> None:
    """Each reader has to work when called on its own, which is why they draw for themselves."""
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    fig.canvas.draw()
    calls = _counting(fig)
    for _ in range(3):
        ensure_rendered(fig)
    assert len(calls) == 3
    plt.close(fig)


def test_inside_the_scope_only_the_first_draws() -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    with one_render(fig):
        calls = _counting(fig)
        for _ in range(5):
            ensure_rendered(fig)
    assert len(calls) == 0, "the scope draws once on entry and suppresses the rest"
    plt.close(fig)


def test_the_scope_is_keyed_to_its_own_figure_and_not_to_a_module_flag() -> None:
    """A `ContextVar` keyed by the figure, so two figures cannot silence each other's renders.

    `guard._AUDITING` is the older shape of this and is the module global it warns about.
    """
    first, ax_a = plt.subplots()
    second, ax_b = plt.subplots()
    for ax in (ax_a, ax_b):
        ax.plot([0.0, 1.0], [0.0, 1.0])
    first.canvas.draw()
    second.canvas.draw()
    with one_render(first):
        calls = _counting(second)
        ensure_rendered(second)
    assert len(calls) == 1, "a scope on one figure must not suppress a render of another"
    plt.close(first)
    plt.close(second)


def test_the_scope_is_released_even_when_the_body_raises() -> None:
    """Otherwise one failed check leaves every later reader measuring a stale figure."""
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(RuntimeError), one_render(fig):
        message = "a check failed inside the sweep"
        raise RuntimeError(message)
    calls = _counting(fig)
    ensure_rendered(fig)
    assert len(calls) == 1, "the scope did not survive its own exception"
    plt.close(fig)


def test_nesting_the_scope_does_not_leave_it_open() -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    with one_render(fig), one_render(fig):
        pass
    calls = _counting(fig)
    ensure_rendered(fig)
    assert len(calls) == 1, "the outer scope was still suppressing after both had exited"
    plt.close(fig)
