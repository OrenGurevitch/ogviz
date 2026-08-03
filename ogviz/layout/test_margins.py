from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import figure_margins, required_margins


def _panel(ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot([0.0, 1.0, 2.0], [1.0, 2.0, 1.5])
    ax.set_ylabel(ylabel, fontweight="bold")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.15)
    return fig


def test_a_pinned_layout_is_sized_by_the_worst_figure_in_the_set() -> None:
    """The value has to fit the WORST figure, and there is no way to know which without rendering.

    Guessed instead it fails in both directions: too generous leaves the tightest figures
    half-empty, and the obvious correction crops the ones whose ink reaches furthest.
    """
    short, long_label = _panel("n"), _panel("a considerably longer axis label")
    per_figure = [figure_margins(fig) for fig in (short, long_label)]
    assert all(margins is not None for margins in per_figure)

    worst = required_margins([short, long_label])
    assert worst.left == pytest.approx(min(m.left for m in per_figure))
    assert worst.right == pytest.approx(max(m.right for m in per_figure))


def test_padding_widens_every_side_and_stays_on_the_page() -> None:
    fig = _panel("n")
    tight = required_margins([fig])
    padded = required_margins([fig], pad=0.01)
    assert padded.left < tight.left
    assert padded.right > tight.right
    assert padded.left >= 0.0 and padded.right <= 1.0


def test_a_blank_page_is_not_measured_as_a_margin() -> None:
    blank = plt.figure(figsize=(4.0, 3.0))
    assert figure_margins(blank) is None
    with pytest.raises(AssertionError, match="at least one figure with ink"):
        required_margins([blank])
