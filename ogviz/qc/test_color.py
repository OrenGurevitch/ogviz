"""`series_confusable_under_cvd`: which legends it reads, and what it will not crash on."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ogviz import legend_pill
from ogviz.qc.color import series_confusable_under_cvd

# A pair that separates for normal vision and merges under deuteranopia — the canonical case, and
# the one the check exists for.
CONFUSABLE = ("#D62728", "#2CA02C")


def _two_series(ax) -> None:
    for color, label in zip(CONFUSABLE, ("alpha", "beta"), strict=True):
        ax.plot([0, 1], [0, 1], color=color, label=label)


def test_a_confusable_pair_is_reported_in_an_axes_legend() -> None:
    """The premise for the figure-legend test below: this pair really is reported somewhere."""
    fig, ax = plt.subplots()
    _two_series(ax)
    legend_pill(ax)
    fig.canvas.draw()
    assert series_confusable_under_cvd(fig), "the pair is confusable at all"
    plt.close(fig)


def test_a_confusable_pair_is_reported_in_a_figure_level_legend() -> None:
    """A legend attached to the figure lives in `fig.legends`; `ax.get_legend()` never returns it.

    `legend_pill` accepts a Figure, so the supported way of giving a grid one shared legend produced
    exactly the legend this check could not see, and the pair passed clean.
    """
    fig, ax = plt.subplots()
    _two_series(ax)
    legend_pill(fig)
    fig.canvas.draw()
    assert ax.get_legend() is None, "the premise: this legend belongs to the figure, not the axes"
    assert fig.legends, "and the figure really carries one"
    assert series_confusable_under_cvd(fig), "a figure legend is read like an axes legend"
    plt.close(fig)


def test_both_kinds_of_legend_on_one_figure_are_read() -> None:
    """A grid with a shared figure legend AND a panel that names its own series."""
    fig, axes = plt.subplots(1, 2)
    _two_series(axes[0])
    _two_series(axes[1])
    legend_pill(axes[1])
    legend_pill(fig)
    fig.canvas.draw()
    found = series_confusable_under_cvd(fig)
    assert len(found) == 2, "one complaint per legend, not one per figure"
    plt.close(fig)


def test_a_legend_of_one_series_says_nothing() -> None:
    """There is no pair to confuse, and a one-entry legend is common on a reference line."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=CONFUSABLE[0], label="only")
    legend_pill(ax)
    fig.canvas.draw()
    assert not series_confusable_under_cvd(fig)
    plt.close(fig)
