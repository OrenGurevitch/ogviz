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


def test_the_set_comes_off_the_figure_so_a_caller_never_assembles_it() -> None:
    """The end-to-end path: a rendered figure in, a colour safe against it out.

    Assembling the set by hand is the failure this pair exists to prevent — too small and the
    winner collides with something the palette constant does not contain, too large and nothing
    clears the threshold at all.
    """
    import matplotlib.pyplot as plt

    from ogviz.color import separated_from, worst_separation
    from ogviz.qc import legend_colors, series_confusable_under_cvd

    fig, ax = plt.subplots()
    for color, label in (("#2E7CE0", "control"), ("#EFA607", "treated")):
        ax.plot([0.0, 1.0], [0.0, 1.0], color=color, label=label)
    ax.legend()
    fig.canvas.draw()

    taken = legend_colors(fig)
    assert set(taken) == {"control", "treated"}, taken

    picked = separated_from(taken.values())
    assert worst_separation(picked, taken.values()) >= 0.18

    ax.plot([0.0, 1.0], [1.0, 0.0], color=picked, label="third")
    ax.legend()
    fig.canvas.draw()
    assert series_confusable_under_cvd(fig) == [], "the check that named the set now passes it"
    plt.close(fig)


def test_a_figure_level_legend_is_read_too() -> None:
    """`ax.get_legend()` never returns one, which is how the check missed them for a while."""
    import matplotlib.pyplot as plt

    from ogviz.qc import legend_colors

    fig, ax = plt.subplots()
    for color, label in (("#2E7CE0", "a"), ("#EFA607", "b")):
        ax.plot([0.0, 1.0], [0.0, 1.0], color=color, label=label)
    fig.legend()
    fig.canvas.draw()
    assert set(legend_colors(fig)) == {"a", "b"}
    plt.close(fig)
