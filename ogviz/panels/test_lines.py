"""The line panel: what a series may be given as, and the three helpers that shape its axes.

`ogviz/panels/lines.py` had no test module of its own until 2026-08-10. It was reached only through
`examples/__main__.py`, which the gallery renders — so it was covered in the sense that a break
would show up as a changed image, and not in the sense that anything stated what it promises. The
gap that found: `Line` typed its points `NDArray` and `line_panel` read `line.x.shape`, so a caller
handing it the plain lists everyone hands matplotlib met an `AttributeError` naming `list`, raised
from inside the `require` call that exists to give them a sentence instead.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.panels.lines import Line, broken_zero, line_panel, money_ticks, value_floor
from ogviz.qc import audit


@pytest.mark.parametrize(
    "points",
    [
        pytest.param(([0, 1, 2], [1.0, 2.0, 3.0]), id="lists"),
        pytest.param(((0, 1, 2), (1.0, 2.0, 3.0)), id="tuples"),
        pytest.param((np.arange(3.0), np.arange(1.0, 4.0)), id="arrays"),
        pytest.param((range(3), [1.0, 2.0, 3.0]), id="a-range"),
    ],
)
def test_a_series_takes_whatever_matplotlib_takes(points) -> None:
    """Array-LIKE, as `bars.Series` and `effect_heatmap` already are. Coerced on the way in."""
    x, y = points
    line = Line(label="a", x=x, y=y, color="#2E7CE0")
    assert isinstance(line.x, np.ndarray) and line.x.dtype == float
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    line_panel(ax, [line])
    assert audit(fig) == []
    plt.close(fig)


def test_mismatched_series_lengths_are_refused_with_a_sentence() -> None:
    """And the sentence is the point — this used to be an AttributeError about `list`."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    with pytest.raises(AssertionError, match="3 x values and 2 y values"):
        line_panel(ax, [Line(label="a", x=[0, 1, 2], y=[1.0, 2.0], color="#2E7CE0")])
    plt.close(fig)


def test_a_panel_with_no_series_is_refused() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    with pytest.raises(AssertionError, match="at least one series"):
        line_panel(ax, [])
    plt.close(fig)


def test_the_floor_sits_below_every_point_it_was_given() -> None:
    """`value_floor` is what `broken_zero` is told to break at, so it must clear the data."""
    lines = [
        Line(label="a", x=[0, 1], y=[4.0, 9.0], color="#2E7CE0"),
        Line(label="b", x=[0, 1], y=[6.0, 7.0], color="#EFA607"),
    ]
    floor = value_floor(lines)
    assert floor < 4.0, "below the lowest point of any series"
    assert floor > 0.0, "but not all the way to zero — that is what the break is for"


def test_a_broken_zero_leaves_the_floor_below_the_data() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    lines = [Line(label="a", x=[0, 1, 2], y=[4.0, 9.0, 6.0], color="#2E7CE0")]
    line_panel(ax, lines)
    floor = value_floor(lines)
    broken_zero(ax, floor=floor)
    assert ax.get_ylim()[0] <= floor, "the axis starts at or below the break"
    plt.close(fig)


def test_money_ticks_label_the_positions_asked_for_on_a_log_axis() -> None:
    """A log axis defaults to decades; the readable answer is the values the data sits at."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    line_panel(ax, [Line(label="a", x=[0.2, 1.0, 5.0], y=[1.0, 2.0, 3.0], color="#2E7CE0")])
    money_ticks(ax, [0.2, 1.0, 5.0])
    fig.canvas.draw()

    assert ax.get_xscale() == "log"
    drawn = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert drawn == ["$0.20", "$1.00", "$5.00"], drawn
    low, high = ax.get_xlim()
    assert low < 0.2 and high > 5.0, "the axis starts before its first tick, or the label collides"
    plt.close(fig)


def test_a_muted_series_is_drawn_behind_the_others() -> None:
    """A baseline should be findable and not competing — which is a z-order claim, so assert it."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    line_panel(
        ax,
        [
            Line(label="lead", x=[0, 1, 2], y=[1.0, 2.0, 3.0], color="#2E7CE0"),
            Line(label="baseline", x=[0, 1, 2], y=[0.5, 0.6, 0.7], color="#E7E5DD", muted=True),
        ],
    )
    by_label = {line.get_label(): line.get_zorder() for line in ax.lines}
    assert by_label["baseline"] < by_label["lead"]
    plt.close(fig)
