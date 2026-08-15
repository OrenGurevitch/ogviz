"""The furniture: rules, baselines, the legend pill, the colour scale.

None of it carries a value of its own, and all of it decides how hard the marks have to work.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.layout.frame import (
    baseline,
    color_scale,
    hairline_grid,
    is_color_scale,
    legend_pill,
    pill_frame,
    zero_baseline,
)
from ogviz.theme import PANEL_FILL

pytestmark = pytest.mark.usefixtures("pinned_font")


def test_a_hairline_grid_sits_under_the_marks() -> None:
    """A rule drawn over the data competes with it."""
    fig, ax = plt.subplots()
    (line,) = ax.plot([0.0, 1.0], [0.0, 1.0])
    hairline_grid(ax, axis="y")
    drawn = [one for one in ax.get_ygridlines() if one.get_visible()]
    assert drawn
    assert all(one.get_zorder() < line.get_zorder() for one in drawn)
    plt.close(fig)


def test_the_grid_goes_on_the_axis_it_was_asked_for() -> None:
    fig, ax = plt.subplots()
    hairline_grid(ax, axis="x")
    assert any(one.get_visible() for one in ax.get_xgridlines())
    assert not any(one.get_visible() for one in ax.get_ygridlines())
    plt.close(fig)


def test_a_baseline_is_heavier_than_the_grid_it_stands_among() -> None:
    """The axis is a boundary the marks stand on, not another rule."""
    fig, ax = plt.subplots()
    hairline_grid(ax, axis="y")
    baseline(ax)
    rule = next(one for one in ax.get_ygridlines() if one.get_visible())
    assert ax.spines["bottom"].get_linewidth() > rule.get_linewidth()
    plt.close(fig)


def test_zero_baseline_draws_at_zero_and_only_when_zero_is_in_view() -> None:
    fig, ax = plt.subplots()
    ax.set_ylim(-1.0, 1.0)
    zero_baseline(ax)
    at_zero = [one for one in ax.lines if list(one.get_ydata()) == [0.0, 0.0]]
    assert at_zero, "a signed panel needs the line its bars are measured from"
    plt.close(fig)


def test_the_pill_replaces_the_boxed_frame() -> None:
    """matplotlib's default is a square outlined box, which reads as a second panel."""
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], label="one")
    pill = pill_frame(ax.legend())
    frame = pill.get_frame()
    assert frame.get_edgecolor()[3] == 0.0, "no outline"
    assert frame.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb(PANEL_FILL)
    plt.close(fig)


def test_legend_pill_works_on_an_axes_and_on_a_figure() -> None:
    """A grid normally wants ONE legend, and that one belongs to the figure."""
    for on_figure in (False, True):
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0], label="one")
        legend = legend_pill(fig if on_figure else ax)
        assert legend.get_frame().get_edgecolor()[3] == 0.0
        assert (legend in fig.legends) is on_figure
        plt.close(fig)


def test_a_colour_scale_is_recognisable_as_furniture() -> None:
    """`is_color_scale` is how the checks tell a key from a panel, and it must not guess."""
    fig, ax = plt.subplots()
    image = ax.imshow([[0.0, 1.0], [1.0, 0.0]])
    bar = color_scale(ax, image, label="effect")
    assert is_color_scale(bar.ax)
    assert not is_color_scale(ax), "the panel it describes is not itself a key"
    plt.close(fig)


def test_a_colour_scale_carries_no_tick_marks() -> None:
    """matplotlib's default slab has them; a thin strip with numbers alone does not."""
    fig, ax = plt.subplots()
    image = ax.imshow([[0.0, 1.0], [1.0, 0.0]])
    bar = color_scale(ax, image, label="effect", ticks=[0.0, 0.5, 1.0])
    fig.canvas.draw()
    lengths = {tick.tick1line.get_markersize() for tick in bar.ax.yaxis.get_major_ticks()}
    assert lengths <= {0.0}, f"tick marks are drawn at {lengths}"
    plt.close(fig)
