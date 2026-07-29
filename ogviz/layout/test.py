from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ogviz.layout import baseline, hairline_grid, save, titled
from ogviz.theme import GRID, MUTED_INK, use_house_style


@pytest.fixture(autouse=True)
def _style() -> None:
    use_house_style()


def test_titled_reports_a_header_bottom_below_the_title() -> None:
    fig = plt.figure(figsize=(8, 6))
    bottom = titled(fig, "Title", subtitle="Subtitle")
    assert 0.0 < bottom < 0.98
    plt.close("all")


def test_the_title_to_subtitle_gap_scales_with_the_figure_height() -> None:
    # A fixed figure-fraction gap collides on a short figure and floats away on a tall one, so
    # the same header must occupy a SMALLER fraction of a taller figure.
    fractions = []
    for height in (4.0, 12.0):
        fig = plt.figure(figsize=(8, height))
        fractions.append(0.98 - titled(fig, "Title", subtitle="Subtitle"))
        plt.close("all")
    assert fractions[0] > 2 * fractions[1]


def test_hairline_grid_is_one_axis_only_and_below_the_data() -> None:
    _fig, ax = plt.subplots()
    hairline_grid(ax, axis="y")
    assert ax.get_axisbelow()
    assert ax.yaxis.get_gridlines()[0].get_visible()
    assert not ax.xaxis.get_gridlines()[0].get_visible()
    plt.close("all")


def test_baseline_weights_the_category_axis_over_the_value_axis() -> None:
    """The category axis reads as a boundary, not as data. It used to be a 2 pt black rule, which
    competed with the marks and — where a tick label sat on it — read as a broken line."""
    _fig, ax = plt.subplots()
    baseline(ax, axis="x")
    assert ax.spines["bottom"].get_linewidth() > ax.spines["left"].get_linewidth()
    assert ax.spines["bottom"].get_linewidth() < 2.0, "not a heavy black bar"
    assert ax.spines["bottom"].get_edgecolor()[:3] == matplotlib.colors.to_rgb(MUTED_INK)
    assert ax.spines["left"].get_edgecolor()[:3] == matplotlib.colors.to_rgb(GRID)
    plt.close("all")


def test_save_writes_both_formats_and_closes_the_figure(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = save(fig, tmp_path / "out", "demo", dpi=60)
    assert [p.name for p in written] == ["demo.png", "demo.svg"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    assert not plt.get_fignums()


def test_a_scatters_extent_comes_from_its_offsets_not_its_marker() -> None:
    """`get_paths()` on a scatter is the MARKER, a unit circle about the origin.

    Reading it as the data reported every point cloud as spanning about -0.5 to 0.5. On values of
    order one that is a small error; on a panel in ppm it put the printed mean at -0.25 on an axis
    running 0.0004 to 0.006, which is off the page.
    """
    import numpy as np

    from ogviz.layout import drawn_value_extent

    _fig, ax = plt.subplots()
    tiny = np.array([0.0011, 0.0030, 0.0050])
    ax.scatter(np.zeros_like(tiny), tiny)
    low, high = drawn_value_extent(ax)
    assert (low, high) == pytest.approx((0.0011, 0.0050)), "the offsets are the data"
    assert abs(low) < 0.5, "the marker outline must not be mistaken for the data"


def test_a_filled_body_still_reports_its_own_shape() -> None:
    """The other half: a fill has no offsets, and its path IS the shape in data coordinates."""

    from ogviz.layout import drawn_value_extent

    _fig, ax = plt.subplots()
    ax.fill_between([0.0, 1.0], [2.0, 2.0], [7.0, 7.0])
    low, high = drawn_value_extent(ax)
    assert (low, high) == pytest.approx((2.0, 7.0))


def test_the_axis_keeps_one_tick_above_the_marks() -> None:
    """An axis brackets its data. Only dropping what sits above leaves a coarse one unreadable.

    Reported on a panel with ticks every 2.0 and violins reaching 3.75: everything above 2.0 was
    removed, so the upper 1.75 of every body had no reference beside it. What is worth removing is
    the LADDER of ticks climbing through the room held open for brackets, not the tick that closes
    the data in.
    """

    from ogviz.layout import ticks_over_data

    _fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.plot([0.0, 1.0], [0.0, 3.75])
    ax.set_ylim(-3.0, 9.0)
    ax.set_yticks([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0])
    ticks_over_data(ax, 3.75)

    kept = [float(tick) for tick in ax.get_yticks()]
    assert 4.0 in kept, "the tick that brackets the top of the marks stays"
    assert 6.0 not in kept and 8.0 not in kept, "the ladder above it goes"


def test_ticks_over_data_measures_the_drawn_marks_when_not_told() -> None:
    """Callers kept handing it the DATA maximum, which is not where a violin's body ends.

    A kernel density body reaches past the largest observation, so the tick above that observation
    is inside the violin — and dropping it is what left the top of a panel unlabelled. Measuring by
    default means a caller cannot make that mistake.
    """

    from ogviz.layout import drawn_value_extent, ticks_over_data

    _fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.fill_between([0.0, 1.0], [0.0, 0.0], [3.75, 3.75])
    ax.set_ylim(-1.0, 9.0)
    ax.set_yticks([0.0, 2.0, 4.0, 6.0, 8.0])

    assert drawn_value_extent(ax)[1] == pytest.approx(3.75)
    ticks_over_data(ax)
    assert 4.0 in [float(tick) for tick in ax.get_yticks()]
    assert 8.0 not in [float(tick) for tick in ax.get_yticks()]
