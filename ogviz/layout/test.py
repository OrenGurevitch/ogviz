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
