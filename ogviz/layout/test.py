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


def test_the_mean_row_is_the_visual_midpoint_on_any_scale() -> None:
    """Midway is a question about the picture, so it is measured in pixels and converted back.

    Averaged in data units the answer is right only while the axis is linear. On a log axis running
    1 to 1000, the data-space midpoint of a gap from 1 to 100 sits 108 px from the middle of a
    308 px gap — and every panel in this package is linear today, which is exactly why the error
    would have waited for the first log axis to appear.
    """
    from ogviz.layout import align_mean_rows

    for scale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(5.0, 6.0))
        ax.set_yscale(scale)
        ax.set_ylim(1.0, 1000.0)
        ax.fill_between([-0.4, 0.4], [100.0, 100.0], [800.0, 800.0])
        row = ax.text(0.0, 50.0, "123", ha="center")
        row.ogviz_mean_row = True
        fig.canvas.draw()

        line = align_mean_rows([ax], floor=1.0)
        assert line is not None
        fig.canvas.draw()
        floor_px, body_px, row_px = (
            float(ax.transData.transform((0.0, value))[1]) for value in (1.0, 100.0, line)
        )
        assert row_px == pytest.approx((floor_px + body_px) / 2, abs=0.5), (
            f"{scale}: the row must sit at the visual middle, not the arithmetic one"
        )
        plt.close(fig)


def test_the_rows_align_whatever_the_grid_shape() -> None:
    """1x2 today, 2x2 today, 1x3 tomorrow — the rule cannot depend on the arrangement."""
    import numpy as np

    from ogviz import group_violins, share_value_limits

    rng = np.random.default_rng(5)
    for rows, columns in ((1, 2), (2, 2), (1, 3), (3, 1), (2, 3)):
        fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.0 * rows), squeeze=False)
        for index, ax in enumerate(axes.flat):
            group_violins(ax, [(0.0, rng.normal(index * 0.7, 1.0, 30), "#E8A838", "#B97C10")])
        share_value_limits(axes.flat)
        fig.canvas.draw()
        heights = {
            round(float(text.get_position()[1]), 6)
            for ax in axes.flat
            for text in ax.texts
            if getattr(text, "ogviz_mean_row", False)
        }
        assert len(heights) == 1, f"{rows}x{columns} put its rows at {len(heights)} heights"
        plt.close(fig)
