"""Panels put on one scale, with their printed rows on one line."""

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import use_house_style


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def test_the_mean_row_is_the_visual_midpoint_on_any_scale() -> None:
    """Midway is a question about the picture, so it is measured in pixels and converted back.

    Averaged in data units the answer is right only while the axis is linear. On a log axis running
    1 to 1000, the data-space midpoint of a gap from 1 to 100 sits 108 px from the middle of a
    308 px gap — and every panel in this package is linear today, which is exactly why the error
    would have waited for the first log axis to appear.
    """
    from ogviz.panels.grid import align_mean_rows

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


def test_a_shared_scale_puts_every_bracket_on_one_line() -> None:
    """The other end of the panel from the mean row, and the same argument.

    Each panel anchors its bracket to ITS OWN data, which is right alone and wrong once the panels
    share a ceiling: the lowest-data panel gets the lowest bracket and still inherits the tallest
    panel's top, so it wears a gap several times its neighbour's. Measured on the six-panel grid
    before this existed: 0.59 above the tightest bracket, 1.84 above the loosest.
    """
    import numpy as np

    from ogviz import group_violins, share_value_limits

    def crossbars(ax):
        return [
            float(np.asarray(line.get_ydata(), dtype=float).max())
            for line in ax.lines
            if getattr(line, "ogviz_bracket", False)
        ]

    rng = np.random.default_rng(9)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0))
    for index, ax in enumerate(axes):
        group_violins(
            ax,
            [
                (0.0, rng.normal(0.0, 1.0, 30), "#E8A838", "#B97C10"),
                (1.0, rng.normal(index * 0.9, 1.0, 30), "#7C9A6E", "#4A6136"),
            ],
            comparisons=[(0.0, 1.0, 0.01)],
        )
    assert len({round(max(crossbars(ax)), 6) for ax in axes}) > 1, "they start out at three heights"

    share_value_limits(axes)
    fig.canvas.draw()
    heights = {round(max(crossbars(ax)), 6) for ax in axes}
    assert len(heights) == 1, "one line for the whole grid"
    gaps = {round(ax.get_ylim()[1] - max(crossbars(ax)), 6) for ax in axes}
    assert len(gaps) == 1, "and therefore one gap to the ceiling"


def test_aligning_a_stack_keeps_its_internal_spacing() -> None:
    """A stack moves as a unit; `bracket_stack` measured the spacing inside it and owns it."""
    import numpy as np

    from ogviz import group_violins
    from ogviz.panels.grid import align_brackets

    rng = np.random.default_rng(10)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, shift in ((left, 0.0), (right, 2.0)):
        group_violins(
            ax,
            [(float(i), rng.normal(shift, 1.0, 25), "#E8A838", "#B97C10") for i in range(3)],
            comparisons=[(0.0, 1.0, 0.001), (0.0, 2.0, 0.01), (1.0, 2.0, 0.04)],
        )
    fig.canvas.draw()

    def spacing(ax):
        tops = sorted(
            float(np.asarray(line.get_ydata(), dtype=float).max())
            for line in ax.lines
            if getattr(line, "ogviz_bracket", False)
        )
        return [round(b - a, 6) for a, b in pairwise(tops)]

    before = spacing(left)
    align_brackets([left, right])
    fig.canvas.draw()
    assert spacing(left) == before, "shifting a stack must not change the gaps inside it"
