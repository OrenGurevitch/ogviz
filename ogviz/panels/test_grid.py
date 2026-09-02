"""Panels put on one scale, with their printed rows on one line."""

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.tags import mark, marked


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
        mark(row, "mean_row")
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
            if marked(text, "mean_row")
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
            if marked(line, "bracket")
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
            if marked(line, "bracket")
        )
        return [round(b - a, 6) for a, b in pairwise(tops)]

    before = spacing(left)
    align_brackets([left, right])
    fig.canvas.draw()
    assert spacing(left) == before, "shifting a stack must not change the gaps inside it"


def test_panels_on_one_scale_carry_one_set_of_ticks() -> None:
    """Reported on a 2x2 that arrived with five rules in one row and eight in the next.

    Each panel chose its ticks from its own data before the scale was shared. The rules are what a
    reader compares panels with, so different rules per panel make the same height look different.
    """
    import numpy as np

    from ogviz import group_violins, share_value_limits
    from ogviz.qc import panels_disagree_about_ticks

    def tick_sets(axes):
        sets = set()
        for ax in axes:
            low, high = ax.get_ylim()
            sets.add(tuple(round(float(t), 6) for t in ax.get_yticks() if low <= t <= high))
        return sets

    rng = np.random.default_rng(12)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0))
    for index, ax in enumerate(axes.flat):
        group_violins(ax, [(0.0, rng.normal(index * 1.2, 1.0, 30), "#E8A838", "#B97C10")])
    fig.canvas.draw()
    assert len(tick_sets(axes.flat)) > 1, "they start out disagreeing"

    share_value_limits(axes.flat)
    fig.canvas.draw()
    assert len(tick_sets(axes.flat)) == 1, "one set for the whole grid"
    assert not panels_disagree_about_ticks(fig)


def test_a_panel_draws_its_own_value_grid_and_can_be_asked_not_to() -> None:
    """Every example had to remember `hairline_grid`, and the two grid examples did not."""
    import numpy as np

    from ogviz import group_violins

    values = np.random.default_rng(3).normal(0.0, 1.0, 30)
    _fig, ax = plt.subplots()
    group_violins(ax, [(0.0, values, "#E8A838", "#B97C10")])
    assert ax.yaxis.get_gridlines()[0].get_visible(), "on by default"

    _fig2, bare = plt.subplots()
    group_violins(bare, [(0.0, values, "#E8A838", "#B97C10")], grid=False)
    assert not bare.yaxis.get_gridlines()[0].get_visible(), "and removable"


def _numbered(ax) -> bool:
    return any(label.get_text() for label in ax.get_yticklabels() if label.get_visible())


def test_a_shared_scale_prints_its_numbers_on_the_edge_panels_only() -> None:
    """Six panels on one scale repeating the same six numbers say nothing six times."""
    import numpy as np

    from ogviz import group_violins, share_value_limits

    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(2, 3)
    for ax in axes.flat:
        group_violins(ax, [(0.0, rng.normal(0.0, 1.0, 30), "#E8A838", "#B97C10")])
    share_value_limits(axes.flat)
    fig.canvas.draw()
    assert all(_numbered(ax) for ax in axes[:, 0]), "the first column carries the numbers"
    assert not any(_numbered(ax) for ax in axes[:, 1:].flat), "the inner columns repeat nothing"


def test_the_edge_is_read_from_the_grid_and_not_guessed_from_its_shape() -> None:
    """The defect in the caller-side version: `axes[:, 1]` is a guess about how wide the grid is.

    Written against a 2-wide grid, where column 1 IS the inner column. On a 2x3 grid it blanks
    the middle column and leaves the right one repeating, with nothing to say it went wrong.
    """
    import numpy as np

    from ogviz import group_violins, share_value_limits

    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(2, 3)
    for ax in axes.flat:
        group_violins(ax, [(0.0, rng.normal(0.0, 1.0, 30), "#E8A838", "#B97C10")])
    share_value_limits(axes.flat, label_edge=False)
    for ax in axes[:, 1]:  # the caller-side way
        ax.set_yticklabels([])
    fig.canvas.draw()
    guessed = [[_numbered(ax) for ax in row] for row in axes]
    assert guessed == [[True, False, True], [True, False, True]], guessed

    share_value_limits(axes.flat)  # the same grid, labelled from each panel's subplotspec
    fig.canvas.draw()
    assert [[_numbered(ax) for ax in row] for row in axes] == [[True, False, False]] * 2


def test_a_panel_labels_its_own_categories() -> None:
    """`bar_panel` always did; this did not, so three examples set the ticks and labels by hand."""
    import numpy as np

    from ogviz import group_violins

    rng = np.random.default_rng(6)
    # The middle group is empty: it is dropped from the drawn positions, and labelling that shorter
    # list would slide "Moria" onto Bree's slot.
    groups = [
        (0.0, rng.normal(0.0, 1.0, 30), "#E8A838", "#B97C10"),
        (1.0, np.array([]), "#7C9A6E", "#4A6136"),
        (2.0, rng.normal(2.0, 1.0, 30), "#6E8CA0", "#3C566B"),
    ]
    fig, ax = plt.subplots()
    group_violins(ax, groups, categories=["Shire", "Bree", "Moria"])
    fig.canvas.draw()
    positions = [float(tick) for tick in ax.get_xticks()]
    placed = dict(zip(positions, [t.get_text() for t in ax.get_xticklabels()], strict=True))
    assert placed == {0.0: "Shire", 1.0: "Bree", 2.0: "Moria"}, placed


def test_a_misspelled_orientation_is_refused_by_the_grid_helpers() -> None:
    """`orientation == "vertical"` made a typo mean "horizontal"; `is_vertical` refuses it, as
    every other orientation-taking function here does."""
    from ogviz.panels.grid import align_ticks, label_shared_scale_once, share_value_limits

    fig, ax = plt.subplots()
    for helper in (share_value_limits, align_ticks):
        with pytest.raises(AssertionError, match="orientation"):
            helper([ax], orientation="sideways")  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="orientation"):
        label_shared_scale_once([ax], orientation="sideways")  # type: ignore[arg-type]
    plt.close(fig)


def test_the_mean_rows_of_a_horizontal_grid_align_along_the_value_axis() -> None:
    """Every step of the answer read y: the extent, the midpoint, and the coordinate moved.

    On a horizontal panel that measures the CATEGORY axis, takes a midpoint of a floor belonging
    to the other axis, and then moves each label across its own violin — three wrong answers
    composing into a row that looks placed. `printed_means` has drawn a horizontal row correctly
    since it was written, so this was the half of the pair that could not settle one, and it is
    why `show_means=True` on a horizontal panel declined without saying so.
    """
    import numpy as np

    from ogviz import group_violins
    from ogviz.panels.grid import align_mean_rows

    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    for ax, shift in zip(axes, (0.0, 3.0), strict=True):
        group_violins(
            ax,
            [(0.0, rng.normal(5.0 + shift, 1.0, 40), "#E8A838", "#B97C10")],
            orientation="horizontal",
            show_means=True,
        )
        ax.set_xlim(0.0, 12.0)
    fig.canvas.draw()

    def rows() -> list[tuple[float, float]]:
        return [t.get_position() for ax in axes for t in ax.texts if marked(t, "mean_row")]

    before = rows()
    assert before[0][0] != pytest.approx(before[1][0]), "premise: the two rows start apart"
    categories = [position[1] for position in before]

    line = align_mean_rows(axes, floor=0.0, orientation="horizontal")
    assert line is not None
    after = rows()
    assert [position[0] for position in after] == pytest.approx([line, line])
    assert [position[1] for position in after] == categories, (
        "the category coordinate is not the row"
    )
    plt.close(fig)
