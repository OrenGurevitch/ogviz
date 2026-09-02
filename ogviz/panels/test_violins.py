"""`group_violins`: the two arguments a repeated-measures panel needs, and what they must refuse.

Both were added for the condition grids, and both are widenings — every default here reproduces
what the panel drew before them, which is what the last test in this file asserts.
"""

from __future__ import annotations

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection

from ogviz.panels.violins import group_violins
from ogviz.qc import audit
from ogviz.tags import marked
from ogviz.theme import identity_colors


def _cohort(size: int = 12) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    own = rng.normal(0.0, 0.7, size)
    return [own + step + rng.normal(0.0, 0.5, size) for step in (0.0, 0.6, 1.2)]


def _clouds(ax) -> list[PathCollection]:
    """The scatters, which are the only PathCollections `group_violins` draws."""
    return [art for art in ax.collections if isinstance(art, PathCollection)]


def test_a_dot_keeps_its_subject_s_colour_in_every_violin() -> None:
    """The whole point of `point_colors`: one subject, one colour, across all three conditions.

    Asserted on the DRAWN facecolours rather than on the argument, because the jitter reorders
    nothing but easily could: the colours are matched to the values by position, so a change that
    sorted the values before scattering would still pass a test that only looked at the input.
    """
    values = _cohort()
    subject = identity_colors(len(values[0]))
    _fig, ax = plt.subplots()
    group_violins(
        ax,
        [(float(i), v, "#888888", "#FFFFFF") for i, v in enumerate(values)],
        point_colors=[subject] * len(values),
    )
    drawn = [cloud.get_facecolor() for cloud in _clouds(ax)]
    assert len(drawn) == len(values), "one cloud per condition"
    for cloud in drawn:
        assert len(cloud) == len(subject), "a colour per dot, not one for the cloud"
    first = drawn[0]
    for other in drawn[1:]:
        assert np.allclose(first, other), "the same subject is the same colour in every violin"


def test_point_colours_that_do_not_match_the_data_are_refused() -> None:
    """Silently recolouring is the failure worth raising over: a dot would carry someone else's
    identity, and the figure would look entirely correct."""
    values = _cohort()
    groups = [(float(i), v, "#888888", "#FFFFFF") for i, v in enumerate(values)]
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="point colours"):
        group_violins(ax, groups, point_colors=[identity_colors(3)] * len(values))
    with pytest.raises(AssertionError, match="indexed against"):
        group_violins(ax, groups, point_colors=[identity_colors(len(values[0]))])
    plt.close(fig)


def test_point_colours_are_indexed_before_empty_groups_are_dropped() -> None:
    """A caller passes one entry per group it HANDED OVER. `group_violins` drops the empty ones, and
    indexing the colours after that shifts every remaining group onto its neighbour's palette."""
    values = _cohort()
    subject = identity_colors(len(values[0]))
    marker = identity_colors(len(values[0]), saturation=0.9)
    fig, ax = plt.subplots()
    group_violins(
        ax,
        [
            (0.0, np.array([]), "#888888", "#FFFFFF"),  # dropped
            (1.0, values[0], "#888888", "#FFFFFF"),
            (2.0, values[1], "#888888", "#FFFFFF"),
        ],
        point_colors=[None, subject, marker],
    )
    clouds = _clouds(ax)
    assert len(clouds) == 2, "the empty group drew nothing"
    expected = matplotlib.colors.to_rgba_array(subject)
    assert np.allclose(clouds[0].get_facecolor()[:, :3], expected[:, :3]), (
        "the first surviving group kept ITS OWN palette, not the dropped group's"
    )
    plt.close(fig)


def test_outline_violins_gives_each_body_its_own_fill_as_an_edge() -> None:
    """And leaves the group's edge colour alone, because that one is the dots' rim."""
    values = _cohort()
    fills = ["#E8B33C", "#5B9BD5", "#3FA372"]
    fig, ax = plt.subplots()
    group_violins(
        ax,
        [
            (float(i), v, fill, "#FFFFFF")
            for i, (v, fill) in enumerate(zip(values, fills, strict=True))
        ],
        outline_violins=True,
    )
    bodies = [art for art in ax.collections if art not in _clouds(ax)]
    edges = [art.get_edgecolor() for art in bodies if len(art.get_edgecolor())]
    wanted = matplotlib.colors.to_rgba_array(fills)[:, :3]
    drawn = np.array([edge[0][:3] for edge in edges[: len(fills)]])
    assert np.allclose(drawn, wanted), "each body is outlined in its own fill"
    plt.close(fig)


def test_the_new_arguments_left_untouched_change_nothing() -> None:
    """The premise the other tests rest on: these are widenings, not a new default look.

    Without this a later change to the defaults would go unnoticed — every test above passes an
    argument explicitly, so none of them would see it.
    """
    values = _cohort()
    groups = [(float(i), v, "#888888", "#4A4A4A") for i, v in enumerate(values)]

    def facecolors():
        fig, ax = plt.subplots()
        group_violins(ax, list(groups))
        out = [cloud.get_facecolor().copy() for cloud in _clouds(ax)]
        plt.close(fig)
        return out

    for cloud in facecolors():
        assert len(cloud) == 1, "one colour for the whole cloud, as before"


def test_a_plain_panel_returns_the_top_of_its_drawn_ink_not_the_data_maximum() -> None:
    """ "Returns the topmost drawn y" meant the bracket ink's top WITH comparisons and the data
    maximum without — which sits below a kernel body's top."""
    from ogviz.layout import drawn_value_extent

    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    values = rng.normal(0.0, 1.0, 40)
    returned = group_violins(ax, [(0.0, values, "#E8A838", "#B97C10")])
    fig.canvas.draw()
    extent = drawn_value_extent(ax)
    assert extent is not None
    assert returned == pytest.approx(extent[1])
    plt.close(fig)


def test_a_constant_series_still_gets_a_panel_with_height() -> None:
    """`max(high - low, 1e-9)` kept the arithmetic safe and drew the figure into a line.

    The limits came out equal to nine decimal places, so every mark landed on one row of pixels
    and the printed mean sat on the frame — a panel that cannot say what the constant IS. A
    constant group is ordinary data: an all-zero condition, a saturated measure.
    """
    from ogviz.tags import marked

    for value in (5.0, 0.0, 1200.0):
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        group_violins(ax, [(0.0, np.full(30, value), "#E8A838", "#B97C10")])
        fig.canvas.draw()
        low, high = (float(v) for v in ax.get_ylim())
        assert high > low, f"a constant series of {value} left the axis with no height"
        row = next(text for text in ax.texts if marked(text, "mean_row"))
        # The row, the marks and the frame are three distinct rows of pixels.
        row_px = ax.transData.transform((0.0, row.get_position()[1]))[1]
        floor_px = ax.transData.transform((0.0, low))[1]
        marks_px = ax.transData.transform((0.0, value))[1]
        assert floor_px < row_px < marks_px
        plt.close(fig)


def _crowded(count: int, seed: int = 5):
    """`count` groups of values small enough that `row_decimals` gives the row seven characters."""
    rng = np.random.default_rng(seed)
    return [
        (float(i), rng.normal(0.0035 + i * 3e-5, 6e-4, 25), "#E8A838", "#B97C10")
        for i in range(count)
    ]


def test_a_mean_row_that_would_collide_shrinks_instead_of_being_refused() -> None:
    """The row could not shrink, so the GATE refused the figure for it.

    A metric whose values are small takes many decimals from `row_decimals`, and six
    seven-character numbers across one panel is wider than the panel. The premise is measured by
    pinning the size: at the full `MEAN_LABEL_SIZE` the row overlaps and `audit` reports it.
    """
    from ogviz.panels.violins import MEAN_ROW_FLOOR_PT
    from ogviz.theme import MEAN_LABEL_SIZE

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    group_violins(ax, _crowded(6), show_means=True, mean_min_fontsize=MEAN_LABEL_SIZE)
    fig.canvas.draw()
    assert [one for one in audit(fig) if "runs into" in one], (
        "premise: with the shrink floored at the full size, the row still collides"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    rows = group_violins(ax, _crowded(6), show_means=True)
    fig.canvas.draw()
    settled = {float(text.get_fontsize()) for text in ax.texts if marked(text, "mean_row")}
    assert len(settled) == 1
    assert MEAN_ROW_FLOOR_PT <= settled.pop() < MEAN_LABEL_SIZE, "it shrank, and not past the floor"
    assert [one for one in audit(fig) if "runs into" in one] == []
    assert rows is not None
    plt.close(fig)


def test_a_roomy_row_is_left_at_the_size_it_was_asked_for() -> None:
    """The shrink is a fallback, not a policy: nothing crowded means nothing moves."""
    from ogviz.theme import MEAN_LABEL_SIZE

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    rng = np.random.default_rng(0)
    roomy = [(float(i), rng.normal(5.0 + i, 1.0, 30), "#E8A838", "#B97C10") for i in range(2)]
    group_violins(ax, roomy, show_means=True)
    fig.canvas.draw()
    sizes = {float(text.get_fontsize()) for text in ax.texts if marked(text, "mean_row")}
    assert sizes == {MEAN_LABEL_SIZE}
    plt.close(fig)


def test_a_grid_reconciles_the_row_to_one_size() -> None:
    """`printed_means` shrinks per PANEL, and a shared scale demands one size across the grid.

    So the shrink created a new way to fail: `mean_rows_unaligned` refuses two sizes on purpose,
    because the row's size is one of the constants that make a shared scale read as one
    comparison. Measured before the reconciliation: 20.0 pt beside 11.5 pt, and the gate said so.
    """
    from ogviz.panels.grid import share_value_limits
    from ogviz.qc.arrangement import mean_rows_unaligned

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    group_violins(axes[0], _crowded(2, seed=6), show_means=True)
    group_violins(axes[1], _crowded(6, seed=6), show_means=True)

    apart = {float(t.get_fontsize()) for ax in axes for t in ax.texts if marked(t, "mean_row")}
    assert len(apart) == 2, f"premise: the two panels shrink differently on their own — {apart}"

    share_value_limits(axes, label_edge=False)
    fig.canvas.draw()
    together = {float(t.get_fontsize()) for ax in axes for t in ax.texts if marked(t, "mean_row")}
    assert together == {min(apart)}, "the smallest is the only size guaranteed to fit both"
    assert mean_rows_unaligned(fig) == []
    plt.close(fig)


def _slot_inches(fig, ax) -> float:
    """Inches per category step in a drawn cell — the size a violin is actually rendered at."""
    fig.canvas.draw()
    low, high = ax.get_xlim()
    return float(ax.get_position().width * fig.get_figwidth() / (high - low))


def _fill(ax, count: int, ylabel: str = "Measurement (units)") -> None:
    rng = np.random.default_rng(0)
    group_violins(
        ax,
        [(float(i), rng.normal(5.0, 1.0, 25), "#E8A838", "#B97C10") for i in range(count)],
        show_means=False,
    )
    ax.set_ylabel(ylabel, fontsize=16)


def test_violin_cells_holds_one_slot_size_across_every_shape() -> None:
    """The invariant the cell exists for: a violin is one physical size in every figure of a set.

    The premise is this package's own gallery, measured: across fourteen violin panels in six
    figures the slot runs 1.755 in to 3.188 in, a 1.82x spread — the same violin at nearly double
    the size in one figure as in another. Nothing is wrong with any one of them; the SET is what
    stops being comparable.
    """
    from ogviz.panels.violins import VIOLIN_SLOT_INCHES, violin_cells

    measured = []
    for count, rows, columns in ((2, 1, 1), (3, 1, 1), (4, 1, 1), (6, 1, 1), (3, 2, 2), (2, 1, 3)):
        fig, axes = violin_cells(count, rows=rows, columns=columns)
        for ax in axes:
            _fill(ax, count)
        measured.append(_slot_inches(fig, axes[0]))
        plt.close(fig)

    assert measured == pytest.approx([VIOLIN_SLOT_INCHES] * len(measured), abs=0.01)


def test_the_canvas_alone_does_not_hold_the_slot() -> None:
    """Which is why `violin_cells` pins the margins rather than leaving that to the caller.

    Sizing the canvas and taking matplotlib's default subplot params leaves the slot varying with
    the group count, because the default margins are FRACTIONS of a canvas that is now a different
    width for each count. A caller handed only `violin_figsize` would believe the slot was held.
    """
    from ogviz.panels.violins import violin_figsize

    loose = []
    for count in (2, 3, 4):
        fig, ax = plt.subplots(figsize=violin_figsize(count))
        _fill(ax, count)
        loose.append(_slot_inches(fig, ax))
        plt.close(fig)

    assert max(loose) / min(loose) > 1.05, (
        f"premise: the canvas alone leaves the slot drifting — {[round(v, 3) for v in loose]}"
    )


def test_a_wider_y_label_does_not_eat_the_slot() -> None:
    """Chrome is a BUDGET, not a measurement of what happened to be there.

    A `tight_layout` sizes the axes around whatever the tick labels need, so a figure with wide
    labels gets a narrower plot than one with short labels on the same canvas — which is the thing
    being pinned, and the reason this must not be run through `tight_layout`.
    """
    from ogviz.panels.violins import violin_cells

    sizes = []
    for label in ("y", "Concentration (15,000 ppb equivalent) measured at the sensor"):
        fig, axes = violin_cells(3)
        _fill(axes[0], 3, ylabel=label)
        sizes.append(_slot_inches(fig, axes[0]))
        plt.close(fig)

    assert sizes[0] == pytest.approx(sizes[1], abs=1e-9)


def test_the_width_grows_by_exactly_one_slot_per_group() -> None:
    """The property that makes the aspect derived rather than chosen."""
    from ogviz.panels.violins import VIOLIN_SLOT_INCHES, violin_figsize

    widths = [violin_figsize(count)[0] for count in (2, 3, 4, 5)]
    steps = [b - a for a, b in pairwise(widths)]
    assert steps == pytest.approx([VIOLIN_SLOT_INCHES] * 3)


def test_a_cell_needs_a_group_and_a_grid_needs_a_cell() -> None:
    from ogviz.panels.violins import violin_figsize

    with pytest.raises(AssertionError, match="at least one group"):
        violin_figsize(0)
    with pytest.raises(AssertionError, match="no cells in it"):
        violin_figsize(3, rows=0)
    with pytest.raises(AssertionError, match="a slot is a width"):
        violin_figsize(3, per_slot=0.0)
