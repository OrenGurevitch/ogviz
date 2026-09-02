"""`group_violins`: the two arguments a repeated-measures panel needs, and what they must refuse.

Both were added for the condition grids, and both are widenings — every default here reproduces
what the panel drew before them, which is what the last test in this file asserts.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection

from ogviz.panels.violins import group_violins
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
