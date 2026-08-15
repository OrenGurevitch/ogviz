"""Checks with a memory: what the layout intended, against what landed."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

pytestmark = pytest.mark.usefixtures("pinned_font")


def test_a_crowded_header_is_reported_and_a_comfortable_one_is_not() -> None:
    """The one pair no overlap rule catches: they are placed by two mechanisms, not one.

    The crowded case is SEARCHED for rather than pinned to a subplot top. The band this fires in
    is only a few pixels wide — above it the figure is comfortable, below it the two touch and
    `text_overlaps` owns the complaint — and where a given top lands in that band depends on text
    layout, which moved between the two matplotlib versions CI runs. Measuring the case is the
    difference between a test that holds on both and one that passes here and fails there.
    """
    from ogviz.qc.arrangement import CROWDED_HEADER_PX, header_crowds_the_panels

    def figure_with(top: float):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.set_title("a panel")
        fig.suptitle("A title", y=0.99)
        fig.subplots_adjust(top=top)
        fig.canvas.draw()
        return fig, ax

    def gap(fig, ax) -> float:
        lowest = min(text.get_window_extent().y0 for text in fig.texts)
        return lowest - ax.title.get_window_extent().y1

    roomy, roomy_ax = figure_with(0.72)
    assert gap(roomy, roomy_ax) > CROWDED_HEADER_PX, "the premise: this one really is comfortable"
    assert header_crowds_the_panels(roomy) == []
    plt.close(roomy)

    for step in range(40):
        fig, ax = figure_with(0.86 + step * 0.002)
        found = gap(fig, ax)
        if 0.0 <= found < CROWDED_HEADER_PX:
            notes = header_crowds_the_panels(fig)
            assert notes and "reads as deliberate" in notes[0], (found, notes)
            plt.close(fig)
            return
        plt.close(fig)
    raise AssertionError("no subplot top put the header in the crowded band; the search is wrong")


def test_it_stays_silent_where_it_cannot_judge_rather_than_inventing_a_complaint() -> None:
    """Panel text ABOVE the header's bottom is a table's top row, or a real overlap.

    Measured naively, two gallery figures come back hundreds of pixels negative and neither is
    crowding anything. `text_overlaps` owns the touching case; this one speaks only about a gap it
    can actually judge.
    """
    from ogviz.qc.arrangement import header_crowds_the_panels

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_axis_off()
    # Text filling the axes to its very top, the shape a table panel has.
    for row in range(6):
        ax.text(0.5, 1.0 - row * 0.16, f"row {row}", ha="center", transform=ax.transAxes)
    fig.suptitle("A title", y=0.5)  # deliberately BELOW the topmost cell
    fig.canvas.draw()
    assert header_crowds_the_panels(fig) == []
    plt.close(fig)


def test_no_gallery_figure_is_reported_as_crowded() -> None:
    """The floor is a regression guard set below everything that ships, not a target.

    The number it was borrowed from would have complained about two correct figures here.
    """
    from ogviz.qc.arrangement import CROWDED_HEADER_PX

    assert CROWDED_HEADER_PX < 17.6, (
        "the tightest shipped gallery figure clears by 17.6 px; a floor at or above that "
        "reports a figure that is correct"
    )


def test_an_axis_run_far_past_the_data_is_reported() -> None:
    """Ticks and gridlines for values nothing reaches, and the marks squashed into what is left."""
    from ogviz.qc.arrangement import unused_value_headroom

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.bar([0, 1, 2], [0.34, 0.48, 0.65])
    ax.set_ylim(0.0, 1.5)
    found = unused_value_headroom(fig)
    assert found and "empty above everything drawn" in found[0], found
    assert "1.5" in found[0] and "0.65" in found[0], "it names both ends of the gap"
    plt.close(fig)


def test_a_fitted_axis_says_nothing() -> None:
    """The premise for the test above, and the thing a threshold has to get right."""
    from ogviz.qc.arrangement import unused_value_headroom

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.bar([0, 1, 2], [0.34, 0.48, 0.65])
    ax.set_ylim(0.0, 0.72)
    assert unused_value_headroom(fig) == []
    plt.close(fig)


def test_headroom_a_bracket_occupies_is_headroom_in_use() -> None:
    """The whole point of the distinction: reserved space is not wasted space.

    `ticks_in_the_headroom` covers the case where the room is spoken for. This must not report the
    same panel a second time, in different words, for having the room at all.
    """
    from ogviz.qc.arrangement import unused_value_headroom
    from ogviz.tags import mark

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.bar([0, 1, 2], [0.34, 0.48, 0.65])
    ax.set_ylim(0.0, 1.5)
    assert unused_value_headroom(fig), "the premise: empty, this panel is reported"

    bracket = ax.plot([0, 2], [1.35, 1.35], color="#333")[0]
    mark(bracket, "bracket")
    assert unused_value_headroom(fig) == [], "a bracket in the room is the room being used"
    plt.close(fig)


def test_in_panel_text_counts_as_using_the_room() -> None:
    """A printed mean or an annotation occupies the axis as surely as a mark does."""
    from ogviz.qc.arrangement import unused_value_headroom

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.bar([0, 1, 2], [0.34, 0.48, 0.65])
    ax.set_ylim(0.0, 1.5)
    assert unused_value_headroom(fig), "the premise"
    ax.text(1.0, 1.35, "n = 40", ha="center")
    assert unused_value_headroom(fig) == []
    plt.close(fig)


def test_an_empty_bottom_is_not_reported() -> None:
    """A floor at zero is a deliberate and often required choice for bars.

    Empty space below the data has an innocent explanation that empty space above it does not, so
    only the top is judged.
    """
    from ogviz.qc.arrangement import unused_value_headroom

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.bar([0, 1, 2], [1.30, 1.42, 1.48])  # every bar starts at zero; the lower axis is empty
    ax.set_ylim(0.0, 1.5)
    assert unused_value_headroom(fig) == []
    plt.close(fig)


def test_no_gallery_figure_is_reported_as_over_tall() -> None:
    """The floor is set clear of the worst correct figure, not fitted to the broken one."""
    from ogviz.qc.arrangement import EMPTY_HEADROOM

    assert EMPTY_HEADROOM > 0.277, (
        "the airiest shipped panel leaves 27.7% of its axis empty above everything drawn; "
        "a floor at or below that refuses a figure that is correct"
    )


def test_a_shared_scale_is_not_reported_as_unused_headroom() -> None:
    """The room belongs to the tallest panel in the group, not to the one being measured.

    A short panel beside a tall one is empty at the top BY CONSTRUCTION, and tightening it is the
    one action that stops the grid being comparable. This is a GATE check, so getting it wrong
    refuses a correct figure — the same distinction `dead_space` was taught to make.
    """
    import numpy as np

    from ogviz import Series, bar_panel
    from ogviz.panels.grid import share_value_limits
    from ogviz.qc.arrangement import unused_value_headroom

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    bar_panel(
        axes[0],
        [Series("a", np.array([0.9, 0.5, 0.4]), "#7C9A6E", np.array([0.30, 0.05, 0.05]))],
        list("abc"),
    )
    bar_panel(
        axes[1],
        [Series("b", np.array([0.3, 0.2, 0.25]), "#E8A838", np.array([0.03, 0.03, 0.03]))],
        list("abc"),
    )
    fig.canvas.draw()
    assert unused_value_headroom(fig) == [], "the premise: neither panel is over-tall on its own"

    share_value_limits(axes, label_edge=False)
    fig.canvas.draw()
    empty = (axes[1].get_ylim()[1] - 0.35) / axes[1].get_ylim()[1]
    assert empty > 0.40, f"the premise: the short panel is now {empty:.0%} empty and would fire"
    assert unused_value_headroom(fig) == []
    plt.close(fig)
