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
