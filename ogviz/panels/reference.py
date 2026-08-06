"""Marks a reader measures the data AGAINST: a level, and a range.

A reference is context rather than a finding, so both are drawn recessive and both are careful about
where their label goes — a label is what makes a reference readable, and it is also the thing most
likely to end up on top of the data it is there to compare with.

They live here rather than in `bars` because nothing about them is a bar. A line panel, a violin
panel and a matrix can all want a published threshold or an agreement band, and reaching into the
bar module for one is how `heatmap` came to import its string formatter from `panels.bars`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ogviz.marks import Z_ERROR
from ogviz.orientation import (
    constant_value_line,
    place_many,
    value_transform,
)
from ogviz.require import require
from ogviz.tags import mark, value_of
from ogviz.theme import MUTED_INK, VALUE_LABEL_SIZE

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.text import Text

    from ogviz.orientation import Orientation

Z_BAND_FILL = 0.5  # a band's fill is context: under the bars, and under the frame
# Derived, not repeated: a reference reads AGAINST the marks, so it sits a fixed step above the
# error bars rather than at a number that has to be kept in agreement with them by hand.
Z_REFERENCE = Z_ERROR + 0.75


def reference_line(
    ax: Axes,
    value: float,
    label: str,
    *,
    orientation: Orientation = "vertical",
    label_side: Literal["left", "right"] = "left",
    span: tuple[float, float] | None = None,
):
    """A dashed comparison level, labelled at whichever end the bars leave room.

    `span` limits it to part of the category axis, in category coordinates. A reference is read
    against something, and sometimes only some of the panel is that something: a ceiling that the
    comparable arms are measured against says nothing about the reference arm standing beside them,
    and drawn across the whole axis it claims a comparison the figure does not make.

    The side is given rather than chosen, because it cannot be chosen here: `bar_panel` draws its
    value labels LAST, after the limits have settled, so at this moment there is nothing to test a
    collision against. A panel whose bars ascend to the right wants `label_side="right"`, and the
    overlap check will say so if the wrong one is picked.
    """
    if span is None:
        threshold = constant_value_line(
            ax, orientation, value, ls="--", color=MUTED_INK, lw=1.4, zorder=Z_REFERENCE
        )
    else:
        low, high = span
        require(
            low < high,
            f"a span runs low to high, got {span}",
        )
        along, across = place_many(orientation, [low, high], [value, value])
        (threshold,) = ax.plot(along, across, ls="--", color=MUTED_INK, lw=1.4, zorder=Z_REFERENCE)
    mark(threshold, "reference")  # must stay readable
    at_left = label_side == "left"
    along, across = place_many(orientation, 0.012 if at_left else 0.988, value)
    drawn = ax.text(
        along,
        across,
        label,
        transform=value_transform(ax, orientation),
        ha="left" if at_left else "right",
        va="bottom",
        fontsize=VALUE_LABEL_SIZE * 0.8,
        color=MUTED_INK,
        zorder=2,
    )
    # This label names the line it sits on; moving it off that line is the opposite of the fix. It
    # is excused against THAT line and nothing else — it slides freely along it, so a bar it lands
    # on is a real defect and has to be reported.
    mark(drawn, "anchored")
    mark(drawn, "anchor", threshold)
    return drawn


LABEL_SLOTS = 24  # candidate positions along a threshold, left to right


def slide_label_clear(ax: Axes, label: Text) -> None:
    """Move a threshold's label along its own line until it sits on nothing.

    It can only be done here, at the end. The label is placed when the line is drawn, and at that
    moment the bars have no final heights and the value labels do not exist — so there is nothing
    to test against. By now everything else is on the page.

    The label slides along the line and never off it: that is the axis it is free on, and the line
    is the thing it names. Both sides of the line are tried, because a threshold above a short bar
    has room underneath it that a threshold below a tall one does not.
    """
    fig = ax.figure
    fig.canvas.draw()
    anchor = value_of(label, "anchor", None)
    obstacles = [
        artist
        for artist in [*ax.patches, *ax.texts, *ax.lines, *ax.collections]
        if artist is not label and artist is not anchor
    ]
    boxes = [artist.get_window_extent() for artist in obstacles]
    original = label.get_position()
    for vertical in ("bottom", "top"):
        for slot in range(LABEL_SLOTS):
            fraction = 0.012 + slot * (0.976 / max(LABEL_SLOTS - 1, 1))
            label.set_position((fraction, original[1]))
            label.set_verticalalignment(vertical)
            label.set_horizontalalignment("left" if fraction < 0.5 else "right")
            fig.canvas.draw()
            if not any(label.get_window_extent().overlaps(box) for box in boxes):
                return
    # Nowhere along the line is clear — a crowded panel, which is normal rather than exceptional.
    # The label goes just outside the axes, level with its own line. It is still unambiguously that
    # line's label, it is legible, and it is the one place in a full panel guaranteed to be empty.
    # `save` writes with a tight bounding box, so the margin it needs comes with it.
    label.set_position((1.012, original[1]))
    label.set_horizontalalignment("left")
    label.set_verticalalignment("center")
    fig.canvas.draw()
