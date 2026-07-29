"""Catch overlapping text before the figure ships, instead of after someone squints at it.

Overlap is the defect that keeps coming back, because it depends on the rendered size of every
string and therefore on the font, the figure size and the data range — none of which a builder
can reason about while writing the call. So measure it: draw the figure, take every visible text
artist's window extent, and report any pair on one row that sits closer than `min_gap`.

The threshold exists because tick labels and legend entries legitimately abut. A few percent of
shared area is kerning; a fifth of a label buried under another is a defect.

Ported from a sibling project, which introduced the check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

DEFAULT_MIN_GAP = 5.0  # px; below this two labels on one row read as one word


def _visible_texts(fig: Figure) -> list[Text]:
    items: list[Text] = list(fig.texts)
    for ax in fig.axes:
        items += [*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.axison:
            # `ax.axis("off")` stops the axis being DRAWN but leaves its tick label artists with
            # positions and `get_visible() is True`. Collecting them anyway reports collisions
            # against labels that are not on the page — which a table, drawn on a bare axes, hits
            # for every tick it never shows.
            items += drawn_tick_labels(ax)
        legend = ax.get_legend()
        if legend is not None:
            items += legend.get_texts()
    return [t for t in items if t.get_visible() and t.get_text().strip()]


SAME_ROW_FRACTION = 0.5  # of the shorter box's height, before two labels count as one row


def drawn_tick_labels(ax: Axes) -> list[Text]:
    """Tick labels for ticks that are actually inside the axis limits.

    A locator generates ticks past the limit — an axis capped at 2.15 still carries a 2.25 tick —
    and matplotlib simply does not draw those. Their label artists keep a position, though, and it
    lies outside the axes, so collecting them reports collisions with whatever sits beyond the
    panel edge. A subtitle was flagged as running into a tick label that is not on the page.
    """
    found: list[Text] = []
    for axis, ticks, limits in (
        (ax.xaxis, ax.get_xticks(), ax.get_xlim()),
        (ax.yaxis, ax.get_yticks(), ax.get_ylim()),
    ):
        low, high = min(limits), max(limits)
        for tick, label in zip(ticks, axis.get_ticklabels(), strict=False):
            if low - 1e-9 <= float(tick) <= high + 1e-9:
                found.append(label)
    return found


def _horizontal_gap(first: Bbox, second: Bbox) -> float | None:
    """Pixels between two boxes ON THE SAME TEXT ROW, or None if they are not.

    "Same row" needs most of the shorter box's height to be shared, not any overlap at all. At the
    axes origin the x tick and the y tick share a sliver of vertical extent while sitting
    diagonally apart, and a bare-overlap test reads that as one row and flags a collision that
    nobody can see.
    """
    shared_height = min(first.y1, second.y1) - max(first.y0, second.y0)
    shorter = min(first.height, second.height)
    if shorter <= 0 or shared_height < SAME_ROW_FRACTION * shorter:
        return None
    return max(first.x0, second.x0) - min(first.x1, second.x1)


def text_overlaps(fig: Figure, *, min_gap: float = DEFAULT_MIN_GAP) -> list[str]:
    """Text pairs sitting so close on one row that they read as a single word.

    This is the SPACING rule, and it is all that is left here. It catches what no overlap test can:
    a tick row where "cognition" ends 3 px before "autonomic" begins shares no pixel and still
    renders as "cognitionautonomic".

    Whether two labels actually collide is `colliding_ink`'s question, answered on rendered pixels.
    A bounding-box overlap rule used to live here too and was wrong in both directions — it reported
    pairs whose boxes intersect while no glyph does, and, because its threshold was a fraction of
    the smaller box's AREA, it stayed silent on a real collision where descenders met ascenders
    across a 3% overlap. Measured, not argued: that case is in `test_overlap.py`.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    boxes = []
    for text in _visible_texts(fig):
        box = text.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            boxes.append((text.get_text().strip().replace("\n", "⏎"), box))

    hits = []
    for index, (first_label, first_box) in enumerate(boxes):
        for second_label, second_box in boxes[index + 1 :]:
            # Runs on every same-row pair, including intersecting ones. An earlier version skipped
            # it whenever the boxes intersected, so a pair overlapping by a little escaped both
            # rules — the worse condition passing while the milder one was caught.
            gap = _horizontal_gap(first_box, second_box)
            if gap is not None and gap < min_gap:
                verb = "touches" if gap >= 0 else "runs into"
                hits.append(f"{first_label!r} {verb} {second_label!r} ({gap:.1f} px apart)")
                continue
    return hits


def assert_no_text_overlap(
    fig: Figure,
    *,
    min_gap: float = DEFAULT_MIN_GAP,
) -> None:
    """Fail the build rather than write a figure whose labels sit on or against each other."""
    hits = text_overlaps(fig, min_gap=min_gap)
    assert not hits, "text collisions: " + " | ".join(hits)


def clipped_artists(fig: Figure) -> list[str]:
    """Artists that fall outside their axes and will be silently cropped.

    The overlap check measures where text IS and knows nothing about clipping, so a bracket line
    that ran past the axis limit passes it with zero problems, which is how a star ended up
    floating over nothing in a shipped figure. matplotlib clips `Line2D` by default and does not
    clip `Text`, so an overflowing stack loses its lines and keeps its labels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    escaped = []
    for ax in fig.axes:
        frame = ax.get_window_extent(renderer)
        for line in ax.lines:
            vertices = np.asarray(line.get_path().vertices)
            if not line.get_clip_on() or vertices.size == 0:
                continue
            box = line.get_window_extent(renderer)
            if box.y1 > frame.y1 + 1 or box.y0 < frame.y0 - 1:
                escaped.append(f"a line runs {box.y1 - frame.y1:.0f} px past the top of its axes")
            elif box.x1 > frame.x1 + 1 or box.x0 < frame.x0 - 1:
                escaped.append("a line runs past the side of its axes")
    return escaped


def assert_nothing_clipped(fig: Figure) -> None:
    """Fail rather than write a figure whose ink was cropped away."""
    escaped = clipped_artists(fig)
    assert not escaped, "clipped out of the axes: " + " | ".join(sorted(set(escaped)))
