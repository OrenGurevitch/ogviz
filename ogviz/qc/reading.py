"""How to read a finished figure — the questions every check asks first.

A check measures what was drawn, so it sees rectangles and lines and has to work out what they MEAN
before it can judge them. That work is shared: which way the panel runs, where its brackets sit,
which artists are marks and which are the frame, whether a label was placed against a particular
thing on purpose. Each of those is asked by several checks and must be answered the same way for all
of them — two checks disagreeing about what a backdrop is would report the same figure two ways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout.collision import quoted
from ogviz.layout.overlap import (
    opaque_backing,
)
from ogviz.orientation import read_orientation
from ogviz.tags import marked, value_of

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def orientation_of(ax: Axes) -> str:
    """The panel's orientation: what it recorded, or failing that what its marks suggest.

    The marks are a fallback for a figure this package did not draw. Inferring it where the answer
    was known produced the worst class of QC failure — a confident complaint about a correct
    figure. A grouped bar panel votes exactly once, and wrongly: `errorbar` keeps its bars and caps
    in a LineCollection, so the only two-point line is the zero baseline, whose y is constant, and
    the panel reads as horizontal. Brackets were then measured along x, none matched the bracket
    shape, and all five stars were reported as having no bracket under them.

    The vote itself is unchanged and still right for what it is: an IQR whisker is a two-point line
    on the group's centre, constant x in a vertical panel and constant y in a horizontal one.
    """
    recorded = read_orientation(ax)
    if recorded is not None:
        return recorded
    vertical = horizontal = 0
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if xdata.size != 2:
            continue
        if float(xdata[0]) == float(xdata[1]):
            vertical += 1
        elif float(ydata[0]) == float(ydata[1]):
            horizontal += 1
    return "horizontal" if horizontal > vertical else "vertical"


def _is_bracket(data: np.ndarray) -> bool:
    """A bracket is [edge, top, top, edge]: two equal middles, two equal lower ends.

    Counting four points alone is not enough — an error bar's cap is also a four-point line, and
    treating those as brackets reported a whole bar panel as an "uneven stack" with brackets 0 px
    apart. The shape is the discriminator.
    """
    if data.size != BRACKET_POINTS:
        return False
    edge_a, top_a, top_b, edge_b = (float(v) for v in data)
    return top_a == top_b and edge_a == edge_b and edge_a < top_a


BRACKET_POINTS = 4  # a bracket is drawn as four points: down, across, down


def bracket_tops_px(ax: Axes) -> list[float]:
    """Where each bracket's outer edge sits, in display pixels along the axis it grows on."""
    axis = 1 if orientation_of(ax) == "vertical" else 0
    tops = []
    for line in ax.lines:
        data = np.asarray(line.get_ydata() if axis == 1 else line.get_xdata(), dtype=float)
        if not _is_bracket(data):
            continue
        point = (0.0, float(np.max(data))) if axis == 1 else (float(np.max(data)), 0.0)
        tops.append(float(ax.transData.transform(point)[axis]))
    return sorted(tops)


def drawn_artists(ax) -> list:
    """Everything on one axes that a reader can see, in one list."""
    from ogviz.layout.overlap import drawn_tick_labels

    items = [*ax.texts, *ax.lines, *ax.patches, *ax.collections]
    if ax.axison:
        items += drawn_tick_labels(ax)
    return [artist for artist in items if artist.get_visible()]


def is_backdrop(artist) -> bool:
    """A shaded region text is MEANT to sit on — a highlighted column, a reference band.

    Not a mark: it carries no value of its own and is there to tint the marks that do. Text over it
    is the design, and the value labels knock out to its colour rather than to the page precisely so
    they stay readable on it.
    """
    return bool(marked(artist, "backdrop"))


def artist_name(artist) -> str:
    text = getattr(artist, "get_text", None)
    if text is not None:
        return repr(quoted(text()))
    return type(artist).__name__


def is_excused(label, other) -> bool:
    """Whether `label` is deliberately placed against `other`, and only against `other`.

    An anchored label is pardoned for touching the thing it labels — a star over its bracket, a
    value printed past its own whisker. It is not pardoned for landing on anything else. Treating
    the flag as a blanket exemption is what let a reference-line label sit on a bar without a word:
    the label is pinned to its line vertically and free to slide along it, so the one axis it could
    actually be wrong on was the one being excused.
    """
    anchor = value_of(label, "anchor", None)
    if anchor is not None and anchor is other:
        return True
    # A star in a stack sits between its own bracket and the next one up, and how close it may come
    # to either is measured by `significance_gaps` and `stack_spacing` — in points of ink, against
    # the glyph's own extents, far more precisely than "these two share a pixel". Contact there is
    # their business, not this check's.
    return bool(marked(label, "bracket_star")) and bool(marked(other, "bracket"))


def knocked_out_over(label, other) -> bool:
    """Whether `label` carries an opaque box that hides `other` where the two overlap.

    A knockout is the sanctioned way for a value label to stay legible over a rule, and it works:
    on the figure that raised this, the label and a reference edge shared 62 px of ink and NONE of
    those pixels were visible — the box covered every one. Reporting them is the "gate that cries
    wolf" failure, and it appeared the moment a band grew dashed edges under labels with boxes.

    Paint order decides it, so a box under the other artist excuses nothing.
    """

    if not hasattr(label, "get_text") or label.get_zorder() < other.get_zorder():
        return False
    return opaque_backing(label) is not None


GAP_TOLERANCE_PX = 1.5  # rendering rounds to the pixel grid; past this it is a real drift


def bracket_spans_px(ax: Axes) -> list[tuple[float, float, float]]:
    """Each bracket as (outer edge, near end, far end) in display pixels.

    The ends matter because two brackets only have to clear each other when they OVERLAP along the
    category axis. Two independent comparisons side by side in one panel — one spanning the first
    pair of groups, one the last — cannot collide however close their heights are, and reporting
    them as a crowded stack is a complaint about a figure that is right.
    """
    upright = orientation_of(ax) == "vertical"
    value_axis, category_axis = (1, 0) if upright else (0, 1)
    spans: list[tuple[float, float, float]] = []
    for line in ax.lines:
        along = np.asarray(line.get_ydata() if upright else line.get_xdata(), dtype=float)
        if not _is_bracket(along):
            continue
        across = np.asarray(line.get_xdata() if upright else line.get_ydata(), dtype=float)
        top = float(np.max(along))
        near, far = float(np.min(across)), float(np.max(across))

        def at(value: float, axis: int, other: float = 0.0) -> float:
            point = (other, value) if axis == 1 else (value, other)
            return float(ax.transData.transform(point)[axis])

        spans.append(
            (
                at(top, value_axis),
                at(near, category_axis),
                at(far, category_axis),
            )
        )
    return sorted(spans)


def filled_marks_over(ax: Axes, box, zorder: float) -> list:
    """Collections above `zorder` whose FILLED OUTLINE actually enters `box`.

    HERE rather than beside either caller, because `buried_baselines` and `raise_buried_lines`
    both need it and a check and its repair disagreeing about what buries a line is exactly the
    failure this module exists to prevent — the gate would report a covered spine the repair then
    declined to lift.

    The narrow pre-filter that closes the collection gap without the box test's false positives.
    Two conditions, and each rules out one half of the problem:

    A POINT CLOUD IS EXCLUDED OUTRIGHT. `point_offsets` identifies it — one marker path repeated at
    many offsets. Its bounding box spans the whole cloud and overlaps the spine while every dot is
    somewhere else, which is exactly what made adding `ax.collections` to the box test fail 17
    tests. A cloud cannot cover a line the way a filled body can.

    THE REST ARE TESTED AS PATHS, not as boxes. `Path.intersects_bbox(..., filled=True)` asks
    whether the shape itself enters the rectangle, which for a `fill_between` band is the question
    — and for a violin body correctly answers no when the body does not reach the axis.

    The z-order condition does most of the work in practice: matplotlib gives a collection zorder 1
    and a spine 2.5, so a default `fill_between` is UNDER the frame and never arrives here. What
    does arrive is a collection someone deliberately raised over the spine, which is the defect.
    """
    from ogviz.layout.collision import point_offsets

    found = []
    for collection in ax.collections:
        if not collection.get_visible() or collection.get_zorder() <= zorder:
            continue
        if point_offsets(collection) is not None:
            continue
        transform = collection.get_transform()
        if any(
            path.transformed(transform).intersects_bbox(box, filled=True)
            for path in collection.get_paths()
        ):
            found.append(collection)
    return found
