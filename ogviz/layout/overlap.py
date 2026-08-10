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
from matplotlib.transforms import Bbox

from ogviz.layout.render import ensure_rendered

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

DEFAULT_MIN_GAP = 5.0  # px; below this two labels on one row read as one word
OPAQUE_ENOUGH = 1.0  # a knockout hides what is under it only when it is fully opaque
HIDDEN_SHARE = 0.10  # of a label's area painted over before it is worth reporting


def _visible_texts(fig: Figure) -> list[Text]:
    """Every label a SPACING check must consider: ticks and legend text included.

    One walker, in `bounds`, so this set and the canvas check's set cannot drift apart.
    """
    from ogviz.layout.bounds import figure_text

    return [text for text, _owner in figure_text(fig, ticks=True, legend=True)]


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
    ensure_rendered(fig)
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
    return hits


def assert_no_text_overlap(
    fig: Figure,
    *,
    min_gap: float = DEFAULT_MIN_GAP,
) -> None:
    """Fail the build rather than write a figure whose labels sit on or against each other.

    Raised rather than asserted: `python -O` removes an `assert` and would take this gate with it.
    """
    hits = text_overlaps(fig, min_gap=min_gap)
    if hits:
        raise AssertionError("text collisions: " + " | ".join(hits))


def clipped_artists(fig: Figure) -> list[str]:
    """Artists that fall outside their axes and will be silently cropped.

    The overlap check measures where text IS and knows nothing about clipping, so a bracket line
    that ran past the axis limit passes it with zero problems, which is how a star ended up
    floating over nothing in a shipped figure. matplotlib clips `Line2D` by default and does not
    clip `Text`, so an overflowing stack loses its lines and keeps its labels.

    Measured from the artist's geometry, which is why the complaint names the two fixes that work.
    Neither `set_clip_path` nor a page-coloured band painted over the overflow changes it —
    `Line2D.get_window_extent` returns the data bounding box whether the artist is clipped or not,
    and this asks where the ink WOULD be. A consumer tried both before shortening the data, and the
    second especially looks like it must have worked.
    """
    ensure_rendered(fig)
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    escaped = []
    for ax in fig.axes:
        frame = ax.get_window_extent(renderer)
        for line in ax.lines:
            vertices = np.asarray(line.get_path().vertices)
            if not line.get_clip_on() or vertices.size == 0:
                continue
            box = line.get_window_extent(renderer)
            # Both directions, not one or the other. The `elif` here meant a line escaping the top
            # AND the side was only ever reported for the top, so a caller shortened the data,
            # re-ran, and met a second complaint that had been there all along.
            over_value = max(box.y1 - frame.y1, frame.y0 - box.y0)
            over_category = max(box.x1 - frame.x1, frame.x0 - box.x0)
            if over_value > 1:
                escaped.append(
                    f"a line runs {over_value:.0f} px past the top or bottom of its axes — "
                    "clipping it does not change this, and neither does painting over it; "
                    "shorten the data or raise the limit"
                )
            if over_category > 1:
                escaped.append(
                    f"a line runs {over_category:.0f} px past the side of its axes — same fix: "
                    "shorten the data or widen the limit"
                )
    return escaped


def assert_nothing_clipped(fig: Figure) -> None:
    """Fail rather than write a figure whose ink was cropped away."""
    escaped = clipped_artists(fig)
    if escaped:
        raise AssertionError("clipped out of the axes: " + " | ".join(sorted(set(escaped))))


def opaque_backing(text: Text) -> Bbox | None:
    """The box a label paints behind itself, when it is opaque enough to hide what is under it."""
    from matplotlib.colors import to_rgba

    patch = text.get_bbox_patch()
    if patch is None or not patch.get_visible():
        return None
    _red, _green, _blue, alpha = to_rgba(patch.get_facecolor(), patch.get_alpha())
    if alpha < OPAQUE_ENOUGH:
        return None
    return patch.get_window_extent()


def text_hidden_behind_knockouts(fig: Figure) -> list[str]:
    """Labels painted over by another label's opaque knockout box.

    A knockout is how a number stays legible over a gridline, drawn as an opaque rectangle behind
    the glyphs. It hides whatever else is under it just as effectively: two labels can each be
    exactly where they belong, and the one drawn second erases the first. Every position-based
    check passes, because position was never the problem — paint order was. It shipped as formulas
    rendering with pieces missing.

    Ordered by what matplotlib actually paints: zorder first, and for equal zorder the later artist,
    which is the order the axes stores them in.
    """
    ensure_rendered(fig)
    texts = [text for text in _visible_texts(fig) if text.get_text().strip()]
    order = {id(text): index for index, text in enumerate(texts)}
    complaints: list[str] = []
    for over in texts:
        backing = opaque_backing(over)
        if backing is None:
            continue
        for under in texts:
            if under is over:
                continue
            above = (over.get_zorder(), order[id(over)]) > (under.get_zorder(), order[id(under)])
            if not above:
                continue
            covered = Bbox.intersection(backing, under.get_window_extent())
            if covered is None or covered.width <= 0 or covered.height <= 0:
                continue
            share = (covered.width * covered.height) / max(
                under.get_window_extent().width * under.get_window_extent().height, 1.0
            )
            if share > HIDDEN_SHARE:
                complaints.append(
                    f"{under.get_text().strip()[:40]!r} is {share:.0%} covered by the knockout "
                    f"behind {over.get_text().strip()[:40]!r} — it is painted over, not overlapped"
                )
    return complaints
