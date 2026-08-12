"""Whether the marks are where the panel put them, and still visible.

Two failures, both silent. A jittered dot can land on the central marks it is meant to sit beside,
which hides the mean line under the cloud that surrounds it. And a line a reader measures against —
the category axis a bar stands on, a threshold the bars are compared to — can end up under the marks
drawn over it, where it survives only in the gaps and reads as broken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout.render import ensure_rendered
from ogviz.qc.reading import (
    filled_marks_over,
    orientation_of,
)
from ogviz.tags import marked, value_of

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def dots_off_the_marks(fig: Figure) -> list[str]:
    """No jittered point may sit on the central marks it is there to leave readable.

    Compares against the lane `points` RECORDED at draw time, not a recomputed one. The lane is a
    step function of y and its steps move when the axes are resized, so a lane recomputed after a
    `tight_layout` can put a dot in a different band than the one it was placed against — which
    reported perfectly placed dots as violations, twice, while I tuned thresholds at it.
    """
    ensure_rendered(fig)
    complaints: list[str] = []
    for ax in fig.axes:
        upright = orientation_of(ax) == "vertical"
        across = 0 if upright else 1
        for collection in ax.collections:
            lane = value_of(collection, "lane", None)
            position = value_of(collection, "position", None)
            if lane is None or position is None:
                continue
            offsets = np.asarray(collection.get_offsets(), dtype=float)
            if offsets.shape[0] != np.asarray(lane).shape[0]:
                continue
            inside = int(np.count_nonzero(np.abs(offsets[:, across] - position) < lane * 0.999))
            if inside:
                complaints.append(f"{inside} dot(s) sit on the central marks")
    return complaints


def _reference_lines(ax: Axes) -> list:
    return [line for line in ax.lines if marked(line, "reference")]


def buried_baselines(fig: Figure) -> list[str]:
    """A spine or a threshold must not be covered by the marks it is there to be read against.

    A bar grows from the category axis, so its base lies exactly along that spine. matplotlib
    draws spines at zorder 2.5 and bars above them, so the axis survives only in the gaps between
    bars and reads as a broken line. Caught by comparing z-order against the marks that actually
    overlap the spine, not by assuming which panel type is being drawn.

    Translucency is no defence and is deliberately not exempted: the house bars are drawn at 0.85,
    and 0.85 over a 1.6-point rule is exactly the washed-out segment this check exists to catch.
    Marks a panel puts UNDER the axis on purpose — a highlight column, a reference band — sit below
    the spine's z-order and never reach this test.

    TWO TESTS, one per artist type, and mixing them is the mistake. A PATCH is judged by its BOX:
    a bar whose box overlaps the spine is a solid rectangle standing on it, so the overlap IS the
    covering. A COLLECTION cannot be judged that way — a scatter's box spans its whole cloud and
    overlaps the spine while every dot is somewhere else. Measured: adding `ax.collections` to the
    box test fails 17 tests, four of them shipped examples, all false positives.

    So collections go through `filled_marks_over`, which excludes point clouds outright and tests
    the rest as PATHS rather than boxes. That closes the gap this had for a `fill_between` band
    covering a spine — invisible here until 2026-08-10 — without the ink test, which is the
    expensive per-artist render reserved for `--thorough`. The z-order condition does most of the
    work: matplotlib gives a collection zorder 1 against a spine's 2.5, so a default band is under
    the frame and never arrives; what arrives is one deliberately raised over it.
    """
    ensure_rendered(fig)
    complaints: list[str] = []
    for ax in fig.axes:
        if not ax.axison:
            continue  # `ax.axis("off")` leaves the spine objects visible but draws none of them
        upright = orientation_of(ax) == "vertical"
        for side, spine in ax.spines.items():
            if not spine.get_visible():
                continue
            spine_box = spine.get_window_extent()
            spine_z = spine.get_zorder()
            buried = [
                patch
                for patch in ax.patches
                if patch.get_zorder() > spine_z and patch.get_window_extent().overlaps(spine_box)
            ] + filled_marks_over(ax, spine_box, spine_z)
            if buried:
                # NAMED, because the cause is invisible from a count. A single
                # `axhspan(..., color=PAGE)` masking a cropped band does this: it spans the full
                # width of the axes, so it erases the spine through the whole headroom, and the
                # figure ships with its axis line stopping partway up.
                which = ", ".join(sorted({type(patch).__name__ for patch in buried}))
                highest = max(patch.get_zorder() for patch in buried)
                complaints.append(
                    f"the {side} spine is covered by {len(buried)} mark(s) drawn over it "
                    f"({which} at zorder {highest:g}, spine at {spine_z:g}) — a full-width band "
                    "belongs under the frame"
                )
        for line in _reference_lines(ax):
            # A threshold is there to be compared against the bars. Behind them it survives only in
            # the gaps between them, and the reader loses the one comparison the line was added to
            # make — the same defect as bars covering the category axis, one artist along.
            line_box = line.get_window_extent()
            over = [
                patch
                for patch in ax.patches
                if patch.get_zorder() > line.get_zorder()
                and patch.get_window_extent().overlaps(line_box)
            ] + filled_marks_over(ax, line_box, line.get_zorder())
            if over:
                # Along the VALUE axis, which is x on a horizontal panel. Read from y regardless,
                # the complaint named the category coordinate: measured, a threshold at 2.5 under
                # the bars of a horizontal panel was reported as "the reference line at 0", sending
                # a reader to look for a line that is not there.
                along = line.get_ydata() if upright else line.get_xdata()
                value = float(np.asarray(along, dtype=float)[0])
                complaints.append(
                    f"the reference line at {value:g} is behind {len(over)} mark(s) — "
                    "a threshold has to stay readable across the panel"
                )
    return complaints
