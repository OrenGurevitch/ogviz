"""Whether a significance mark says what it appears to say.

A star is read against its bracket, so the checks here are about that relationship: every star the
same distance above its own line, no star touching one, and a stack whose steps are even and wider
than the gap between a star and its own bracket. Consistency is the subject rather than any absolute
number — a `tight_layout` moves the whole stack together and that is harmless; one star at a
different distance from the rest is what a reader notices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz import units
from ogviz.layout.collision import quoted
from ogviz.layout.render import ensure_rendered
from ogviz.qc.reading import (
    GAP_TOLERANCE_PX,
    bracket_spans_px,
    bracket_tops_px,
    orientation_of,
)
from ogviz.significance import STACK_GAP_PX, ink_extents_points
from ogviz.tags import marked

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text


def _looks_like_stars(text: str) -> bool:
    """A row of asterisks, however `spaced_stars` spaced them. For figures with no tag to read."""
    stripped = text.strip()
    return bool(stripped) and set(stripped.split()) == {"*"}


def _stars(ax: Axes) -> list[Text]:
    """Every label `bracket_stack` placed over a bracket, whatever it says.

    Read from the tag the stack already sets, not from the wording. Matching asterisks instead
    excluded `"n.s."` — so the one label in a stack that is not a row of asterisks was the one
    exempt from the check that they all sit at the same height, and a figure shipped with `n.s.`
    visibly below its neighbours while the audit stayed silent. `label_for` means a project can
    print any wording it likes, so no list of strings could have covered this.

    The wording test remains for a figure this package did not draw, where there is no tag.
    A forest strip marks whole rows and flags those, so they are skipped.
    """
    tagged = [t for t in ax.texts if marked(t, "bracket_star")]
    candidates = tagged or [t for t in ax.texts if _looks_like_stars(t.get_text())]
    return [t for t in candidates if not marked(t, "column_star")]


MIN_STAR_GAP_PX = 1.0  # below this the glyph is touching its own bracket


def significance_gaps(fig: Figure) -> list[str]:
    """Every star must sit the same distance above its own bracket, and never on it.

    CONSISTENCY is the check, not the absolute number. A `tight_layout` after the marks are drawn
    rescales the axes, and the gap is a pixel quantity converted through the transform, so the
    whole stack drifts a little together — harmless, and even. What is not harmless is one star
    sitting at a different distance from the rest, which is what a reader notices and what a lone
    "*" did for as long as `spaced_stars` existed: `TextPath` counts a space as an empty contour at
    the origin, so the ink bottom of "* * *" measured as zero and every spaced star sat 7.7 pt low
    while a single one was placed correctly.

    Measured from the glyph's INK, never its layout box.
    """
    ensure_rendered(fig)
    px_per_point = units.px_per_point(fig)
    complaints: list[str] = []
    for ax in fig.axes:
        tops = bracket_tops_px(ax)
        gaps: list[tuple[str, float]] = []
        for star in _stars(ax):
            axis = 1 if orientation_of(ax) == "vertical" else 0
            size = star.get_fontsize()
            # In the weight it was DRAWN in, not the default. A lighter weight is a different glyph
            # shape with a different ink bottom, so measuring bold against a star set otherwise
            # offsets this whole check by the difference — on a rule whose job is to notice a
            # difference of a pixel. `bracket_stack` takes `fontweight` as an argument, and
            # `settle_bracket_labels` has read it back correctly since it was written; this did not.
            ink_low, _ = ink_extents_points(
                star.get_text(), float(size), axis=axis, weight=str(star.get_fontweight())
            )
            baseline_px = float(ax.transData.transform(star.get_position())[axis])
            ink_bottom_px = baseline_px + ink_low * px_per_point
            below = [t for t in tops if t <= ink_bottom_px + GAP_TOLERANCE_PX]
            if not below:
                complaints.append(f"the star {quoted(star.get_text())!r} has no bracket under it")
                continue
            gaps.append((star.get_text(), ink_bottom_px - max(below)))
        for label, gap in gaps:
            if gap < MIN_STAR_GAP_PX:
                complaints.append(
                    f"the star {quoted(label)!r} is {gap:.1f} px from its bracket — touching"
                )
        if len(gaps) > 1:
            spread = max(g for _l, g in gaps) - min(g for _l, g in gaps)
            if spread > GAP_TOLERANCE_PX:
                complaints.append(
                    "stars sit at different distances from their brackets: "
                    + ", ".join(f"{g:.1f}" for _l, g in gaps)
                    + " px"
                )
    return complaints


def _levels(tops: list[float]) -> list[float]:
    """The distinct heights brackets sit at, brackets on one line counted once.

    A panel with one comparison per category puts every bracket at the SAME height — siblings on one
    line, which is what `significance_row` draws. Treating each bracket as a step of its own then
    reported "brackets are 0 px apart", a complaint about the very thing that made the row a row.

    Clustering by height covers all three arrangements with one rule: a stack has one bracket per
    level, a row has one level, and a row of stacks has several brackets on each of several levels.
    """
    levels: list[float] = []
    for top in sorted(tops):
        if not levels or top - levels[-1] > GAP_TOLERANCE_PX:
            levels.append(top)
    return levels


def stack_spacing(fig: Figure) -> list[str]:
    """Stacked brackets must be even, and further apart than a star is from its own line.

    Judged per COLUMN of brackets that actually overlap along the category axis. Two independent
    comparisons drawn side by side — one over the first pair of groups, one over the last — share no
    x and cannot collide however close their heights come, and were reported as a crowded stack.
    """
    ensure_rendered(fig)
    complaints: list[str] = []
    for ax in fig.axes:
        for column in _overlapping_columns(bracket_spans_px(ax)):
            tops = _levels(sorted(column))
            if len(tops) < 2:
                continue
            steps = np.diff(tops)
            if float(steps.std()) > GAP_TOLERANCE_PX * 2:
                complaints.append(
                    f"brackets are unevenly stacked: steps {np.round(steps, 1).tolist()} px"
                )
            if float(steps.min()) < STACK_GAP_PX * 0.5:
                complaints.append(
                    f"brackets are {steps.min():.0f} px apart, closer than a star is to its "
                    "own line"
                )
    return complaints


def _overlapping_columns(spans: list[tuple[float, float, float]]) -> list[list[float]]:
    """Group brackets into the sets that share category space, and return each set's heights.

    A bracket joins a column when it overlaps ANY bracket already in it, so a wide bracket spanning
    two narrow ones below it puts all three in one column — which is right, since the wide one has
    to clear both.

    MERGED TRANSITIVELY. The first spelling stopped at the first column a span touched, so a wide
    bracket arriving after two disjoint narrow ones joined the first and was never compared with
    the second: `[(10, 0, 1), (11, 3, 4), (12, 0, 4)]` came back as two columns, `[10, 12]` and
    `[11]`, and the docstring above described what the code did not do. Every column the new span
    touches is folded into one.
    """
    columns: list[tuple[float, float, list[float]]] = []
    for top, near, far in spans:
        touching = [
            index for index, (low, high, _) in enumerate(columns) if near <= high and low <= far
        ]
        merged_low, merged_high, merged_tops = near, far, [top]
        for index in reversed(touching):
            low, high, tops = columns.pop(index)
            merged_low, merged_high = min(merged_low, low), max(merged_high, high)
            merged_tops = [*tops, *merged_tops]
        columns.append((merged_low, merged_high, merged_tops))
    return [tops for _low, _high, tops in columns]
