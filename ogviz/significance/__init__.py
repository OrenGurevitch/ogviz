"""Significance brackets whose stars belong to the line beneath them.

The problem this module exists for: asterisks float high inside their layout box. At 18 pt the
visible ink of `***` runs roughly 5-13 pt above the baseline while the box also covers a
descender ~5 pt *below* it. Anchoring with `va="bottom"` therefore parks the glyph ~10 pt above
its own bracket even when the box is one pixel away, and in a stack of brackets the star reads
as belonging to the line ABOVE it. Nudging the y offset by hand fixes one figure and breaks the
next, because the error is in points while the nudge is in data units.

`bracket_stack` measures the glyph's ink with `TextPath` and anchors THAT: the ink bottom sits
`INK_GAP_PX` above its own bracket and `STACK_GAP_PX` below the next. Both are pixels, so the
asymmetry survives any font size, figure size or data range.

Ported from the sibling project that solved this first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from ogviz.orientation import (
    is_vertical,
    place_many,
    stamp_orientation,
)
from ogviz.theme import INK, MUTED_INK, STAR_SIZE, WORD_LABEL_SIZE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes

    from ogviz.orientation import Orientation

INK_GAP_PX = 3.0  # star ink to its OWN bracket
STACK_GAP_PX = 26.0  # star ink to the NEXT bracket up — must stay >> INK_GAP_PX
TICK_FRACTION = 0.022  # bracket end-tick length, as a fraction of the data span


def stars(p: float) -> str:
    """Conventional significance stars; "n.s." when nothing clears 0.05.

    A p outside [0, 1] is refused rather than mapped: everything below 0.001 becomes three stars,
    so a sign error or an uninitialised value upstream would otherwise print as the most
    significant result on the figure.
    """
    assert 0.0 <= p <= 1.0, f"p must be in [0, 1], got {p}"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


STAR_SEPARATOR = "\u2009\u2009"  # two thin spaces: wider than one space, tighter than two


def spaced_stars(p: float) -> str:
    """Stars with breathing room, which reads better than "***" at any size. "n.s." unchanged.

    The separator is two THIN spaces rather than a word space: an asterisk is narrow and sits
    high, so a full space between two of them reads as a gap in the label rather than as spacing.
    """
    label = stars(p)
    return STAR_SEPARATOR.join(label) if set(label) <= {"*"} else label


def ink_bounds_points(text: str, fontsize: float, *, weight: str = "bold") -> tuple[float, float]:
    """(bottom, top) of the visible ink relative to the baseline, in points."""
    return ink_extents_points(text, fontsize, axis=1, weight=weight)


def ink_extents_points(
    text: str, fontsize: float, *, axis: int, weight: str = "bold"
) -> tuple[float, float]:
    """Ink bounds along one axis: vertical uses the baseline-relative height, horizontal the width.

    A bracket that grows rightward must be cleared by the label's WIDTH. Reusing the vertical ink
    there offsets a star by its own height, which lands it back on its bracket.

    The VERTICAL measurement drops whitespace first. `TextPath` emits an empty contour at the
    origin for a space, and `get_extents` counts it, so the ink bottom of "* * *" measured as 0.0
    instead of 7.7 pt, so every spaced star was placed 7.7 pt too low while a lone "*" was placed
    correctly. Width keeps the spaces, since they are part of how wide the label is.
    """
    font = FontProperties(size=fontsize, weight=weight)
    if axis == 0:
        box = TextPath((0, 0), text, prop=font).get_extents()
        return float(box.x0), float(box.x1)
    inked = "".join(character for character in text if not character.isspace())
    if not inked:
        return 0.0, 0.0
    box = TextPath((0, 0), inked, prop=font).get_extents()
    return float(box.y0), float(box.y1)


WORD_LABEL_RATIO = WORD_LABEL_SIZE / STAR_SIZE


def label_size(label: str, star_size: float) -> float:
    """How big a bracket label is set, decided by what it IS rather than by the caller.

    "***" is three glyphs and "n.s." is a word. Set at one size they do not read as one family: the
    word carries several times the ink and dominates a panel where some comparisons clear the
    threshold and some do not, which inverts the emphasis — the non-result shouts.

    Sized here rather than passed in, because the caller does not choose the text either: it comes
    from `stars(p)` or from a project's own `label_for`, and the rule should hold for both.
    """
    inked = label.replace("\u2009", "").replace(" ", "")
    if inked and set(inked) == {"*"}:
        return star_size
    return star_size * WORD_LABEL_RATIO


def bracket_stack(
    ax: Axes,
    comparisons: Sequence[tuple[float, float, float]],
    *,
    start: float,
    span: float,
    fontsize: float = STAR_SIZE,
    linewidth: float = 1.6,
    line_color: str = MUTED_INK,
    text_color: str = INK,
    fontweight: str = "bold",
    spaced: bool = True,
    label_for: Callable[[float], str] | None = None,
    orientation: Orientation = "vertical",
    draw: bool = True,
) -> float:
    """Stack brackets bottom-up, each star anchored by its ink. Returns the topmost ink top.

    `comparisons` is [(x_left, x_right, p), ...] in the order they should stack, lowest first.
    `start` is the data y of the first bracket and `span` sets the end-tick length.

    `draw=False` measures where the stack would reach and adds nothing, so a caller can reserve
    exactly the room it needs. Predicting that room arithmetically is circular — the stack
    advances in pixels, and reserving more room rescales the axis, which changes what a pixel is
    worth — so measure it instead.

    ogviz owns where the label sits, never what it says: a project that writes "" for
    non-significant where another writes "n.s." passes its own `label_for`.
    """
    if not comparisons:
        return start
    # Before anything is drawn, and even for `draw=False`: the panel this stack belongs to may have
    # nothing else that reveals which way it runs, and a bracket is the artist QC most needs to
    # measure along the right axis.
    stamp_orientation(ax, orientation)
    figure = ax.figure
    assert figure is not None, "the axes must belong to a figure"
    figure.canvas.draw()  # realise a renderer so transforms are meaningful
    to_data = ax.transData.inverted()
    to_pixels = ax.transData
    px_per_pt = figure.dpi / 72.0
    tick = span * TICK_FRACTION
    upright = is_vertical(orientation)
    # The ink measurement is along the axis the bracket grows on. Vertical brackets stack
    # upward, so that is screen y; horizontal ones grow rightward, so it is screen x.
    axis = 1 if upright else 0
    if label_for is None:
        label_for = spaced_stars if spaced else stars

    y = top = start
    for x_left, x_right, p in comparisons:
        label = label_for(p)
        if not label:
            continue
        bracket = None
        if draw:
            (bracket,) = ax.plot(
                *place_many(
                    orientation,
                    [x_left, x_left, x_right, x_right],
                    [y - tick, y, y, y - tick],
                ),
                color=line_color,
                lw=linewidth,
                zorder=10,
            )
            bracket.ogviz_bracket = True  # type: ignore[attr-defined]
        size = label_size(label, fontsize)
        ink_low, ink_high = (
            v * px_per_pt
            # The same weight the label is set in: the placement is measured from the glyph's ink,
            # and a lighter weight is a different glyph shape. Measuring bold and drawing regular
            # would offset every label in the stack by the difference.
            for v in ink_extents_points(label, size, axis=axis, weight=fontweight)
        )
        # Place the BASELINE so that the ink bottom lands INK_GAP_PX beyond this bracket.
        anchor = to_pixels.transform(place_many(orientation, 0.0, y))[axis]
        baseline_px = anchor + INK_GAP_PX - ink_low

        def at(pixels: float) -> float:
            point = (pixels, 0.0) if axis == 0 else (0.0, pixels)
            return float(to_data.transform(point)[axis])

        label_x, label_y = place_many(orientation, (x_left + x_right) / 2, at(baseline_px))
        if draw:
            drawn = ax.text(
                label_x,
                label_y,
                label,
                ha="center" if upright else "left",
                va="baseline" if upright else "center",
                fontsize=size,
                fontweight=fontweight,
                color=text_color,
                zorder=10,
            )
            # Placed against its own bracket on purpose; `significance_gaps` owns that
            # relationship and measures it far more precisely than a box test could.
            drawn.ogviz_bracket_star = True  # type: ignore[attr-defined]
            drawn.ogviz_anchored = True  # type: ignore[attr-defined]
            drawn.ogviz_anchor = bracket  # type: ignore[attr-defined]  # its own line, only
        top = at(baseline_px + ink_high)
        y = at(baseline_px + ink_high + STACK_GAP_PX)
    return top


def significance_row(
    ax: Axes,
    comparisons: Sequence[tuple[float, float, float]],
    *,
    start: float,
    span: float,
    **kwargs,
) -> float:
    """Every bracket at ONE height, for one comparison per category across a panel.

    `bracket_stack` stacks bottom-up, right for several comparisons WITHIN a group and wrong for a
    panel of seven categories each carrying its own test: those brackets are siblings and belong on
    one line. Getting that from the stack meant seven separate calls of one comparison each — the
    workaround a project wrote, and the one that surfaced the orientation bug, since seven
    one-bracket stacks is not a shape the checks were built to read.

    Returns the topmost ink top, so a caller can reserve room the same way.
    """
    tops = [
        bracket_stack(ax, [comparison], start=start, span=span, **kwargs)
        for comparison in comparisons
    ]
    return max(tops, default=start)
