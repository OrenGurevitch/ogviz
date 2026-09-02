"""How much of the figure carries ink, and where the empty parts are.

A figure that wastes half its area is not broken — no check fires, nothing overlaps, and it looks
fine in isolation. It is simply smaller than it should be: the marks are smaller than the page
allows, the type is smaller than it could be, and the reader gets less for the same space. The
fault is invisible precisely because emptiness looks like breathing room.

Measured by rasterising, not by reasoning about artists. Every question here is "does this pixel
differ from the page colour", and the render already answers it for lines, fills, text, images and
anything a project draws that this module has never heard of. Bounding boxes cannot answer it: a
panel holding one rising line is three-quarters empty, and its artists' bounding boxes cover the
whole panel.

Two scales, because they have different fixes:

  the OUTER margin   ink stops well short of the canvas edge -> the axes can grow, or the figure
                     can be cropped. `trim_margins` does this one, and it is safe: it moves the
                     axes inside the page and changes no data.
  a PANEL's own      ink fills only part of an axes -> the limits are too generous, or the panel
  empty band        is taller than its content. Reported, never applied: tightening a value axis
                     changes how big an effect looks, which is the caller's call and not a
                     layout decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout.bounds import panel_prefix
from ogviz.layout.raster import INK_TOLERANCE, frame_rgb, ink_of
from ogviz.require import require
from ogviz.tags import value_of

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

# Overall ink coverage is deliberately NOT a complaint. It measures the mark type, not the layout:
# one well-fitted line covers about 3% of any canvas and a dense scatter covers 40%, and neither
# number says whether space is being wasted. `coverage` is reported for interest; the signals that
# actually mean something are emptiness where ink was expected — the outer margin and each panel's
# own empty bands.
GENEROUS_BAND = 0.18  # an empty strip wider than this share of the panel is worth reporting
LOOSE_MARGIN_PX = 8.0  # outer white space past this is worth reclaiming


def ink_mask(fig: Figure, *, tolerance: int = INK_TOLERANCE) -> NDArray[np.bool_]:
    """True where the rendered figure differs from its own page colour.

    Kept as a name because it is what this module's questions are phrased in; the reading and the
    tolerance both come from `layout.raster`, which is the one place that decides what counts as a
    pixel of ink. They used to be decided here AND in `layout.ink`, at 12 and at 10.
    """
    return ink_of(frame_rgb(fig), tolerance=tolerance)


def _ink_bounds(mask: NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
    """(left, right, top, bottom) of the inked region in array coordinates, or None if blank."""
    columns = np.flatnonzero(mask.any(axis=0))
    rows = np.flatnonzero(mask.any(axis=1))
    if not columns.size or not rows.size:
        return None
    return int(columns[0]), int(columns[-1]), int(rows[0]), int(rows[-1])


@dataclass(frozen=True)
class Density:
    """What the render says about how well the figure uses its page."""

    coverage: float  # share of the whole canvas carrying ink
    left: float  # unused pixels on each side, before any ink
    right: float
    top: float
    bottom: float

    def wasted_margin(self) -> float:
        return max(self.left, self.right, self.top, self.bottom)


def measure(fig: Figure) -> Density:
    """Ink coverage and the unused band on each side of the canvas, in pixels.

    Each band runs from that edge of the canvas to that edge of the INK, so on a figure with no ink
    the four bands are each the whole canvas and they overlap: left and right both read the full
    width, top and bottom both the full height. They are not summable there, and a caller that
    sums them concludes the figure is twice its own size. `figure_margins` answers None for that
    case and `dead_space` names it in one note; nothing should read these four for a blank page.
    """
    mask = ink_mask(fig)
    height, width = mask.shape
    bounds = _ink_bounds(mask)
    if bounds is None:
        return Density(0.0, float(width), float(width), float(height), float(height))
    left, right, top, bottom = bounds
    return Density(
        coverage=float(mask.mean()),
        left=float(left),
        right=float(width - 1 - right),
        top=float(top),
        bottom=float(height - 1 - bottom),
    )


def data_ink_mask(fig: Figure, ax: Axes) -> NDArray[np.bool_]:
    """Pixels that exist only because the DATA was drawn, found by rendering twice.

    A raster cannot tell a gridline from a curve, and it does not have to: hide the marks, render,
    restore them, render again, and the difference is exactly the ink the marks contribute. This
    is what makes a panel's emptiness measurable at all — a full-width gridline and a boxed frame
    put ink on every edge of every panel, so measuring the raw render reports every panel as
    perfectly full no matter how little data it holds.
    """
    from ogviz.layout.collision import decoration_ids

    skip = decoration_ids(ax)
    marks = [
        artist
        for artist in [*ax.lines, *ax.collections, *ax.patches, *ax.images]
        if id(artist) not in skip
    ]
    was_visible = [artist.get_visible() for artist in marks]
    for artist in marks:
        artist.set_visible(False)
    try:
        without = ink_mask(fig)
    finally:
        # `zip(marks, was_visible)`, plainly. It read `zip(was_visible and marks, ...)`, where the
        # guard can only ever change the answer in the case where both lists are empty and the zip
        # is empty anyway — an expression that looks like it is defending against something.
        for artist, state in zip(marks, was_visible, strict=True):
            artist.set_visible(state)
    return ink_mask(fig) & ~without


def panel_emptiness(fig: Figure, ax: Axes) -> dict[str, float]:
    """The share of each edge of one axes that holds no ink.

    Read inside the axes' own rectangle, so a tall panel whose data occupies the lower third
    reports `top` near 0.66 — which is the number that says "your y-limit is too generous", and
    the one a bounding box can never produce.
    """
    mask = data_ink_mask(fig, ax)
    height, _width = mask.shape
    box = ax.get_window_extent()
    # Display y grows upward; the array's first row is the TOP of the image.
    x0, x1 = int(max(box.x0, 0)), int(min(box.x1, mask.shape[1]))
    y0, y1 = int(max(height - box.y1, 0)), int(min(height - box.y0, height))
    inside = mask[y0:y1, x0:x1]
    if not inside.size:
        # The axes does not land on the canvas at all. It used to report every edge as 0.0 empty,
        # which is the answer a perfectly FULL panel gives — so the one panel that is genuinely
        # broken was the one `dead_space` had nothing to say about. Empty throughout is both the
        # honest reading and the one that gets reported.
        return {"coverage": 0.0, "left": 1.0, "right": 1.0, "top": 1.0, "bottom": 1.0}
    bounds = _ink_bounds(inside)
    if bounds is None:
        return {"coverage": 0.0, "left": 1.0, "right": 1.0, "top": 1.0, "bottom": 1.0}
    left, right, top, bottom = bounds
    rows, columns = inside.shape
    return {
        "coverage": float(inside.mean()),
        "left": left / columns,
        "right": (columns - 1 - right) / columns,
        "top": top / rows,
        "bottom": (rows - 1 - bottom) / rows,
    }


def dead_space(fig: Figure) -> list[str]:
    """Advisory notes about space the figure is not using. Never fails a build.

    Advisory on purpose. Every item here is a judgement — a deliberately airy figure is a real
    choice, and a panel with headroom for a significance bracket is not wasting it. These read as
    "look at this", which is what a density report can honestly claim.
    """
    notes: list[str] = []
    overall = measure(fig)
    if overall.coverage == 0.0:
        # ONE note, not four. Every band is measured to the edge of the ink, so with no ink each
        # of the four is the whole canvas — and the four notes then said "the canvas past the left
        # edge of the ink is unused" about an edge that does not exist, twice over per axis. The
        # fact is a single one and it is not about margins. The PANEL loop still runs: an axes
        # placed off the canvas leaves a blank page and is exactly the case worth reporting.
        notes.append("the figure has no ink on it")
    else:
        for side in ("left", "right", "top", "bottom"):
            unused = getattr(overall, side)
            if unused > LOOSE_MARGIN_PX:
                notes.append(
                    f"{unused:.0f} px of the canvas past the {side} edge of the ink is unused"
                )
    for ax in fig.axes:
        panel = panel_emptiness(fig, ax)
        shared = value_of(ax, "shared_scale")
        for side in ("left", "right", "top", "bottom"):
            if panel[side] > GENEROUS_BAND:
                where = f"{panel_prefix(fig, ax)}the {side} {panel[side]:.0%} of the panel is empty"
                # A panel on a SHARED scale is empty at one end by construction — the tallest panel
                # in the grid set the limit. Reported in the same words as a loose limit, the note
                # told a reader to tighten the one axis that must not be tightened, since doing so
                # is what stops the grid being comparable at all.
                notes.append(
                    f"{where} — it shares a value scale with {shared} panels, so this is the "
                    "tallest panel's headroom rather than a loose limit"
                    if shared and _axis_along(ax, side) == "value"
                    else f"{where} — the {_axis_along(ax, side)} limit is generous"
                )
    return notes


def _axis_along(ax: Axes, side: str) -> str:
    """Which axis a given EDGE of this panel belongs to — the value one, or the category one.

    Read from the orientation the panel recorded rather than assumed. Written out as
    `'value' if side in ('top', 'bottom') else 'category'`, the note named the wrong axis on every
    horizontal panel: `bar_panel(orientation="horizontal")` runs its categories down the left, so
    an empty band at the TOP is a generous category limit and one at the RIGHT is the value limit.
    A reader told to tighten the wrong limit tightens the one carrying the comparison.

    `None` for axes this package did not draw, where the vertical convention is the better guess
    and is all that can be said.
    """
    from ogviz.orientation import read_orientation

    upright = read_orientation(ax) != "horizontal"
    vertical_edge = side in ("top", "bottom")
    return "value" if vertical_edge == upright else "category"


def trim_margins(fig: Figure, *, pad_px: float = 6.0) -> bool:
    """Grow the axes into unused outer margin. Returns whether anything moved.

    The safe half of the fix: this moves the axes within the page and touches no limit, no font
    and no datum, so the figure says exactly what it said before at a larger size. The unsafe half
    — tightening a panel's own limits — is reported by `dead_space` and left to the caller, because
    a value axis decides how big an effect looks.

    Skips figures whose axes were not placed by a subplot grid, since there is no grid to adjust.
    """
    if not fig.axes or any(ax.get_subplotspec() is None for ax in fig.axes):
        return False
    before = measure(fig)
    if before.wasted_margin() <= LOOSE_MARGIN_PX:
        return False
    width, height = fig.get_size_inches() * fig.dpi
    current = fig.subplotpars

    # GROW ONLY, per side. Each side moves by its own unused margin less `pad_px`, and that
    # difference is NEGATIVE on a side whose ink already reaches within `pad_px` of the edge — so
    # the axes was pulled inward there. The gate above tests only the WIDEST side, which meant one
    # wasteful side licensed a shrink on the other three: a figure with 40 px spare on the left and
    # 2 px on the right had its right edge moved IN by 4 px, undoing part of what the trim was for.
    def gain(unused: float, extent: float) -> float:
        return max(0.0, (unused - pad_px) / extent)

    moved = {
        "left": max(0.0, current.left - gain(before.left, width)),
        "right": min(1.0, current.right + gain(before.right, width)),
        "bottom": max(0.0, current.bottom - gain(before.bottom, height)),
        "top": min(1.0, current.top + gain(before.top, height)),
    }
    if moved["left"] >= moved["right"] or moved["bottom"] >= moved["top"]:
        return False  # the trim would invert the axes; leave it alone
    fig.subplots_adjust(**moved)
    return True


@dataclass(frozen=True)
class Margins:
    """The room one figure's ink actually needs, per side, as a figure FRACTION.

    Fractions rather than pixels so the numbers go straight into `subplots_adjust`, which is what a
    pinned layout sets. `left` and `bottom` are distances from those edges; `right` and `top` are
    the coordinates of the far edges, matching `subplots_adjust`'s own convention so a caller can
    pass them without inverting anything.
    """

    left: float
    right: float
    bottom: float
    top: float

    def padded(self, pad: float) -> Margins:
        """The same margins with `pad` of slack on every side, clamped to the page."""
        return Margins(
            left=max(0.0, self.left - pad),
            right=min(1.0, self.right + pad),
            bottom=max(0.0, self.bottom - pad),
            top=min(1.0, self.top + pad),
        )


def figure_margins(fig: Figure, *, tolerance: int = INK_TOLERANCE) -> Margins | None:
    """Where this figure's ink actually reaches, per side. None for a blank page."""
    mask = ink_mask(fig, tolerance=tolerance)
    bounds = _ink_bounds(mask)
    if bounds is None:
        return None
    left, right, top, bottom = bounds
    height, width = mask.shape
    # The mask's rows run top-down and a figure fraction runs bottom-up.
    return Margins(
        left=left / width,
        right=(right + 1) / width,
        bottom=1.0 - (bottom + 1) / height,
        top=1.0 - top / height,
    )


def required_margins(figures: Iterable[Figure], *, pad: float = 0.0) -> Margins:
    """The margins that fit EVERY figure in a set — the widest demand on each side.

    For a pinned layout. Wanting one axes rectangle across a set rules out `tight_layout`, so the
    margins get pinned, and the pinned value has to fit the WORST figure in the set. There is no way
    to know which that is without rendering all of them, so it gets guessed — and a guess fails in
    both directions: too generous leaves the tightest figures looking half-empty, and the obvious
    correction crops the ones whose ink reaches furthest.

    Takes FIGURES, and renders each here. Measuring PNGs already on disk gives a contaminated
    answer: a partial regeneration leaves stale files from the previous candidate, and two separate
    attempts in that project reported the same count and looked like a floor when they were not.

    `pad` is slack added on every side, in figure fractions.

    Pair this with `save(..., crop=False)`. Measuring margins and then writing with the default
    `bbox_inches="tight"` cancels the whole exercise: the crop resizes each file to its own ink, so
    the pinned rectangle never reaches disk.
    """
    measured = [margins for margins in map(figure_margins, figures) if margins is not None]
    require(
        measured,
        "required_margins needs at least one figure with ink on it",
    )
    widest = Margins(
        left=min(m.left for m in measured),
        right=max(m.right for m in measured),
        bottom=min(m.bottom for m in measured),
        top=max(m.top for m in measured),
    )
    return widest.padded(pad) if pad else widest
