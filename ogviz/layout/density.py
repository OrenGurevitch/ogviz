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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

INK_TOLERANCE = 12  # 0-255 per channel; below this a pixel is the page, not a mark
# Overall ink coverage is deliberately NOT a complaint. It measures the mark type, not the layout:
# one well-fitted line covers about 3% of any canvas and a dense scatter covers 40%, and neither
# number says whether space is being wasted. `coverage` is reported for interest; the signals that
# actually mean something are emptiness where ink was expected — the outer margin and each panel's
# own empty bands.
GENEROUS_BAND = 0.18  # an empty strip wider than this share of the panel is worth reporting
LOOSE_MARGIN_PX = 8.0  # outer white space past this is worth reclaiming


def ink_mask(fig: Figure, *, tolerance: int = INK_TOLERANCE) -> NDArray[np.bool_]:
    """True where the rendered figure differs from its own page colour.

    The page colour is read from the corner pixel of the render rather than from rcParams, so a
    figure that set its own facecolor, or that was saved with one, is measured against what it
    actually is.
    """
    fig.canvas.draw()
    # Only a raster canvas can be read back. Asserted rather than assumed: on a vector backend the
    # attribute is simply absent, and the failure would otherwise be an AttributeError from inside
    # a QC helper rather than a sentence naming the cause.
    read_back = getattr(fig.canvas, "buffer_rgba", None)
    assert read_back is not None, (
        "measuring ink needs a raster canvas — run under Agg (matplotlib.use('Agg')), which is "
        f"what the figure builders do; this figure has a {type(fig.canvas).__name__}"
    )
    buffer = np.asarray(read_back(), dtype=np.int16)[:, :, :3]
    page = buffer[0, 0, :]
    return np.any(np.abs(buffer - page) > tolerance, axis=2)


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
    """Ink coverage and the unused band on each side of the canvas, in pixels."""
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
    from ogviz.layout.collision import _decoration

    skip = _decoration(ax)
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
        for artist, state in zip(was_visible and marks, was_visible, strict=True):
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
        return {"coverage": 0.0, "left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}
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
    for side in ("left", "right", "top", "bottom"):
        unused = getattr(overall, side)
        if unused > LOOSE_MARGIN_PX:
            notes.append(f"{unused:.0f} px of the canvas past the {side} edge of the ink is unused")
    for index, ax in enumerate(fig.axes):
        panel = panel_emptiness(fig, ax)
        for side in ("left", "right", "top", "bottom"):
            if panel[side] > GENEROUS_BAND:
                notes.append(
                    f"axes {index}: the {side} {panel[side]:.0%} of the panel is empty — the "
                    f"{'value' if side in ('top', 'bottom') else 'category'} limit is generous"
                )
    return notes


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
    moved = {
        "left": max(0.0, current.left - (before.left - pad_px) / width),
        "right": min(1.0, current.right + (before.right - pad_px) / width),
        "bottom": max(0.0, current.bottom - (before.bottom - pad_px) / height),
        "top": min(1.0, current.top + (before.top - pad_px) / height),
    }
    if moved["left"] >= moved["right"] or moved["bottom"] >= moved["top"]:
        return False  # the trim would invert the axes; leave it alone
    fig.subplots_adjust(**moved)
    return True
