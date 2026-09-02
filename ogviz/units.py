"""One place that converts between the coordinate systems, because mixing them is the bug.

Every layout question in this package is asked in one of five systems, and the answer is wrong if
it is computed in a different one:

  data          what the numbers mean. Non-linear on a log axis
  axes fraction 0 to 1 across the panel. Linear whatever the scale
  display px    what the reader sees. The only system in which "looks halfway" is true
  points        physical, 1/72 inch. Type sizes and line widths live here
  em            relative to a type size. What a gap "beside a label" should be measured in

The failures this session were all one mistake: answering a question posed in one system using
arithmetic from another. A midpoint averaged in DATA units is the visual midpoint only while the
axis is linear — on a log axis from 1 to 1000 it lands a third of the way along. A gap chosen as a
fraction of the data span changes meaning when the limits change. A search that walks a fixed
number of PIXELS covers a tenth of one panel and the whole of another.

ProPlot and UltraPlot solved this by specifying every spacing in physical units and converting
once; matplotlib itself keeps the transforms but leaves the choice to the caller. This module is
the small version of that idea: name the system, convert through the transform that is defined for
it, and never divide two numbers that came from different ones.

Nothing here is clever. It exists so the conversion is written once and can be read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ogviz.require import require

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

# A SubFigure carries a dpi of its own (delegated to its parent), and `ax.figure` is typed as
# either — so every caller that reaches for a dpi has this union in hand. Writing it out here is
# what centralising the conversion turned up: four sites did `.dpi` inline, where the union was
# accepted silently, and the question of whether a sub-figure is a figure for this purpose was
# never asked. It is: the answer is its dpi, and that is all any of this needs.
Unit = Literal["px", "pt", "in", "cm", "mm", "em"]
POINTS_PER_INCH = 72.0
CM_PER_INCH = 2.54


def px_per_point(fig: Figure | SubFigure) -> float:
    """How many display pixels one typographic point is on this figure.

    The FACTOR, not a converted value, because that is the shape most of the callers need: a
    placement that measures ink in points and then works in pixels reads it once and multiplies
    several times, and `to_px` per multiplication would re-read the dpi each time and read less
    clearly at the site. Five modules had `dpi / 72.0` written out for exactly this.
    """
    return fig.dpi / POINTS_PER_INCH


def to_px(value: float, unit: Unit, *, fig: Figure | SubFigure, em: float | None = None) -> float:
    """A length in any physical unit, as display pixels on this figure.

    `em` is the type size the value is relative to, and is required for `"em"` — an em with no font
    behind it is not a length.
    """
    dpi = fig.dpi
    if unit == "px":
        return value
    if unit == "pt":
        return value * px_per_point(fig)
    if unit == "in":
        return value * dpi
    if unit == "cm":
        return value / CM_PER_INCH * dpi
    if unit == "mm":
        return value / 10.0 / CM_PER_INCH * dpi
    # An unknown unit used to fall through to the em branch: a typo was reported as a missing type
    # size, and with `em=` given it returned a number in no unit at all.
    require(unit == "em", f"unknown unit {unit!r}; one of px, pt, in, cm, mm, em")
    require(em is not None, "an em is relative to a type size; pass the size it is relative to")
    return value * float(em or 0.0) * px_per_point(fig)


def to_points(pixels: float, *, fig: Figure | SubFigure) -> float:
    """Display pixels back to typographic points — the direction this module could not go.

    It converted TO pixels and never back, so every caller that had a measured extent and needed a
    type size wrote `px / fig.dpi * 72.0` itself: the caption's wrap target, the panel wrapper, the
    bar panel's slot width. A module whose whole claim is that the conversion is written once has to
    be able to answer it in both directions, or the half it cannot answer goes on being written out.
    """
    return pixels / px_per_point(fig)


def inches_to_points(inches: float) -> float:
    """Figure inches as points. The one conversion with no figure in it — 72 is the definition."""
    return inches * POINTS_PER_INCH


def points_to_inches(points: float) -> float:
    """The inverse of `inches_to_points`, for a figure size computed from a type size."""
    return points / POINTS_PER_INCH


def value_to_px(ax: Axes, value: float, *, orientation: str = "vertical") -> float:
    """A data value as a display pixel, through the axis's own transform.

    Through the transform rather than by scaling a ratio, because a ratio only exists on a linear
    axis. This is defined on every scale the axis can have.
    """
    point = (0.0, value) if orientation == "vertical" else (value, 0.0)
    index = 1 if orientation == "vertical" else 0
    return float(ax.transData.transform(point)[index])


def px_to_value(ax: Axes, pixels: float, *, orientation: str = "vertical") -> float:
    """The inverse of `value_to_px`, so a decision made in pixels can be applied in data."""
    point = (0.0, pixels) if orientation == "vertical" else (pixels, 0.0)
    index = 1 if orientation == "vertical" else 0
    return float(ax.transData.inverted().transform(point)[index])


def midpoint(ax: Axes, low: float, high: float, *, orientation: str = "vertical") -> float:
    """The data value halfway between two data values AS THE READER SEES IT.

    The whole reason this module exists. `(low + high) / 2` is the visual midpoint only on a linear
    axis; on a log axis running 1 to 1000, the arithmetic midpoint of 1 and 100 is 50.5, which sits
    a third of the way up a gap the eye reads as centred at 10.
    """
    low_px = value_to_px(ax, low, orientation=orientation)
    high_px = value_to_px(ax, high, orientation=orientation)
    return px_to_value(ax, (low_px + high_px) / 2.0, orientation=orientation)


def panel_px(ax: Axes, *, orientation: str = "vertical") -> float:
    """The panel's extent in pixels ALONG ITS VALUE AXIS — the scale a search should be sized to.

    `orientation` names which way the PANEL runs, as everywhere else in this module, so it decides
    which side is the value axis rather than which side to measure: a vertical panel's values run up
    the height, a horizontal one's run across the width. That is what makes a threshold written
    against this survive `orientation="horizontal"` unchanged.

    A caller who wants one FIXED side whatever the panel is doing does not want this — they want
    `ax.get_window_extent().width`, which is a line and says so. It was recorded as a defect that
    the nearest candidate caller reached for the width regardless of orientation and did not use
    this; that caller was right, and the two are different questions rather than one badly named.
    """
    box = ax.get_window_extent()
    return float(box.height if orientation == "vertical" else box.width)
