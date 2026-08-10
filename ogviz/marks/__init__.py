"""The violin and its central marks, drawn in one fixed order.

Stacking is named here so no caller re-decides it: violin body < points < IQR bar < mean line <
median dot. Getting this wrong is invisible in code review and obvious in the figure — a mean
line drawn under the IQR bar reads as passing behind it.

The mean line's halo is the canvas colour and only slightly wider than the line. A wide *white*
halo under a dark IQR bar punches a pale gap through the bar, which reads as the mean line
going behind it — the exact bug the ordering was supposed to prevent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patheffects as path_effects
import numpy as np
from scipy.stats import gaussian_kde

from ogviz import units
from ogviz.orientation import (
    is_vertical,
    place_many,
    require_linear_value_axis,
    violin_orientation_kwarg,
)
from ogviz.require import require
from ogviz.tags import mark
from ogviz.theme import INK, page_color

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from ogviz.orientation import Orientation

# The central marks are a SET, and this is the order they stack in. Exported from the top-level
# namespace as well as here, because a caller writing its own mark needs the ordering in front of
# it: one that hand-rolled the box, whiskers, median dot and mean line drew the mean at 11 over the
# median at 10, and the mean line covered the median dot. A primitive that is easy to reimplement
# will be reimplemented, and the reimplementation does not carry the invariant.
# An error bar belongs with the marks, not with bars: a line panel, a slopegraph and a matrix can
# all want one. It kept `Z_ERROR` company in `panels/bars.py` until 2026-08-01, which is also what
# made `Z_REFERENCE` end up defined twice — once derived from `Z_ERROR` there and once hardcoded in
# `panels/reference.py`, because deriving it would have been an import cycle.
Z_ERROR = 4
ERROR_CAPSIZE = 4.0
ERROR_LINEWIDTH = 1.4

Z_VIOLIN = 2
Z_POINTS = 3
Z_IQR = 4
Z_MEAN_LINE = 6
Z_MEDIAN_DOT = 8

VIOLIN_WIDTH = 0.62
VIOLIN_ALPHA = 0.35
POINT_SIZE = 30
POINT_ALPHA = 0.85
POINT_EDGE_WIDTH = 0.8
JITTER_FILL = 0.82  # fraction of the local half-width dots may occupy (<1 keeps them inside)
CENTER_GAP = 0.06  # min x-gap dots keep from the centre marks where the violin is wide
BOX_COLOR = "#6E6E6E"
MEAN_HALF_WIDTH = 0.15
# The widths of the central marks, named once. `iqr_box` DRAWS at these and `central_clearance`
# reserves a lane against them, and they were two pairs of bare literals in two functions — which is
# how the clearance came to reserve the IQR bar's width along the thin whisker.
BOX_WIDTH = 5.5
WHISKER_WIDTH = 1.5
MEDIAN_SIZE = 7.5
MEAN_LINEWIDTH = 3.2


def violin(
    ax: Axes,
    values: NDArray[np.float64],
    position: float,
    color: str,
    *,
    width: float = VIOLIN_WIDTH,
    alpha: float = VIOLIN_ALPHA,
    edge_color: str | None = None,
    edge_width: float = 1.3,
    zorder: float = Z_VIOLIN,
    orientation: Orientation = "vertical",
) -> None:
    """Filled kernel-density body, no matplotlib extras.

    `width` is an argument because it is a layout fact, not a house fact: three conditions in a
    cell want a wider body than two groups in a tall panel. Anything a project must be able to
    set per panel belongs in the signature; the default is the house value.
    """
    parts = ax.violinplot(
        [values],
        positions=[position],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=width,
        **violin_orientation_kwarg(orientation),  # type: ignore[arg-type]
    )
    for body in parts["bodies"]:  # type: ignore[union-attr]
        body.set_facecolor(color)
        body.set_alpha(alpha)
        body.set_edgecolor(edge_color if edge_color is not None else "none")
        if edge_color is not None:
            body.set_linewidth(edge_width)
        body.set_zorder(zorder)


def jitter_x(
    values: NDArray[np.float64],
    position: float,
    rng: np.random.Generator,
    *,
    width: float = VIOLIN_WIDTH,
    fill: float = JITTER_FILL,
    center_gap: float = CENTER_GAP,
    clearance: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Category-axis coordinate per point, spread across the violin's OWN width at its value.

    Uniform jitter puts dots outside a narrow violin and clumps them inside a wide one. Spreading
    by the same Gaussian KDE the body is drawn from keeps every dot inside the shape it belongs
    to, and a central lane keeps the box, the mean line and the median dot readable.

    `clearance` is the per-point half-width that lane must have, which is NOT one number: a dot
    level with the mean has to clear the mean line, and one out in a tail only has to clear the
    thin whisker. When it is given it is a FLOOR — a dot is pushed just outside the marks rather
    than squeezed onto them, even where the body is too narrow to hold it. Squeezing is what put
    dots on top of the mean line and the median.
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2 or float(np.ptp(v)) == 0.0:
        return np.full(len(v), float(position))
    kde = gaussian_kde(v)
    densest = max(float(kde(np.linspace(v.min(), v.max(), 200)).max()), float(kde(v).max()))
    half_width = kde(v) / densest * (width / 2) * fill
    if clearance is None:
        inner = np.minimum(center_gap, half_width * 0.9)
    else:
        inner = np.asarray(clearance, dtype=np.float64)
        half_width = np.maximum(half_width, inner * 1.06)  # always somewhere to put the dot
    side = np.where(rng.random(len(v)) < 0.5, -1.0, 1.0)
    return position + side * rng.uniform(inner, half_width)


# The central marks a dot may have to keep clear of, named. `"iqr"` covers the whisker, the box and
# the median dot, because `iqr_box` draws all three and a caller either has that mark or does not.
MARK_NAMES: tuple[str, ...] = ("iqr", "mean")


def central_clearance(
    ax: Axes,
    values: NDArray[np.float64],
    *,
    point_size: float = POINT_SIZE,
    mean_half_width: float = MEAN_HALF_WIDTH,
    mean_linewidth: float = MEAN_LINEWIDTH,
    box_linewidth: float = BOX_WIDTH,
    whisker_linewidth: float = WHISKER_WIDTH,
    median_size: float = MEDIAN_SIZE,
    orientation: Orientation = "vertical",
    drawn: Collection[str] | None = None,
) -> NDArray[np.float64]:
    """Per-point half-width, in data units, that a dot must keep clear of the central marks.

    The marks do not all span the whole violin. The whisker runs its full height but is thin; the
    IQR bar is thicker and spans Q1-Q3; the mean line is by far the widest but occupies only its
    own linewidth in y; the median dot only its own diameter. So the lane a dot has to respect
    depends on where the dot IS — which is why one `center_gap` could not do this, and why dots
    kept landing on the mean line, whose reach is more than twice that gap.

    THE WHISKER CASE DID NOT WORK until 2026-08-06. The line that was supposed to narrow the lane
    outside Q1-Q3 read `np.where(inside_the_box, box_linewidth / 2, lane)` where `lane` was already
    `box_linewidth / 2` everywhere — both branches identical, so it did nothing at all and there was
    no whisker width in the signature to do it with. Measured: a dot out in a tail reserved exactly
    as much room as a dot against the IQR bar, 3.7x the ink it was actually avoiding, so every tail
    was pushed needlessly wide of a shape the jitter is supposed to follow.

    `drawn` NAMES THE MARKS THAT ARE ACTUALLY THERE — any of `"iqr"` and `"mean"`; omit it and both
    are assumed, which is what every existing caller gets. It exists because the lane is reserved
    for the full mark set, and a panel that hand-assembles `violin` + `points` + `mean_line` and
    deliberately draws no IQR box was still holding room for a bar that is not on the figure. Two
    consumers assemble exactly that trio. The dots were pushed wide of nothing, which is invisible
    in the sense that matters: the figure looks fine and the jitter no longer follows the shape it
    is supposed to describe.

    `widths_of` answers the neighbouring question — how WIDE the marks are, read off the kwargs
    they were drawn with — and the two compose: pass `widths_of(box_kwargs, mean_kwargs)` for the
    sizes and `drawn={...}` for which exist.

    Reads the axes transform, so the limits must be set before this is called.
    """
    require_linear_value_axis(ax, orientation, "central_clearance")
    present = set(MARK_NAMES if drawn is None else drawn)
    unknown = present - set(MARK_NAMES)
    require(not unknown, f"central_clearance does not know the mark(s) {sorted(unknown)}")

    v = np.asarray(values, dtype=np.float64)
    to_data_x, to_data_y = _data_per_point(ax, orientation)
    dot_radius_x = to_data_x * float(np.sqrt(point_size)) / 2.0
    dot_radius_y = to_data_y * float(np.sqrt(point_size)) / 2.0

    q1, median, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    mean = float(np.mean(v))

    # A mark that was not drawn reserves nothing. The dot's own radius is added at the end
    # regardless, so a lane of zero still keeps a dot from straddling the centre line.
    lane = np.zeros(len(v))
    if "iqr" in present:
        lane = np.full(len(v), to_data_x * whisker_linewidth / 2.0)  # the thin whisker, everywhere
        lane = np.where((v >= q1) & (v <= q3), to_data_x * box_linewidth / 2.0, lane)
        near_median = np.abs(v - median) <= (to_data_y * median_size / 2.0 + dot_radius_y)
        lane = np.where(near_median, to_data_x * median_size / 2.0, lane)
    if "mean" in present:
        near_mean = np.abs(v - mean) <= (to_data_y * mean_linewidth / 2.0 + dot_radius_y)
        lane = np.where(near_mean, mean_half_width, lane)
    return lane + dot_radius_x


def _data_per_point(ax: Axes, orientation: Orientation) -> tuple[float, float]:
    """Data units per typographic point, on the (category, value) axes."""
    figure = ax.figure
    assert figure is not None, "the axes must belong to a figure"
    px_per_point = units.px_per_point(figure)
    origin = ax.transData.transform((0.0, 0.0))
    unit = ax.transData.transform((1.0, 1.0))
    px_per_data_x = abs(unit[0] - origin[0]) or 1.0
    px_per_data_y = abs(unit[1] - origin[1]) or 1.0
    across, along = px_per_data_x, px_per_data_y
    if not is_vertical(orientation):
        across, along = along, across
    return px_per_point / across, px_per_point / along


def points(
    ax: Axes,
    values: NDArray[np.float64],
    position: float,
    color: str | Sequence[str],
    edge_color: str,
    rng: np.random.Generator,
    *,
    size: float = POINT_SIZE,
    alpha: float = POINT_ALPHA,
    edge_width: float = POINT_EDGE_WIDTH,
    width: float = VIOLIN_WIDTH,
    fill: float = JITTER_FILL,
    center_gap: float = CENTER_GAP,
    clear_central_marks: bool = True,
    mark_widths: Mapping[str, float] | None = None,
    marks_drawn: Collection[str] | None = None,
    orientation: Orientation = "vertical",
) -> None:
    """One dot per observation, jittered inside the violin body.

    `color` is one colour for the cloud, or ONE COLOUR PER OBSERVATION in the order `values` are
    given — which is how a dot carries an identity rather than a group. In a repeated-measures
    panel the same subject appears in every violin, and colouring by subject lets a reader follow
    one of them across the conditions; a single colour can only restate the fill they already sit
    in. The sequence is passed to the scatter untouched, so it must be as long as `values`.

    `width`, `fill` and `center_gap` must match the violin this scatter sits in, or the dots
    spread outside a narrow body or clump inside a wide one.

    `mark_widths` is the same requirement for the CENTRAL marks, and it exists because the defaults
    agreeing is not the same as them matching. A caller drawing `iqr_box(box_width=9.0)` and leaving
    this alone reserves a lane for a 5.5-wide bar, and its dots sit on the box — the same class of
    failure as the whisker lane that did nothing, one step out. `widths_of` builds it from the
    kwargs a caller already passed to the marks themselves, so there is one place that knows which
    of `iqr_box`'s names answers which of `central_clearance`'s.

    HAND-ASSEMBLING A PANEL: `group_violins` passes both of these for you. Calling `violin` +
    `points` + `mean_line` yourself does not, and the lane then reserves room for the full mark set
    including an IQR box that was never drawn — dots pushed away from a bar that is not there. Two
    consumers assemble exactly that trio. Say what you drew:

        points(ax, v, x, fill, edge, rng,
               mark_widths=widths_of(box_kwargs, mean_kwargs),   # how wide the marks are
               marks_drawn={"mean"})                             # which marks exist at all

    The two answer different halves and neither implies the other. `mark_widths` was the only one
    available, so a panel with no box had to say so by passing zero widths for the box, the whisker
    AND the median — three numbers to express one fact, and forgetting the median left a lane for a
    dot that is not drawn. Measured on a 200-point sample: naming `{"mean"}` frees a fifth of the
    reserved lane.
    """
    clearance = (
        central_clearance(
            ax,
            values,
            point_size=size,
            orientation=orientation,
            drawn=marks_drawn,
            **(mark_widths or {}),
        )
        if clear_central_marks and len(np.asarray(values)) > 1
        else None
    )
    spread = jitter_x(
        values,
        position,
        rng,
        width=width,
        fill=fill,
        center_gap=center_gap,
        clearance=clearance,
    )
    horizontal, vertical = place_many(orientation, spread, values)
    drawn = ax.scatter(
        horizontal,
        vertical,
        color=color,
        s=size,
        alpha=alpha,
        edgecolor=edge_color,
        linewidth=edge_width,
        zorder=Z_POINTS,
    )
    # Record what was actually reserved. The lane is a step function of y whose steps move when
    # the axes are resized, so recomputing it after a `tight_layout` can assign a dot a different
    # lane than the one it was placed against — and QC then reports a correctly placed dot. The
    # only trustworthy check is against the value that was used.
    mark(drawn, "lane", clearance)
    mark(drawn, "position", float(position))


# `iqr_box` and `mean_line` name their widths for what they DRAW; `central_clearance` names them for
# what a dot must AVOID. The two vocabularies are not the same and the translation between them is
# exactly the kind of thing that gets written twice and drifts, so it is written here.
_WIDTH_NAMES = {
    "box_width": "box_linewidth",
    "whisker_width": "whisker_linewidth",
    "median_size": "median_size",
    "half_width": "mean_half_width",
    "linewidth": "mean_linewidth",
}


def widths_of(*mark_kwargs: Mapping[str, object] | None) -> dict[str, float]:
    """The clearance a dot needs, read off the kwargs the marks were actually drawn with.

    Takes the mappings a caller passed to `iqr_box` and `mean_line` and returns what
    `central_clearance` calls the same quantities. Anything that is not a width — a colour, a fill —
    is ignored, so a caller can hand over its whole kwargs mapping without filtering it first.
    """
    found: dict[str, float] = {}
    for given in mark_kwargs:
        for name, value in (given or {}).items():
            target = _WIDTH_NAMES.get(name)
            if target is not None and isinstance(value, (int, float)):
                found[target] = float(value)
    return found


def iqr_box(
    ax: Axes,
    values: NDArray[np.float64],
    position: float,
    *,
    color: str = BOX_COLOR,
    box_width: float = BOX_WIDTH,
    whisker_width: float = WHISKER_WIDTH,
    median_size: float = MEDIAN_SIZE,
    median_fill: str | None = None,
    orientation: Orientation = "vertical",
) -> None:
    """Q1-Q3 bar on a 1.5x IQR whisker, with the median as a dot on top."""
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    spread = q3 - q1
    low = max(float(values.min()), q1 - 1.5 * spread)
    high = min(float(values.max()), q3 + 1.5 * spread)
    ax.plot(
        *place_many(orientation, [position] * 2, [low, high]),
        color=color,
        lw=whisker_width,
        zorder=Z_IQR,
        solid_capstyle="round",
    )
    ax.plot(
        *place_many(orientation, [position] * 2, [q1, q3]),
        color=color,
        lw=box_width,
        zorder=Z_IQR,
        solid_capstyle="round",
    )
    ax.plot(
        *place_many(orientation, [position], [median]),
        "o",
        mfc=median_fill if median_fill is not None else page_color(),
        mec=color,
        mew=1.6,
        ms=median_size,
        zorder=Z_MEDIAN_DOT,
    )


def mean_line(
    ax: Axes,
    values: NDArray[np.float64],
    position: float,
    *,
    half_width: float = MEAN_HALF_WIDTH,
    color: str = INK,
    linewidth: float = MEAN_LINEWIDTH,
    halo: str | None = None,
    orientation: Orientation = "vertical",
) -> None:
    """Mean as a short ink line, over the IQR bar and under the median dot.

    The halo is the page colour, read at draw time so it follows `use_house_style(canvas=...)`;
    pass your own, or "none" to drop it.
    """
    m = float(np.mean(values))
    ax.plot(
        *place_many(orientation, [position - half_width, position + half_width], [m, m]),
        color=color,
        lw=linewidth,
        solid_capstyle="round",
        zorder=Z_MEAN_LINE,
        path_effects=[
            path_effects.withStroke(
                linewidth=linewidth * 1.75, foreground=halo if halo is not None else page_color()
            )
        ],
    )


def _draw_error_bars(
    ax: Axes,
    positions: NDArray[np.float64],
    centres: NDArray[np.float64],
    lengths: NDArray[np.float64],
    *,
    orientation: Orientation,
    color: str = INK,
    linewidth: float = ERROR_LINEWIDTH,
    capsize: float = ERROR_CAPSIZE,
    zorder: float = Z_ERROR,
) -> None:
    """One drawing of an interval, in matplotlib's own (2, N) below/above LENGTHS."""
    horizontal, vertical = place_many(orientation, positions, centres)
    upright = is_vertical(orientation)
    ax.errorbar(
        horizontal,
        vertical,
        yerr=lengths if upright else None,
        xerr=None if upright else lengths,
        fmt="none",
        ecolor=color,
        elinewidth=linewidth,
        capsize=capsize,
        capthick=linewidth,
        zorder=zorder,
    )


def error_bars(
    ax: Axes,
    positions: ArrayLike,
    centre: ArrayLike,
    low: ArrayLike,
    high: ArrayLike,
    *,
    orientation: Orientation = "vertical",
    color: str = INK,
    linewidth: float = ERROR_LINEWIDTH,
    capsize: float = ERROR_CAPSIZE,
    zorder: float = Z_ERROR,
) -> None:
    """Intervals from absolute BOUNDS, in the house style.

    Bounds rather than lengths because that is what every statistics library hands back — a
    bootstrap returns the 2.5th and 97.5th percentiles, not distances from the mean. matplotlib
    wants `yerr=[[centre - low], [high - centre]]`, a (2, N) of lengths in below/above order, and
    that subtraction is the step consumers were repeating and could get backwards: swapping the
    rows draws a plausible interval on the wrong side of every point, and nothing errors.

    Asymmetric intervals are the normal case here, which is why there is no single-length shortcut.
    """
    place = np.asarray(positions, dtype=float)
    middle = np.asarray(centre, dtype=float)
    lower = np.asarray(low, dtype=float)
    upper = np.asarray(high, dtype=float)
    if not place.shape == middle.shape == lower.shape == upper.shape:
        raise AssertionError(
            f"positions {place.shape}, centre {middle.shape}, low {lower.shape}, "
            f"high {upper.shape} must all describe the same points"
        )
    below = middle - lower
    above = upper - middle
    # A bound on the wrong side is a caller error, not something to draw politely: matplotlib takes
    # a negative length without complaint and renders the cap inside the interval. Raised rather
    # than asserted so `python -O` cannot delete the refusal.
    if not (np.all(below >= 0.0) and np.all(above >= 0.0)):
        raise AssertionError(
            "low must be at or below centre and high at or above it; got "
            f"{np.count_nonzero(below < 0)} inverted lower and {np.count_nonzero(above < 0)} "
            "upper. Passing lengths where bounds are wanted looks exactly like this."
        )
    _draw_error_bars(
        ax,
        place,
        middle,
        np.vstack([below, above]),
        orientation=orientation,
        color=color,
        linewidth=linewidth,
        capsize=capsize,
        zorder=zorder,
    )
