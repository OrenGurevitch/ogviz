"""Pairwise relationships, each with its estimate read on one shared scale underneath.

The case this exists for: three or four variables that might all be measuring the same thing, and
the question of whether any one of them stands in for the others. Taken two at a time, each pair
gets a scatter, and under the scatter the correlation itself — a dot at the estimate, a bar for
its interval, one row per subset of the data.

The strip is the point. Numbers printed inside a panel read as clutter and have to be decoded one
at a time; the same estimates drawn on a shared axis are read at a glance — how far from zero, how
precise, and whether the interval covers it. Every strip in the figure shares one scale, so the
panels compare to each other and not only to themselves.

Stratifying is why the rows exist. A correlation pooled across two groups that are separated on
both axes can carry a sign no within-group trend shares, and a reader given only the pooled number
cannot see that. Drawing the pooled row and the per-group rows on the same axis makes the
disagreement, or the agreement, the thing you notice first.

The statistics are the caller's. This module places marks at numbers it is handed and never
computes a correlation, an interval, or a trend's significance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import patheffects
from matplotlib.ticker import MaxNLocator

from ogviz.layout import hairline_grid
from ogviz.require import require
from ogviz.significance import stars
from ogviz.tags import mark
from ogviz.theme import AXIS_LABEL_SIZE, INK, STAR_SIZE, TICK_SIZE, page_color

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

POINT_SIZE = 80
POINT_ALPHA = 0.5  # translucent on purpose: where the cloud is dense the overlap reads as depth,
# and a trend line drawn through it stays legible instead of being buried under solid dots
POINT_EDGE_WIDTH = 1.0
TREND_WIDTH = 4.4
POOLED_WIDTH = 4.8
HALO_WIDTH = 3.0  # added to a trend line's width, in the page colour, so it reads over the cloud
# matplotlib scales a dash pattern BY the line width, so (5, 4) at width 3.2 is a 29-point cycle —
# longer than a legend handle, which then shows one fragment of a dash instead of a dashed line.
# A short cycle repeats inside the handle.
POOLED_DASH = (0, (2.2, 1.3))
ESTIMATE_WIDTH = 4.2
ESTIMATE_MARKER_SIZE = 11
ESTIMATE_MARKER_EDGE = 1.6
MINIMUM_FOR_A_TREND = 3  # two points define a line through themselves and say nothing
STRIP_TICKS = 5  # before pruning the two end ticks
# Just OUTSIDE the axes. Inside, the column has to be squeezed between the widest interval and the
# right spine, and the width available depends on the font — it fitted in Arial and collided with an
# interval bar in DejaVu. Outside, the lane is always there, and the shared scale that makes the
# strips comparable is not disturbed to make room.
#
# The cost is a requirement on whoever lays the figure out: the RIGHTMOST panel needs that lane to
# exist at the page edge too. A caller that pushed `right` to 0.985 sent its last column of stars
# 17 px off the canvas, where a plain save crops them and a tight save silently grows the page.
# `text_off_canvas` is the check that catches it.
STAR_COLUMN = 1.015
# A column star is smaller than a bracket star, and this is how much. Measured 2026-08-13 on the
# default three-column layout at 12.8 in, with `label_for` producing the widest realistic label
# (`q=0.001 ***`), which is the case that decides it — a bare `***` clears the 73 px gutter at
# every size tried. Past the RIGHT-HAND edge of the canvas, for the rightmost column:
#
#     x STAR_SIZE   0.50   0.62   0.75   1.00
#     clearance    +52px  +37px  +15px  -22px   <- runs off the page at full size
#
# So the ceiling is between 0.75 and 1.00, and 0.62 is one step inside it. The star also has to
# read as smaller than a bracket star, because a bracket star marks a COMPARISON between two
# positions and this marks a row; at 0.75 the two are close enough to be mistaken for each other.
COLUMN_STAR = 0.62
STRIP_PAD = 1.12  # the shared estimate scale reaches this far past the widest interval
SCATTER_TO_STRIP = (3.0, 1.15)  # how the height of one column is split


@dataclass(frozen=True)
class Cloud:
    """One subset's points in a scatter, in the colours the project gives it."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    fill: str
    edge: str
    label: str
    trend: bool = True


@dataclass(frozen=True)
class Estimate:
    """One row of a strip: where the estimate sits, and how far its interval reaches.

    `fill` is the dot's face, for the case where a row's colour scheme wants a light dot with a
    dark rim. Left out, the dot is drawn in `color` throughout.
    """

    label: str
    value: float
    interval: tuple[float, float]
    color: str
    fill: str | None = None
    p: float | None = None  # given, the strip prints its stars in a column at the right

    def __post_init__(self) -> None:
        low, high = self.interval
        require(
            low <= high,
            f"{self.label}: interval {self.interval} runs backwards",
        )
        require(
            low <= self.value <= high,
            f"{self.label}: the estimate {self.value:g} sits outside its interval {self.interval}",
        )
        require(
            self.p is None or 0.0 <= self.p <= 1.0,
            f"{self.label}: p={self.p} is not a p-value",
        )


@dataclass(frozen=True)
class Leg:
    """One pair of variables: the clouds to scatter, and the estimates to strip beneath them."""

    x_label: str
    y_label: str
    clouds: tuple[Cloud, ...]
    estimates: tuple[Estimate, ...] = field(default_factory=tuple)


def trend_line(
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    color: str,
    dashed: bool = False,
    width: float | None = None,
) -> tuple[float, float] | None:
    """A least-squares guide over the paired points, haloed so it survives the cloud.

    Returns the line's right-hand end, where a label for it would go, or None when there were too
    few points to fit one. A guide, not an inference: nothing here tests whether the slope differs
    from zero, and a caller that wants to say so has to have measured it.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(finite)) < MINIMUM_FOR_A_TREND:
        return None
    x_fit, y_fit = x[finite], y[finite]
    # A least-squares line needs the x values to differ. Given points stacked on one x, `polyfit`
    # returns a meaningless slope with a RankWarning — and `strict` promotes only DeprecationWarning
    # to an error, so it passed the gate and the panel drew a confident guide through a column of
    # points. Nothing to fit is not a fit; the caller gets the same None as too few points.
    if float(np.ptp(x_fit)) == 0.0:
        return None
    slope, intercept = (float(v) for v in np.polyfit(x_fit, y_fit, 1))
    ends = np.array([x_fit.min(), x_fit.max()], dtype=float)
    line_width = width if width is not None else (POOLED_WIDTH if dashed else TREND_WIDTH)
    ax.plot(
        ends,
        slope * ends + intercept,
        color=color,
        linewidth=line_width,
        linestyle=POOLED_DASH if dashed else "-",
        solid_capstyle="round",
        zorder=5,
        path_effects=[
            patheffects.withStroke(linewidth=line_width + HALO_WIDTH, foreground=page_color())
        ],
    )
    return float(ends[-1]), float(slope * ends[-1] + intercept)


def scatter_panel(
    ax: Axes,
    leg: Leg,
    *,
    pooled_color: str | None = INK,
    point_size: float = POINT_SIZE,
    point_alpha: float = POINT_ALPHA,
) -> None:
    """The observations and their trends. No numbers and no box over the data.

    `pooled_color` adds a dashed trend fitted to every cloud at once, which is the line the strip's
    pooled row corresponds to. Pass None where pooling the subsets would not mean anything.
    """
    require(
        leg.clouds,
        f"the {leg.x_label} / {leg.y_label} leg has nothing to scatter",
    )
    for cloud in leg.clouds:
        require(
            cloud.x.shape == cloud.y.shape,
            f"{cloud.label}: {cloud.x.shape[0]} x values and {cloud.y.shape[0]} y values",
        )
        ax.scatter(
            cloud.x,
            cloud.y,
            s=point_size,
            alpha=point_alpha,
            color=cloud.fill,
            edgecolor=cloud.edge,
            linewidth=POINT_EDGE_WIDTH,
            label=cloud.label,
            zorder=3,
        )
        if cloud.trend:
            trend_line(ax, cloud.x, cloud.y, color=cloud.edge)
    if pooled_color is not None:
        every_x = np.concatenate([cloud.x for cloud in leg.clouds])
        every_y = np.concatenate([cloud.y for cloud in leg.clouds])
        trend_line(ax, every_x, every_y, color=pooled_color, dashed=True)

    ax.set_xlabel(leg.x_label)
    ax.set_ylabel(leg.y_label)
    ax.margins(y=0.10)
    hairline_grid(ax, axis="y")
    ax.spines["left"].set(linewidth=1.0)
    ax.spines["bottom"].set(linewidth=2.0, color=INK)


def _star_column(
    ax: Axes,
    estimates: Sequence[Estimate],
    label_for: Callable[[float], str] | None = None,
) -> None:
    """Stars for the rows that carry a p, aligned in one column at the strip's right edge.

    A column rather than a mark beside each dot: the dots sit at their estimates, which are all
    over the axis, so stars placed next to them would be scattered and would collide with the
    intervals they belong to. In a column the reader compares rows down one line, and the axis
    still shows how far each estimate is from the reference — which is the thing a star cannot say.

    The row's own colour, so a star is unambiguous about which estimate it belongs to.

    `label_for` decides what the column SAYS, the same separation `bracket_stack` makes: ogviz owns
    where the mark sits and never what it reads. Stars alone are categorical, and a project whose
    intervals clear zero while nothing survives its multiplicity correction needs the number as well
    — with stars only, that figure reads as findings it does not have. Such a project passes
    `lambda q: f"q={q:.3f} {stars(q)}"`.

    A longer label is wider, and the column sits just outside the panel; `text_off_canvas` is the
    check that catches a rightmost column running off the page.
    """
    marked = [(row, e) for row, e in enumerate(estimates) if e.p is not None]
    if not marked:
        return
    for row, estimate in marked:
        assert estimate.p is not None  # narrowed by the comprehension above
        glyphs = (label_for or stars)(estimate.p)
        if not glyphs:
            continue
        drawn = ax.text(
            STAR_COLUMN,
            row,
            glyphs,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=STAR_SIZE * COLUMN_STAR,
            fontweight="bold",
            color=estimate.color,
            zorder=6,
        )
        # A strip star marks a ROW, not a comparison between two positions, so it has no bracket
        # and must not be measured against one. Marked rather than inferred: "this axes has no
        # brackets" would also excuse a violin panel whose brackets all failed to draw.
        mark(drawn, "column_star")
        mark(drawn, "anchored")


def estimate_strip(
    ax: Axes,
    estimates: Sequence[Estimate],
    *,
    limits: tuple[float, float],
    name_the_rows: bool = True,
    reference: float | None = 0.0,
    label_size: float = TICK_SIZE - 3.5,
    label_for: Callable[[float], str] | None = None,
    sort_by: Literal["given", "value", "magnitude"] = "given",
) -> None:
    """The estimates as dot-and-interval marks on a shared scale — a FOREST PLOT, standalone or in
    a column of them.

    Worth saying under that name: a sorted forest of point estimates with confidence whiskers, a
    zero rule and a significance column is exactly this function, and a project that did not know
    that hand-rolled one beside it. Colour per row comes from the `Estimate`, so colouring by
    significance, by direction or by group is done where the rows are built.

    Rows are drawn bottom-up in the order given, so the sequence reads the way a legend does — and
    that is the trap `sort_by` exists for: a caller who sorts its own list descending gets the
    largest effect at the BOTTOM. `"magnitude"` puts the largest at the top, which is the ordering a
    forest is normally read in, and `"value"` sorts signed so the negatives gather at one end.

    The reference line is where "no relationship" sits; pass None for a quantity that has no such
    value.
    """
    require(
        estimates,
        "a strip with no estimates in it",
    )
    if sort_by != "given":
        key = (
            (lambda e: abs(e.value - (reference or 0.0)))
            if sort_by == "magnitude"
            else (lambda e: e.value)
        )
        # Ascending, because row 0 is drawn at the BOTTOM: ascending by magnitude puts the largest
        # effect on the top row.
        estimates = sorted(estimates, key=key)
    if reference is not None:
        ax.axvline(reference, color=INK, linewidth=1.4, zorder=1)
    for row, estimate in enumerate(estimates):
        low, high = estimate.interval
        ax.plot(
            [low, high],
            [row, row],
            color=estimate.color,
            linewidth=ESTIMATE_WIDTH,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            [estimate.value],
            [row],
            marker="o",
            markersize=ESTIMATE_MARKER_SIZE,
            markerfacecolor=estimate.fill if estimate.fill is not None else estimate.color,
            markeredgecolor=estimate.color,
            markeredgewidth=ESTIMATE_MARKER_EDGE,
            zorder=4,
        )

    _star_column(ax, estimates, label_for)
    ax.set_xlim(*limits)
    # Strips stand side by side on one scale, so the tick at one strip's right edge and the tick
    # at its neighbour's left edge are a gutter apart and collide as soon as the scale reaches a
    # round number. Pruning both ends is what keeps the row readable at any width.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=STRIP_TICKS, prune="both"))
    ax.set_ylim(-0.7, len(estimates) - 0.3)
    ax.set_yticks(range(len(estimates)))
    ax.set_yticklabels(
        [estimate.label for estimate in estimates] if name_the_rows else [],
        fontsize=label_size,
    )
    ax.tick_params(axis="y", length=0.0)
    ax.tick_params(axis="x", labelsize=label_size - 1)
    hairline_grid(ax, axis="x")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set(linewidth=1.6, color=INK)


def shared_limits(
    legs: Sequence[Leg], *, pad: float = STRIP_PAD, reference: float | None = 0.0
) -> tuple[float, float]:
    """One scale wide enough for every interval in the figure, centred on the reference.

    Centred rather than fitted to the data, because a strip whose reference line sits off-centre
    reads as though the estimates lean one way before any of them has been looked at.

    `reference` is what "no relationship" is on this quantity, and it was hardcoded to zero — the
    same number `estimate_strip` takes as an argument and documents passing something else for. A
    ratio has its null at 1.0, and the composed figure centred it on zero regardless, so the case
    the standalone function supports was the case the composed one silently could not draw. None
    centres on zero, since a quantity with no null has no other centre to prefer.
    """
    middle = reference or 0.0
    reach = max(
        (abs(bound - middle) for leg in legs for e in leg.estimates for bound in e.interval),
        default=1.0,
    )
    return (middle - pad * reach, middle + pad * reach)


def coupling_panels(
    fig: Figure,
    legs: Sequence[Leg],
    *,
    estimate_axis_label: str | None = None,
    pooled_color: str | None = INK,
    limits: tuple[float, float] | None = None,
    height_ratios: tuple[float, float] = SCATTER_TO_STRIP,
    width_space: float = 0.26,
    height_space: float = 0.42,
    label_for: Callable[[float], str] | None = None,
    reference: float | None = 0.0,
) -> None:
    """One column per pair: the scatter above, its estimates below, all strips on one scale.

    Row labels are printed on the leftmost strip only. Repeating them under every column costs the
    width the panels need and tells the reader nothing they did not learn from the first.

    `reference` is where "no relationship" sits, passed to every strip AND to the shared scale so
    the two agree. There was no way to say it here: the strips took the default zero and the scale
    was centred on zero regardless, so a ratio-valued estimate — null at 1.0 — could not be drawn
    by this function although `estimate_strip` supports it directly.

    A LEG WITH NO ESTIMATES gets a scatter spanning both rows rather than a hole under it. The
    lower cell was simply never filled, and an empty gridspec cell in a finished figure reads as a
    rendering failure. `panel_grid` treats exactly this as worth reporting; here it said nothing.
    """
    require(
        legs,
        "coupling_panels needs at least one leg",
    )
    grid = fig.add_gridspec(2, len(legs), height_ratios=height_ratios)
    scale = limits if limits is not None else shared_limits(legs, reference=reference)
    for column, leg in enumerate(legs):
        cell = grid[0, column] if leg.estimates else grid[:, column]
        scatter_panel(fig.add_subplot(cell), leg, pooled_color=pooled_color)
        if leg.estimates:
            estimate_strip(
                fig.add_subplot(grid[1, column]),
                leg.estimates,
                limits=scale,
                name_the_rows=column == 0,
                label_for=label_for,
                reference=reference,
            )
    if estimate_axis_label is not None:
        fig.supxlabel(estimate_axis_label, fontsize=AXIS_LABEL_SIZE)
    fig.subplots_adjust(wspace=width_space, hspace=height_space)
