"""Series over a continuous axis: heavy strokes, one dot per observation, a broken zero.

The shape benchmark charts use. Every choice in it is about a reader looking at a slide from
across a room rather than at a page from a foot away.

  heavy strokes and large dots  the line is the finding, so it gets the ink; a 1-point line with
                                a 3-point marker is a print convention and disappears projected
  a dot per observation         these axes are usually five measured points, not a sampled curve,
                                and the dots say where the measurements actually are — without
                                them a reader interpolates a claim that was never made
  a broken zero                 a range of 50-65% plotted from zero wastes three quarters of the
                                panel, and plotted from 50 lies about the size of the gap. The
                                break says "the axis is cut here" in the one place a reader looks
  vertical rules only           the x positions are the comparison; horizontal rules would compete
  a legend pill                 sits over the plot where the space is, without a box fighting the
                                marks for attention

`broken_zero` is the piece worth reading twice. It draws the axis from a floor near the data with a
zigzag between that floor and a zero tick, so the panel spends its height on the range that varies
while the reader is told, in the figure, that the scale is cut. A truncated axis with no such mark
is the most common honest-looking way to overstate a difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from ogviz.layout import hairline_grid, legend_pill
from ogviz.require import require
from ogviz.theme import GRID, INK, MUTED_INK

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

LINE_WIDTH = 3.4
MARKER_SIZE = 11.0
MARKER_EDGE = 1.6
BREAK_HEIGHT = 0.045  # share of the panel the zigzag occupies
BREAK_WIDTH_PX = 9.0
FLOOR_GAP = 0.10  # share of the data range left below the lowest point
TICK_MARGIN = 1.18  # multiplicative pad beyond the outermost tick, on a log axis


@dataclass(frozen=True)
class Line:
    """One series: its points, its colour, and the name that goes in the legend."""

    label: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    color: str
    marker: str = "o"
    muted: bool = False  # a comparison series, drawn back so the others read forward
    order: int = field(default=0, compare=False)


def money_ticks(ax: Axes, positions: Sequence[float], *, decimals: int = 2) -> None:
    """Label a log x-axis at the values that matter, in dollars, with no minor ticks.

    A log axis defaults to decade labels, which on a $0.20-$5.00 range means two labels and a
    forest of unlabelled minor ticks. Naming the positions the data actually sits at is what makes
    the axis readable.
    """
    ax.set_xscale("log")
    ax.set_xticks(list(positions))
    # Start the axis BEFORE its first tick. With the axis beginning exactly at the first tick, that
    # tick's label is centred on the corner and lands on the y axis's own bottom label — which on a
    # broken-zero panel is the "0". Padding in log space keeps the spacing even.
    low, high = min(positions), max(positions)
    ax.set_xlim(low / TICK_MARGIN, high * TICK_MARGIN)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.{decimals}f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0.0)


def broken_zero(ax: Axes, *, floor: float, zero_gap: float | None = None) -> None:
    """Cut the value axis below `floor`, INTERRUPTING the spine, with a zero tick beneath the cut.

    Say the range is 50-65. Plotting from zero spends three quarters of the panel on empty space;
    plotting from 50 with a plain axis silently triples every visible difference. This does the
    second and admits it.

    The break is a gap in the spine, not a mark drawn over it. Laying a zigzag on top of an
    unbroken line says nothing — the axis is still visibly continuous underneath, and the squiggle
    reads as decoration that happens to sit there. The spine is split into a long run covering the
    data and a stub carrying the zero tick, and the zigzag is drawn across the space between them,
    so the axis is genuinely discontinuous at the point where the scale is.
    """
    low, high = ax.get_ylim()
    span = high - low
    gap = zero_gap if zero_gap is not None else span * BREAK_HEIGHT * 2.0
    bottom = floor - gap
    ax.set_ylim(bottom, high)
    ticks = [tick for tick in ax.get_yticks() if floor <= tick <= high]
    ax.set_yticks([bottom, *ticks])
    ax.set_yticklabels(["0", *[_tick_text(tick) for tick in ticks]])

    # The spine now runs only over the range that carries data...
    ax.spines["left"].set_bounds(floor, high)
    # ...and a stub holds the zero tick, with the break between the two.
    stub_top = bottom + gap * 0.30
    ax.plot(
        [0.0, 0.0],
        [bottom, stub_top],
        transform=ax.get_yaxis_transform(),
        color=ax.spines["left"].get_edgecolor(),
        lw=ax.spines["left"].get_linewidth(),
        clip_on=False,
        zorder=3,
        solid_capstyle="butt",
    )
    _draw_break(ax, low=stub_top, high=floor)


def _draw_break(ax: Axes, *, low: float, high: float) -> None:
    """The zigzag that fills the gap in the spine, sized in pixels so it never distorts."""
    middle = (low + high) / 2
    reach = (high - low) * 0.5
    wobble = BREAK_WIDTH_PX / max(ax.get_window_extent().width, 1.0)
    ax.plot(
        [-wobble, wobble, -wobble, wobble],
        [middle - reach, middle - reach * 0.33, middle + reach * 0.33, middle + reach],
        transform=ax.get_yaxis_transform(),
        color=ax.spines["left"].get_edgecolor(),
        lw=ax.spines["left"].get_linewidth(),
        clip_on=False,
        zorder=4,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def _tick_text(value: float) -> str:
    return f"{value:g}"


def line_panel(
    ax: Axes,
    lines: Sequence[Line],
    *,
    x_ticks: Sequence[float] | None = None,
    money: bool = False,
    legend: bool = True,
    legend_loc: str = "lower right",
    legend_below: bool = False,
    line_width: float = LINE_WIDTH,
    marker_size: float = MARKER_SIZE,
) -> None:
    """Draw every series, with the rules running the way the comparison does.

    Muted series are drawn first and in a light colour, so a baseline stays visible without
    competing — the reader should see the leading lines first and find the baseline when looking
    for it.
    """
    require(
        lines,
        "line_panel needs at least one series",
    )
    for line in lines:
        require(
            line.x.shape == line.y.shape,
            f"{line.label}: {line.x.shape[0]} x values and {line.y.shape[0]} y values",
        )
    for line in sorted(lines, key=lambda item: (not item.muted, item.order)):
        ax.plot(
            line.x,
            line.y,
            marker=line.marker,
            color=line.color,
            lw=line_width,
            ms=marker_size,
            mec=line.color,
            mew=MARKER_EDGE,
            label=line.label,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2 if line.muted else 4,
        )
    if money:
        if x_ticks is None:  # written out rather than `require`d: this narrows for the call below
            raise AssertionError("money ticks need the positions to label")
        money_ticks(ax, x_ticks)
    elif x_ticks is not None:
        ax.set_xticks(list(x_ticks))

    # Vertical rules only: the x positions are what the series are being compared across.
    hairline_grid(ax, axis="x")
    ax.spines["bottom"].set(linewidth=1.6, color=INK)
    ax.spines["left"].set(linewidth=1.6, color=INK)
    ax.tick_params(colors=MUTED_INK)
    if legend:
        if legend_below:
            # One row under the panel: with four or more series a stacked pill eats the corner the
            # data wants, and a row below is read in the same sweep as the x axis.
            legend_pill(
                ax,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=len(lines),
                columnspacing=2.2,
            )
        else:
            legend_pill(ax, loc=legend_loc)


def value_floor(lines: Sequence[Line], *, gap: float = FLOOR_GAP) -> float:
    """A floor just below the lowest point, for `broken_zero` to cut the axis at."""
    every = np.concatenate([line.y for line in lines])
    low, high = float(every.min()), float(every.max())
    return low - gap * max(high - low, 1e-9)


def series_colors(count: int) -> tuple[str, ...]:
    """The benchmark-chart order: warm lead, amber, blue, then a near-grey for a baseline."""
    # The fourth was a violet that collapses onto the blue beside it under deuteranopia.
    wheel = ("#E8552D", "#F0A800", "#2E7CE0", "#9B3B8F", "#14A97C")
    return tuple(wheel[index % len(wheel)] for index in range(count))


# The colour for a `Line(muted=True)`. Named rather than left as `GRID` at the call site, because
# what a caller wants to say is "this series is the baseline", not "this series is the colour the
# gridlines happen to be" — the two are the same value today and are not the same decision.
MUTED_SERIES = GRID
