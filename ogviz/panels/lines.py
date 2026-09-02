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
from ogviz.layout.ticks import format_value
from ogviz.require import require
from ogviz.theme import GRID, INK, LINE_SERIES, MUTED_INK

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

LINE_WIDTH = 3.4
MARKER_SIZE = 11.0
MARKER_EDGE = 1.6
# Half the gap the break opens, as a share of the value span BEFORE the axis is extended for it:
# `broken_zero` opens `2 * BREAK_HEIGHT` of that span, the stub takes 0.30 of the gap, and the
# zigzag fills the rest — 0.063 of the pre-break span. (The comment used to say "share of the
# panel the zigzag occupies", which is none of those numbers.)
BREAK_HEIGHT = 0.045
BREAK_WIDTH_PX = 9.0
FLOOR_GAP = 0.10  # share of the data range left below the lowest point
TICK_MARGIN = 1.18  # multiplicative pad beyond the outermost tick, on a log axis


@dataclass(frozen=True)
class Line:
    """One series: its points, its colour, and the name that goes in the legend.

    `x` and `y` are ARRAY-LIKE and coerced here, which is what the rest of this package's inputs
    already do — `bars.Series` types its values `ArrayLike`, and `effect_heatmap` takes nested
    lists. This one declared `NDArray` and meant it: `line_panel` read `line.x.shape`, so passing
    the plain lists everyone passes matplotlib raised `AttributeError: 'list' object has no
    attribute 'shape'` from inside a `require` call — an internal attribute error where the caller
    should have met a message, which is the failure `require` exists to prevent.

    Coerced in `__post_init__` rather than by the caller, because the frozen dataclass is the one
    place that can guarantee every reader downstream sees an array.
    """

    label: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    color: str
    marker: str = "o"
    muted: bool = False  # a comparison series, drawn back so the others read forward
    order: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        for name in ("x", "y"):
            # `object.__setattr__` because the dataclass is frozen, which is the documented way to
            # normalise a field on a frozen instance.
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))


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
    """A tick label that groups its thousands, as every other number this package prints does.

    Was `f"{value:g}"`: a broken axis over five-figure values printed `52000`, and the package's own
    `ungrouped_thousands` then refused the figure — five complaints on one panel, measured.
    `spectrogram` had already met and fixed the same defect with `format_value`.
    """
    return format_value(value)


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

    Muted series are drawn first and BEHIND the others (a lower z-order), so a baseline stays
    visible without competing — the reader should see the leading lines first and find the
    baseline when looking for it. The colour stays the caller's: pass `MUTED_SERIES` as a muted
    line's colour to have it recede as well. (This said "in a light colour" while setting no
    colour at all, so a saturated muted line was a competing line drawn behind the others.)
    """
    require(
        lines,
        "line_panel needs at least one series",
    )
    for line in lines:
        require(
            line.x.shape == line.y.shape,
            # The SHAPES, not their first axis: `shape[0]` is itself an IndexError on a 0-d array,
            # so the message meant to name the mismatch raised instead of reporting it.
            f"{line.label}: x has shape {line.x.shape} and y has shape {line.y.shape}",
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
    """The first `count` colours of the line palette, in benchmark-chart order.

    Warm lead, amber, blue, plum, teal, then a deep navy, an oxblood and a khaki —
    `theme.LINE_SERIES`, which is where the palette lives now and where the note on how those last
    three were chosen lives with it. This docstring used to describe "a near-grey for a baseline" as
    the fourth, and there is no near-grey in the palette at all: the baseline colour is
    `MUTED_SERIES` below, a different constant for a different job, so a reader picking a baseline
    from that sentence got a saturated plum.

    A count past the palette is REFUSED rather than wrapped. It used to index modulo the length, so
    `series_colors(6)` handed the first and sixth series the same colour — two lines a reader takes
    for one, in a figure that passes the whole gate, because `indistinguishable_series` skips pairs
    that are already close for normal vision and two identical colours are the limit of that.

    Eight is where it stops, and the ceiling is real rather than a shortage of effort: of 4320
    candidate colours, the 1019 that stay distinct from all five originals under three simulated
    deficiencies contain no new hue family at all. A ninth series wants explicit colours, or a
    second panel.
    """
    require(count > 0, f"series_colors needs a count of at least one, got {count}")
    require(
        count <= len(LINE_SERIES),
        f"the line palette has {len(LINE_SERIES)} distinct colours and {count} were asked for. "
        "Past that they would repeat, and two series in one colour read as one series. Pass "
        "explicit colours, or split the panel",
    )
    return LINE_SERIES[:count]


# The colour for a `Line(muted=True)`. Named rather than left as `GRID` at the call site, because
# what a caller wants to say is "this series is the baseline", not "this series is the colour the
# gridlines happen to be" — the two are the same value today and are not the same decision.
MUTED_SERIES = GRID
