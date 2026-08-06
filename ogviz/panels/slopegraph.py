"""Several series across a few ordered stages, drawn so the SHAPE of each is what a reader compares.

A slopegraph is a line panel with a categorical x axis, and the difference is not cosmetic. On a
continuous axis a reader reads levels off the y axis; here they read the slope between two stages,
so every choice serves the comparison between series rather than the reading of any one value:

  the stages are evenly spaced, whatever they represent, because uneven spacing makes a slope mean
  something the data did not say;
  every series gets a marked point at every stage, so a crossing is unambiguous about which series
  went where;
  the spread, when given, is a band rather than a whisker per point — whiskers at N stages by M
  series is a thicket, and the band reads as one object per series.

Where the labels go is the open problem. Series ending within a hair of each other need a set-wise
placement algorithm — the thing `directlabels` and `ggrepel` do, which `annotate_clear` does not —
so this panel takes a legend by default and offers end labels for the case where they are far
enough apart to place naively. `crowded_ends` says which case a given figure is in, rather than
leaving it to be discovered in the rendered PNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout import hairline_grid, legend_pill
from ogviz.layout.stacking import place_end_labels
from ogviz.orientation import stamp_orientation
from ogviz.require import require
from ogviz.theme import INK, MUTED_INK

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

MARKER_SIZE = 7.0
LINE_WIDTH = 2.4
BAND_ALPHA = 0.16
END_LABEL_PAD = 0.06  # of the stage span, between the last point and its label
CROWDED_PX = 12.0  # end labels closer than this cannot be placed without solving them together


@dataclass(frozen=True)
class Strand:
    """One series across the stages, and optionally the spread around it.

    `spread` is (low, high) per stage — absolute bounds, the same convention as `error_bars`, since
    that is what a bootstrap or a quantile returns.
    """

    label: str
    values: Sequence[float]
    color: str
    spread: Sequence[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        require(
            len(self.values),
            f"{self.label}: a strand with no values",
        )
        require(
            self.spread is None or len(self.spread) == len(self.values),
            f"{self.label}: {len(self.spread or ())} spreads for {len(self.values)} values",
        )


def crowded_ends(strands: Sequence[Strand], ax: Axes, *, gap_px: float = CROWDED_PX) -> list[str]:
    """Pairs of strands whose last points are too close for their labels to be placed separately.

    Answered before drawing rather than discovered afterwards. Placing end labels one at a time is
    only correct when they do not compete, and a caller that knows its series converge can reach for
    a legend instead of finding out from the rendered figure.
    """
    ends = sorted(
        (float(ax.transData.transform((0.0, float(strand.values[-1])))[1]), strand.label)
        for strand in strands
    )
    return [
        f"{first[1]!r} and {second[1]!r} end {second[0] - first[0]:.0f} px apart"
        for first, second in pairwise(ends)
        if second[0] - first[0] < gap_px
    ]


def slopegraph(
    ax: Axes,
    strands: Sequence[Strand],
    stages: Sequence[str],
    *,
    end_labels: bool = False,
    legend: bool = True,
    grid: bool = True,
    marker_size: float = MARKER_SIZE,
    line_width: float = LINE_WIDTH,
    band_alpha: float = BAND_ALPHA,
    label_size: float | None = None,
) -> list[str]:
    """Draw every strand across the stages. Returns any end-label crowding it had to work around.

    `end_labels` names each strand at its last point instead of in a legend, which reads better when
    the series separate and is unusable when they do not. The return value is `crowded_ends`, so a
    caller can assert on it or switch to the legend; asking for end labels does not silently produce
    a pile of them.
    """
    require(
        strands,
        "a slopegraph with no strands in it",
    )
    require(
        len(stages) >= 2,
        "a slopegraph needs at least two stages",
    )
    for strand in strands:
        require(
            len(strand.values) == len(stages),
            f"{strand.label}: {len(strand.values)} values across {len(stages)} stages",
        )

    stamp_orientation(ax, "vertical")
    positions = np.arange(len(stages), dtype=float)
    if grid:
        hairline_grid(ax, axis="y")

    for strand in strands:
        values = np.asarray(strand.values, dtype=float)
        if strand.spread is not None:
            bounds = np.asarray(strand.spread, dtype=float)
            require(
                bounds.shape == (len(values), 2),
                f"{strand.label}: spread must be (low, high) per stage, got {bounds.shape}",
            )
            ax.fill_between(
                positions,
                bounds[:, 0],
                bounds[:, 1],
                color=strand.color,
                alpha=band_alpha,
                linewidth=0.0,
                zorder=2,
            )
        ax.plot(
            positions,
            values,
            color=strand.color,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            markerfacecolor=strand.color,
            markeredgecolor=INK,
            markeredgewidth=0.8,
            label=strand.label,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(list(stages), fontsize=label_size)
    # Evenly spaced and a margin at each end, so the first and last markers are not on the frame.
    ax.set_xlim(-0.35, len(stages) - 0.65)
    ax.tick_params(axis="y", colors=MUTED_INK)

    ax.figure.canvas.draw()  # the crowding question is asked in pixels
    crowding = crowded_ends(strands, ax)
    if end_labels:
        # Solved together, not one at a time: the labels compete for one column, and placing them
        # greedily is what made a legend the better option before `stack_without_overlap` existed.
        place_end_labels(
            ax,
            [strand.label for strand in strands],
            [float(strand.values[-1]) for strand in strands],
            x=len(stages) - 1 + END_LABEL_PAD,
            colors=[strand.color for strand in strands],
            fontsize=label_size,
        )
    elif legend:
        legend_pill(ax, loc="best")
    return crowding


def null_distance(scores: Sequence[float], nulls: Sequence[float]) -> NDArray[np.float64]:
    """Each score as its distance ABOVE its own null, so metrics with different units share an axis.

    An OOS correlation is at chance around 0 and an AUC at chance around 0.5; plotted raw on one
    axis they cannot be compared, and plotted as "score" with a single zero rule the AUC looks
    enormous. Subtracting each metric's OWN null puts every one on an axis where zero is chance.

    The subtraction is trivial and the pairing is what goes wrong — one null for a whole family of
    metrics, or a null vector in a different order from the scores. Both are shape errors here.
    """
    measured = np.asarray(scores, dtype=float)
    chance = np.asarray(nulls, dtype=float)
    require(
        measured.shape == chance.shape,
        f"{measured.shape} scores against {chance.shape} nulls — every metric needs its own null, "
        "and one null broadcast across a family of metrics is the mistake this refuses",
    )
    return measured - chance
