"""Placing a SET of labels that all want the same space, by solving them together.

`clear_position` searches for somewhere free for ONE label. Run it once per label and each call is
blind to the others: the first label takes the best spot, the second takes the next, and five series
ending within a hair of each other produce a pile that looks deliberate. That greedy failure is the
reason series-end labels were done with a legend instead.

The end-of-line case is one-dimensional, which is what makes an exact answer possible. Every label
wants the same x — just past its series' last point — so only the y's compete, and the problem is:
put the labels as close as possible to where they belong, without overlapping, keeping their order.
Written down, that is

    minimise  sum (placed_i - wanted_i)^2   subject to   placed_(i+1) - placed_i >= separation_i

which is isotonic regression after substituting out the separations, and pool-adjacent-violators
solves it exactly in one pass. No iteration count, no force constant, no convergence to check —
this is the arrangement with the least total movement, not a good-looking one.

Order is preserved on purpose. A solver free to reorder could shuffle a label past its neighbour and
save a few pixels, and the reader would then follow a leader line that crosses another. The label
above stays above.

What this deliberately does NOT do is general 2-D repulsion. `adjustText`, `ggrepel` and `textalloc`
do that, iteratively and approximately, for labels scattered across a panel. Where the labels share
a column, that machinery is a worse answer than the exact one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.require import require
from ogviz.tags import mark

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.text import Text
    from numpy.typing import NDArray

LEADER_COLOR = "#B4B0A4"
LEADER_WIDTH = 1.0
MOVED_TO_LEAD = 1.0  # a label moved by more than this many pixels gets a leader line


def _isotonic(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """The nearest non-decreasing sequence in least squares, by pool-adjacent-violators.

    Each block holds a run of points that have been merged; a block's fitted value is its mean, and
    two adjacent blocks are merged whenever the earlier one sits above the later one. The result is
    exact rather than iterative, which is the whole reason the placement is stated this way.
    """
    totals: list[float] = []
    counts: list[int] = []
    for value in values:
        totals.append(float(value))
        counts.append(1)
        while len(totals) > 1 and totals[-2] / counts[-2] > totals[-1] / counts[-1]:
            merged_total = totals.pop()
            merged_count = counts.pop()
            totals[-1] += merged_total
            counts[-1] += merged_count
    fitted = np.concatenate(
        [np.full(count, total / count) for total, count in zip(totals, counts, strict=True)]
    )
    return fitted


def stack_without_overlap(
    wanted: Sequence[float], sizes: Sequence[float], *, gap: float = 0.0
) -> NDArray[np.float64]:
    """Positions as near `wanted` as possible with no two labels overlapping, order kept.

    `sizes` is each label's full extent along the axis, in the same units as `wanted`, so two
    neighbours must end up at least half of each of their sizes apart, plus `gap`.

    Returns positions in the order given, not in sorted order — a caller holds labels, not ranks.
    """
    desired = np.asarray(wanted, dtype=float)
    extents = np.asarray(sizes, dtype=float)
    require(
        desired.shape == extents.shape,
        f"{desired.shape} positions against {extents.shape} sizes",
    )
    require(
        np.all(extents >= 0.0),
        "a label cannot have negative extent",
    )
    if desired.size <= 1:
        return desired.copy()

    order = np.argsort(desired, kind="stable")
    sorted_wanted = desired[order]
    sorted_sizes = extents[order]

    # The separation each neighbouring pair must keep, and the running total of it. Subtracting the
    # running total turns "must be at least S apart" into "must be non-decreasing", which is the
    # form pool-adjacent-violators solves.
    separations = (sorted_sizes[:-1] + sorted_sizes[1:]) / 2.0 + gap
    offsets = np.concatenate([[0.0], np.cumsum(separations)])
    placed = _isotonic(sorted_wanted - offsets) + offsets

    settled = np.empty_like(desired)
    settled[order] = placed
    return settled


def place_end_labels(
    ax: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    x: float | None = None,
    colors: Sequence[str] | None = None,
    pad_px: float = 4.0,
    fontsize: float | None = None,
    leaders: bool = True,
) -> list[Text]:
    """Name each series beside its last point, moving the ones that collide as little as possible.

    Placed OUTSIDE the axes deliberately: that is where a series-end label belongs, and it is the
    reason `clear_position` cannot be used here — its search rejects every candidate past the frame.
    Nothing clips text, so this draws; the figure needs room made for it, and `text_off_canvas`
    reports the case where none was.

    A label that had to move gets a leader line back to the value it names, because a label beside
    the wrong line is worse than no label.
    """
    require(
        len(labels) == len(values),
        f"{len(labels)} labels for {len(values)} values",
    )
    require(
        colors is None or len(colors) == len(labels),
        f"{len(colors or ())} colours for {len(labels)} labels",
    )
    if not labels:
        return []

    figure = ax.figure
    assert figure is not None, "the axes must belong to a figure"
    figure.canvas.draw()  # sizes are measured from the render, not guessed from the font size

    anchor_x = float(ax.get_xlim()[1]) if x is None else float(x)
    drawn: list[Text] = []
    for index, label in enumerate(labels):
        drawn.append(
            ax.text(
                anchor_x,
                float(values[index]),
                label,
                ha="left",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=None if colors is None else colors[index],
                clip_on=False,
            )
        )
    figure.canvas.draw()

    # Heights in DATA units, measured per label: a two-line label needs twice the room of a
    # one-line one, and a size in points would have to be converted through a transform that the
    # next `tight_layout` invalidates.
    to_data = ax.transData.inverted()
    heights = []
    for text in drawn:
        box = text.get_window_extent()
        low = to_data.transform((0.0, box.y0))[1]
        high = to_data.transform((0.0, box.y1 + pad_px))[1]
        heights.append(abs(float(high - low)))

    settled = stack_without_overlap([float(value) for value in values], heights)
    for text, value, position in zip(drawn, values, settled, strict=True):
        text.set_position((anchor_x, float(position)))
        # Deliberately placed against the series it names; the general "is this label on the data"
        # rule would drag it back into the panel.
        mark(text, "anchored")
        if not leaders:
            continue
        moved_px = abs(
            ax.transData.transform((0.0, float(position)))[1]
            - ax.transData.transform((0.0, float(value)))[1]
        )
        if moved_px > MOVED_TO_LEAD:
            line = ax.plot(
                [anchor_x, anchor_x],
                [float(value), float(position)],
                color=LEADER_COLOR,
                linewidth=LEADER_WIDTH,
                zorder=1,
                clip_on=False,
            )[0]
            mark(line, "backdrop")
    return drawn
