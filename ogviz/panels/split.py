"""Two half-violins back to back, for a paired comparison at each category.

The case this exists for: the same quantity measured two ways on the same subjects — two sensors,
two pipelines, two raters. Side-by-side violins put a gap between things that should be read as
one number measured twice, and a difference of a fraction of a percent disappears into that gap.
Sharing a spine puts the two distributions edge to edge, where a real difference in shape is
visible and an identical pair looks identical.

Each half keeps its own IQR bar, mean line, median dot and printed mean, so the halves are
comparable and neither borrows the other's summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout import hairline_grid
from ogviz.layout.ticks import groups_its_ticks
from ogviz.marks import (
    BOX_COLOR,
    BOX_WIDTH,
    MEAN_HALF_WIDTH,
    MEAN_LINEWIDTH,
    MEDIAN_EDGE_WIDTH,
    MEDIAN_SIZE,
    VIOLIN_ALPHA,
    Z_IQR,
    Z_MEAN_LINE,
    Z_MEDIAN_DOT,
    Z_VIOLIN,
)
from ogviz.orientation import violin_orientation_kwarg
from ogviz.panels.grid import align_mean_rows
from ogviz.panels.violins import MEAN_ROW_FLOOR_PT, _shrink_until_clear, constant_span
from ogviz.require import require
from ogviz.tags import mark
from ogviz.theme import KNOCKOUT_PAD, VALUE_LABEL_SIZE, page_color

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.text import Text
    from numpy.typing import NDArray

Side = int  # -1 for the left half, +1 for the right
SPINE_GAP = 0.004  # keeps the two halves from sharing a pixel column on the centre line
MARK_OFFSET = 0.055  # how far each half's marks sit from the shared spine
PAIR_WIDTH = 0.86  # the width of a PAIR, so each half is half of it and neighbours stay clear


def half_violin(
    ax: Axes,
    values: NDArray[np.float64],
    position: float,
    color: str,
    side: Side,
    *,
    width: float = PAIR_WIDTH,
    alpha: float = VIOLIN_ALPHA,
) -> None:
    """One side of a violin, clipped at the centre line.

    `width` is the width of the whole PAIR, so each half gets half of it. Passing the single-violin
    width here is what made the first version overlap its neighbours: a half as wide as a whole
    violin puts 1.24 units of body into a 1.0-unit slot.

    matplotlib has no half violin, so this draws the whole body and clamps the vertices on one
    side back to the centre. Clipping the vertices rather than the artist keeps the outline exact:
    a rectangular clip path would cut the body off at the axes' pixel grid instead of at the
    position.
    """
    require(
        side in (-1, 1),
        f"side must be -1 (left) or +1 (right), got {side}",
    )
    parts = ax.violinplot(
        [values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        **violin_orientation_kwarg("vertical"),  # type: ignore[arg-type]
    )
    edge = position + side * SPINE_GAP
    for body in parts["bodies"]:  # type: ignore[union-attr]
        vertices = body.get_paths()[0].vertices
        if side < 0:
            vertices[:, 0] = np.minimum(vertices[:, 0], edge)
        else:
            vertices[:, 0] = np.maximum(vertices[:, 0], edge)
        body.set_facecolor(color)
        body.set_alpha(alpha)
        body.set_edgecolor("none")
        body.set_zorder(Z_VIOLIN)


def half_marks(
    ax: Axes,
    values: NDArray[np.float64],
    position: float,
    side: Side,
    color: str,
    *,
    offset: float = MARK_OFFSET,
    mean_half_width: float = MEAN_HALF_WIDTH,
) -> None:
    """The IQR bar, mean line and median dot for one half, set off from the shared spine."""
    q1, median, q3 = (float(x) for x in np.percentile(values, [25, 50, 75]))
    at = position + side * offset
    ax.plot([at, at], [q1, q3], color=BOX_COLOR, lw=BOX_WIDTH, zorder=Z_IQR, solid_capstyle="round")
    ax.plot(
        [position, position + side * mean_half_width * 2],
        [float(np.mean(values))] * 2,
        color=color,
        lw=MEAN_LINEWIDTH,
        solid_capstyle="round",
        zorder=Z_MEAN_LINE,
    )
    ax.plot(
        [at],
        [median],
        "o",
        mfc=page_color(),
        mec=BOX_COLOR,
        mew=MEDIAN_EDGE_WIDTH,
        ms=MEDIAN_SIZE,
        zorder=Z_MEDIAN_DOT,
    )


LEGEND_HEADROOM = 0.24  # of the value span, held above the marks so the legend sits clear of them


@groups_its_ticks
def split_violins(
    ax: Axes,
    categories: Sequence[str],
    left: Sequence[NDArray[np.float64]],
    right: Sequence[NDArray[np.float64]],
    *,
    left_color: str,
    right_color: str,
    width: float = PAIR_WIDTH,
    show_means: bool = True,
    mean_decimals: int | None = None,
    display_scale: float = 1.0,
    bottom_pad: float = 0.28,
    grid: bool = True,
    headroom: float = 0.12,
    labels: tuple[str, str] | None = None,
    mean_weight: str = "bold",
    mean_color: str | None = None,
) -> None:
    """A back-to-back pair of half-violins per category, each half with its own marks.

    `left` and `right` hold one array per category, in the same order as `categories`; they do not
    have to be the same length as each other, since a paired measurement can drop samples on one
    side.

    `labels` names the two halves, left then right, as a legend in the panel's top-left corner, in
    headroom it holds open above the marks. A
    split violin says nothing about which half is which without it: the colour is the only cue, and
    the printed means repeat it rather than explain it. Left as None, no legend is drawn, since a
    caller may be naming the halves in a figure-level legend or a caption already.

    `mean_weight` and `mean_color` set the printed means: bold in each half's own colour by
    default, or any weight and one ink colour for both when the pair should read as one row.
    """
    require(
        len(left) == len(right) == len(categories),
        f"{len(categories)} categories, {len(left)} left and {len(right)} right series",
    )
    require(
        len(categories),
        "split_violins needs at least one category",
    )
    for name, side in (("left", left), ("right", right)):
        for index, values in enumerate(side):
            array = np.asarray(values, dtype=float)
            # EMPTY IS REFUSED BY NAME, and it has to be said out loud because the docstring above
            # invites it: the two sides need not be the same length, since a paired measurement can
            # drop samples on one side. Dropping ALL of them is a different thing, and it used to
            # surface as `IndexError: list index out of range` from inside the violin — an error
            # naming nothing the caller wrote.
            require(
                array.size,
                f"the {name} series for {categories[index]!r} is empty; a half-violin needs "
                "observations to have a shape. Drop the category, or say what an empty side means.",
            )
            missing = int(np.count_nonzero(~np.isfinite(array)))
            require(
                not missing,
                f"{name} series for {categories[index]!r} has {missing} non-finite value(s); drop "
                "or impute them in the project, where the choice is visible.",
            )

    if grid:
        hairline_grid(ax, axis="y")
    positions = np.arange(len(categories), dtype=float)
    every = np.concatenate([np.asarray(v, dtype=float) for v in (*left, *right)])
    low, high = float(every.min()), float(every.max())
    span = high - low or constant_span(high)
    room = max(headroom, LEGEND_HEADROOM) if labels is not None else headroom
    ax.set_ylim(low - bottom_pad * span, high + room * span)
    ax.set_xlim(-0.5 - width / 2, positions[-1] + 0.5 + width / 2)

    for at, values in zip(positions, left, strict=True):
        half_violin(ax, np.asarray(values, dtype=float), at, left_color, -1, width=width)
        half_marks(ax, np.asarray(values, dtype=float), at, -1, left_color)
    for at, values in zip(positions, right, strict=True):
        half_violin(ax, np.asarray(values, dtype=float), at, right_color, +1, width=width)
        half_marks(ax, np.asarray(values, dtype=float), at, +1, right_color)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(categories))
    mark(ax, "category_axis")
    if show_means:
        _printed_pairs(
            ax,
            positions,
            left,
            right,
            left_color=left_color,
            right_color=right_color,
            y=low - bottom_pad * span * 0.5,  # settled exactly by `align_mean_rows` below
            decimals=mean_decimals,
            scale=display_scale,
            weight=mean_weight,
            color=mean_color,
        )
        # Same rule as every other panel: midway between the frame and the lowest drawn mark. A
        # fraction chosen here would be a third answer to one question.
        align_mean_rows([ax], floor=ax.get_ylim()[0])
    if labels is not None:
        _name_the_halves(ax, labels, (left_color, right_color))


def _name_the_halves(ax: Axes, labels: tuple[str, str], colors: tuple[str, str]) -> None:
    """A two-entry legend, each swatch in its half's colour at the strength the body is drawn."""
    from matplotlib.patches import Patch

    from ogviz.layout import legend_pill

    handles = [
        Patch(facecolor=color, edgecolor=color, alpha=VIOLIN_ALPHA, label=label)
        for label, color in zip(labels, colors, strict=True)
    ]
    legend_pill(ax, handles=handles, loc="upper left", ncol=2)


def _printed_pairs(
    ax: Axes,
    positions: NDArray[np.float64],
    left: Sequence[NDArray[np.float64]],
    right: Sequence[NDArray[np.float64]],
    *,
    left_color: str,
    right_color: str,
    y: float,
    decimals: int | None,
    scale: float,
    weight: str = "bold",
    color: str | None = None,
) -> None:
    """Both means under each category, each in its own half's colour so neither is ambiguous.

    THE ROW SHRINKS UNTIL IT STOPS COLLIDING WITH ITSELF, by the same measured loop
    `printed_means` uses and down to the same floor. It was set at one size, so a metric of order
    0.003 — five characters a number, two numbers a category — ran its neighbours together by tens
    of pixels on a four-category panel and the gate refused a figure that a smaller row would have
    fixed. A row with room is left at the size it was set at.
    """
    from ogviz.layout.ticks import format_value, row_decimals

    means = [float(np.mean(v)) * scale for v in (*left, *right)]
    if decimals is None:
        decimals = row_decimals(means)
    size = VALUE_LABEL_SIZE * 0.85
    drawn: list[Text] = []
    for at, low_values, high_values in zip(positions, left, right, strict=True):
        for side, values, half_color in (
            (-1, low_values, left_color),
            (1, high_values, right_color),
        ):
            printed = ax.text(
                at + side * 0.22,
                y,
                format_value(float(np.mean(values)), scale=scale, decimals=decimals),
                ha="center",
                va="center",
                fontsize=size,
                fontweight=weight,
                color=color or half_color,
                zorder=9,
                bbox={
                    "facecolor": page_color(),
                    "edgecolor": "none",
                    "pad": KNOCKOUT_PAD,
                    "boxstyle": "square",
                },
            )
            mark(printed, "mean_row")
            drawn.append(printed)
    _shrink_until_clear(ax, drawn, size, min(MEAN_ROW_FLOOR_PT, size))
