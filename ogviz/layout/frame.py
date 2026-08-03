"""The furniture around the marks: rules, baselines and the legend pill.

None of it carries a value. All of it decides how hard the marks have to work to be seen, which is
why it is one module rather than a line in each builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ogviz.theme import GRID, MUTED_INK, PANEL_FILL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.text import Text


def hairline_grid(ax: Axes, *, axis: Literal["x", "y"] = "y") -> None:
    """Hairline rules on one axis only, under the data."""
    ax.grid(visible=False)
    ax.grid(visible=True, axis=axis, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def baseline(ax: Axes, *, axis: Literal["x", "y"] = "x") -> None:
    """A quiet rule on the category axis, and a hairline on the value axis.

    Not a heavy black bar. The category axis is a boundary, not data: at 2 pt of ink it competes
    with the marks for attention and, where a tick label sits on it, reads as a broken line rather
    than an axis.
    """
    near, far = ("bottom", "left") if axis == "x" else ("left", "bottom")
    ax.spines[near].set(linewidth=1.2, color=MUTED_INK)
    ax.spines[far].set(linewidth=1.0, color=GRID)


TITLE_CLEARANCE = 1.35  # an axes title needs its own height plus the pad under it


def zero_baseline(ax: Axes) -> None:
    """Heavy ink line at y=0, for bars that grow from zero rather than from the frame."""
    ax.axhline(0.0, color=MUTED_INK, linewidth=1.4, zorder=4)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set(linewidth=1.0, color=GRID)
    ax.tick_params(axis="x", length=0.0)


def pill_frame(legend: Legend) -> Legend:
    """Give an EXISTING legend the soft filled pill. Returned so it can be used inline.

    Separate from `legend_pill` because a caller that has already built its legend — with its own
    handles, order and anchor — should not have to hand those to a helper just to restyle it.
    """
    legend.get_frame().set(
        facecolor=PANEL_FILL, edgecolor="none", boxstyle="round,pad=0.5,rounding_size=0.5"
    )
    return legend


def legend_pill(target: Axes | Figure, **kwargs: object) -> Legend:
    """Create a legend on an Axes or Figure and give it the pill."""
    return pill_frame(target.legend(frameon=True, **kwargs))  # type: ignore[arg-type]


CATEGORY_ROW_STEP = 0.075  # of the axes height, between one row of category labels and the next


def label_rows(
    ax: Axes,
    positions: Sequence[float],
    rows: Sequence[Sequence[str]],
    *,
    sizes: Sequence[float] | None = None,
    colors: Sequence[Sequence[str] | str] | None = None,
    weights: Sequence[str] | None = None,
    step: float = CATEGORY_ROW_STEP,
    first_row: float = -0.02,
) -> list[Text]:
    """Stack several rows of labels under the category axis, each row a level of grouping.

    One row of tick labels can only say what each bar IS. A figure often has more to say about the
    same axis — what each bar is, then which model it belongs to, then how much data that model
    saw — and each of those is a level a reader takes in separately.

    A label may be given per position or once for the whole row, so a row naming a group spanning
    several bars is `["trained on 50 subjects"]` at the midpoint rather than the same string
    repeated. Positions and labels must agree in length otherwise.

    Placed in the axes' x-coordinate / figure-fraction transform, so the rows stay put when the
    value axis rescales. A caller writing this by hand reaches for `ax.get_xaxis_transform()` and
    picks the offsets one at a time, which is how three rows come out unevenly spaced.
    """
    drawn: list[Text] = []
    for index, labels in enumerate(rows):
        assert len(labels) in (1, len(positions)), (
            f"row {index} has {len(labels)} labels for {len(positions)} positions"
        )
        places = [sum(positions) / len(positions)] if len(labels) == 1 else list(positions)
        row_color = None if colors is None else colors[index]
        for place, label in zip(places, labels, strict=True):
            drawn.append(
                ax.text(
                    place,
                    first_row - index * step,
                    label,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=None if sizes is None else sizes[index],
                    fontweight="normal" if weights is None else weights[index],
                    color=_row_color(row_color, places, place),
                )
            )
    return drawn


def _row_color(row_color, places: list[float], place: float) -> str | None:
    """One colour for the whole row, or one per position."""
    if row_color is None or isinstance(row_color, str):
        return row_color
    return row_color[places.index(place)]
