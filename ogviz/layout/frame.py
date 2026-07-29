"""The furniture around the marks: rules, baselines and the legend pill.

None of it carries a value. All of it decides how hard the marks have to work to be seen, which is
why it is one module rather than a line in each builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ogviz.theme import GRID, MUTED_INK, PANEL_FILL

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend


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
