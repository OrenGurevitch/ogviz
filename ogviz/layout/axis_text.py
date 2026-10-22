"""The type of a bar or violin panel's axes: how big, and which title is always there.

A project stated this rule for its bar figures in three requests, and each one was a figure that
had shipped without it. A category axis with no title ("Model", "Method", "Group") leaves the
reader to guess what the names under the bars are. The ticks and the title were too small to read
at the size the figure is printed. And the title was set in regular weight while the panel title
beside it was bold, so the axis looked like an afterthought.

So the rule is one call. `style_axis_text` sets the tick sizes and a BOLD title on the category
axis, at one of two scales: `TEMPLATE_AXIS_TEXT` for a panel in a grid, `LARGE_AXIS_TEXT` for a
figure of one row, where the same type has more room. The sizes are the ones that project settled
on after measuring what its gate would accept, and a project that needs another passes its own
`AxisText`.

The title is the caller's words, since the package cannot know what the categories are. What it
owns is that there is one: `category_axis_untitled` in `ogviz.qc` reports a bar panel without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ogviz.orientation import is_vertical, read_orientation
from ogviz.theme import INK

if TYPE_CHECKING:
    from matplotlib.axes import Axes

TITLE_PAD = 6.0  # points between the tick labels and the category axis title


@dataclass(frozen=True)
class AxisText:
    """Point sizes for a panel's category tick labels, value tick labels and category axis title."""

    category_tick: float
    value_tick: float
    category_title: float


TEMPLATE_AXIS_TEXT = AxisText(category_tick=14.0, value_tick=13.0, category_title=14.0)
LARGE_AXIS_TEXT = AxisText(category_tick=17.0, value_tick=16.0, category_title=17.0)
# A grid's small cells.
SMALL_AXIS_TEXT = AxisText(category_tick=10.0, value_tick=9.5, category_title=10.5)


def style_axis_text(
    ax: Axes,
    category_title: str | None = None,
    *,
    text: AxisText = TEMPLATE_AXIS_TEXT,
    ticks: bool = True,
) -> None:
    """Set the tick sizes and a bold category axis title, whichever way the panel runs.

    The category axis is the one the names sit on: x for an upright panel, y for a horizontal one.
    `category_title` is left alone when None, so a panel that shares its title with the one below
    it (a grid labels its bottom row only) can still take the tick sizes. Call it after the ticks
    and labels are set, since the sizes apply to the labels that exist. `ticks=False` sets the title
    alone, for a panel whose tick labels were already sized to fit their slots.
    """
    upright = is_vertical(read_orientation(ax) or "vertical")
    category, value = ("x", "y") if upright else ("y", "x")
    if ticks:
        ax.tick_params(axis=category, labelsize=text.category_tick)
        ax.tick_params(axis=value, labelsize=text.value_tick)
    if category_title is not None:
        label = ax.set_xlabel if upright else ax.set_ylabel
        label(
            category_title,
            fontsize=text.category_title,
            fontweight="bold",
            color=INK,
            labelpad=TITLE_PAD,
        )
