"""A row of panels, with an optional caption in its own reserved row.

Named `panel_row`, not `panels`: `ogviz.panels` is the subpackage holding `group_violins` and
`bar_panel`, so a function of that name at the top level was shadowed by the module and
`ogviz.panels(2, caption=...)` raised `TypeError: 'module' object is not callable` — while being
listed in `__all__` and documented as callable.

The caption lives in its own gridspec row — an invisible axes below the panel row — rather than
at a chosen `fig.text` y-coordinate, which overlaps eventually, gets nudged, and overlaps again
at a different figure size.

KNOWN LIMIT, measured not assumed: the reserved row is sized from the caption's own line count,
so an axes whose decorations grow DOWNWARD past their allotment can still reach it. A two-line
x-label does, and `test_a_two_line_x_label_still_reaches_the_caption_row` holds that case as an
xfail. `constrained` layout reserves for decorations correctly but then ignores the row height
ratios and pushes the caption back up into the tick labels, so it is not the fix either. Until
this is solved, keep x-labels to one line under a caption, and let `save`'s overlap check catch
it if you do not.

Captions default off. A manuscript figure carries its title, axes and legend, and what the marks
mean belongs in the project's README. A reproducibility document is the case that wants the
caption on the image, where it travels with the file.

Ported from a sibling project, which introduced the reserved-row idea.
Two things changed on the way. Its wrap used a calibrated ~143 char-points-per-inch constant;
`TextPath` measures the string exactly and needs no renderer, so the constant is gone. And its
row positions were fixed fractions, which the overlap check falsified: a two-line x-label grows
downward into the caption row, so the caption could still be collided with. The layout engine
reserves for axis decorations, so it places the rows now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from ogviz.theme import MUTED_INK

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

CAPTION_SIZE = 8.0
CAPTION_CLEARANCE_PX = 6.0  # between the lowest panel decoration and the caption's first line
CAPTION_LINE_SPACING = 1.45
LEFT_MARGIN = 0.055
TOP_MARGIN = 0.88  # above this is the suptitle band
LEGEND_RIGHT = 0.80  # panels end here when a right-hand legend is reserved
FULL_RIGHT = 0.985


def text_width_points(text: str, fontsize: float) -> float:
    """Rendered width of `text` in points, measured from the glyph outlines."""
    if not text:
        return 0.0
    return float(TextPath((0, 0), text, prop=FontProperties(size=fontsize)).get_extents().width)


def wrap_to_width(text: str, width_points: float, fontsize: float) -> list[str]:
    """Greedy word wrap against measured glyph width, so no line exceeds `width_points`."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and text_width_points(candidate, fontsize) > width_points:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def panel_row(
    count: int,
    *,
    caption: str | None = None,
    legend: bool = False,
    share_y: bool = True,
    width: float = 12.8,
    panel_height: float = 5.0,
    caption_size: float = CAPTION_SIZE,
) -> tuple[Figure, list[Axes]]:
    """`count` side-by-side panels, plus a reserved caption row when `caption` is given.

    `legend=True` keeps the right margin free for a figure-level legend the caller adds at
    `bbox_to_anchor=(LEGEND_RIGHT, 0.5)`.
    """
    assert count >= 1, "panels needs at least one panel"
    right = LEGEND_RIGHT if legend else FULL_RIGHT

    lines: list[str] = []
    caption_height = 0.0
    if caption:
        panel_width_points = width * (right - LEFT_MARGIN) * 72.0
        lines = wrap_to_width(caption, panel_width_points, caption_size)
        caption_height = (caption_size * CAPTION_LINE_SPACING / 72.0) * len(lines) + 0.32

    figure = plt.figure(figsize=(width, panel_height + caption_height))
    rows = 2 if caption else 1
    ratios = [panel_height, caption_height] if caption else [panel_height]
    grid = figure.add_gridspec(
        rows,
        count,
        height_ratios=ratios,
        left=LEFT_MARGIN,
        right=right,
        top=TOP_MARGIN,
        bottom=0.10,
        hspace=0.42,
        wspace=0.06,
    )

    axes: list[Axes] = []
    for index in range(count):
        shared = axes[0] if (share_y and index) else None
        axes.append(figure.add_subplot(grid[0, index], sharey=shared))

    if caption:
        caption_axes = figure.add_subplot(grid[1, :])
        caption_axes.axis("off")
        # axis("off") alone leaves tick LABELS, which reappear on the next draw.
        caption_axes.set_xticks([])
        caption_axes.set_yticks([])
        drawn = caption_axes.text(
            0.5,
            1.0,
            "\n".join(lines),
            ha="center",
            va="top",
            transform=caption_axes.transAxes,
            fontsize=caption_size,
            color=MUTED_INK,
            linespacing=CAPTION_LINE_SPACING,
        )
        drawn.ogviz_caption_row = True  # type: ignore[attr-defined]  # `settle_caption` finds it
    return figure, axes


def settle_caption(fig: Figure, *, gap_px: float = CAPTION_CLEARANCE_PX) -> bool:
    """Push the caption below whatever the panels actually grew downward. Returns whether it moved.

    The reservation cannot be right when `panel_row` makes it: the row is sized from the caption's
    own line count, and the caller has not plotted yet — the x-label that reaches into it does not
    exist. A two-line x-label was enough, and the case was held open as an xfail for exactly that
    reason.

    So it is settled afterwards, from the rendered panels: take the lowest ink of every panel
    including its decorations, and drop the caption below it. Measured, so a three-line label or a
    rotated tick row is handled by the same code that handles none.
    """
    fig.canvas.draw()
    captions = [
        text for ax in fig.axes for text in ax.texts if getattr(text, "ogviz_caption_row", False)
    ]
    if not captions:
        return False
    panels = [ax for ax in fig.axes if ax.axison]
    if not panels:
        return False
    # `get_tightbbox` returns None for an axes with nothing drawn in it; such a panel has no
    # decorations to reach the caption and simply does not constrain it.
    boxes = [ax.get_tightbbox() for ax in panels]
    reaching = [float(box.y0) for box in boxes if box is not None]
    if not reaching:
        return False
    lowest = min(reaching)
    moved = False
    for caption_text in captions:
        top = float(caption_text.get_window_extent().y1)
        overlap = top - (lowest - gap_px)
        if overlap <= 0:
            continue
        axes = caption_text.axes
        assert axes is not None
        shift = overlap / max(axes.get_window_extent().height, 1.0)
        x, y = caption_text.get_position()
        caption_text.set_position((x, float(y) - shift))
        moved = True
    if moved:
        fig.canvas.draw()
    return moved
