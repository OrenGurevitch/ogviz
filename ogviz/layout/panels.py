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

from ogviz import units
from ogviz.require import require
from ogviz.tags import mark, marked, value_of
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
    """Rendered width of `text` in points — the WIDEST line, measured from the glyph outlines.

    Split here rather than handed to `TextPath` whole, because what `TextPath` does with a newline
    has changed between matplotlib versions: 3.11 breaks the line, older versions measure the string
    as one run and return roughly twice the true width. `table_panel` sizes each column with this,
    so on the older behaviour a two-line header reserved about double the width it needed and
    visibly ballooned its column — worked around downstream by forbidding multi-line headers, a
    caller-side rule that existed only because of this.

    Measuring per line makes the answer the same on every version, which a layout number has to be.
    """
    if not text:
        return 0.0
    return max(_line_width_points(line, fontsize) for line in text.split("\n"))


def _line_width_points(line: str, fontsize: float) -> float:
    if not line:
        return 0.0
    return float(TextPath((0, 0), line, prop=FontProperties(size=fontsize)).get_extents().width)


def wrap_to_width(text: str, width_points: float, fontsize: float) -> list[str]:
    """Greedy word wrap against measured glyph width, so no line exceeds `width_points`.

    A newline the caller wrote is a HARD break and survives: each side is wrapped on its own.
    `text.split()` alone collapses them, so a two-line heading came back as one long line and the
    author's own layout was silently discarded.
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if current and text_width_points(candidate, fontsize) > width_points:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines or [""]


DEFAULT_COLUMN_WIDTH = 6.5  # inches of text column a figure is normally placed at
DEFAULT_PAGE_HEIGHT = 9.0  # inches of usable page height beside a caption
CELL_ASPECT = 1.32  # width : height of one cell, the shape a square-ish panel reads as


def rows_that_fit(
    ncol: int,
    *,
    width: float | None = None,
    cell_aspect: float = CELL_ASPECT,
    column_width: float = DEFAULT_COLUMN_WIDTH,
    page_height: float = DEFAULT_PAGE_HEIGHT,
) -> int:
    """How many rows still fit the page once the figure is placed at `column_width`.

    A figure wider than the column it is dropped into is scaled down to fit, and its height goes
    down with it — so what limits the row count is the page height AFTER that scaling. Written out,
    with `w` the figure's own width:

        cell_height   = w / ncol / aspect
        figure_height = cell_height * rows
        placed_height = figure_height * column_width / w      <- the scaling to the column
                      = rows * column_width / (ncol * aspect)  <- and `w` is gone

    **The figure's own width cancels, and that is the answer rather than a bug in it.** Any width
    reaches the page at the same size, so the only things that decide the row count are the shape of
    a cell, how many sit side by side, and how much page there is. `width` is therefore ignored, and
    is kept only so the existing calls keep working — measured on 2026-08-06, `width=4.0`, `12.8`
    and `40.0` all returned 3, which is right and reads as broken. It used to be a REQUIRED argument
    that could not affect the result, which is the kind of thing a caller tunes for an afternoon.

    The number matters because of what happens when it is exceeded. A grid that runs onto a second
    page gets fixed by shrinking the cell, and the cell is shared: one grid of seven cells needing a
    fourth row can pin the cell height for a whole document, leaving every smaller grid a
    letterbox. The constraint then lives in a different figure, which is what makes it invisible.

    So the fix for a grid that will not fit is to take a cell OUT, never to shrink the cell.
    """
    del width  # documented above: it cancels out of the arithmetic entirely
    require(ncol >= 1, f"a grid needs at least one column, got {ncol}")
    require(cell_aspect > 0, f"a cell needs a positive aspect, got {cell_aspect}")
    require(
        column_width > 0 and page_height > 0,
        f"the page must have a size, got {column_width} in wide and {page_height} in tall",
    )
    return int(page_height * ncol * cell_aspect / column_width)


def panel_grid(
    count: int,
    *,
    ncol: int = 2,
    width: float = 12.8,
    cell_aspect: float = CELL_ASPECT,
    column_width: float = DEFAULT_COLUMN_WIDTH,
    page_height: float = DEFAULT_PAGE_HEIGHT,
    hspace: float = 0.42,
    wspace: float = 0.22,
) -> tuple[Figure, list[Axes]]:
    """`count` panels on a grid `ncol` wide, sized so a SET of grids places consistently.

    `width` is the TOTAL figure width, as in `panel_row`, and that is the whole point. Sizing a grid
    as `cell_width * ncol` instead makes a one-cell grid half the width of a four-cell one, and a
    document placing both at one column magnifies the small one, so its type and its marks come
    out oversized for no reason visible in the figure itself.

    `cell_aspect` rather than a cell height, because aspect is what a reader perceives; the height
    falls out of `width / ncol`.

    Returns the figure and `count` axes in reading order. Extra slots are left empty and REPORTED —
    see `grid_warnings`, which is also a QC check, since both of these are invisible until the
    figure is drawn.
    """
    require(
        count >= 1,
        f"a grid needs at least one panel, got {count}",
    )
    require(
        ncol >= 1,
        f"a grid needs at least one column, got {ncol}",
    )
    columns = min(ncol, count)
    rows = -(-count // columns)  # ceiling
    cell_height = width / columns / cell_aspect

    figure = plt.figure(figsize=(width, cell_height * rows))
    grid = figure.add_gridspec(rows, columns, hspace=hspace, wspace=wspace)
    axes = [figure.add_subplot(grid[index // columns, index % columns]) for index in range(count)]
    mark(figure, "grid_page", (columns, rows, width, cell_aspect, column_width, page_height))
    return figure, axes


def grid_warnings(fig: Figure) -> list[str]:
    """What a grid's shape will cost, for a figure `panel_grid` built.

    Two things, both invisible until the figure is drawn and both cheap to say:

    an EMPTY SLOT — seven panels two columns wide is four rows with a hole in it, which is the
    signal to reconsider the column count or the panel set rather than to ship the hole;

    a grid TALLER THAN THE PAGE at its column width, naming the row count that would fit. This is
    the one that spreads: the instinct is to shrink the cell, and the cell is shared with every
    other grid in the set.
    """
    page = value_of(fig, "grid_page")
    if page is None:
        return []
    columns, rows, width, cell_aspect, column_width, page_height = page
    drawn = len([ax for ax in fig.axes if ax.get_subplotspec() is not None])
    complaints: list[str] = []
    empty = columns * rows - drawn
    if empty:
        complaints.append(
            f"{drawn} panels on a {rows}x{columns} grid leaves {empty} slot(s) empty — "
            "another column count, or another panel, closes it"
        )
    allowed = rows_that_fit(
        columns,
        width=width,
        cell_aspect=cell_aspect,
        column_width=column_width,
        page_height=page_height,
    )
    if rows > allowed:
        complaints.append(
            f"{rows} rows will not fit {page_height:g} in of page at a {column_width:g} in column; "
            f"{allowed} would. Take a panel out rather than shrinking the cell, which is shared "
            "with every other grid in the set"
        )
    return complaints


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
    require(
        count >= 1,
        "panels needs at least one panel",
    )
    right = LEGEND_RIGHT if legend else FULL_RIGHT

    lines: list[str] = []
    caption_height = 0.0
    if caption:
        panel_width_points = units.inches_to_points(width * (right - LEFT_MARGIN))
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
        mark(drawn, "caption_row")  # `settle_caption` finds it
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
    captions = [text for ax in fig.axes for text in ax.texts if marked(text, "caption_row")]
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


def wrap_to_panel(ax: Axes, text: str, fontsize: float, *, fraction: float = 1.0) -> list[str]:
    """`wrap_to_width` against an AXES, which is what a caller actually has.

    The width has to be measured in points and an axes reports pixels, so wrapping to a panel means
    a transform and a dpi conversion — two more chances to be out by a factor. A consumer wrote its
    own `wrap_to_panel` for exactly this.

    `fraction` wraps to part of the panel, for a note meant to sit under one half of it.
    """
    require(
        0.0 < fraction <= 1.0,
        f"a fraction of the panel, got {fraction}",
    )
    figure = ax.get_figure(root=True)
    assert figure is not None, "the axes must belong to a figure"
    figure.canvas.draw()
    width_points = units.to_points(ax.get_window_extent().width, fig=figure) * fraction
    return wrap_to_width(text, width_points, fontsize)


BAR_INCHES = 1.9  # of figure width per bar, measured against the house type sizes


def width_for_bars(count: int, *, minimum: float = 12.0, per_bar: float = BAR_INCHES) -> float:
    """How wide a bar panel has to be to hold `count` bars without its labels colliding.

    A figure that held six bars does not hold eight. The gate says so — it refused a 12-inch panel
    of eight bars with six real collisions — but saying so after the fact leaves the caller to
    invent the rule, and the rule is one multiplication.

    `per_bar` is measured against the house type sizes: a value label, an arm name and a group row
    all have to clear their neighbours, and 1.9 in is what does it. A project setting smaller type
    can pass its own and still have one number to change rather than a width to re-guess.
    """
    require(
        count >= 1,
        f"a panel needs at least one bar, got {count}",
    )
    return max(minimum, per_bar * count)
