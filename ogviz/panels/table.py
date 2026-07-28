"""A comparison table drawn as a figure: rows of measurements, columns of things being compared.

Rendered rather than typeset because it ships as an image — into a slide, a README, a post — where
an HTML table cannot go and a screenshot of one looks like a screenshot. Drawing it means it shares
the figure's type, colour and page with every chart beside it.

What the layout is doing:

  a row label and a sub-label   the row says what was measured, the smaller line under it says on
                                what — "Agentic coding" and the benchmark are different facts and
                                the reader wants the first one first
  a highlighted column          the subject of the table, outlined the whole way down, so a reader
                                who reads nothing else knows which column the page is about
  shaded cells                  the best value per row, shaded rather than bolded alone, because
                                bold at a glance across twelve rows is invisible
  an em dash for missing        never a blank and never a zero. A blank reads as an oversight and a
                                zero is a measurement that was never taken
  sub-labelled values           a cell may carry its own qualifier ("with tools"), which is what
                                lets one row hold two conditions without becoming two rows

Column widths come from the longest string that has to fit, measured, not guessed — a table whose
header collides with the column beside it is the failure this replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle

from ogviz.layout.panels import text_width_points
from ogviz.theme import GRID, INK, MUTED_INK, SERIES, page_color

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

MISSING = "—"  # em dash: measured and absent, not zero and not forgotten
HEADER_SIZE = 12.0
LABEL_SIZE = 11.5
SUBLABEL_SIZE = 9.5
VALUE_SIZE = 12.0
VALUE_SUB_SIZE = 8.5
ROW_RULE_WIDTH = 0.8
HIGHLIGHT_RADIUS = 0.012
TINT_STRENGTH = 0.84  # how far a highlight colour is blended toward the page for a cell fill
LABEL_COLUMN_SHARE = 0.26  # of the table width, before measurement widens it
CELL_PAD_PT = 18.0


@dataclass(frozen=True)
class Cell:
    """One measurement. `sub` is the condition it was taken under, printed under the value."""

    value: str = MISSING
    sub: str | None = None
    best: bool = False  # shade it: the strongest value in this row

    def is_missing(self) -> bool:
        return self.value == MISSING


@dataclass(frozen=True)
class Row:
    """One measured thing, and its result in each column."""

    label: str
    cells: tuple[Cell, ...]
    sub: str | None = None  # the benchmark, instrument or dataset the row was measured on
    height: float = field(default=1.0)  # rows carrying two conditions ask for more


def _row_text_width(row: Row) -> float:
    return max(
        text_width_points(row.label, LABEL_SIZE),
        text_width_points(row.sub or "", SUBLABEL_SIZE),
    )


def _column_text_width(header: str, rows: Sequence[Row], index: int) -> float:
    widths = [text_width_points(header, HEADER_SIZE)]
    for row in rows:
        cell = row.cells[index]
        widths.append(text_width_points(cell.value, VALUE_SIZE))
        widths.append(text_width_points(cell.sub or "", VALUE_SUB_SIZE))
    return max(widths)


def tint(color: str, *, strength: float = TINT_STRENGTH) -> tuple[float, float, float, float]:
    """`color` blended toward the page — the cell fill that matches a given highlight.

    Derived rather than named so a caller who passes their own highlight gets a shade that belongs
    with it. A second hex constant would be a colour that has to be re-chosen, and forgotten, every
    time the first one changes.
    """
    red, green, blue, _alpha = to_rgba(color)
    page = to_rgba(page_color())
    blend = tuple(
        channel * (1.0 - strength) + page[index] * strength
        for index, channel in enumerate((red, green, blue))
    )
    return (blend[0], blend[1], blend[2], 1.0)


def table_panel(
    ax: Axes,
    headers: Sequence[str],
    rows: Sequence[Row],
    *,
    highlight: int | None = None,
    highlight_color: str = SERIES[0],
    shade: str | None = None,
    rule_color: str = GRID,
) -> None:
    """Draw the table across `ax`, sizing every column to the widest string it must hold.

    `highlight` is the column index the table is about; it gets a rounded outline running the full
    height. Cells marked `best` are shaded in `shade` — including ones outside the highlighted
    column, because a table that can only ever flatter its own subject is not worth drawing.
    """
    assert rows, "a table needs at least one row"
    for row in rows:
        assert len(row.cells) == len(headers), (
            f"row {row.label!r} has {len(row.cells)} cells for {len(headers)} columns"
        )
    assert highlight is None or 0 <= highlight < len(headers), (
        f"highlight {highlight} is not a column index"
    )
    cell_fill = tint(highlight_color) if shade is None else shade

    label_width = max(_row_text_width(row) for row in rows) + CELL_PAD_PT
    column_widths = [_column_text_width(h, rows, i) + CELL_PAD_PT for i, h in enumerate(headers)]
    total = label_width + sum(column_widths)
    label_share = label_width / total
    shares = [width / total for width in column_widths]
    edges = [label_share]
    for share in shares:
        edges.append(edges[-1] + share)
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(headers))]

    heights = [row.height for row in rows]
    header_height = 1.2
    unit = 1.0 / (header_height + sum(heights))
    tops = [1.0 - header_height * unit]
    for height in heights:
        tops.append(tops[-1] - height * unit)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    def cell_text(*args: object, **kwargs: object):
        """Draw a string and mark it as belonging to its cell.

        Every string in a table sits on top of that cell's shading, which is the entire design. The
        general "is this label on the data" check would report all of them and try to move each one
        somewhere emptier, which for a table means nowhere.
        """
        drawn = ax.text(*args, **kwargs)  # type: ignore[arg-type]
        drawn.ogviz_anchored = True  # type: ignore[attr-defined]
        return drawn

    if highlight is not None:
        ax.add_patch(
            FancyBboxPatch(
                (edges[highlight], 0.0),
                shares[highlight],
                1.0,
                boxstyle=f"round,pad=0,rounding_size={HIGHLIGHT_RADIUS}",
                transform=ax.transAxes,
                facecolor="none",
                edgecolor=highlight_color,
                linewidth=1.8,
                zorder=5,
            )
        )

    for index, (header, centre) in enumerate(zip(headers, centres, strict=True)):
        cell_text(
            centre,
            1.0 - header_height * unit * 0.45,
            header,
            ha="center",
            va="center",
            fontsize=HEADER_SIZE,
            color=INK,
            zorder=6,
        )
        del index

    for row_index, row in enumerate(rows):
        top, bottom = tops[row_index], tops[row_index + 1]
        middle = (top + bottom) / 2
        ax.plot(
            [0.0, 1.0], [top, top], color=rule_color, lw=ROW_RULE_WIDTH, zorder=1, clip_on=False
        )
        cell_text(
            0.0,
            middle + (0.012 if row.sub else 0.0),
            row.label,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE,
            color=INK,
            zorder=6,
        )
        if row.sub:
            cell_text(
                0.0,
                middle - 0.020,
                row.sub,
                ha="left",
                va="center",
                fontsize=SUBLABEL_SIZE,
                color=MUTED_INK,
                zorder=6,
            )
        for column, (cell, centre) in enumerate(zip(row.cells, centres, strict=True)):
            if cell.best and not cell.is_missing():
                fill = Rectangle(
                    (edges[column], bottom),
                    shares[column],
                    top - bottom,
                    transform=ax.transAxes,
                    facecolor=cell_fill,
                    edgecolor="none",
                    zorder=0,
                )
                # the cell tint; its value is meant to sit on it
                fill.ogviz_backdrop = True  # type: ignore[attr-defined]
                ax.add_patch(fill)
            cell_text(
                centre,
                middle + (0.012 if cell.sub else 0.0),
                cell.value,
                ha="center",
                va="center",
                fontsize=VALUE_SIZE,
                fontweight="bold" if cell.best else "normal",
                color=MUTED_INK if cell.is_missing() else INK,
                zorder=6,
            )
            if cell.sub:
                cell_text(
                    centre,
                    middle - 0.020,
                    cell.sub,
                    ha="center",
                    va="center",
                    fontsize=VALUE_SUB_SIZE,
                    color=MUTED_INK,
                    zorder=6,
                )
    ax.plot(
        [0.0, 1.0],
        [tops[-1], tops[-1]],
        color=rule_color,
        lw=ROW_RULE_WIDTH,
        zorder=1,
        clip_on=False,
    )
