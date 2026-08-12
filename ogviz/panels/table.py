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
from typing import TYPE_CHECKING, Literal, NamedTuple

from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle

from ogviz.layout.panels import text_width_points
from ogviz.require import require
from ogviz.tags import mark
from ogviz.theme import BAD, GOOD, GRID, INK, MUTED_INK, SERIES, family_for, page_color

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
# A row label used to be drawn flush against x=0.0, which is poor typography on its own — text
# hard against the edge of a figure — and became a defect once a row could be outlined: the frame's
# 1.8 pt stroke ran straight through the first glyph of the name, and `colliding_ink` said so, at 23
# px of shared ink. The label is indented instead of the frame being nudged, because the frame has
# to enclose the row and there was no gap for it to sit in.
LABEL_INDENT = 0.008
TINT_STRENGTH = 0.84  # how far a highlight colour is blended toward the page for a cell fill
CELL_PAD_PT = 18.0


Tone = Literal["good", "bad", "neutral"]
# Semantic name in, house colour out. A caller says what a cell MEANS and the palette stays this
# package's decision, so a green tick reads the same green in every repo — which is the whole reason
# `Cell` takes a tone rather than a hex string.
TONE_INK: dict[str, str] = {"good": GOOD, "bad": BAD, "neutral": MUTED_INK}


@dataclass(frozen=True)
class Cell:
    """One measurement. `sub` is the condition it was taken under, printed under the value.

    `tone` colours the value by what it MEANS — a green `YES`, a red `NO` — which is the ordinary
    way to make a conditions table scannable and had no route before: `best` shades a background,
    and that reads as "the strongest value here", not as "has" against "lacks".

    Colour is never the only signal. `theme.YES` and `theme.NO` differ in SHAPE as well, which is
    what makes the table readable to the reader who cannot separate the two hues — and is why this
    is a tone on a cell that already carries a string, rather than a colour on its own.
    """

    value: str = MISSING
    sub: str | None = None
    best: bool = False  # shade it: the strongest value in this row
    tone: Tone | None = None

    def is_missing(self) -> bool:
        return self.value == MISSING

    def ink(self) -> str:
        """The colour this cell's value is printed in."""
        if self.is_missing():
            return MUTED_INK
        return TONE_INK[self.tone] if self.tone is not None else INK


@dataclass(frozen=True)
class Row:
    """One measured thing, and its result in each column."""

    label: str
    cells: tuple[Cell, ...]
    sub: str | None = None  # the benchmark, instrument or dataset the row was measured on
    height: float = field(default=1.0)  # rows carrying two conditions ask for more


def _row_text_width(row: Row, scale: float = 1.0) -> float:
    return max(
        text_width_points(row.label, LABEL_SIZE * scale),
        text_width_points(row.sub or "", SUBLABEL_SIZE * scale),
    )


def _column_text_width(header: str, rows: Sequence[Row], index: int, scale: float = 1.0) -> float:
    # Scaled here as well as at the draw, or a table set larger keeps the column widths of the
    # smaller one and its own headers run into the column beside them.
    widths = [text_width_points(header, HEADER_SIZE * scale)]
    for row in rows:
        cell = row.cells[index]
        widths.append(text_width_points(cell.value, VALUE_SIZE * scale))
        widths.append(text_width_points(cell.sub or "", VALUE_SUB_SIZE * scale))
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


HEADER_HEIGHT = 1.2  # in row units: the header band is a fifth taller than an ordinary row


class Layout(NamedTuple):
    """Where every column and row edge lands, in axes fractions.

    Split out of `table_panel`, which was 168 statements — by a wide margin the longest function in
    the package, and it had grown by three arguments in a week. This half is a pure function of the
    inputs: no axes, no artists, nothing drawn. That is what makes it separable, and it is also the
    half worth testing directly, since every collision in a table is a consequence of these numbers.
    """

    shares: list[float]  # width of each value column
    edges: list[float]  # left edge of each value column, plus the right edge of the last
    centres: list[float]  # centre of each value column
    tops: list[float]  # top of the header band, then the top of each row
    unit: float  # one row height, in axes fractions


def _measure(headers: Sequence[str], rows: Sequence[Row], font_scale: float) -> Layout:
    """Column widths from the widest string each must hold, and row tops from the row heights."""
    label_width = max(_row_text_width(row, font_scale) for row in rows) + CELL_PAD_PT
    column_widths = [
        _column_text_width(h, rows, i, font_scale) + CELL_PAD_PT for i, h in enumerate(headers)
    ]
    total = label_width + sum(column_widths)
    shares = [width / total for width in column_widths]

    edges = [label_width / total]
    for share in shares:
        edges.append(edges[-1] + share)
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(headers))]

    heights = [row.height for row in rows]
    unit = 1.0 / (HEADER_HEIGHT + sum(heights))
    tops = [1.0 - HEADER_HEIGHT * unit]
    for height in heights:
        tops.append(tops[-1] - height * unit)
    return Layout(shares, edges, centres, tops, unit)


def table_panel(
    ax: Axes,
    headers: Sequence[str],
    rows: Sequence[Row],
    *,
    highlight: int | None = None,
    highlight_row: int | None = None,
    highlight_color: str = SERIES[0],
    shade: str | None = None,
    rule_color: str = GRID,
    font_scale: float = 1.0,
) -> None:
    """Draw the table across `ax`, sizing every column to the widest string it must hold.

    `highlight` is the column index the table is about; it gets a rounded outline running the full
    height. Cells marked `best` are shaded in `shade` — including ones outside the highlighted
    column, because a table that can only ever flatter its own subject is not worth drawing.

    `highlight_row` is the same claim about a ROW, for a table with entities as rows and attributes
    across. It exists because `highlight` is a COLUMN index and nothing said so loudly enough: a
    table transposed from metrics-down to arms-across kept its `highlight=0`, which then outlined
    the first METRIC and read as "this table is about that measurement". The call stayed valid and
    rendered happily, which is what made it dangerous — no assertion can catch a correct index that
    now means something else, so the fix is to have the other axis available at all.

    `font_scale` multiplies every type size in the table together. A table is set in points on a
    canvas whose height comes from its row count, so the same sizes read large on a six-row table
    and small on a twenty-row one; that ratio is the caller's to set and the only alternative was
    monkey-patching this module's constants. Note that the usual cause of an unreadable table is the
    ASPECT RATIO rather than the type size — `type_too_small` in `ogviz.qc` reports which.
    """
    require(rows, "a table needs at least one row")
    for row in rows:
        require(
            len(row.cells) == len(headers),
            f"row {row.label!r} has {len(row.cells)} cells for {len(headers)} columns",
        )
    require(
        highlight is None or 0 <= highlight < len(headers),
        f"highlight {highlight} is not a column index",
    )
    require(
        highlight_row is None or 0 <= highlight_row < len(rows),
        f"highlight_row {highlight_row} is not a row index of {len(rows)} rows",
    )
    require(font_scale > 0.0, f"font_scale multiplies the type sizes, got {font_scale}")
    header_size = HEADER_SIZE * font_scale
    label_size = LABEL_SIZE * font_scale
    sublabel_size = SUBLABEL_SIZE * font_scale
    value_size = VALUE_SIZE * font_scale
    value_sub_size = VALUE_SUB_SIZE * font_scale
    cell_fill = tint(highlight_color) if shade is None else shade

    grid = _measure(headers, rows, font_scale)
    shares, edges, centres, tops, unit = grid.shares, grid.edges, grid.centres, grid.tops, grid.unit
    header_height = HEADER_HEIGHT

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    def cell_text(x: float, y: float, body: str, **kwargs: object):
        """Draw a string, in a font that can render it, and mark it as belonging to its cell.

        Every string in a table sits on top of that cell's shading, which is the entire design. The
        general "is this label on the data" check would report all of them and try to move each one
        somewhere emptier, which for a table means nowhere.

        The family is asked per string rather than set for the table, because it is only the
        exceptional string that needs it: `theme.YES` and `theme.NO` are exactly the characters the
        display stack has no glyph for, and setting the whole table in the fallback would change the
        type of every table to accommodate two ticks. `family_for` returns None for everything else.
        """
        family = family_for(body)
        if family is not None:
            kwargs = {**kwargs, "fontfamily": family}
        drawn = ax.text(x, y, body, **kwargs)  # type: ignore[arg-type]
        mark(drawn, "anchored")
        return drawn

    def outline(x: float, y: float, width: float, height: float) -> None:
        """The rounded frame that says which part of the table the figure is about."""
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={HIGHLIGHT_RADIUS}",
                transform=ax.transAxes,
                facecolor="none",
                edgecolor=highlight_color,
                linewidth=1.8,
                zorder=5,
                clip_on=False,  # a frame at the very edge would lose half its stroke
            )
        )

    if highlight is not None:
        outline(edges[highlight], 0.0, shares[highlight], 1.0)
    if highlight_row is not None:
        # The full width, the way the column outline runs the full height — the two are the same
        # claim about the two axes of the table, so they are drawn by the same call.
        outline(0.0, tops[highlight_row + 1], 1.0, tops[highlight_row] - tops[highlight_row + 1])

    for header, centre in zip(headers, centres, strict=True):
        cell_text(
            centre,
            1.0 - header_height * unit * 0.45,
            header,
            ha="center",
            va="center",
            fontsize=header_size,
            color=INK,
            zorder=6,
        )

    for row_index, row in enumerate(rows):
        top, bottom = tops[row_index], tops[row_index + 1]
        middle = (top + bottom) / 2
        ax.plot(
            [0.0, 1.0], [top, top], color=rule_color, lw=ROW_RULE_WIDTH, zorder=1, clip_on=False
        )
        cell_text(
            LABEL_INDENT,
            middle + (0.012 if row.sub else 0.0),
            row.label,
            ha="left",
            va="center",
            fontsize=label_size,
            color=INK,
            zorder=6,
        )
        if row.sub:
            cell_text(
                LABEL_INDENT,
                middle - 0.020,
                row.sub,
                ha="left",
                va="center",
                fontsize=sublabel_size,
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
                mark(fill, "backdrop")
                ax.add_patch(fill)
            cell_text(
                centre,
                middle + (0.012 if cell.sub else 0.0),
                cell.value,
                ha="center",
                va="center",
                fontsize=value_size,
                fontweight="bold" if cell.best else "normal",
                color=cell.ink(),
                zorder=6,
            )
            if cell.sub:
                cell_text(
                    centre,
                    middle - 0.020,
                    cell.sub,
                    ha="center",
                    va="center",
                    fontsize=value_sub_size,
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
