"""A matrix of effects, coloured by sign and magnitude, with the number in every cell.

The panel for "many measures against many groups" — Cohen's d per region per modality, a correlation
matrix, any signed quantity where zero is the neutral value. Three things make it different from
`imshow` with a colourbar, and each is a place a hand-rolled version goes wrong:

  The scale is SYMMETRIC about the neutral value, so equal effects in opposite directions get equal
  colour. An automatic scale over the data lands the neutral value off-centre, and the reader then
  sees a red cell where the number says nothing is happening.

  The number is printed IN the cell, in a colour chosen per cell from the fill behind it. A single
  ink colour is unreadable on the dark end of a diverging map, and the usual fix — printing
  everything white — is unreadable on the pale middle.

  A missing cell is drawn as missing rather than as neutral. `nan` shaded like zero is a cell that
  reads as a measured null, which is the one thing it is not.

A colourbar is deliberately absent. The number is in the cell, so the bar would be a second, less
precise encoding of what the reader can already read exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Rectangle

from ogviz.panels.bars import typeset
from ogviz.theme import INK, MUTED_INK, page_color

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

# Blue-to-page-to-amber, and the middle stop is the PAGE colour rather than white: a white midpoint
# on the warm canvas reads as a hole punched in the figure.
DIVERGING = ("#3C566B", "#6E8CA0", None, "#E8A838", "#B97C10")
MISSING_FILL = "#EAE7DE"
MISSING_MARK = "—"
# Below this relative luminance a fill is dark enough that ink on it stops being legible. 0.55 was
# chosen against the ends of the map above, where the darkest stops sit near 0.30.
DARK_FILL = 0.55
CELL_TEXT_SIZE = 13.0
STAR_SIZE = 11.0


def diverging_map(colors: Sequence[str | None] = DIVERGING) -> LinearSegmentedColormap:
    """The house diverging map, with the page colour standing in for any `None` stop."""
    stops = [page_color() if color is None else color for color in colors]
    return LinearSegmentedColormap.from_list("ogviz_diverging", stops)


def _luminance(color: tuple[float, float, float, float]) -> float:
    """Relative luminance, so the cell decides what colour its own number is."""
    red, green, blue, _alpha = color
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def effect_heatmap(
    ax: Axes,
    values: NDArray[np.float64],
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    p_values: NDArray[np.float64] | None = None,
    neutral: float = 0.0,
    reach: float | None = None,
    value_format: str = "{:+.2f}",
    label_for: Callable[[float], str] | None = None,
    row_dividers: Sequence[int] = (),
    column_dividers: Sequence[int] = (),
    colors: Sequence[str | None] = DIVERGING,
    text_size: float = CELL_TEXT_SIZE,
) -> None:
    """Draw `values` as a diverging matrix with the number, and optionally its stars, in each cell.

    `reach` is how far the colour scale runs either side of `neutral`; left out, it is the largest
    departure present, so the scale fits the data and stays symmetric. Pass it to hold one scale
    across several figures — a matrix whose colours mean something different from its neighbour's is
    worse than no colour at all.

    `row_dividers` and `column_dividers` are indices to rule a line BEFORE, for a matrix whose rows
    fall into groups.
    """
    grid = np.asarray(values, dtype=float)
    assert grid.ndim == 2, f"effect_heatmap needs a 2-D matrix, got shape {grid.shape}"
    assert grid.shape == (len(row_labels), len(column_labels)), (
        f"matrix {grid.shape} against {len(row_labels)} rows and {len(column_labels)} columns"
    )
    assert p_values is None or np.asarray(p_values).shape == grid.shape, (
        f"p-values {np.asarray(p_values).shape} do not match the matrix {grid.shape}"
    )

    departures = np.abs(grid - neutral)
    if reach is None:
        widest = float(np.nanmax(departures)) if np.any(np.isfinite(departures)) else 1.0
        reach = widest if widest > 0.0 else 1.0
    assert reach > 0.0, f"the scale must reach somewhere, got {reach}"

    colormap = diverging_map(colors)
    scale = Normalize(vmin=neutral - reach, vmax=neutral + reach)
    rows, columns = grid.shape

    for row in range(rows):
        for column in range(columns):
            value = grid[row, column]
            missing = not np.isfinite(value)
            fill = to_rgba(MISSING_FILL) if missing else colormap(scale(value))
            # The cell is a BACKDROP, not a mark: it carries no value of its own and exists to
            # tint the number printed on it. Untagged, every cell reported its own number as
            # sitting on the data and the audit returned 46 complaints about a correct panel.
            patch = _cell(column, row, facecolor=fill)
            patch.ogviz_backdrop = True  # type: ignore[attr-defined]
            ax.add_patch(patch)
            # Per cell, from the fill actually behind it: one ink colour is unreadable at the dark
            # end of the map and white is unreadable in the pale middle.
            ink = page_color() if _luminance(fill) < DARK_FILL else INK
            printed = MISSING_MARK if missing else typeset(value_format.format(value))
            number = ax.text(
                column,
                row,
                printed,
                ha="center",
                va="center",
                fontsize=text_size,
                fontweight="bold",
                color=MUTED_INK if missing else ink,
                zorder=3,
            )
            # Printed on its own cell on purpose, which is the whole design of the panel.
            number.ogviz_anchored = True  # type: ignore[attr-defined]
            if p_values is None or missing:
                continue
            p = float(np.asarray(p_values)[row, column])
            if not np.isfinite(p):
                continue
            glyphs = _stars_for(p, label_for)
            if glyphs:
                marked = ax.text(
                    column,
                    row + 0.30,
                    glyphs,
                    ha="center",
                    va="center",
                    fontsize=STAR_SIZE,
                    fontweight="bold",
                    color=ink,
                    zorder=3,
                )
                # It marks its own cell, so the general "is this label on the data" rule must leave
                # it where it is; `ogviz_column_star` also keeps it out of the bracket checks, which
                # measure a star against a bracket this one does not have.
                marked.ogviz_column_star = True  # type: ignore[attr-defined]
                marked.ogviz_anchored = True  # type: ignore[attr-defined]

    for index in row_dividers:
        ax.axhline(index - 0.5, color=INK, linewidth=1.8, zorder=4)
    for index in column_dividers:
        ax.axvline(index - 0.5, color=INK, linewidth=1.8, zorder=4)

    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # first row at the TOP, the way a matrix is written
    ax.set_xticks(range(columns))
    ax.set_xticklabels(list(column_labels))
    ax.set_yticks(range(rows))
    ax.set_yticklabels(list(row_labels))
    ax.tick_params(length=0.0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _cell(column: int, row: int, *, facecolor) -> Rectangle:
    return Rectangle(
        (column - 0.5, row - 0.5),
        1.0,
        1.0,
        facecolor=facecolor,
        edgecolor=page_color(),
        linewidth=1.5,
        zorder=2,
    )


def _stars_for(p: float, label_for: Callable[[float], str] | None) -> str:
    from ogviz.significance import stars

    marked = (label_for or stars)(p)
    return "" if marked == "n.s." and label_for is None else marked
