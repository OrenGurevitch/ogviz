"""Headers, frames and saving — the parts every builder otherwise reinvents.

`titled` returns where the header ends so a caller sizes its axes from the measured header rather
than a guessed `rect` top. The title-to-subtitle gap is derived from the type size and the figure
height in points: a fixed figure-fraction gap collides with the title on a short figure and floats
away from it on a tall one.

Captions are OFF unless a caller asks for one, and `ogviz.layout.caption` is where they live. A
figure normally carries title, subtitle, axes and legend, and what the marks mean belongs in the
surrounding text; a caption baked into an image is a second copy that drifts. The exception is a
figure that travels as a file — a slide, a shared PNG — where there is no surrounding text to
carry it, and there the caption has to be right, which is what that module is for.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from ogviz.layout.caption import caption, overflowing_text
from ogviz.layout.collision import (
    annotate_clear,
    clear_position,
    hits_data,
    point_offsets,
    text_over_data,
)
from ogviz.layout.density import dead_space, trim_margins
from ogviz.layout.density import measure as measure_density
from ogviz.layout.overlap import (
    assert_no_text_overlap,
    assert_nothing_clipped,
    clipped_artists,
    text_overlaps,
)
from ogviz.layout.panels import panel_row, text_width_points, wrap_to_width
from ogviz.layout.ticks import auto_decimals, format_value, round_ticks, value_ticks

__all__ = [
    "annotate_clear",
    "assert_no_text_overlap",
    "assert_nothing_clipped",
    "auto_decimals",
    "baseline",
    "caption",
    "clear_position",
    "clipped_artists",
    "dead_space",
    "drawn_value_extent",
    "fit_under_header",
    "format_value",
    "hairline_grid",
    "hits_data",
    "legend_pill",
    "measure_density",
    "overflowing_text",
    "panel_row",
    "pill_frame",
    "round_ticks",
    "save",
    "text_over_data",
    "text_overlaps",
    "text_width_points",
    "ticks_over_data",
    "titled",
    "trim_margins",
    "value_ticks",
    "wrap_to_width",
    "zero_baseline",
]
from ogviz.theme import (
    GRID,
    INK,
    MUTED_INK,
    PANEL_FILL,
    SUBTITLE_SIZE,
    TITLE_SIZE,
    glyphs_must_render,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend

    from ogviz.orientation import Orientation


def titled(
    fig: Figure,
    title: str,
    *,
    subtitle: str | None = None,
    title_y: float = 0.98,
    size: float = TITLE_SIZE,
    subtitle_size: float = SUBTITLE_SIZE,
) -> float:
    """Centred bold-sans title with a serif grey subtitle. Returns the header's bottom.

    `size` is an argument because a title is sized against its figure, not against the house: the
    27 pt default suits a single tall panel and swamps a 12.8-inch report row, where it grows wide
    enough to reach the y tick labels.
    """
    figure_points = fig.get_figheight() * 72.0
    fig.suptitle(title, fontsize=size, fontweight="bold", color=INK, y=title_y)
    header_bottom = title_y - size * 1.25 / figure_points
    if subtitle is not None:
        fig.text(
            0.5,
            header_bottom,
            subtitle,
            ha="center",
            va="top",
            fontsize=subtitle_size,
            family="serif",
            color=MUTED_INK,
        )
        header_bottom -= subtitle_size * 1.35 / figure_points
    return header_bottom


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


def ticks_over_data(ax: Axes, data_high: float, *, orientation: Orientation = "vertical") -> None:
    """Drop value ticks that fall in the room reserved above the data.

    A panel grows its value axis to fit a bracket stack, and the locator then puts ticks up there
    because it sees axis, not meaning. Those ticks and their gridlines say a measurement could sit
    at that height when nothing can — the space is layout, held open for the brackets.

    It also makes panels disagree with each other for no reason a reader can see: one whose stack
    happens to clear a round number carries an extra rule and its neighbour does not. That is the
    inconsistency this removes.
    """
    upright = orientation == "vertical"
    ticks = ax.get_yticks() if upright else ax.get_xticks()
    kept = [float(tick) for tick in ticks if float(tick) <= data_high + 1e-9]
    if not kept or len(kept) == len(ticks):
        return
    # `set_yticks` FIXES the locator, and matplotlib then grows the view to contain every fixed
    # tick. Dropping the ticks above the data therefore dragged the floor down to the lowest
    # remaining one — on a panel whose ticks ran to zero, the axis reframed itself from zero and
    # the violins ended up in the top third of a panel that had been fitted to them. Restore the
    # limits, which were already correct before the ticks were touched.
    limits = ax.get_ylim() if upright else ax.get_xlim()
    if upright:
        ax.set_yticks(kept)
        ax.set_ylim(*limits)
    else:
        ax.set_xticks(kept)
        ax.set_xlim(*limits)


def drawn_value_extent(ax: Axes) -> tuple[float, float] | None:
    """The lowest and highest value any MARK reaches, in data units, or None if nothing is drawn.

    Reading `collection.get_paths()` is the trap, and it cost a panel its layout. For a filled body
    the path IS the shape in data coordinates. For a scatter it is the MARKER OUTLINE — a unit
    circle about the origin, reused at every offset — so a panel of points reports its extent as
    roughly -0.5 to 0.5 whatever the data says. On values of order one that looks plausible; on
    values of order 0.001 it puts the answer nowhere near the panel.

    So: offsets when a collection has them, path vertices when it does not.
    """
    lows: list[float] = []
    highs: list[float] = []
    for collection in ax.collections:
        offsets = point_offsets(collection)
        if offsets is not None:
            lows.append(float(offsets[:, 1].min()))
            highs.append(float(offsets[:, 1].max()))
            continue
        for path in collection.get_paths():
            vertices = np.asarray(path.vertices, dtype=float)
            if vertices.size:
                lows.append(float(vertices[:, 1].min()))
                highs.append(float(vertices[:, 1].max()))
    if not lows:
        return None
    return min(lows), max(highs)


def fit_under_header(
    fig: Figure,
    header_bottom: float,
    *,
    bottom: float = 0.0,
    gap: float = 0.014,
) -> bool:
    """Lay the panels out and PIN their top just under the header `titled` reported.

    `tight_layout(rect=...)` treats the rect as room it may use, not a top edge it must reach: with
    a legend anchored below the axes it leaves a band of empty page between the panels and the
    title — 92 px on a 680 px figure, which reads as a mistake rather than as breathing room. Doing
    the layout and then pinning the top closes it.

    Returns whether `tight_layout` actually ran. It refuses, with a warning and no effect, when the
    axis decorations cannot fit the rect — a two-line x tick label is enough to cause it (FIXME §8)
    — and the figure then keeps default margins. Usually fine, never silent: a caller that needs
    the layout to have happened can check.

    `gap` is the only free number left, and it is the space BETWEEN the subtitle and the panels
    rather than a guess at where the header ends; `titled` measures that and returns it.

    An axes title grows UPWARD out of its axes, so pinning the subplot top without reserving for it
    drives the titles into the subtitle — which is what pinning a 2x2 grid of named panels did on
    the first attempt. The reservation is measured from the drawn titles rather than assumed from a
    font size, because a title that wrapped to two lines needs twice the room and says so only once
    it has been laid out.
    """
    # matplotlib warns and then does nothing when the decorations will not fit the rect, leaving
    # the figure on default margins. That is survivable — the top pin below is applied either way,
    # and the overlap checks still run on whatever came out — but it must not be mistaken for a
    # layout that succeeded. Caught so it cannot be lost in a build log, and reported through the
    # return value so a caller can act on it.
    applied = True
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always", UserWarning)
        fig.tight_layout(rect=(0.0, bottom, 1.0, header_bottom))
        applied = not any("Tight layout not applied" in str(one.message) for one in raised)
    # Recorded on the figure as well as returned, because the return value went unread for a week
    # and the whole point was that this should not pass unnoticed.
    fig.ogviz_layout_refused = not applied  # type: ignore[attr-defined]
    fig.canvas.draw()
    figure_px = fig.get_figheight() * fig.dpi
    titles_px = max(
        (float(ax.title.get_window_extent().height) for ax in fig.axes if ax.get_title().strip()),
        default=0.0,
    )
    reserved = titles_px / figure_px * TITLE_CLEARANCE
    fig.subplots_adjust(top=max(0.05, header_bottom - gap - reserved))
    return applied


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


def save(
    fig: Figure,
    directory: Path,
    name: str,
    *,
    dpi: int = 200,
    check_overlap: bool = True,
    formats: Sequence[str] = ("png", "svg"),
    close: bool = True,
) -> list[Path]:
    """Write `<directory>/<name>.<ext>` per format, on the figure's own canvas, and check it.

    Both checks fail the build rather than write a broken figure: a missing glyph renders as a
    tofu box and overlapping labels render as mush, and both otherwise ship unnoticed because a
    figure build scrolls past. Pass `check_overlap=False` for a panel whose text legitimately
    abuts, such as a rendered table, and `close=False` to keep working on the figure.
    """
    assert formats, "save needs at least one format"
    directory.mkdir(parents=True, exist_ok=True)
    if check_overlap:
        from ogviz.qc import assert_clean

        assert_clean(fig)
    canvas = fig.get_facecolor()
    paths = [directory / f"{name}.{extension}" for extension in formats]
    with glyphs_must_render():
        for path in paths:
            fig.savefig(path, bbox_inches="tight", facecolor=canvas, dpi=dpi)
    if close:
        plt.close(fig)
    return paths
