"""Headers, frames and saving — the parts every builder otherwise reinvents.

`titled` returns where the header ends so a caller sizes its axes from the measured header
rather than a guessed `rect` top. The title-to-subtitle gap is derived from the type size and
the figure height in points: a fixed figure-fraction gap collides with the title on a short
figure and floats away from it on a tall one.

No caption helper. Figures carry title, subtitle, axes, legend and data; what the marks mean,
which test was run, n and the exact p belong in the project's README, where they are maintained
and read. A caption baked into an image is a second copy that drifts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

from ogviz.layout.caption import caption, overflowing_text
from ogviz.layout.collision import (
    annotate_clear,
    clear_position,
    hits_data,
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
    "align_mean_rows",
    "annotate_clear",
    "assert_no_text_overlap",
    "assert_nothing_clipped",
    "auto_decimals",
    "baseline",
    "caption",
    "clear_position",
    "clipped_artists",
    "dead_space",
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
    "share_value_limits",
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


def ticks_over_data(ax, data_high: float, *, orientation: str = "vertical") -> None:
    """Drop value ticks that fall in the room reserved above the data.

    A panel grows its value axis to fit a bracket stack, and the locator then puts ticks up there
    because it sees axis, not meaning. Those ticks and their gridlines say a measurement could sit
    at that height when nothing can — the space is layout, held open for the brackets.

    It also makes panels disagree with each other for no reason a reader can see: one whose stack
    happens to clear a round number carries an extra rule and its neighbour does not. That is the
    inconsistency this removes.
    """
    ticks = ax.get_yticks() if orientation == "vertical" else ax.get_xticks()
    kept = [float(tick) for tick in ticks if float(tick) <= data_high + 1e-9]
    if not kept or len(kept) == len(ticks):
        return
    if orientation == "vertical":
        ax.set_yticks(kept)
    else:
        ax.set_xticks(kept)


def share_value_limits(axes, *, orientation: str = "vertical") -> tuple[float, float]:
    """Put every panel on one value scale: the union of the limits they each worked out.

    For a grid of comparable panels, which have to share a scale to be read against each other. The
    scale is the union of what the panels ALREADY fitted, not a number chosen in advance — a violin
    panel measures the headroom its bracket stack needs and grows the axis to suit, and a caller who
    then overwrites that with a guess has thrown the measurement away.

    That is the bug this replaces. A grid of one-comparison panels was given headroom sized for a
    three-bracket stack, so every panel carried two brackets' worth of empty page between its stars
    and its title. Ask each panel what it needs and take the widest answer, and a grid of
    single-bracket panels gets exactly one bracket's room.

    Returns the shared (low, high).
    """
    panels = list(axes)
    assert panels, "share_value_limits needs at least one axes"
    reader = (lambda ax: ax.get_ylim()) if orientation == "vertical" else (lambda ax: ax.get_xlim())
    spans = [reader(ax) for ax in panels]
    low = min(bounds[0] for bounds in spans)
    high = max(bounds[1] for bounds in spans)
    for ax in panels:
        if orientation == "vertical":
            ax.set_ylim(low, high)
        else:
            ax.set_xlim(low, high)
    if orientation == "vertical":
        align_mean_rows(panels, floor=low)
    return low, high


def align_mean_rows(axes, *, floor: float) -> float | None:
    """Put every panel's printed means on ONE line, and return that line.

    A panel places its means in the middle of the margin below its own data. Once the panels share
    a scale that is wrong: the floor is common and the lowest violin is not, so the row sits at a
    different height in each panel and the eye reads four different rows where there is one kind of
    number. The gap from a row to the frame stops meaning anything.

    The line is the midpoint between the floor and the lowest mark ACROSS the panels, so it clears
    the deepest violin in the grid and is identical everywhere. Returns None where no panel prints
    means.
    """
    rows = [text for ax in axes for text in ax.texts if getattr(text, "ogviz_mean_row", False)]
    if not rows:
        return None
    lowest = min(
        (
            float(np.asarray(path.vertices, dtype=float)[:, 1].min())
            for ax in axes
            for collection in ax.collections
            for path in collection.get_paths()
            if np.asarray(path.vertices, dtype=float).size
        ),
        default=None,
    )
    if lowest is None:
        return None
    line = (floor + lowest) / 2.0
    for text in rows:
        text.set_position((text.get_position()[0], line))
    return line


def fit_under_header(
    fig: Figure,
    header_bottom: float,
    *,
    bottom: float = 0.0,
    gap: float = 0.014,
) -> None:
    """Lay the panels out and PIN their top just under the header `titled` reported.

    `tight_layout(rect=...)` treats the rect as room it may use, not a top edge it must reach: with
    a legend anchored below the axes it leaves a band of empty page between the panels and the
    title — 92 px on a 680 px figure, which reads as a mistake rather than as breathing room. Doing
    the layout and then pinning the top closes it.

    `gap` is the only free number left, and it is the space BETWEEN the subtitle and the panels
    rather than a guess at where the header ends; `titled` measures that and returns it.

    An axes title grows UPWARD out of its axes, so pinning the subplot top without reserving for it
    drives the titles into the subtitle — which is what pinning a 2x2 grid of named panels did on
    the first attempt. The reservation is measured from the drawn titles rather than assumed from a
    font size, because a title that wrapped to two lines needs twice the room and says so only once
    it has been laid out.
    """
    fig.tight_layout(rect=(0.0, bottom, 1.0, header_bottom))
    fig.canvas.draw()
    figure_px = fig.get_figheight() * fig.dpi
    titles_px = max(
        (float(ax.title.get_window_extent().height) for ax in fig.axes if ax.get_title().strip()),
        default=0.0,
    )
    reserved = titles_px / figure_px * TITLE_CLEARANCE
    fig.subplots_adjust(top=max(0.05, header_bottom - gap - reserved))


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
