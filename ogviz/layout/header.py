"""The title band, and fitting the panels under it.

`titled` returns where the header ends, so a caller sizes its panels from a measured header rather
than a guessed `rect` top. The title-to-subtitle gap is derived from the type size and the figure
height in points: a fixed figure-fraction gap collides with the title on a short figure and floats
away from it on a tall one.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ogviz.theme import INK, MUTED_INK, SUBTITLE_SIZE, TITLE_SIZE

if TYPE_CHECKING:
    from matplotlib.figure import Figure

TITLE_CLEARANCE = 1.35  # an axes title needs its own height plus the pad under it


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
