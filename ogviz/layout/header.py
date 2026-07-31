"""The title band, and fitting the panels under it.

`titled` returns where the header ends, so a caller sizes its panels from a measured header rather
than a guessed `rect` top. The title-to-subtitle gap is derived from the type size and the figure
height in points: a fixed figure-fraction gap collides with the title on a short figure and floats
away from it on a tall one.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from ogviz.theme import INK, MUTED_INK, SUBTITLE_SIZE, TITLE_SIZE

if TYPE_CHECKING:
    from matplotlib.figure import Figure

TITLE_CLEARANCE = 1.35  # an axes title needs its own height plus the pad under it


LEFT_ALIGNED = "ogviz_header_left"  # marks a header line that hangs off the panels' left edge


def panel_left_edge(fig: Figure) -> float:
    """Where the leftmost panel starts, in figure coordinates.

    The anchor a left-aligned header hangs from. A `fig.suptitle` sits in FIGURE coordinates and a
    panel in AXES coordinates, and the two do not line up — a title at x=0.05 beside an axes that
    begins at 0.11 reads as a mistake, which is why a caller doing this by hand ends up reading
    `ax.get_position().x0` after a draw.
    """
    boxes = [ax.get_position().x0 for ax in fig.axes if ax.get_visible()]
    return float(min(boxes)) if boxes else float(fig.subplotpars.left)


def settle_header(fig: Figure) -> list[str]:
    """Re-anchor every left-aligned header line to the panels as they NOW are.

    A left-aligned header is measured against the panels, and `tight_layout` moves the panels after
    the header is written — so the alignment the caller asked for is stale by the time the figure is
    saved. The same failure `settle_caption` and `settle_bracket_labels` exist for, and the reason
    the alignment is stored as intent and resolved once instead of computed at call time.
    """
    left = panel_left_edge(fig)
    moved: list[str] = []
    for text in fig.texts:
        if not getattr(text, LEFT_ALIGNED, False):
            continue
        x, y = text.get_position()
        if abs(x - left) < 1e-9:
            continue
        text.set_position((left, y))
        moved.append(f"re-anchored the header line {text.get_text()[:40]!r} to the panels")
    return moved


def titled(
    fig: Figure,
    title: str,
    *,
    subtitle: str | None = None,
    title_y: float = 0.98,
    size: float = TITLE_SIZE,
    subtitle_size: float = SUBTITLE_SIZE,
    align: Literal["center", "left"] = "center",
) -> float:
    """Bold-sans title with a serif grey subtitle. Returns the header's bottom.

    `size` is an argument because a title is sized against its figure, not against the house: the
    27 pt default suits a single tall panel and swamps a 12.8-inch report row, where it grows wide
    enough to reach the y tick labels.

    `align="left"` hangs both lines off the leftmost panel's left edge rather than centring them on
    the canvas. It is a different look, not a variant of the same one: the centred header belongs to
    a figure read as a plate, and the left-aligned one to a figure read as the top of a page. Two
    consumer scripts were carrying about eight lines each of hand-placed `fig.text` for it, and
    hand-placing it is what makes it drift when the layout runs afterwards.
    """
    figure_points = fig.get_figheight() * 72.0
    horizontal = "left" if align == "left" else "center"
    x = panel_left_edge(fig) if align == "left" else 0.5
    heading = fig.suptitle(
        title, fontsize=size, fontweight="bold", color=INK, x=x, y=title_y, ha=horizontal
    )
    header_bottom = title_y - size * 1.25 / figure_points
    lines = [heading]
    if subtitle is not None:
        lines.append(
            fig.text(
                x,
                header_bottom,
                subtitle,
                ha=horizontal,
                va="top",
                fontsize=subtitle_size,
                family="serif",
                color=MUTED_INK,
            )
        )
        header_bottom -= subtitle_size * 1.35 / figure_points
    if align == "left":
        for line in lines:
            setattr(line, LEFT_ALIGNED, True)
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
    axis decorations cannot fit the rect — a two-line x tick label is enough to cause it
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
