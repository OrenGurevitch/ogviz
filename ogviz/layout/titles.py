"""Panel titles sized as a set, and refused when no size fits.

A title wider than its panel lands on the title beside it, and it does so at full size while a
thumbnail hides it. Two projects each wrote the same fix: measure the rendered titles and step the
type down until none is wider than the panel it names. Both also learned the two details below.

TITLES SHRINK TOGETHER. A grid is read across, and one title in 13.5 pt beside another in 15 reads
as a mistake. The fit finds the largest size at which EVERY title fits and sets them all to it.

RUNNING OUT OF ROOM RAISES. Returning at the smallest size would leave a title overrunning at a
size nobody chose, on a figure where nothing else may say so. The wording is the author's call, and
no size hides a title that is too long; a second line is what one project found put the whole set
back at its full size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ogviz.layout.render import ensure_rendered
from ogviz.require import require

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

STEP = 0.5  # points the set shrinks by on each attempt
PAD_PX = 2.0  # clear space a title keeps inside its panel's width


def axes_titles(ax: Axes) -> list[Text]:
    """The title artists of `ax` that hold text, whichever of the three positions carries it.

    matplotlib keeps a `loc="left"` or `loc="right"` title in a private attribute and exposes only
    its string through `get_title`, whose default asks for the CENTRE title alone. Code that read
    `ax.title` and `ax.get_title()` therefore never saw a left-aligned title: the header spacing
    reserved nothing for it, and every overlap and off-canvas check walked past it. A figure of
    left-aligned panel titles passed the gate with its subtitle printed across them.
    """
    return [
        text
        for text in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None))
        if text is not None and text.get_text().strip()
    ]


def title_text(ax: Axes) -> str:
    """The panel's title as a reader sees it: the first of its centre, left and right titles."""
    found = axes_titles(ax)
    return found[0].get_text().strip() if found else ""


def fit_titles(
    fig: Figure,
    axes: Iterable[Axes] | None = None,
    *,
    largest: float | None = None,
    smallest: float = 10.0,
    pad_px: float = PAD_PX,
) -> float:
    """Set every panel title to the largest size at which none is wider than its panel.

    `largest` defaults to the biggest title now on the figure, so a call can only shrink what the
    author chose and never enlarge it. Returns the size used. Raises when `smallest` is not enough,
    naming the title and by how many pixels it still overruns.

    Draw after the layout is final: the panel's width is part of the measurement, and a panel whose
    aspect is fixed (`set_aspect("equal", adjustable="box")`) is only squared at draw time, so a fit
    run before the first draw measures the wider box and concludes a title fits that then overruns.
    """
    panels = list(fig.axes if axes is None else axes)
    titled = [(ax, text) for ax in panels for text in axes_titles(ax)]
    if not titled:
        return largest if largest is not None else smallest
    ensure_rendered(fig)
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    size = largest if largest is not None else max(float(text.get_fontsize()) for _, text in titled)
    require(
        size >= smallest, f"the titles are already at {size:g} pt, below the {smallest:g} pt floor"
    )
    while True:
        for _, text in titled:
            text.set_fontsize(size)
        over = max(
            text.get_window_extent(renderer).width - ax.get_window_extent(renderer).width
            for ax, text in titled
        )
        if over <= -pad_px:
            return size
        if size - STEP < smallest:
            worst = max(
                titled,
                key=lambda pair: (
                    pair[1].get_window_extent(renderer).width
                    - pair[0].get_window_extent(renderer).width
                ),
            )[1]
            raise AssertionError(
                f"{worst.get_text()!r} is still {over:.0f} px wider than its panel at {size:g} pt. "
                "Shorten the title or give it a second line; no size hides a title this long."
            )
        size -= STEP
