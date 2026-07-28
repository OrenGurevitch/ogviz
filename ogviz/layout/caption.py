"""A heading above the figure and a source note below it, neither able to exceed the figure width.

Captions are off unless asked for. A figure carries its title, axes and legend, and what the marks
mean belongs in the surrounding text; a caption baked into the image is for the case where the
file travels alone — a slide, a shared PNG, a reproducibility bundle — and there it has to be
right.

The shape is the one used in economic-research figures: a bold sentence ABOVE the plot that says
what the reader should take away, and a small grey block BELOW carrying the source and the caveats.
The heading is a claim, the note is provenance, and putting the claim above means it is read first.

The width guarantee is the point of this module, and it is a measurement, not a calculation. Every
wrap based on estimated character widths eventually spills — a long URL, a chemical name, a font
substituted on another machine, a `-` that the wrapper treated as a break and the renderer did not.
So: wrap, draw, MEASURE the rendered box, and if it is still too wide, narrow the target and go
again. `_fit_within` loops until the rendered text fits or the text can no longer be narrowed, and
`overflowing_text` fails the build for the second case rather than shipping a figure that runs off
its own canvas.

An unbreakable word longer than the figure is the one case wrapping cannot solve. It is reported,
never silently shrunk to fit: a caption at 4 pt is not a fixed caption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ogviz.layout.panels import text_width_points, wrap_to_width
from ogviz.theme import INK, MUTED_INK

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.text import Text

HEADING_SIZE = 13.0
NOTE_SIZE = 9.0
LINE_SPACING = 1.5
SIDE_MARGIN = 0.055  # share of figure width kept clear on each side
FIT_ATTEMPTS = 6  # re-wraps allowed before the text is declared unwrappable
FIT_SHRINK = 0.94  # how much the wrap target narrows on each attempt
OVERFLOW_TOLERANCE_PX = 1.0


def _rendered_width_px(fig: Figure, text: Text) -> float:
    fig.canvas.draw()
    return float(text.get_window_extent().width)


def _fit_within(fig: Figure, text: Text, body: str, size: float, limit_px: float) -> bool:
    """Re-wrap until the DRAWN text fits `limit_px`. Returns whether it does.

    The loop exists because the wrap and the renderer do not have to agree. `wrap_to_width`
    measures glyph outlines, which is exact for the font it was given — and the figure may be
    rendered with a different one, or hold a token with no break in it. Measuring what was actually
    drawn is the only check that cannot be argued with.
    """
    target_pt = limit_px / fig.dpi * 72.0
    for _attempt in range(FIT_ATTEMPTS):
        if _rendered_width_px(fig, text) <= limit_px + OVERFLOW_TOLERANCE_PX:
            return True
        target_pt *= FIT_SHRINK
        text.set_text("\n".join(wrap_to_width(body, target_pt, size)))
    return _rendered_width_px(fig, text) <= limit_px + OVERFLOW_TOLERANCE_PX


def caption(
    fig: Figure,
    note: str | None = None,
    *,
    heading: str | None = None,
    note_size: float = NOTE_SIZE,
    heading_size: float = HEADING_SIZE,
    margin: float = SIDE_MARGIN,
) -> None:
    """Put `heading` above the figure and `note` below it, both wrapped to the figure width.

    Space is reserved by moving the axes, so neither block can land on a panel: the note pushes the
    subplot area up by exactly the height it needs, and the heading pushes it down. Both are
    measured after drawing and re-wrapped if the render disagrees with the wrap.

    Either may be left out. With neither, this does nothing.
    """
    if note is None and heading is None:
        return
    width_px = fig.get_figwidth() * fig.dpi
    height_px = fig.get_figheight() * fig.dpi
    limit_px = width_px * (1.0 - 2.0 * margin)
    available_pt = limit_px / fig.dpi * 72.0

    if heading is not None:
        lines = wrap_to_width(heading, available_pt, heading_size)
        drawn = fig.text(
            margin,
            0.995,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=heading_size,
            fontweight="bold",
            color=INK,
            linespacing=LINE_SPACING,
        )
        _fit_within(fig, drawn, heading, heading_size, limit_px)
        used = _rendered_width_px(fig, drawn) and float(drawn.get_window_extent().height)
        fig.subplots_adjust(top=min(0.97, 1.0 - (used + 0.022 * height_px) / height_px))

    if note is not None:
        lines = wrap_to_width(note, available_pt, note_size)
        drawn = fig.text(
            margin,
            0.012,
            "\n".join(lines),
            ha="left",
            va="bottom",
            fontsize=note_size,
            color=MUTED_INK,
            linespacing=LINE_SPACING,
        )
        _fit_within(fig, drawn, note, note_size, limit_px)
        used = float(drawn.get_window_extent().height)
        fig.subplots_adjust(bottom=max(0.06, (used + 0.030 * height_px) / height_px))


def longest_unbreakable(text: str, size: float) -> float:
    """Width in points of the widest single word — what wrapping can never get below."""
    return max((text_width_points(word, size) for word in text.split()), default=0.0)


def overflowing_text(fig: Figure) -> list[str]:
    """Any figure-level text wider than the canvas it sits on.

    The last line of defence for the caption's width guarantee, and it applies to every figure-level
    string, not only captions: a title long enough to run off the page fails here too. Reported
    against the CANVAS rather than against a margin, so it fires only on genuine overflow and not
    on a deliberately full-bleed heading.
    """
    fig.canvas.draw()
    width_px = fig.get_figwidth() * fig.dpi
    complaints: list[str] = []
    for text in fig.texts:
        content = text.get_text().strip()
        if not content or not text.get_visible():
            continue
        size = float(text.get_fontsize())
        drawn = float(text.get_window_extent().width)
        if drawn > width_px + OVERFLOW_TOLERANCE_PX:
            # Name the word that cannot be broken, not the first line of the block. The first line
            # is usually short and innocent, and quoting it sends the reader looking in the wrong
            # place for a problem that is three lines down.
            culprit = max(
                content.split(), key=lambda word: text_width_points(word, size), default=""
            )
            complaints.append(
                f"a caption is {drawn - width_px:.0f} px wider than the figure: {culprit[:60]!r} "
                "is one word and cannot be wrapped"
            )
    return complaints
