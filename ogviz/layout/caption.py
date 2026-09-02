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

from ogviz import units
from ogviz.layout.bounds import figure_text
from ogviz.layout.panels import text_width_points, wrap_to_width
from ogviz.layout.render import ensure_rendered
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
    ensure_rendered(fig)
    return float(text.get_window_extent().width)


def _fit_within(fig: Figure, text: Text, body: str, size: float, limit_px: float) -> bool:
    """Re-wrap until the DRAWN text fits `limit_px`. Returns whether it does.

    The loop exists because the wrap and the renderer do not have to agree. `wrap_to_width`
    measures glyph outlines, which is exact for the font it was given — and the figure may be
    rendered with a different one, or hold a token with no break in it. Measuring what was actually
    drawn is the only check that cannot be argued with.
    """
    target_pt = units.to_points(limit_px, fig=fig)
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
) -> list[str]:
    """Put `heading` above the figure and `note` below it, both wrapped to the figure width.

    Space is reserved by moving the axes, so neither block can land on a panel: the note pushes the
    subplot area up by exactly the height it needs, and the heading pushes it down. Both are
    measured after drawing and re-wrapped if the render disagrees with the wrap.

    Either may be left out. With neither, this does nothing.

    RETURNS WHAT IT COULD NOT DO, empty when there was nothing. Two things can go wrong here and
    both used to be silent at this call:

    - the text cannot be narrowed to the canvas. `_fit_within` has always returned whether it
      managed it and both call sites threw the answer away. The gate still catches it —
      `overflowing_text` is the last line of defence and fails the build — but a caller assembling
      a figure has no way to ask before saving, which is the whole shape of a returned complaint;
    - the figure has a LAYOUT ENGINE. `subplots_adjust` is refused outright under `constrained` or
      `tight` layout: matplotlib warns and does not move the axes, so the caption reserves nothing
      and lands on the panels. Measured on a `layout="constrained"` figure — the note's own
      `bottom` never moves. The consequence is caught downstream by `text_over_data`, one step
      later and in words about the note rather than about the layout.

    A refusal is NOT marked on the figure: `layout_refused` is `fit_under_header`'s tag and the
    gate fails a figure carrying it, so writing it here would newly reject every figure that pairs
    a managed layout with a caption. Reported to the caller, checked by the gate on its
    consequence.
    """
    if note is None and heading is None:
        return []
    unreserved: list[str] = []
    managed = fig.get_layout_engine() is not None
    if managed:
        unreserved.append(
            f"the figure's {type(fig.get_layout_engine()).__name__} refuses subplots_adjust, so "
            "no room was reserved for the caption"
        )
    width_px = fig.get_figwidth() * fig.dpi
    height_px = fig.get_figheight() * fig.dpi
    limit_px = width_px * (1.0 - 2.0 * margin)
    available_pt = units.to_points(limit_px, fig=fig)

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
        if not _fit_within(fig, drawn, heading, heading_size, limit_px):
            unreserved.append("the heading is wider than the canvas and cannot be wrapped narrower")
        # `_fit_within` may have re-wrapped, so the height has to be measured from a fresh render.
        # This read `used = _rendered_width_px(...) and float(...height)`, using `and` to sequence a
        # draw before a measurement — which quietly means "if the rendered WIDTH is 0.0, reserve the
        # width instead of the height". It has never fired, because a drawn heading has a width; it
        # is one empty string away from reserving nothing and putting the heading on the panels.
        #
        # THE DRAW IS BELT AND BRACES, not the fix it looks like. Measured: `Text.get_window_extent`
        # recomputes its layout from the current string, so an extent read straight after
        # `set_text` is already the new one — which is why the note branch below has never needed
        # a draw of its own and must not be given one to match. Kept because a figure this size
        # redraws in milliseconds and the cost of being wrong here is a header on the panels.
        fig.canvas.draw()
        used = float(drawn.get_window_extent().height)
        if not managed:
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
        if not _fit_within(fig, drawn, note, note_size, limit_px):
            unreserved.append("the note is wider than the canvas and cannot be wrapped narrower")
        used = float(drawn.get_window_extent().height)
        if not managed:
            fig.subplots_adjust(bottom=max(0.06, (used + 0.030 * height_px) / height_px))
    return unreserved


def longest_unbreakable(text: str, size: float) -> float:
    """Width in points of the widest single word — what wrapping can never get below."""
    return max((text_width_points(word, size) for word in text.split()), default=0.0)


def overflowing_text(fig: Figure) -> list[str]:
    """Any text wider than the canvas it sits on, wherever it lives.

    The last line of defence for the caption's width guarantee, and it applies to every string, not
    only captions: a title long enough to run off the page fails here too. Reported against the
    CANVAS rather than against a margin, so it fires only on genuine overflow and not on a
    deliberately full-bleed heading.

    It read `fig.texts` alone until an axes title running past its panel went unreported. A title
    set with `ax.set_title` is an AXES-level artist and never appears in `fig.texts`, so the name
    promised a check the function did not perform, and a consumer wrote its own `titles_fit`.
    """
    ensure_rendered(fig)
    width_px = fig.get_figwidth() * fig.dpi
    complaints: list[str] = []
    for text, _owner in figure_text(fig):
        content = text.get_text().strip()
        if not content or not text.get_visible():
            continue
        size = float(text.get_fontsize())
        drawn = float(text.get_window_extent().width)
        if drawn > width_px + OVERFLOW_TOLERANCE_PX:
            # TWO DIFFERENT PROBLEMS, and this reported both as the second one. Naming the widest
            # word rather than the first line is right — the first line is usually short and
            # innocent, and quoting it sends the reader three lines away from the fault. But
            # "is one word and cannot be wrapped" was said whatever the case, so a long TITLE that
            # simply had not been wrapped was reported as an unbreakable word: measured on a 4 in
            # panel, a 69-character title came back as "'Autonomic' is one word and cannot be
            # wrapped", which is false — `wrap_to_panel` breaks that title into three lines
            # happily. A caller acting on it shortens a word that was never the problem.
            #
            # So the WORD is measured against the canvas too, and it decides which of the two
            # sentences is true.
            culprit = max(
                content.split(), key=lambda word: text_width_points(word, size), default=""
            )
            widest_word_px = text_width_points(culprit, size) * units.px_per_point(fig)
            over = drawn - width_px
            if widest_word_px > width_px:
                complaints.append(
                    f"a label is {over:.0f} px wider than the figure: {culprit[:60]!r} is one word "
                    "and cannot be wrapped, so it needs shorter wording or smaller type"
                )
            else:
                complaints.append(
                    f"a label is {over:.0f} px wider than the figure and has not been wrapped — "
                    f"its longest word ({culprit[:40]!r}) does fit, so `wrap_to_panel` or "
                    "`wrap_to_width` will place it"
                )
    return complaints
