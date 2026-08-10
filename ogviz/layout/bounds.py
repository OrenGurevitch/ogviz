"""Whether a label stays inside the thing that is supposed to contain it.

Three containment questions, easy to conflate and each with its own failure:

  text vs the CANVAS    a label running off the page. Cropped when the figure is saved plainly,
                        and with `bbox_inches="tight"` not cropped at all — the page GROWS to fit
                        it, so a 7x4 figure writes a 822x352 image and a set of figures meant to
                        share a size no longer does. Measured: both, on the same label.
  text vs the CANVAS,   a caption or title too long to fit however it is placed
  by width
  text vs its OWN PANEL a label sitting exactly where it belongs and reaching across the panel
                        next door. Every position-based check passes, because position was never
                        the problem.

`clipped_artists` answers a fourth, different question — LINES escaping their AXES — and answers
it for lines only, which is right: matplotlib clips `Line2D` by default and never clips `Text`, so
the line vanishes and the label stays. The consequence is that text is the one thing that gate
cannot see, and a label ran off a saved page with nothing raised.

Measured against `fig.bbox`, never `fig.canvas.get_width_height()`. The two agree until
something changes the dpi and then disagree, and the disagreement reports a label as clipped "in
819x500" while it sits comfortably inside the figure — enough to block a generator from rendering
at all. A window extent is in display pixels; the only bbox in the same units is the figure's own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ogviz.layout.collision import quoted
from ogviz.layout.frame import is_color_scale
from ogviz.layout.render import ensure_rendered

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

EDGE_TOLERANCE_PX = 1.0  # rendering rounds to the pixel grid


def panel_text(ax: Axes, *, ticks: bool = False, legend: bool = False) -> Iterator[Text]:
    """Every label an axes owns: its title, its axis labels, its own texts, and what is asked for.

    Tick labels and legend text are OPTIONAL rather than always-or-never, because the two checks
    that walk a figure's text want different sets and each reason is a measurement:

    Tick labels are out for the canvas check. An ordinary panel puts its end ticks a few pixels past
    the canvas — 19 px on a plain `subplots` — because `bbox_inches="tight"` is expected to absorb
    exactly that. Counting them reported four complaints on a figure with nothing wrong. They are IN
    for the spacing check, where two ticks running together is the commonest collision there is.

    `ax.axis("off")` stops the axis being DRAWN but leaves its tick label artists visible with
    positions, so those are read through `drawn_tick_labels` — a table on a bare axes otherwise
    collides with every tick it never shows.

    This existed twice with different contents until 2026-07-31, once here and once in `overlap`,
    and the sets disagreed about ticks, legends and invisible axes. Two walkers means a check
    silently covers a different figure from its neighbour, with each exclusion documented in the
    file that does not apply it.
    """
    from ogviz.layout.overlap import drawn_tick_labels

    yield ax.title
    yield ax.xaxis.label
    yield ax.yaxis.label
    yield from ax.texts
    if ticks and ax.axison:
        yield from drawn_tick_labels(ax)
    if legend:
        drawn = ax.get_legend()
        if drawn is not None:
            yield from drawn.get_texts()


def figure_text(
    fig: Figure, *, ticks: bool = False, legend: bool = True
) -> Iterator[tuple[Text, Axes | None]]:
    """Every visible label in the figure, paired with the axes that owns it, or None.

    Legend text is IN by default: a legend that runs off the page is as cropped as any other label,
    and it was outside the only walker that asked.
    """
    for text in fig.texts:
        if text.get_visible() and text.get_text().strip():
            yield text, None
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for text in panel_text(ax, ticks=ticks, legend=legend):
            if text.get_visible() and text.get_text().strip():
                yield text, ax


def _panel_name(fig: Figure, ax: Axes, text: Text) -> str:
    """Which panel, said in the way a reader can act on.

    "wider than the panel it belongs to" leaves someone with a six-panel mosaic grepping for the
    string, and a label repeated across panels is ambiguous outright. A title identifies a panel to
    a person; an index identifies it to anyone counting in reading order.

    The title is NOT used when the offending label is that title, which is the commonest case of
    this complaint: naming the panel by the string already being quoted produces "X is wider than
    its panel ('X')", which tells a reader nothing they did not have.

    Neither is an INDEX used on a single-panel figure. "panel 0 of 1 (in reading order)" is three
    facts a reader already has and one — "of 1" — that says the identification was never in doubt.
    """
    title = ax.get_title().strip()
    if title and text is not ax.title:
        return f"its panel ({quoted(title)!r})"
    if len(fig.axes) == 1:
        return "its panel"
    try:
        return f"panel {fig.axes.index(ax)} of {len(fig.axes)} (in reading order)"
    except ValueError:  # an axes not on this figure; nothing useful to say
        return "the panel it belongs to"


def _overflow_consequence(fig: Figure) -> str:
    """What a too-wide label actually collides with, which depends on having a neighbour.

    "so it reaches across the one beside it" was said unconditionally, and on a single-panel figure
    there is nothing beside it — the complaint asserted a collision that could not happen and sent
    a reader looking for it. What happens there instead is that the label runs into the margin, and
    off the page if it runs far enough, which is a different edit.
    """
    if len(fig.axes) == 1:
        return ", so it runs into the margin"
    return ", so it reaches across the one beside it"


def _rotation_hint(text: Text) -> str:
    """Which way to edit, for a label whose width is its line COUNT.

    A rotated label — `set_label` on a vertical colourbar is the usual source — is reported as too
    wide, correctly. The natural remedy is to reflow it onto more lines, and for rotated text that
    makes the overflow LARGER, because line count runs along the width. The complaint is true and
    reads as advice to do the wrong thing, so it says which direction helps.
    """
    rotation = float(text.get_rotation()) % 180.0
    if 45.0 < rotation < 135.0:
        return " — it is ROTATED, so adding lines widens it further; shorten it instead"
    return ""


def text_off_canvas(fig: Figure) -> list[str]:
    """Any label whose ink leaves the page.

    The defect `clipped_artists` structurally cannot see, since it tests lines against their
    axes and matplotlib never clips text at all.
    """
    ensure_rendered(fig)
    page = fig.bbox
    complaints: list[str] = []
    for text, _owner in figure_text(fig):
        box = text.get_window_extent()
        over = max(
            box.x0 * -1 + page.x0,
            box.x1 - page.x1,
            box.y0 * -1 + page.y0,
            box.y1 - page.y1,
        )
        if over > EDGE_TOLERANCE_PX:
            complaints.append(
                f"{quoted(text.get_text())!r} runs {over:.0f} px off the page — it is cropped when "
                "the figure is saved plainly, and grows the page when it is saved tight"
            )
    return complaints


def text_wider_than_its_panel(fig: Figure) -> list[str]:
    """Any label an axes owns that is wider than the axes itself.

    A panel's own title or annotation reaching past its panel is legible and lands on its
    neighbour — a 356 px sub-line crossed the panel next door in a real figure while every
    position-based check passed.

    """
    ensure_rendered(fig)
    complaints: list[str] = []
    for ax in fig.axes:
        # A colour scale is a key, not a panel: an 18 px strip whose label is drawn beside it on
        # purpose. Measuring that label against the strip reports every colourbar ever drawn.
        if not ax.get_visible() or is_color_scale(ax):
            continue
        panel = ax.get_window_extent()
        for text in panel_text(ax):
            if not text.get_visible() or not text.get_text().strip():
                continue
            width = float(text.get_window_extent().width)
            if width > panel.width + EDGE_TOLERANCE_PX:
                complaints.append(
                    f"{quoted(text.get_text())!r} is {width - panel.width:.0f} px wider than "
                    f"{_panel_name(fig, ax, text)}"
                    + _overflow_consequence(fig)
                    + _rotation_hint(text)
                )
    return complaints
