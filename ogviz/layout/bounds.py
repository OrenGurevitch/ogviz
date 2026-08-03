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


def text_off_canvas(fig: Figure) -> list[str]:
    """Any label whose ink leaves the page.

    The defect `clipped_artists` structurally cannot see, since it tests lines against their
    axes and matplotlib never clips text at all.
    """
    fig.canvas.draw()
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
    fig.canvas.draw()
    complaints: list[str] = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        panel = ax.get_window_extent()
        for text in panel_text(ax):
            if not text.get_visible() or not text.get_text().strip():
                continue
            width = float(text.get_window_extent().width)
            if width > panel.width + EDGE_TOLERANCE_PX:
                complaints.append(
                    f"{quoted(text.get_text())!r} is {width - panel.width:.0f} px wider than the "
                    "panel it belongs to, so it reaches across the one beside it"
                )
    return complaints
