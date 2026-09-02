"""The furniture around the marks: rules, baselines and the legend pill.

None of it carries a value. All of it decides how hard the marks have to work to be seen, which is
why it is one module rather than a line in each builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ogviz.require import require
from ogviz.theme import GRID, MUTED_INK, PANEL_FILL, TICK_SIZE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.text import Text


# A scale is a key, not a panel: its numbers are set below the axis ticks so it reads as
# subordinate to the marks it describes. MEASURED against the tick size rather than fixed,
# so `font_scale`-style changes to the house type carry through instead of leaving the bar
# at whatever looked right once.
SCALE_TYPE = 0.85


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


def zero_baseline(ax: Axes, *, axis: Literal["x", "y"] = "x") -> None:
    """Heavy ink line at zero on the VALUE axis, for bars that grow from zero rather than the frame.

    `axis` is the CATEGORY axis, as in `baseline` beside it: `"x"` for an upright panel, where the
    rule runs across at y = 0, and `"y"` for a horizontal one, where it runs up at x = 0. It was
    written for the upright case only and drew `axhline` whatever the panel, so on a horizontal bar
    panel the rule the bars are measured from landed across the categories.

    Drawn wherever zero is — matplotlib clips it when zero is out of view, and a signed panel that
    hides zero is the caller's decision, not this function's.
    """
    if axis == "x":
        ax.axhline(0.0, color=MUTED_INK, linewidth=1.4, zorder=4)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set(linewidth=1.0, color=GRID)
    else:
        ax.axvline(0.0, color=MUTED_INK, linewidth=1.4, zorder=4)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set(linewidth=1.0, color=GRID)
    ax.tick_params(axis=axis, length=0.0)


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


CATEGORY_ROW_STEP = 0.075  # of the axes height, between one row of category labels and the next


def label_rows(
    ax: Axes,
    positions: Sequence[float],
    rows: Sequence[Sequence[str]],
    *,
    sizes: Sequence[float] | None = None,
    colors: Sequence[Sequence[str] | str] | None = None,
    weights: Sequence[str] | None = None,
    step: float = CATEGORY_ROW_STEP,
    first_row: float = -0.02,
) -> list[Text]:
    """Stack several rows of labels under the category axis, each row a level of grouping.

    One row of tick labels can only say what each bar IS. A figure often has more to say about the
    same axis — what each bar is, then which model it belongs to, then how much data that model
    saw — and each of those is a level a reader takes in separately.

    A label may be given per position or once for the whole row, so a row naming a group spanning
    several bars is `["trained on 50 subjects"]` at the midpoint rather than the same string
    repeated. Positions and labels must agree in length otherwise.

    Placed in the axes' x-coordinate / figure-fraction transform, so the rows stay put when the
    value axis rescales. A caller writing this by hand reaches for `ax.get_xaxis_transform()` and
    picks the offsets one at a time, which is how three rows come out unevenly spaced.
    """
    # Checked before the loop: a row given ONE label spans the positions and is placed at their
    # midpoint, so zero positions divided by zero rather than saying what was missing.
    require(len(positions) > 0, "label_rows needs at least one position")
    drawn: list[Text] = []
    for index, labels in enumerate(rows):
        require(
            len(labels) in (1, len(positions)),
            f"row {index} has {len(labels)} labels for {len(positions)} positions",
        )
        places = [sum(positions) / len(positions)] if len(labels) == 1 else list(positions)
        row_color = None if colors is None else colors[index]
        for column, (place, label) in enumerate(zip(places, labels, strict=True)):
            drawn.append(
                ax.text(
                    place,
                    first_row - index * step,
                    label,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=None if sizes is None else sizes[index],
                    fontweight="normal" if weights is None else weights[index],
                    color=_row_color(row_color, column),
                )
            )
    return drawn


def _row_color(row_color, column: int) -> str | None:
    """One colour for the whole row, or one per position.

    By the label's INDEX, not by looking its position up in the list. `places.index(place)` returns
    the FIRST match, so two categories sharing a position — which `bar_panel`'s `positions=` exists
    to allow, and which is exactly how a figure sets a non-comparable arm apart — both took the
    first one's colour, and the last row's colour was never used.
    """
    if row_color is None or isinstance(row_color, str):
        return row_color
    return row_color[column]


def color_scale(
    ax: Axes,
    mappable: ScalarMappable,
    *,
    label: str | None = None,
    ticks: Sequence[float] | None = None,
    tick_labels: Sequence[str] | None = None,
    width: float = 0.030,
    pad: float = 0.02,
) -> Colorbar:
    """The key that says what the colours MEAN, set in the house style.

    A colour scale is furniture, like the legend pill: it carries no value of its own and decides
    how hard the marks have to work. matplotlib's default bar is a boxed, black-outlined slab about
    twice this wide with tick marks on it, which on a warm page reads as a second panel competing
    with the one it describes. This is a thin strip with no outline and no tick marks — the numbers
    alone — in the muted ink the axes already use.

    `ticks` names the values worth reading rather than letting matplotlib choose: on a diverging
    scale the three that matter are the two ends and the neutral point, and an automatic locator
    puts five evenly spaced numbers there instead, none of which is the neutral value.

    Returns the `Colorbar` so a caller can do more to it. Its axes is a real axes on the figure, so
    every QC check sees it — which is the point, and was checked: a scale added this way leaves the
    gate clean rather than being excused from it.
    """
    from ogviz.layout.ticks import typeset

    figure = ax.get_figure(root=True)
    require(figure is not None, "the axes must belong to a figure")
    bar = figure.colorbar(mappable, ax=ax, fraction=width, pad=pad, ticks=ticks)  # type: ignore[union-attr]
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0.0, colors=MUTED_INK, labelsize=TICK_SIZE * SCALE_TYPE)
    if tick_labels is not None:
        # Passed through `typeset`, so a negative bound on the scale carries the same minus sign as
        # every other number in the figure. A bar labelled with an ASCII hyphen beside axis ticks
        # using U+2212 is exactly what `one_minus_sign` exists to catch, and it would be catching
        # this package rather than a caller. The caller supplies the WORDING; this owns the glyph.
        bar.set_ticklabels([typeset(text) for text in tick_labels])
    if label:
        bar.set_label(label, color=MUTED_INK, size=TICK_SIZE * SCALE_TYPE)
    return bar


# matplotlib's own name for the axes a colourbar lives on. Public in the sense that matters — it is
# what `Colorbar` sets and what every backend reads — where `cax._colorbar` is the private half of
# the same fact. Both are checked, because a figure this package did not draw has only the first.
COLOR_SCALE_LABEL = "<colorbar>"


def is_color_scale(ax: Axes) -> bool:
    """Whether this axes is a colour scale rather than a panel.

    A scale is a KEY. It is a strip a few pixels wide with its label drawn BESIDE it, and the checks
    that reason about panels have to know the difference: `text_wider_than_its_panel` measured a
    colourbar's own label against the strip it labels and reported it as 3 px too wide — true, and
    true of every colourbar ever drawn, because the label cannot fit inside a 18 px strip and is not
    meant to. That is the gate crying wolf, and it appeared the moment `effect_heatmap` and
    `spectrogram` started drawing scales by default.

    Read from the artist rather than tracked in this module, so it is right for a figure this
    package never touched — a project running `python -m ogviz.qc` over its own work gets the same
    answer as one built with `color_scale`.
    """
    return ax.get_label() == COLOR_SCALE_LABEL or hasattr(ax, "_colorbar")
