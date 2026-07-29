"""The bar panel, with the value labels placed where they cannot be buried.

Two projects grew this independently and each solved half of it. One put
the label above the whisker cap in a page-coloured box, so the label masks the reference line it
crosses instead of colliding with it. The other makes placement
sign-aware, because a negative bar grows downward and a label placed above it lands inside the
bar. `bar_panel` does both, for one series or several.

Errors accept three shapes per series: `None`, a length-N array (symmetric), or a 2xN array
(lower, upper) for an asymmetric CI. The label always clears the whisker cap, so a caller cannot
produce a label sitting on top of its own error bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.patches import FancyBboxPatch

from ogviz.layout.frame import hairline_grid
from ogviz.layout.panels import text_width_points
from ogviz.layout.ticks import typeset
from ogviz.orientation import (
    category_limits,
    category_tick_labels,
    category_ticks,
    constant_value_line,
    is_vertical,
    place_many,
    stamp_orientation,
    value_span,
    value_transform,
)
from ogviz.theme import (
    INK,
    KNOCKOUT_PAD,
    MUTED_INK,
    REFERENCE,
    VALUE_LABEL_SIZE,
    page_color,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from ogviz.orientation import Orientation

BAR_WIDTH = 0.62  # one series; grouped series divide this between them
BAR_ALPHA = 0.85
ERROR_CAPSIZE = 4.0
ERROR_LINEWIDTH = 1.4
LABEL_PAD_FRACTION = 0.02  # of the value span, between the whisker cap and the label
CATEGORY_MARGIN = 0.22  # slack past the outermost bar on the value axis
HIGHLIGHT_FILL = "#EFEDE4"  # the shaded column behind a highlighted category
BAR_ROUNDING = 0.05  # corner radius of a rounded bar top, as a fraction of the tallest bar


def _rounded_bars(
    ax: Axes,
    positions: NDArray[np.float64],
    values: NDArray[np.float64],
    width: float,
    entry: Series,
    *,
    span_hint: float,
) -> None:
    """Bars with softened tops. matplotlib has no rounded bar, so each is a FancyBboxPatch."""
    colors = [entry.color] * len(values) if isinstance(entry.color, str) else list(entry.color)
    radius = BAR_ROUNDING * max(span_hint, 1e-9)
    for index, (position, value, color) in enumerate(zip(positions, values, colors, strict=True)):
        ax.add_patch(
            FancyBboxPatch(
                (position - width / 2, 0.0),
                width,
                value,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color,
                edgecolor="none",
                mutation_aspect=1,
                zorder=Z_BAR,
                # Only the first patch carries the label: one artist per BAR would put the series
                # in the legend once per category.
                label=entry.label if index == 0 else "_nolegend_",
            )
        )


Z_BAR = 3
Z_ERROR = 4
Z_LABEL = 6
# A bar grows FROM the category axis, so its base sits exactly on that spine. matplotlib draws a
# spine at zorder 2.5, under the bars, and the line then survives only in the gaps between them —
# it reads as a broken axis. The axis is a boundary the bars stand on; it belongs on top of them.
Z_BASELINE = Z_ERROR + 0.5
# A threshold is drawn OVER the bars. It exists to be read against them, so burying it behind them
# leaves it visible only in the gaps — the same defect as bars covering the category axis, and it
# costs the reader the one comparison the line was added to make. Under the value labels, which
# knock it out where they cross.
Z_REFERENCE = Z_ERROR + 0.75


@dataclass(frozen=True)
class Series:
    """One bar series. `color` may be a single colour or one per category."""

    label: str
    values: ArrayLike
    color: str | Sequence[str]
    errors: ArrayLike | None = None


def _error_pairs(values: NDArray[np.float64], errors: ArrayLike | None) -> NDArray[np.float64]:
    """(2, N) lower/upper error magnitudes, whatever shape the caller passed."""
    if errors is None:
        return np.zeros((2, len(values)), dtype=float)
    array = np.asarray(errors, dtype=float)
    if array.ndim == 1:
        assert len(array) == len(values), f"errors {len(array)} != values {len(values)}"
        return np.vstack([array, array])
    assert array.shape == (2, len(values)), f"asymmetric errors must be (2, {len(values)})"
    return array


def _auto_decimals(values: NDArray[np.float64]) -> int:
    largest = max((abs(v) for v in values if math.isfinite(v) and v != 0), default=1.0)
    return int(min(max(2 - math.floor(math.log10(largest)), 0), 6))


def _format_of(values: NDArray[np.float64], given: str | None) -> str:
    return given if given is not None else default_value_format(values)


def default_value_format(values: NDArray[np.float64]) -> str:
    """`+0.30` where the series has negatives, `0.30` where it does not.

    A leading plus earns its place only when some other bar carries a minus: the pair then reads as
    two directions. On a series that is positive throughout it is decoration on every label.
    """
    finite = np.asarray(values, dtype=float)
    signed = "+" if bool(np.any(finite[np.isfinite(finite)] < 0.0)) else ""
    return f"{{:{signed}.{_auto_decimals(values)}f}}"


def value_labels(
    ax: Axes,
    positions: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    errors: NDArray[np.float64] | None = None,
    value_format: str | None = None,
    halo: bool = True,
    emphasis: int | None = None,
    knockout_colors: Sequence[str] | None = None,
    orientation: Orientation = "vertical",
    slot_points: float | None = None,
) -> None:
    """Print each bar's value beyond its free end, clear of the whisker cap.

    A negative bar grows downward, so its label goes below it — placing every label above would
    bury the negative ones inside their own bars.

    The knockout behind the text is an opaque box in the page colour, not a stroke around the
    glyphs. A stroke follows the glyph contours, so a dashed reference line crossing the label
    still shows through the gaps between and inside the digits — visible in the figure this
    replaced. On a plain page the box is the page and invisible; where something runs behind the
    label it removes it cleanly.
    """
    # A label belongs to one bar and must not spill onto its neighbour: printed over the bar beside
    # it, a number reads as that bar's. Fitted to the slot rather than assumed to fit, because
    # whether it does depends on the font — these labels fitted in Arial and reached 66 px onto the
    # next bar in DejaVu, which is the wider font the figures are checked against.
    fitted_size = VALUE_LABEL_SIZE
    if slot_points is not None:
        spelling = _format_of(values, value_format)
        widest = max(
            text_width_points(typeset(spelling.format(value)), VALUE_LABEL_SIZE) for value in values
        )
        if widest > slot_points:
            fitted_size = max(VALUE_LABEL_SIZE * slot_points / widest, VALUE_LABEL_SIZE * 0.6)
    low_limit, high_limit = value_span(ax, orientation)
    span = float(high_limit - low_limit)
    caps = _error_pairs(values, errors)
    upright = is_vertical(orientation)
    if value_format is None:
        value_format = default_value_format(values)
    for index, (position, value, low, high) in enumerate(
        zip(positions, values, caps[0], caps[1], strict=True)
    ):
        beyond = value < 0.0  # a negative bar grows away from zero the other way
        end = value - low if beyond else value + high
        end += (-1.0 if beyond else 1.0) * span * LABEL_PAD_FRACTION
        horizontal, vertical = place_many(orientation, float(position), end)
        drawn = ax.text(
            horizontal,
            vertical,
            typeset(value_format.format(value)),
            ha=("center" if upright else ("right" if beyond else "left")),
            va=(("top" if beyond else "bottom") if upright else "center"),
            fontsize=fitted_size,
            fontweight="bold" if emphasis is None or index == emphasis else "normal",
            color=INK if emphasis is None or index == emphasis else MUTED_INK,
            zorder=Z_LABEL,
            bbox=(
                {
                    "facecolor": (knockout_colors[index] if knockout_colors else page_color()),
                    "edgecolor": "none",
                    "pad": KNOCKOUT_PAD,
                    "boxstyle": "square",
                }
                if halo
                else None
            ),
        )
        # Placed against its own bar and whisker on purpose, a measured pad beyond the free end.
        # `test_label_clears_the_whisker_cap` owns that clearance; the general "is this label on
        # the data" check must not drag the label away from the bar it is labelling.
        drawn.ogviz_anchored = True  # type: ignore[attr-defined]


def reference_line(
    ax: Axes,
    value: float,
    label: str,
    *,
    orientation: Orientation = "vertical",
    label_side: Literal["left", "right"] = "left",
):
    """A dashed comparison level, labelled at whichever end the bars leave room.

    The side is given rather than chosen, because it cannot be chosen here: `bar_panel` draws its
    value labels LAST, after the limits have settled, so at this moment there is nothing to test a
    collision against. A panel whose bars ascend to the right wants `label_side="right"`, and the
    overlap check will say so if the wrong one is picked.
    """
    threshold = constant_value_line(
        ax, orientation, value, ls="--", color=MUTED_INK, lw=1.4, zorder=Z_REFERENCE
    )
    threshold.ogviz_reference = True  # type: ignore[attr-defined]  # must stay readable
    at_left = label_side == "left"
    along, across = place_many(orientation, 0.012 if at_left else 0.988, value)
    drawn = ax.text(
        along,
        across,
        label,
        transform=value_transform(ax, orientation),
        ha="left" if at_left else "right",
        va="bottom",
        fontsize=VALUE_LABEL_SIZE * 0.8,
        color=MUTED_INK,
        zorder=2,
    )
    # This label names the line it sits on; moving it off that line is the opposite of the fix. It
    # is excused against THAT line and nothing else — it slides freely along it, so a bar it lands
    # on is a real defect and has to be reported.
    drawn.ogviz_anchored = True  # type: ignore[attr-defined]
    drawn.ogviz_anchor = threshold  # type: ignore[attr-defined]
    return drawn


LABEL_SLOTS = 24  # candidate positions along a threshold, left to right


def _slot_points(ax: Axes, each: float, orientation: Orientation) -> float:
    """How wide one bar's slot is, in points — the room a label printed on it may occupy."""
    along = ax.transData.transform((each, 0))[0] - ax.transData.transform((0, 0))[0]
    if not is_vertical(orientation):
        along = ax.transData.transform((0, each))[1] - ax.transData.transform((0, 0))[1]
    return abs(float(along)) / ax.figure.dpi * 72.0


def _slide_clear(ax: Axes, label) -> None:
    """Move a threshold's label along its own line until it sits on nothing.

    It can only be done here, at the end. The label is placed when the line is drawn, and at that
    moment the bars have no final heights and the value labels do not exist — so there is nothing
    to test against. By now everything else is on the page.

    The label slides along the line and never off it: that is the axis it is free on, and the line
    is the thing it names. Both sides of the line are tried, because a threshold above a short bar
    has room underneath it that a threshold below a tall one does not.
    """
    fig = ax.figure
    fig.canvas.draw()
    anchor = getattr(label, "ogviz_anchor", None)
    obstacles = [
        artist
        for artist in [*ax.patches, *ax.texts, *ax.lines, *ax.collections]
        if artist is not label and artist is not anchor
    ]
    boxes = [artist.get_window_extent() for artist in obstacles]
    original = label.get_position()
    for vertical in ("bottom", "top"):
        for slot in range(LABEL_SLOTS):
            fraction = 0.012 + slot * (0.976 / max(LABEL_SLOTS - 1, 1))
            label.set_position((fraction, original[1]))
            label.set_va(vertical)
            label.set_ha("left" if fraction < 0.5 else "right")
            fig.canvas.draw()
            if not any(label.get_window_extent().overlaps(box) for box in boxes):
                return
    # Nowhere along the line is clear — a crowded panel, which is normal rather than exceptional.
    # The label goes just outside the axes, level with its own line. It is still unambiguously that
    # line's label, it is legible, and it is the one place in a full panel guaranteed to be empty.
    # `save` writes with a tight bounding box, so the margin it needs comes with it.
    label.set_position((1.012, original[1]))
    label.set_ha("left")
    label.set_va("center")
    fig.canvas.draw()


def bar_panel(
    ax: Axes,
    series: Sequence[Series],
    categories: Sequence[str],
    *,
    width: float = BAR_WIDTH,
    value_format: str | None = None,
    show_values: bool = True,
    reference: tuple[float, str] | None = None,
    reference_side: Literal["left", "right"] = "left",
    grid: bool = True,
    reference_band: tuple[float, float, str] | None = None,
    highlight: int | None = None,
    emphasis: int | None = None,
    rounded: bool = False,
    orientation: Orientation = "vertical",
) -> None:
    """Draw one or several bar series over shared categories, with sign-aware value labels.

    `series` carries its own colour and errors; `categories` labels the x axis. Grouped series
    divide `width` between them, so two series stay inside the space one series would occupy.
    `reference` is (value, label) for a dashed comparison level, and `reference_band` is
    (low, high, label) for a shaded one — a published range has width, and drawing it as a single
    line claims a precision the source did not have.

    `highlight` shades one category's column, to say which one the figure is about without
    claiming it won. `emphasis` prints one bar's value in bold — the ranking, where the whiskers
    carry the uncertainty. `rounded` softens the bar tops.

    `orientation="horizontal"` draws the same panel with the categories down the left, which is
    what a long category name needs — the alternative is rotated tick labels that collide.
    """
    assert series, "bar_panel needs at least one series"
    assert len(categories), "bar_panel needs at least one category"
    stamp_orientation(ax, orientation)
    upright = is_vertical(orientation)
    count = len(series)
    centres = np.arange(len(categories), dtype=float)
    each = width / count

    for index, entry in enumerate(series):
        values = np.asarray(entry.values, dtype=float)
        assert len(values) == len(categories), (
            f"series {entry.label!r} has {len(values)} values for {len(categories)} categories"
        )
        if not isinstance(entry.color, str):
            assert len(entry.color) == len(values), (
                f"series {entry.label!r} has {len(entry.color)} colours for {len(values)} bars; "
                "pass one colour, or one per bar."
            )
        missing = int(np.count_nonzero(~np.isfinite(values)))
        assert not missing, (
            f"series {entry.label!r} has {missing} non-finite value(s) of {len(values)}. "
            "A non-finite bar draws nothing and leaves a gap that reads as a zero — drop or "
            "impute them in the project, where the choice is visible."
        )
        offset = (index - (count - 1) / 2) * each
        positions = centres + offset
        caps = _error_pairs(values, entry.errors)
        # Explicit branches rather than a **{name: value} splat: the keyword genuinely differs
        # between the two calls, and a dynamic key defeats the type checker and the reader alike.
        shared = {
            "color": entry.color,
            "alpha": BAR_ALPHA,
            "edgecolor": "none",
            "label": entry.label,
            "zorder": Z_BAR,
        }
        if rounded and upright:
            _rounded_bars(
                ax, positions, values, each * 0.92, entry, span_hint=float(np.max(np.abs(values)))
            )
        elif upright:
            ax.bar(positions, values, width=each * 0.92, **shared)  # type: ignore[arg-type]
        else:
            ax.barh(positions, values, height=each * 0.92, **shared)  # type: ignore[arg-type]
        if entry.errors is not None:
            error_horizontal, error_vertical = place_many(orientation, positions, values)
            ax.errorbar(
                error_horizontal,
                error_vertical,
                yerr=caps if upright else None,
                xerr=None if upright else caps,
                fmt="none",
                ecolor=INK,
                elinewidth=ERROR_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                capthick=ERROR_LINEWIDTH,
                zorder=Z_ERROR,
            )

    category_ticks(ax, orientation)(centres)
    category_tick_labels(ax, orientation)(list(categories))
    category_limits(ax, orientation)(centres[0] - 0.5 - width / 2, centres[-1] + 0.5 + width / 2)
    ax.spines["bottom" if upright else "left"].set_zorder(Z_BASELINE)
    if upright:
        ax.margins(y=CATEGORY_MARGIN)
    else:
        ax.margins(x=CATEGORY_MARGIN)
    if highlight is not None:
        assert 0 <= highlight < len(categories), f"highlight {highlight} is not a category index"
        shade = ax.axvspan if upright else ax.axhspan
        column = shade(highlight - 0.46, highlight + 0.46, color=HIGHLIGHT_FILL, zorder=0)
        column.ogviz_backdrop = True  # type: ignore[attr-defined]
    if grid:
        hairline_grid(ax, axis="y" if upright else "x")
    threshold_label = None
    if reference is not None:
        threshold_label = reference_line(
            ax, *reference, orientation=orientation, label_side=reference_side
        )
    if reference_band is not None:
        low_edge, high_edge, band_label = reference_band
        shade = ax.axhspan if upright else ax.axvspan
        band = shade(low_edge, high_edge, color=REFERENCE, zorder=1)
        band.ogviz_backdrop = True  # type: ignore[attr-defined]
        along, across = place_many(orientation, 0.012, (low_edge + high_edge) / 2)
        banded = ax.text(
            along,
            across,
            band_label,
            transform=value_transform(ax, orientation),
            ha="left",
            va="center",
            fontsize=VALUE_LABEL_SIZE * 0.8,
            color=INK,
            zorder=4,
        )
        banded.ogviz_anchored = True  # type: ignore[attr-defined]  # names its own band

    # Labels last: placement reads ax.get_ylim(), which margins and the reference line both move.
    if show_values:
        # Decided across every series, not per series: one panel showing "+0.30" beside "0.30"
        # reads as two different quantities.
        panel_format = value_format or default_value_format(
            np.concatenate([np.asarray(entry.values, dtype=float) for entry in series])
        )
        for index, entry in enumerate(series):
            values = np.asarray(entry.values, dtype=float)
            offset = (index - (count - 1) / 2) * each
            value_labels(
                ax,
                centres + offset,
                values,
                errors=_error_pairs(values, entry.errors),
                value_format=panel_format,
                orientation=orientation,
                slot_points=_slot_points(ax, each, orientation),
                emphasis=emphasis,
                # A label sitting over the highlighted column must knock out to the SHADE, not to
                # the page, or its box reads as a white patch punched in the shading.
                knockout_colors=[
                    HIGHLIGHT_FILL if position == highlight else page_color()
                    for position in range(len(categories))
                ],
            )
    if threshold_label is not None:
        _slide_clear(ax, threshold_label)
