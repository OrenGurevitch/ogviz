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

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.patches import FancyBboxPatch

from ogviz.layout.frame import hairline_grid
from ogviz.layout.panels import text_width_points
from ogviz.layout.ticks import typeset
from ogviz.marks import (
    Z_ERROR,
    _draw_error_bars,
)
from ogviz.orientation import (
    category_limits,
    category_tick_labels,
    category_ticks,
    is_vertical,
    place_many,
    stamp_orientation,
    value_span,
)
from ogviz.panels.reference import reference_line, slide_label_clear
from ogviz.tags import mark
from ogviz.theme import (
    INK,
    KNOCKOUT_PAD,
    MUTED_INK,
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
LABEL_PAD_FRACTION = 0.02  # of the value span, between the whisker cap and the label
CATEGORY_MARGIN = 0.22  # slack past the outermost bar on the value axis
HIGHLIGHT_FILL = "#EFEDE4"  # the shaded column behind a highlighted category
BAR_ROUNDING = 0.16  # corner radius, as a fraction of the BAR'S OWN WIDTH


def _data_aspect(ax: Axes, span_hint: float) -> float:
    """Data units of y per data unit of x, as the page sees them.

    `FancyBboxPatch` applies `rounding_size` in x and scales y by `mutation_aspect`, so this is what
    turns a radius stated in x units into a corner that is round rather than an ellipse.
    """
    box = ax.get_window_extent()
    if not box.width or not box.height:
        return 1.0
    low, high = ax.get_ylim()
    y_span = abs(high - low) or max(span_hint, 1e-9)
    x_low, x_high = ax.get_xlim()
    x_span = abs(x_high - x_low) or 1.0
    return float((y_span / box.height) / (x_span / box.width))


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
    # The radius is a fraction of the bar's WIDTH, and the patch is told the data aspect so the
    # corner comes out round on the page. It used to be a fraction of the tallest VALUE, which is
    # only sensible when the value axis happens to be order-1: `rounding_size` applies to both axes,
    # so on a counts axis the corner measured 394,460% of the bar's own width — a lozenge rather
    # than a bar. The look was arithmetic rather than taste.
    radius = BAR_ROUNDING * width
    aspect = _data_aspect(ax, span_hint)
    for index, (position, value, color) in enumerate(zip(positions, values, colors, strict=True)):
        ax.add_patch(
            FancyBboxPatch(
                (position - width / 2, 0.0),
                width,
                value,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color,
                edgecolor="none",
                mutation_aspect=aspect,
                zorder=Z_BAR,
                # Only the first patch carries the label: one artist per BAR would put the series
                # in the legend once per category.
                label=entry.label if index == 0 else "_nolegend_",
            )
        )


Z_BAR = 3
Z_LABEL = 6
# A bar grows FROM the category axis, so its base sits exactly on that spine. matplotlib draws a
# spine at zorder 2.5, under the bars, and the line then survives only in the gaps between them —
# it reads as a broken axis. The axis is a boundary the bars stand on; it belongs on top of them.
Z_BASELINE = Z_ERROR + 0.5
# A threshold is drawn OVER the bars. It exists to be read against them, so burying it behind them
# leaves it visible only in the gaps — the same defect as bars covering the category axis, and it
# costs the reader the one comparison the line was added to make. Under the value labels, which
# knock it out where they cross.


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
    # Grouped from 1000 up, the house default. Below that the comma changes nothing, so it costs a
    # small series nothing and saves a large one from being counted digit by digit.
    return f"{{:{signed},.{_auto_decimals(values)}f}}"


STROKE_HALO_WIDTH = 2.6  # points of page colour drawn around the glyphs, not around their box


def _knockout_style(halo: object) -> str:
    """Which knockout a caller asked for, accepting the old boolean.

    A BOX is the default and is right over a gridline: it is opaque, square and cheap to read. It is
    wrong over a shaded region — a rectangle punches a visible hole in a reference band wherever a
    label crosses it, and the band stops reading as continuous. A STROKE follows the glyph outlines,
    so it clears the digits and lets the band show through the gaps inside and between them.

    The gate has accepted a stroke as a knockout since 2026-08-01; this is the other half, which the
    package could not draw. A house project had to hand-roll its value labels to get one.
    """
    if isinstance(halo, bool):
        return "box" if halo else "none"
    assert halo in ("box", "stroke", "none"), f"unknown knockout {halo!r}"
    return halo


def value_labels(
    ax: Axes,
    positions: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    errors: NDArray[np.float64] | None = None,
    value_format: str | None = None,
    halo: Literal["box", "stroke", "none"] | bool = "box",
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
                if _knockout_style(halo) == "box"
                else None
            ),
            path_effects=(
                [
                    path_effects.withStroke(
                        linewidth=STROKE_HALO_WIDTH,
                        foreground=(knockout_colors[index] if knockout_colors else page_color()),
                    )
                ]
                if _knockout_style(halo) == "stroke"
                else None
            ),
        )
        # Placed against its own bar and whisker on purpose, a measured pad beyond the free end.
        # `test_label_clears_the_whisker_cap` owns that clearance; the general "is this label on
        # the data" check must not drag the label away from the bar it is labelling.
        mark(drawn, "anchored")


def _slot_points(ax: Axes, each: float, orientation: Orientation) -> float:
    """How wide one bar's slot is, in points — the room a label printed on it may occupy."""
    along = ax.transData.transform((each, 0))[0] - ax.transData.transform((0, 0))[0]
    if not is_vertical(orientation):
        along = ax.transData.transform((0, each))[1] - ax.transData.transform((0, 0))[1]
    return abs(float(along)) / ax.figure.dpi * 72.0


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
    positions: Sequence[float] | None = None,
    grid: bool = True,
    highlight: int | tuple[int, int] | None = None,
    emphasis: int | None = None,
    rounded: bool = False,
    orientation: Orientation = "vertical",
) -> None:
    """Draw one or several bar series over shared categories, with sign-aware value labels.

    `series` carries its own colour and errors; `categories` labels the x axis. Grouped series
    divide `width` between them, so two series stay inside the space one series would occupy.
    `reference` is (value, label) for a dashed comparison level.

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
    centres = (
        np.arange(len(categories), dtype=float)
        if positions is None
        else np.asarray(positions, dtype=float)
    )
    assert centres.shape == (len(categories),), (
        f"{centres.size} positions for {len(categories)} categories"
    )
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
        bars_at = centres + offset
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
                ax, bars_at, values, each * 0.92, entry, span_hint=float(np.max(np.abs(values)))
            )
        elif upright:
            ax.bar(bars_at, values, width=each * 0.92, **shared)  # type: ignore[arg-type]
        else:
            ax.barh(bars_at, values, height=each * 0.92, **shared)  # type: ignore[arg-type]
        if entry.errors is not None:
            _draw_error_bars(ax, bars_at, values, caps, orientation=orientation)

    category_ticks(ax, orientation)(centres)
    category_tick_labels(ax, orientation)(list(categories))
    category_limits(ax, orientation)(centres[0] - 0.5 - width / 2, centres[-1] + 0.5 + width / 2)
    ax.spines["bottom" if upright else "left"].set_zorder(Z_BASELINE)
    if upright:
        ax.margins(y=CATEGORY_MARGIN)
    else:
        ax.margins(x=CATEGORY_MARGIN)
    if highlight is not None:
        # One category or a RANGE of them. A range is how a figure says "these belong together and
        # that one does not" — a set of comparable arms beside a reference that is not comparable —
        # and a single index could not say it, so a consumer drew its own `axvspan` instead.
        first, last = (highlight, highlight) if isinstance(highlight, int) else highlight
        assert 0 <= first <= last < len(categories), (
            f"highlight {highlight} is not a category index or range of them"
        )
        shade = ax.axvspan if upright else ax.axhspan
        column = shade(centres[first] - 0.46, centres[last] + 0.46, color=HIGHLIGHT_FILL, zorder=0)
        mark(column, "backdrop")
    if grid:
        hairline_grid(ax, axis="y" if upright else "x")
    threshold_label = None
    if reference is not None:
        threshold_label = reference_line(
            ax, *reference, orientation=orientation, label_side=reference_side
        )

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
        slide_label_clear(ax, threshold_label)
