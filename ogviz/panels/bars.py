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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path

from ogviz import units
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
    place,
    place_many,
    stamp_orientation,
    value_span,
)
from ogviz.panels.reference import reference_band, reference_line, slide_label_clear
from ogviz.require import require
from ogviz.tags import mark
from ogviz.theme import (
    INK,
    KNOCKOUT_PAD,
    MUTED_INK,
    VALUE_LABEL_SIZE,
    page_color,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from ogviz.orientation import Orientation

BAR_WIDTH = 0.62  # one series; grouped series divide this between them
BAR_ALPHA = 0.85
LABEL_PAD_FRACTION = 0.02  # of the value span, between the whisker cap and the label
CATEGORY_MARGIN = 0.22  # slack past the outermost bar on the value axis
HIGHLIGHT_FILL = "#EFEDE4"  # the shaded column behind a highlighted category
BAR_ROUNDING = 0.16  # corner radius, as a fraction of the BAR'S OWN WIDTH

# `bar_panel` takes a `reference_band` ARGUMENT, which shadows the function inside its body. The
# alias is how the panel reaches the one implementation without either public name changing.
_draw_reference_band = reference_band


K = 0.5523  # the circle-to-bezier constant, for a quarter turn
_SQUARE_END = [Path.MOVETO, *([Path.LINETO] * 3), Path.CLOSEPOLY]
_ROUND_END = [
    Path.MOVETO,
    Path.LINETO,
    *([Path.CURVE4] * 3),
    Path.LINETO,
    *([Path.CURVE4] * 3),
    Path.LINETO,
    Path.CLOSEPOLY,
]


def rounded_free_end(fraction: float, *, along: Literal["x", "y"] = "y"):
    """A boxstyle rounding the end AWAY from zero and leaving the base square.

    `boxstyle="round"` rounds all four corners, so a rounded bar curved away from its own baseline:
    the foot left a visible gap where the bar should meet zero, and the base read as narrower than
    the bar. A bar encodes its value as the distance from zero, so the one corner that must not be
    softened is the one at zero. The docstring above always said "softened TOPS"; the implementation
    rounded four corners for as long as the option existed.

    A CALLABLE BOXSTYLE rather than a hand-built path, because `FancyBboxPatch` divides y by
    `mutation_aspect` before calling this and multiplies it back afterwards. Inside here a circle is
    a circle, so one radius serves both axes — and the corner stays round when a later
    `tight_layout` resizes the axes, which a path baked in data coordinates would not.

    `along` is which way the bar GROWS, so a horizontal panel gets the same treatment transposed
    rather than nothing at all. And the radius arrives as a FRACTION of the bar's own thickness,
    worked out from the box this is handed: stated as an absolute it would have to be converted
    through `mutation_aspect` by the caller, in the one direction where that scaling applies.

    PUBLIC because drawing your own bars is a legitimate reason not to use `bar_panel` — per-bar
    tints, an outlined leader — and that caller still wants the corner. It was private, and a
    project needing it copied the body instead, with a note saying reaching for a private name was
    the more fragile of the two. The copy had already drifted: it approximates each corner with a
    QUADRATIC, where a quarter circle needs a cubic and the constant below. Which is the argument
    for exporting it rather than for the copier having been careless — a private function that
    others need is a copy that diverges silently.
    """

    def boxstyle(x0: float, y0: float, width: float, height: float, mutation_size: float) -> Path:
        del mutation_size  # the radius is a fraction of the bar, not of the font
        grows_up = along == "y"
        # ONE construction in the bar's own frame: `reach` runs foot -> tip, `across` runs one
        # side of the bar to the other, and `at` puts a point back. Written out a second time for
        # the transpose, the two halves disagreed about which corner was the square one.
        foot, tip = (y0, y0 + height) if grows_up else (x0, x0 + width)
        side, other = (x0, x0 + width) if grows_up else (y0, y0 + height)
        thickness, reach = abs(other - side), abs(tip - foot)

        def at(along_: float, across: float) -> tuple[float, float]:
            return (across, along_) if grows_up else (along_, across)

        r = min(fraction * thickness, thickness / 2.0, reach)
        if r <= 0.0:
            corners = [at(foot, side), at(tip, side), at(tip, other), at(foot, other)]
            return Path([*corners, corners[0]], _SQUARE_END)

        run = r if tip >= foot else -r  # a negative bar grows the other way; its free end is there
        step = r if other >= side else -r
        near, shy = tip - run, tip - run * (1.0 - K)
        pull = step * (1.0 - K)
        return Path(
            [
                at(foot, side),
                at(near, side),
                at(shy, side),
                at(tip, side + pull),
                at(tip, side + step),
                at(tip, other - step),
                at(tip, other - pull),
                at(shy, other),
                at(near, other),
                at(foot, other),
                at(foot, side),
            ],
            _ROUND_END,
        )

    return boxstyle


def _data_aspect(ax: Axes) -> float:
    """Data units of y per data unit of x, as the page sees them.

    `FancyBboxPatch` applies its box style in x and scales y by `mutation_aspect`, so this is what
    turns a radius stated in one axis's units into a corner that is round rather than an ellipse.

    A degenerate axis has no aspect to report and gets 1.0. It used to fall back to the tallest
    value in the series, floored at 1e-9 — so a series of all zeros produced 1e-9 and an aspect
    nine orders of magnitude out. Nothing can be known about the shape of a panel with no extent,
    and guessing is what made the guard worse than its absence.
    """
    box = ax.get_window_extent()
    if not box.width or not box.height:
        return 1.0
    low, high = ax.get_ylim()
    x_low, x_high = ax.get_xlim()
    y_span, x_span = abs(high - low), abs(x_high - x_low)
    if not y_span or not x_span:
        return 1.0
    return float((y_span / box.height) / (x_span / box.width))


class _RoundedBar(FancyBboxPatch):
    """A `FancyBboxPatch` whose data aspect is read from its axes when the path is BUILT.

    `mutation_aspect` was computed once, while the bars were being added — before `bar_panel` set
    the category limits, added its margins, or drew a reference — so it described a panel that no
    longer existed by the time anything was rendered. On an order-1 axis the error was small. On a
    counts axis it was total: the aspect stayed at about 1.6 where the final panel's was in the
    hundreds, the corner's y extent came out at a fraction of a pixel, and a `rounded=True` bar
    drew square. Measured 2026-09-01 on a single bar of 38,000: 0 px of corner against 27 px for
    the same bar at 0.61. The test meant to catch it measured the value label instead (see
    `test_a_rounded_corner_is_the_same_size_on_any_axis`), so the regression shipped.

    `FancyBboxPatch.get_path` calls `get_mutation_aspect()` each time, so overriding it is enough:
    the corner now follows the limits AND a later resize, which is what `rounded_free_end`'s
    docstring already promised.
    """

    def get_mutation_aspect(self) -> float:  # type: ignore[override]
        ax = self.axes
        return _data_aspect(cast("Axes", ax)) if ax is not None else 1.0


def _rounded_bars(
    ax: Axes,
    positions: NDArray[np.float64],
    values: NDArray[np.float64],
    thickness: float,
    entry: Series,
    *,
    orientation: Orientation,
) -> None:
    """Bars with a softened free end. matplotlib has no rounded bar, so each is a FancyBboxPatch.

    The radius is a fraction of the bar's own THICKNESS, and the patch reads the data aspect from
    its axes as it is drawn (`_RoundedBar`) so the corner comes out round on the page. It used to
    be a fraction of the tallest VALUE, which is only sensible when the value axis happens to be
    order-1: the style applies to both axes, so on a counts axis the corner measured 394,460% of the
    bar's own width — a lozenge rather than a bar. The look was arithmetic rather than taste.

    `alpha` is here because the plain `ax.bar` path has always passed `BAR_ALPHA` and this did not,
    so the same hex colour rendered at two different opacities depending on `rounded` — which made
    a shape option quietly into a colour one, and gave the colour-vision check a different pair to
    measure than the one the caller chose.
    """
    colors = [entry.color] * len(values) if isinstance(entry.color, str) else list(entry.color)
    upright = is_vertical(orientation)
    for index, (position, value, color) in enumerate(zip(positions, values, colors, strict=True)):
        corner = place(orientation, float(position) - thickness / 2, 0.0)
        size = place(orientation, thickness, float(value))
        ax.add_patch(
            _RoundedBar(
                corner,
                size[0],
                size[1],
                # A CALLABLE is what matplotlib documents for a custom box and what it accepts;
                # the stubs type this parameter as `BoxStyle | str` only, so the ignore is a gap in
                # them rather than a doubt about the call. Registering a `BoxStyle` subclass instead
                # would mean subclassing `BoxStyle._Base`, which is private and which matplotlib
                # deprecated in favour of exactly this.
                boxstyle=rounded_free_end(BAR_ROUNDING, along="y" if upright else "x"),  # type: ignore[arg-type]
                facecolor=color,
                alpha=BAR_ALPHA,
                edgecolor="none",
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
# The z-order for a threshold — drawn over the bars it is read against, under the value labels that
# knock it out where they cross — was described here for months after the constant itself moved to
# `panels/reference.py` as `Z_REFERENCE`. A comment explaining a line that is not there is worse
# than no comment: it is confidently about the wrong thing, and a reader trusts it.


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
        require(
            len(array) == len(values),
            f"errors {len(array)} != values {len(values)}",
        )
        return np.vstack([array, array])
    require(
        array.shape == (2, len(values)),
        f"asymmetric errors must be (2, {len(values)})",
    )
    return array


def _auto_decimals(values: NDArray[np.float64]) -> int:
    """`layout.ticks.auto_decimals`, asked about a whole SERIES rather than one value.

    The clamp used to be written out a second time here, three lines that had to stay in agreement
    with the ones in `ticks` by nobody's arrangement. What is genuinely different is only which
    number to ask about: a row of labels takes its precision from the largest value in the row, so
    they read as one measurement rather than as several.
    """
    from ogviz.layout.ticks import row_decimals

    return row_decimals(values)


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
    if halo not in ("box", "stroke", "none"):  # written out: this narrows `halo` to a str
        raise AssertionError(f"unknown knockout {halo!r}")
    return str(halo)


def value_labels(
    ax: Axes,
    positions: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    errors: NDArray[np.float64] | None = None,
    value_format: str | None = None,
    halo: Literal["box", "stroke", "none"] | bool = "box",
    emphasis: int | Iterable[int] | None = None,
    knockout_colors: Sequence[str] | None = None,
    orientation: Orientation = "vertical",
    slot_points: float | None = None,
) -> None:
    """Print each bar's value beyond its free end, clear of the whisker cap.

    A negative bar grows downward, so its label goes below it — placing every label above would
    bury the negative ones inside their own bars.

    `halo` is which knockout goes behind the text, and `_knockout_style` is where the choice
    between them is argued. The short of it: `"box"` is the default and is right over a gridline
    or a dashed rule, because a stroke follows the glyph contours and the line shows through the
    gaps inside and between the digits. `"stroke"` is right over a shaded band, where a box punches
    a visible hole. On a plain page the box is the page colour and invisible either way.

    This docstring used to say the knockout "is an opaque box, not a stroke" and give only the
    first half of that argument — written before `halo` existed and left standing after, so the
    paragraph a caller reads first told them not to use the parameter the signature offers.

    `emphasis` is which labels are the ones to read: None means every label is (the default look),
    an index means that one, and an EMPTY collection means none of them — which is not the same
    thing as None and is what a grouped panel needs for the series the emphasis is not in. Passing
    None there instead would have made that whole series bold beside one muted one.
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
    require(
        knockout_colors is None or len(knockout_colors) == len(values),
        f"{len(knockout_colors or ())} knockout colours for {len(values)} labels; "
        "pass one per label, or none.",
    )
    picked: set[int] | None = None
    if emphasis is not None:
        picked = {emphasis} if isinstance(emphasis, int) else set(emphasis)
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
            fontweight="bold" if picked is None or index in picked else "normal",
            color=INK if picked is None or index in picked else MUTED_INK,
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


def _emphasis_within(
    emphasis: int | tuple[int, int] | None, series_index: int
) -> int | tuple[int, ...] | None:
    """`bar_panel`'s emphasis, as the series at `series_index` should be told it.

    An empty tuple rather than None for a series that holds no emphasised bar: None means "every
    label is one to read", so passing it through would have left that whole series bold beside the
    muted one, which is the opposite of what emphasis is for.
    """
    if emphasis is None:
        return None
    if isinstance(emphasis, int):
        return emphasis  # a category, emphasised in every series
    which, category = emphasis
    return (category,) if which == series_index else ()


def _slot_points(ax: Axes, each: float, orientation: Orientation) -> float:
    """How wide one bar's slot is, in points — the room a label printed on it may occupy."""
    figure = ax.get_figure()
    assert figure is not None, "the axes must belong to a figure"
    along = ax.transData.transform((each, 0))[0] - ax.transData.transform((0, 0))[0]
    if not is_vertical(orientation):
        along = ax.transData.transform((0, each))[1] - ax.transData.transform((0, 0))[1]
    return units.to_points(abs(float(along)), fig=figure)


def bar_panel(
    ax: Axes,
    series: Sequence[Series],
    categories: Sequence[str],
    *,
    width: float = BAR_WIDTH,
    value_format: str | None = None,
    show_values: bool = True,
    reference: tuple[float, str] | None = None,
    reference_band: tuple[float, float, str] | None = None,
    reference_side: Literal["left", "right"] = "left",
    positions: Sequence[float] | None = None,
    grid: bool = True,
    highlight: int | tuple[int, int] | None = None,
    emphasis: int | tuple[int, int] | None = None,
    rounded: bool = False,
    orientation: Orientation = "vertical",
) -> None:
    """Draw one or several bar series over shared categories, with sign-aware value labels.

    `series` carries its own colour and errors; `categories` labels the x axis. Grouped series
    divide `width` between them, so two series stay inside the space one series would occupy.
    `reference` is (value, label) for a dashed comparison level, and `reference_band` is
    (low, high, label) for one that is a RANGE — a published agreement interval, a tolerance, an
    acceptance window. A level and a range are different claims and a figure may carry both.

    `highlight` shades one category's column, to say which one the figure is about without
    claiming it won. `rounded` softens each bar's free end, and works either way round.

    `emphasis` prints a value in bold and mutes the rest — the ranking, where the whiskers carry
    the uncertainty. A CATEGORY index emphasises that category in every series, which is the
    reading "look at this one" for a grouped panel; `(series, category)` names a single bar. It
    took a category index only, and passed it into each series unchanged, so on a grouped panel it
    silently emphasised one bar per series while the docstring said "one bar's value".

    `orientation="horizontal"` draws the same panel with the categories down the left, which is
    what a long category name needs — the alternative is rotated tick labels that collide.
    """
    require(
        series,
        "bar_panel needs at least one series",
    )
    require(
        len(categories),
        "bar_panel needs at least one category",
    )
    # The SHAPE of `emphasis` is checked before its values are compared, because the comparison
    # is what a wrong shape crashes in: a list reached `0 <= [0, 1]` and raised a bare TypeError
    # naming int and list, and a one- or three-tuple raised IndexError from inside the message
    # meant to explain it. `bool` is excluded for the reason `widths_of` excludes it — True is an
    # `int`, so `emphasis=True` silently emphasised category 1.
    if isinstance(emphasis, tuple):
        require(
            len(emphasis) == 2,
            f"emphasis {emphasis} is a (series, category) pair or a category index, "
            f"not a {len(emphasis)}-tuple",
        )
        require(
            0 <= emphasis[0] < len(series) and 0 <= emphasis[1] < len(categories),
            f"emphasis {emphasis} is not a (series, category) pair of this panel — "
            f"{len(series)} series, {len(categories)} categories",
        )
    elif emphasis is not None:
        require(
            isinstance(emphasis, int) and not isinstance(emphasis, bool),
            f"emphasis is a category index or a (series, category) pair, got {emphasis!r}",
        )
        require(
            0 <= emphasis < len(categories),
            f"emphasis {emphasis} is not a category index of {len(categories)}",
        )
    stamp_orientation(ax, orientation)
    upright = is_vertical(orientation)
    count = len(series)
    centres = (
        np.arange(len(categories), dtype=float)
        if positions is None
        else np.asarray(positions, dtype=float)
    )
    require(
        centres.shape == (len(categories),),
        f"{centres.size} positions for {len(categories)} categories",
    )
    each = width / count

    for index, entry in enumerate(series):
        values = np.asarray(entry.values, dtype=float)
        require(
            len(values) == len(categories),
            f"series {entry.label!r} has {len(values)} values for {len(categories)} categories",
        )
        if not isinstance(entry.color, str):
            require(
                len(entry.color) == len(values),
                f"series {entry.label!r} has {len(entry.color)} colours for {len(values)} bars; "
                "pass one colour, or one per bar.",
            )
        missing = int(np.count_nonzero(~np.isfinite(values)))
        require(
            not missing,
            f"series {entry.label!r} has {missing} non-finite value(s) of {len(values)}. "
            "A non-finite bar draws nothing and leaves a gap that reads as a zero — drop or "
            "impute them in the project, where the choice is visible.",
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
        if rounded:
            # Both ways round. `rounded` was tested together with `upright`, so a horizontal panel
            # asking for it fell through to the plain `barh` and got square bars with no word said.
            _rounded_bars(ax, bars_at, values, each * 0.92, entry, orientation=orientation)
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
    # One category or a RANGE of them. A range is how a figure says "these belong together and that
    # one does not" — a set of comparable arms beside a reference that is not comparable — and a
    # single index could not say it, so a consumer drew its own `axvspan` instead.
    #
    # Worked out ONCE, into the set of shaded indices, because two places need it and the second one
    # got it wrong: the value labels' knockout compared `position == highlight`, an int against
    # something the signature allows to be a tuple, so for a RANGE the comparison was always false
    # and every label over the shading knocked out to the page instead. Measured on `highlight=(0,
    # 1)`: three labels, three page-coloured boxes, none of them the shade — a white patch punched
    # in the shading at every bar, which is the exact defect the knockout colours exist to prevent.
    shaded: set[int] = set()
    if highlight is not None:
        first, last = (highlight, highlight) if isinstance(highlight, int) else highlight
        require(
            0 <= first <= last < len(categories),
            f"highlight {highlight} is not a category index or range of them",
        )
        shaded = set(range(first, last + 1))
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
    if reference_band is not None:
        # Delegated, not repeated. This drew its own band until 2026-07-31 — solid fill, label
        # centred INSIDE it — which is the design `reference_band` was written to replace, and it
        # stayed reachable through this argument. Two implementations of one mark in one file is
        # how a caller gets the rejected design by picking the other door.
        low_edge, high_edge, band_label = reference_band
        _draw_reference_band(ax, low_edge, high_edge, band_label, orientation=orientation)

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
                emphasis=_emphasis_within(emphasis, index),
                # A label sitting over the highlighted column must knock out to the SHADE, not to
                # the page, or its box reads as a white patch punched in the shading. Read from the
                # set worked out above, so a RANGE is covered as well as a single index.
                knockout_colors=[
                    HIGHLIGHT_FILL if position in shaded else page_color()
                    for position in range(len(categories))
                ],
            )
    if threshold_label is not None:
        slide_label_clear(ax, threshold_label)
