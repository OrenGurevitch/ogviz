"""The group-comparison violin panel — the figure every project draws.

`group_violins` is the whole panel in one call: bodies, jittered points, IQR boxes, mean lines,
the printed group means, the y-limits, and the significance brackets. It exists because the
alternative is each project re-deriving the same four numbers, and getting the headroom wrong
in a different way each time.

Headroom is the part that is easy to get wrong by hand. The axis grows by `SPAN_HEADROOM` of the
data span when a bracket is present, and the first bracket sits `BRACKET_INSET` below the new
top, so the bracket is always clear of the highest observation. Clearance and headroom come from
the same number, which is why a caller cannot place a bracket on a violin's tail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout import hairline_grid, ticks_over_data
from ogviz.marks import VIOLIN_WIDTH, iqr_box, mean_line, points, violin
from ogviz.orientation import (
    category_limits,
    category_tick_labels,
    category_ticks,
    is_vertical,
    place_many,
    stamp_orientation,
    value_limits,
    value_span,
)
from ogviz.panels.grid import align_mean_rows
from ogviz.significance import bracket_stack
from ogviz.theme import INK, KNOCKOUT_PAD, MEAN_LABEL_SIZE, page_color

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.text import Text
    from numpy.typing import NDArray

    from ogviz.orientation import Orientation

BOTTOM_PAD = 0.20  # the margin left below the data, as a fraction of the span
LIMIT_DRIFT = 1e-9  # the value axis is settled; presentation must not move it
# The printed means sit in the MIDDLE of that margin, not at a fraction of their own choosing. Two
# independent constants drift: raise the pad and the row stays put, crowding the violins it belongs
# to; halve the pad and the row lands under the axis. One number, and the row is centred by
# construction between the lowest mark and the frame.
MEAN_ROW_OFFSET = BOTTOM_PAD / 2.0
SPAN_HEADROOM = 0.30  # extra top room for the FIRST bracket
PLAIN_HEADROOM = 0.20
BRACKET_INSET = 0.12  # first bracket, below the expanded top
STACK_FIT_MARGIN = 0.02  # slack per attempt while growing the axis to fit a bracket stack


def printed_means(
    ax: Axes,
    positions: Sequence[float],
    values: Sequence[float],
    row: float,
    *,
    fontsize: float = MEAN_LABEL_SIZE,
    decimals: int | None = None,
    orientation: Orientation = "vertical",
    scale: float = 1.0,
    thousands_separator: bool = False,
) -> list[Text]:
    """A row of numbers under the marks they describe, ONE format for the whole row.

    The row's format is the part with logic in it, and the reason this is public rather than private
    to `group_violins`: letting each value pick its own decimals gives a ragged
    "0.0887 / 0.545 / 1.57", which reads as three different measurements rather than three values
    of one measurement. The count comes from the largest value in the row and applies to all of it.

    `values` are the numbers to print, not the samples they came from — a caller printing medians,
    a difference, or a count wants the same row. `group_violins` computes its means and calls this.

    Every label is tagged `ogviz_mean_row`, which is how `align_mean_rows` finds them and puts a
    grid's rows on one line.
    """
    from ogviz.layout.ticks import auto_decimals, format_value

    assert len(positions) == len(values), f"{len(positions)} positions for {len(values)} values"
    if decimals is None:
        decimals = auto_decimals(max((abs(value * scale) for value in values), default=1.0))
    drawn: list[Text] = []
    for position, value in zip(positions, values, strict=True):
        horizontal, vertical = place_many(orientation, position, row)
        label = ax.text(
            horizontal,
            vertical,
            format_value(
                value,
                scale=scale,
                decimals=decimals,
                thousands_separator=thousands_separator,
            ),
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#333333",
            zorder=9,
            bbox={
                "facecolor": page_color(),
                "edgecolor": "none",
                "pad": KNOCKOUT_PAD,
                "boxstyle": "square",
            },
        )
        # Tagged so a grid can put every panel's row on one line. On a shared scale the floor is
        # common but each panel's lowest violin is not, so a row placed from a panel's own data
        # lands at a different height in every panel — which is the thing a shared scale exists to
        # prevent.
        label.ogviz_mean_row = True  # type: ignore[attr-defined]
        drawn.append(label)
    return drawn


def _finish(ax: Axes, high: float, orientation: Orientation, *, grid: bool) -> None:
    """Tidy the ticks and the mean row, and check the limits survived it.

    A POST-CONDITION, not a style check. This function fitted the value axis to the data and its
    bracket stack; anything done afterwards is presentation and must not move it. `set_yticks`
    does: it fixes the locator, and matplotlib then grows the view to contain every fixed tick, so
    dropping the ticks above the data reframed one panel from zero and left its violins in the top
    third of an axis that had been fitted to them. Nothing in QC noticed, because no check asked
    whether the axis still framed the data.

    Asserted here rather than checked later because this is the one place that knows what the
    limits are supposed to be.
    """
    stamp_orientation(ax, orientation)
    if grid:
        # Drawn by the panel, not by the caller. Every example had to remember `hairline_grid`, and
        # the two grid examples did not — six panels shipped with no rule to read a violin against.
        # A default that has to be re-applied at every call site is not a default.
        hairline_grid(ax, axis="y" if is_vertical(orientation) else "x")
    before = value_span(ax, orientation)
    # No argument: it measures the drawn marks. Handing it `high`, the data maximum, dropped
    # the tick a violin's own body reaches past.
    ticks_over_data(ax, orientation=orientation)
    _settle_mean_row(ax, orientation)
    after = value_span(ax, orientation)
    drift = max(abs(a - b) for a, b in zip(after, before, strict=True))
    assert drift <= abs(before[1] - before[0]) * LIMIT_DRIFT, (
        f"the value axis moved after it was fitted: {before} -> {after}. Something below this "
        "point changed the limits, and the panel no longer frames the data it was sized to."
    )


def _settle_mean_row(ax: Axes, orientation: Orientation) -> None:
    """Put this panel's printed means midway between the frame and the lowest DRAWN mark.

    One code path with the grid, deliberately. The row was placed from the data minimum, and a
    violin's body extends past its data: the kernel's tail reaches below the smallest observation,
    so the row sat closer to the violin than to the frame — by half a unit on a panel with a long
    tail — while a grid, which measured the rendered body, put it in the middle. Two ways of
    answering "midway" is how a single panel and a grid of panels came to disagree.
    """
    if not is_vertical(orientation):
        return  # a horizontal panel prints no mean row
    align_mean_rows([ax], floor=ax.get_ylim()[0])


def _fit_bracket_stack(
    ax: Axes,
    comparisons: Sequence[tuple[float, float, float]],
    *,
    low: float,
    high: float,
    span: float,
    bottom_pad: float,
    lift: float,
    base_headroom: float,
    label_for: Callable[[float], str] | None,
    orientation: Orientation,
    bracket_arguments: Mapping[str, object],
) -> tuple[float, float]:
    """Grow the value axis until the whole bracket stack fits. Returns (headroom, stack start).

    The first bracket is anchored to the DATA (`high + lift * span`), not to the axis top. That
    matters: with it anchored to the top, raising the headroom lifted the stack by exactly as much
    as it lifted the ceiling, while also stretching the data-per-pixel so the stack grew — the
    loop could never converge, and the first version of this fix asserted on every panel with
    three brackets instead of fixing them.

    Anchored to the data, each pass strictly reduces the overshoot as long as the stack is shorter
    than the axes, so two or three passes settle it. Measured with `draw=False` rather than
    predicted, because the arithmetic is circular.
    """
    headroom = base_headroom
    start = high + lift * span
    for _attempt in range(12):
        top = high + headroom * span
        value_limits(ax, orientation)(low - bottom_pad * span, top)
        reached = bracket_stack(
            ax,
            comparisons,
            start=start,
            span=span,
            label_for=label_for,
            orientation=orientation,
            draw=False,
            **bracket_arguments,  # type: ignore[arg-type]
        )
        if reached <= top:
            return headroom, start
        headroom += (reached - top) / span + STACK_FIT_MARGIN
    raise AssertionError(
        f"{len(comparisons)} brackets will not fit above the data at this figure size — the stack "
        "is taller than the axes. Make the figure taller along the value axis, or draw fewer "
        "comparisons."
    )


def group_violins(
    ax: Axes,
    groups: Sequence[tuple[float, NDArray[np.float64], str, str]],
    *,
    comparisons: Sequence[tuple[float, float, float]] = (),
    categories: Sequence[str] | None = None,
    category_fontsize: float | None = None,
    anchor_value: float | None = None,
    seed: int | np.random.Generator = 0,
    show_means: bool | None = None,
    mean_fontsize: float = MEAN_LABEL_SIZE,
    mean_decimals: int | None = None,
    display_scale: float = 1.0,
    thousands_separator: bool = False,
    label_for: Callable[[float], str] | None = None,
    mean_row_offset: float = MEAN_ROW_OFFSET,
    bottom_pad: float = BOTTOM_PAD,
    grid: bool = True,
    headroom: float | None = None,
    bracket_inset: float = BRACKET_INSET,
    violin_kwargs: Mapping[str, object] | None = None,
    point_kwargs: Mapping[str, object] | None = None,
    box_kwargs: Mapping[str, object] | None = None,
    mean_kwargs: Mapping[str, object] | None = None,
    bracket_kwargs: Mapping[str, object] | None = None,
    orientation: Orientation = "vertical",
) -> float:
    """Draw a full group-comparison panel. Returns the topmost drawn y in data units.

    `groups` is [(x position, values, fill colour, edge colour)].
    `comparisons` is [(x_left, x_right, p)] for significance brackets, lowest first.
    `anchor_value` forces a value into the range (0 for a zero-anchored measure).
    `label_for` overrides what a bracket's label says, the way `bracket_stack` does.
    `display_scale` converts the stored unit to the printed one — a quantity stored in ppm and
    written in ppb — and applies to the printed means. Pair it with `value_ticks(scale=...)`
    so the axis and the means agree. The data is never touched.

    The `*_kwargs` reach the marks, so a panel whose layout wants a wider body passes
    `violin_kwargs={"width": 0.8}, point_kwargs={"width": 0.8}` and gets it — a composite that
    hardcodes its parts' defaults is not reusable, it is one project's figure with a public name.
    The spacing fractions are arguments for the same reason.
    """
    # In a vertical panel the printed means get a row beneath the violins. A horizontal panel
    # has no such room: the space past the low end of the value axis is where the category tick
    # labels sit, so the default is off there rather than colliding by default.
    if show_means is None:
        show_means = orientation == "vertical"

    populated = [(p, np.asarray(v, dtype=float), f, e) for p, v, f, e in groups if len(v)]
    assert populated, "group_violins needs at least one non-empty group"
    seats = [p for p, _v, _f, _e in populated]
    assert len(set(seats)) == len(seats), (
        f"two groups share a position in {seats}. They would be drawn on top of each other — two "
        "violins, two clouds of dots, two mean lines, and no way to read either."
    )
    for position, values, _f, _e in populated:
        missing = int(np.count_nonzero(~np.isfinite(values)))
        assert not missing, (
            f"group at x={position} has {missing} non-finite value(s) of {len(values)}. "
            "Drop or impute them in the project, where the choice is visible — a plot that "
            "silently omits them shows an n nobody wrote down."
        )
    rng = np.random.default_rng(seed)  # a Generator passes straight through
    # Built once and given to BOTH `bracket_stack` calls. The first measures how much headroom the
    # stack needs and draws nothing; typography reaching only the drawing call would reserve room
    # for a 20 pt star and then set a 28 pt one in it.
    bracket_arguments: dict[str, object] = {"text_color": INK, **(bracket_kwargs or {})}
    violin_kwargs = {"orientation": orientation, **dict(violin_kwargs or {})}
    point_kwargs = {"orientation": orientation, **dict(point_kwargs or {})}
    box_kwargs = {"orientation": orientation, **dict(box_kwargs or {})}
    mean_kwargs = {"orientation": orientation, **dict(mean_kwargs or {})}

    series = [v for _p, v, _f, _e in populated]
    if anchor_value is not None:
        series.append(np.array([anchor_value]))
    every = np.concatenate(series)
    low, high = float(every.min()), float(every.max())
    span = max(high - low, 1e-9)

    # BOTH limits first, then marks. `points` sizes its central lane from the axes transform, so
    # the scale has to be final before a dot is placed. The category axis matters as much as the
    # value axis and is easier to miss: matplotlib autoscales it only once the violin is added,
    # so a lane computed mid-draw is measured against provisional limits and every dot clears the
    # wrong distance — which is how they ended up back on the marks.
    stack_start = high
    if headroom is None:
        # A fixed fraction covered ONE bracket, so from the third up the stack ran past the
        # limit — and because matplotlib clips Line2D but not Text, the line vanished while its
        # star stayed, leaving a star floating over nothing.
        if comparisons:
            headroom, stack_start = _fit_bracket_stack(
                ax,
                comparisons,
                low=low,
                high=high,
                span=span,
                bottom_pad=bottom_pad,
                lift=SPAN_HEADROOM - bracket_inset,
                base_headroom=SPAN_HEADROOM,
                label_for=label_for,
                orientation=orientation,
                bracket_arguments=bracket_arguments,
            )
        else:
            headroom, stack_start = PLAIN_HEADROOM, high
    top = high + headroom * span
    value_limits(ax, orientation)(low - bottom_pad * span, top)

    body = float(violin_kwargs.get("width", VIOLIN_WIDTH))  # type: ignore[arg-type]
    places = [p for p, _v, _f, _e in populated]
    category_limits(ax, orientation)(min(places) - body, max(places) + body)

    for position, values, fill, edge in populated:
        violin(ax, values, position, fill, **violin_kwargs)  # type: ignore[arg-type]
        points(ax, values, position, fill, edge, rng, **point_kwargs)  # type: ignore[arg-type]
        iqr_box(ax, values, position, **box_kwargs)  # type: ignore[arg-type]
        mean_line(ax, values, position, **mean_kwargs)  # type: ignore[arg-type]

    if show_means:
        printed_means(
            ax,
            [position for position, *_rest in populated],
            [float(np.mean(sample)) for _p, sample, _f, _e in populated],
            low - mean_row_offset * span,
            fontsize=mean_fontsize,
            decimals=mean_decimals,
            orientation=orientation,
            scale=display_scale,
            thousands_separator=thousands_separator,
        )

    if categories is not None:
        # `bar_panel` has always labelled its own categories and this did not, so every caller set
        # the ticks and the labels by hand — the same four lines in three examples, and two panels
        # of the same kind with different contracts.
        assert len(categories) == len(groups), (
            f"{len(categories)} categories for {len(groups)} groups"
        )
        # Every group's position, not `places` — an empty group is dropped from `places`, and
        # labelling that shorter list would slide every later name onto the wrong violin.
        category_ticks(ax, orientation)([position for position, *_rest in groups])
        category_tick_labels(ax, orientation)(
            list(categories),
            **({} if category_fontsize is None else {"fontsize": category_fontsize}),
        )

    if not comparisons:
        _finish(ax, high, orientation, grid=grid)
        return high
    reached = bracket_stack(
        ax,
        comparisons,
        start=stack_start,
        span=span,
        label_for=label_for,
        orientation=orientation,
        **bracket_arguments,  # type: ignore[arg-type]
    )
    limit = value_span(ax, orientation)[1]
    assert reached <= limit + 1e-9, (
        f"the bracket stack reaches {reached:.4g} but the axis stops at {limit:.4g}. matplotlib "
        "clips the bracket LINES and not their stars, so this would have shipped as stars "
        "floating over nothing. Raise `headroom`."
    )
    _finish(ax, high, orientation, grid=grid)
    return reached
