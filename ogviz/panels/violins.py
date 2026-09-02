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

from ogviz.layout import drawn_value_extent, hairline_grid, ticks_over_data
from ogviz.marks import VIOLIN_WIDTH, iqr_box, mean_line, points, violin, widths_of
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
from ogviz.require import require
from ogviz.significance import bracket_stack
from ogviz.tags import mark
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
# What a panel's value span becomes when every observation is the same number. A `max(high - low,
# 1e-9)` floor kept the arithmetic safe and drew the figure into a line: the limits came out equal
# to nine decimal places, every mark landed on one row of pixels, and the printed mean sat on the
# frame. A constant group is ordinary data — an all-zero condition, a saturated measure — and the
# panel still has to say WHAT the constant is, which needs an axis with height.
#
# A tenth of the magnitude, floored, in the spirit of matplotlib's own `nonsingular` (which
# expands a degenerate range by 5% of the value, or to +/-1 at zero). It is a span, so this
# module's own pads still apply on top of it. MEASURED on a plain panel: a constant series of 5.0
# gives limits of 4.9 to 5.1 where they were 5.0 to 5.0, one of 0.0 gives -0.02 to 0.02, and one
# of 1200.0 gives 1176 to 1224 — the mean row, the violin and the frame three distinct rows of
# pixels in every case, and the axis ticks stating what the constant IS.
CONSTANT_SPAN_FLOOR = 0.1


def constant_span(value: float) -> float:
    """The nominal span for a panel whose data is one repeated number."""
    return max(abs(value) * 0.1, CONSTANT_SPAN_FLOOR)


def printed_means(
    ax: Axes,
    positions: Sequence[float],
    values: Sequence[float],
    row: float,
    *,
    fontsize: float = MEAN_LABEL_SIZE,
    weight: str = "bold",
    decimals: int | None = None,
    orientation: Orientation = "vertical",
    scale: float = 1.0,
    thousands_separator: bool = True,
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
    from ogviz.layout.ticks import format_value, row_decimals

    require(
        len(positions) == len(values),
        f"{len(positions)} positions for {len(values)} values",
    )
    if decimals is None:
        decimals = row_decimals(values, scale=scale)
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
            # Bold by default because the printed mean IS the number a single panel reports. A GRID
            # is the case for turning it down: the same row repeats in every cell, so at full weight
            # six copies of it become the loudest thing on the figure and the reader's eye goes to
            # the numbers rather than to the shapes the grid was drawn to compare.
            fontweight=weight,
            # The house ink. This was `"#333333"` — a fourth black, in a package whose reason for
            # existing opens on a project that shipped two of them in one paper. It sits between
            # `INK` and `MUTED_INK`, 0.21 from one and 0.27 from the other, so it was not either of
            # them by accident or on purpose. `INK` is the one that follows from the design: the
            # printed mean IS the number the panel exists to report, which is why `MEAN_LABEL_SIZE`
            # is larger than a label that merely annotates a mark, and a headline number does not
            # get set in a grey.
            color=INK,
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
        mark(label, "mean_row")
        drawn.append(label)
    return drawn


def _finish(ax: Axes, orientation: Orientation, *, grid: bool) -> None:
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
    require(
        drift <= abs(before[1] - before[0]) * LIMIT_DRIFT,
        f"the value axis moved after it was fitted: {before} -> {after}. Something below this "
        "point changed the limits, and the panel no longer frames the data it was sized to.",
    )


def _settle_mean_row(ax: Axes, orientation: Orientation) -> None:
    """Put this panel's printed means midway between the frame and the lowest DRAWN mark.

    One code path with the grid, deliberately. The row was placed from the data minimum, and a
    violin's body extends past its data: the kernel's tail reaches below the smallest observation,
    so the row sat closer to the violin than to the frame — by half a unit on a panel with a long
    tail — while a grid, which measured the rendered body, put it in the middle. Two ways of
    answering "midway" is how a single panel and a grid of panels came to disagree.
    """
    # Whichever way the panel runs. It returned early on a horizontal one, on the grounds that
    # such a panel prints no mean row — true of the DEFAULT and not of `show_means=True`, which
    # drew the row and then left it wherever the offset put it, out among the category tick
    # labels. `align_mean_rows` reads the value axis now, so there is nothing to decline.
    align_mean_rows([ax], floor=value_span(ax, orientation)[0], orientation=orientation)


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
    point_colors: Sequence[Sequence[str] | None] | None = None,
    outline_violins: bool = False,
    categories: Sequence[str] | None = None,
    category_fontsize: float | None = None,
    anchor_value: float | None = None,
    seed: int | np.random.Generator = 0,
    show_means: bool | None = None,
    mean_fontsize: float = MEAN_LABEL_SIZE,
    mean_weight: str = "bold",
    mean_decimals: int | None = None,
    display_scale: float = 1.0,
    thousands_separator: bool = True,
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
    `point_colors` is one entry PER GROUP, each either None for the group's own fill or one colour
    per observation — a repeated-measures panel colours its dots by subject so a reader can follow
    one across the conditions. It is indexed against `groups` as given, before empty groups are
    dropped, so a caller need not know which of its groups turned out to have data.

    `outline_violins` gives each body an outline in its OWN FILL COLOUR, for the pale-fill look: a
    fill dropped to a low alpha so the dots on top stay legible needs an edge, or a narrow waist
    fades out and the body stops reading as one shape. It is off by default and it is a per-group
    fact, which is why it is a flag here rather than an `edge_color` in `violin_kwargs` — those
    kwargs are one dict for every group, so a caller could only give all the bodies one edge. The
    group's own `edge colour` is NOT used, because that one is the dots' rim: on a crowded cloud
    the rim wants the page colour, to separate overlapping dots without darkening them.
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

    supplied = [None] * len(groups) if point_colors is None else list(point_colors)
    require(
        len(supplied) == len(groups),
        f"point_colors has {len(supplied)} entries for {len(groups)} groups. It is indexed against "
        "the groups, so a shorter list silently recolours the wrong ones.",
    )
    kept = [
        ((p, np.asarray(v, dtype=float), f, e), dots)
        for (p, v, f, e), dots in zip(groups, supplied, strict=True)
        if len(v)
    ]
    populated = [group for group, _dots in kept]
    dot_colors = [dots for _group, dots in kept]
    for (position, values, _f, _e), dots in kept:
        require(
            dots is None or len(dots) == len(values),
            f"group at x={position} has {len(values)} values and {len(dots or ())} point colours. "
            "The colours are matched to the values by position, so a mismatch would give a dot "
            "somebody else's identity rather than raise.",
        )
    require(
        populated,
        "group_violins needs at least one non-empty group",
    )
    seats = [p for p, _v, _f, _e in populated]
    require(
        len(set(seats)) == len(seats),
        f"two groups share a position in {seats}. They would be drawn on top of each other — two "
        "violins, two clouds of dots, two mean lines, and no way to read either.",
    )
    for position, values, _f, _e in populated:
        missing = int(np.count_nonzero(~np.isfinite(values)))
        require(
            not missing,
            f"group at x={position} has {missing} non-finite value(s) of {len(values)}. "
            "Drop or impute them in the project, where the choice is visible — a plot that "
            "silently omits them shows an n nobody wrote down.",
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
    # The lane the dots keep clear has to be the lane the OTHER marks actually occupy. This is the
    # one place that holds all four sets of kwargs, so it is the one place that can say so: a caller
    # widening the IQR bar through `box_kwargs` and nothing else would otherwise get dots placed
    # against the default width, sitting on the bar. `setdefault`, so an explicit `mark_widths` in
    # `point_kwargs` still wins.
    point_kwargs.setdefault("mark_widths", widths_of(box_kwargs, mean_kwargs))

    series = [v for _p, v, _f, _e in populated]
    if anchor_value is not None:
        series.append(np.array([anchor_value]))
    every = np.concatenate(series)
    low, high = float(every.min()), float(every.max())
    span = high - low or constant_span(high)

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

    for (position, values, fill, edge), dots in zip(populated, dot_colors, strict=True):
        body = dict(violin_kwargs)
        if outline_violins:
            body.setdefault("edge_color", fill)  # an explicit one in violin_kwargs still wins
        violin(ax, values, position, fill, **body)  # type: ignore[arg-type]
        points(ax, values, position, fill if dots is None else dots, edge, rng, **point_kwargs)  # type: ignore[arg-type]
        iqr_box(ax, values, position, **box_kwargs)  # type: ignore[arg-type]
        mean_line(ax, values, position, **mean_kwargs)  # type: ignore[arg-type]

    if show_means:
        printed_means(
            ax,
            [position for position, *_rest in populated],
            [float(np.mean(sample)) for _p, sample, _f, _e in populated],
            low - mean_row_offset * span,
            fontsize=mean_fontsize,
            weight=mean_weight,
            decimals=mean_decimals,
            orientation=orientation,
            scale=display_scale,
            thousands_separator=thousands_separator,
        )

    if categories is not None:
        # `bar_panel` has always labelled its own categories and this did not, so every caller set
        # the ticks and the labels by hand — the same four lines in three examples, and two panels
        # of the same kind with different contracts.
        require(
            len(categories) == len(groups),
            f"{len(categories)} categories for {len(groups)} groups",
        )
        # Every group's position, not `places` — an empty group is dropped from `places`, and
        # labelling that shorter list would slide every later name onto the wrong violin.
        category_ticks(ax, orientation)([position for position, *_rest in groups])
        category_tick_labels(ax, orientation)(
            list(categories),
            **({} if category_fontsize is None else {"fontsize": category_fontsize}),
        )

    if not comparisons:
        _finish(ax, orientation, grid=grid)
        # The topmost DRAWN value, as the docstring promises — not the data maximum, which sits
        # below a violin body's top on any panel with a kernel body. With brackets the answer was
        # the bracket ink's top; without them it was the data, so the same word meant two things.
        drawn = drawn_value_extent(ax, orientation=orientation)
        return drawn[1] if drawn is not None else high
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
    require(
        reached <= limit + 1e-9,
        f"the bracket stack reaches {reached:.4g} but the axis stops at {limit:.4g}. matplotlib "
        "clips the bracket LINES and not their stars, so this would have shipped as stars "
        "floating over nothing. Raise `headroom`.",
    )
    _finish(ax, orientation, grid=grid)
    return reached
