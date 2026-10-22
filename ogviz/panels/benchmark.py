"""The benchmark bar template: who leads, in which group, drawn so a skim lands on the answer.

Three projects drew the same figure and converged on it: a few large panels for the headline
metrics over a grid of small ones for the rest, one bar per method, the method's hue held across
every panel, the leading bar at full strength and the others faded toward the page. What each of
them added to the plain `bar_panel` is here, so a project states its arms and its numbers and
nothing else.

THE PARTS, each usable alone:

- `featured_grid` builds the layout, so the large panels get the height their two-line names and
  titles need and the small rows get the gap theirs do. One `hspace` is a fraction of the AVERAGE
  row height, so a single grid gave the small rows the large row's gap.
- `leaders` decides who leads, from the values and the metric's direction, so the marking cannot
  disagree with the bars. A tie leads together: a 0.7994 beside a 0.7987 shows two leaders rather
  than crowning a 0.0007 difference. What counts as a tie is the caller's `tied`, because the
  right test is a paired interval and this package does not own the statistics.
- `group_bands` tints the stretch behind each group of arms and names it, so "ours" and "the
  conventionally trained baselines" read as two sets without a legend.
- `stripe_bars` draws an ensemble in both members' colours, in equal diagonal bands, so it reads as
  the two combined and neither colour wins the bar.
- `mark_leaders` outlines the leader and sets its name and interval in ink while the rest stay
  muted. Four cues repeat one signal (saturation, outline, bold value, bold name) because one signal
  repeated four ways survives a reader who is skimming.
- `panel_letters` puts A, B, C above each panel's corner, clear of the tick labels.

COLOUR CARRIES WHICH ARM; RANK RIDES ON SATURATION. A bar changes strength from panel to panel and
never changes hue, so an arm stays findable across the grid while the leader is what the eye lands
on first. The fade is derived from the arm's colour, so a project that changes a colour does not
have to re-choose the pale one.

VALUE LABELS ARE PRINTED LAST (`Benchmark.label`), because their height depends on the final
limits. Printed before `share_value_limits`, they sat at each panel's provisional top and drifted
once the scale moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter

from ogviz import units
from ogviz.layout.render import ensure_rendered
from ogviz.layout.ticks import format_value
from ogviz.orientation import read_orientation
from ogviz.panels.bars import (
    BAR_ALPHA,
    BAR_WIDTH,
    Z_BAR,
    Series,
    _slot_points,
    bar_panel,
    value_labels,
)
from ogviz.panels.table import tint
from ogviz.require import require
from ogviz.tags import mark, marked
from ogviz.theme import INK, MUTED_INK, page_color

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

FADE = 0.52  # how far a non-leading bar is blended toward the page; hue is kept
GROUP_GAP = (
    0.6  # extra category distance between the last bar of one group and the first of the next
)
BAND_PAD = 0.46  # a band reaches this far past its outer bars' centres; a bar is `BAR_WIDTH` wide
BAND_FRACTION = 0.90  # a group's band is its first colour blended this far toward the page
BAND_NAME_SIZE = 14.0
BAND_NAME_SHRINK = 0.8  # the smaller size a band name falls back to when no name fits at full size
NAME_MARGIN_PX = 6.0  # clear space a band name keeps from each band edge
Z_BAND = 0.05  # under the gridlines, under everything
Z_BAND_NAME = 6
LEADER_OUTLINE = 2.0
ERROR_WIDTH = 2.4  # whisker weight in points: thick enough to read over a bar and its outline
ERROR_CAPSIZE = 6.0
STRIPES = 6  # equal diagonal bands per striped bar, alternating the bar's own colour and the second
LETTER_SIZE = 22.0
LETTER_RISE_PT = 10.0  # how far above the axes' top edge a panel letter sits
LETTER_GAP_PT = 6.0  # clear space between a letter and the outermost text to its right
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Arm:
    """One method on the benchmark: its name, its colour, and the set it belongs to.

    `second_color` marks an ensemble, drawn in both members' colours. `group` names the band the
    arm sits in; arms of one group must be adjacent, which `group_positions` checks.
    """

    label: str
    color: str
    group: str | None = None
    second_color: str | None = None


def leaders(
    values: ArrayLike,
    *,
    lower_is_better: bool = False,
    tied: Callable[[int, int], bool] | None = None,
) -> frozenset[int]:
    """Indices of the leading bar and of every bar `tied` with it.

    `tied(best, other)` answers whether `other` is not separated from the best bar. Left as None,
    only the single best bar leads. The rule belongs to the caller: the defensible test is a paired
    interval over the same cases, and an unpaired overlap of two intervals is a weaker one that a
    project should choose knowingly. Values are compared unrounded; round only when printing.
    """
    array = np.asarray(values, dtype=float)
    require(array.size, "leaders needs at least one value")
    require(np.isfinite(array).all(), "leaders needs finite values; a missing arm cannot lead")
    best = int(np.argmin(array) if lower_is_better else np.argmax(array))
    others = {i for i in range(array.size) if i != best and tied is not None and tied(best, i)}
    return frozenset({best, *others})


def group_positions(groups: Sequence[str | None], *, gap: float = GROUP_GAP) -> list[float]:
    """Category positions that leave `gap` extra room between one group and the next.

    Groups must be contiguous and in order of first appearance: a group split in two would be drawn
    as two bands under one name.
    """
    require(all(group is not None for group in groups), "every arm needs a group, or none does")
    order = list(dict.fromkeys(groups))
    seen = [order.index(group) for group in groups]  # type: ignore[arg-type]
    require(
        seen == sorted(seen),
        f"groups must be contiguous, each arm beside the others of its group: {list(groups)}",
    )
    return [index + gap * rank for index, rank in enumerate(seen)]


def group_bands(
    ax: Axes,
    groups: Sequence[str],
    positions: Sequence[float],
    *,
    colors: Mapping[str, str],
    names: Mapping[str, Sequence[str]] | None = None,
    size: float = BAND_NAME_SIZE,
) -> list[tuple[str, float, float]]:
    """Tint the stretch behind each group and, if `names` is given, print its name in the band.

    `names` maps a group to candidate names, longest first. The longest that fits the band at full
    size wins; if none does, each is tried at a smaller size, and if none fits then the call
    raises, because a one-bar band is too narrow for a long name and the wording is the author's
    call. The names are measured against the panel as it is now, so call this after the figure's
    width is settled. They sit at the top of the panel: leave that room above the tallest value.

    The bands are tagged as backdrops, so a label sitting on one is not reported as sitting on
    data. Returns each group's `(name, left, right)` in data coordinates.
    """
    spans = [
        (
            group,
            min(x for x, g in zip(positions, groups, strict=True) if g == group) - BAND_PAD,
            max(x for x, g in zip(positions, groups, strict=True) if g == group) + BAND_PAD,
        )
        for group in dict.fromkeys(groups)
    ]
    for group, left, right in spans:
        band = Rectangle(
            (left, 0.0),
            right - left,
            1.0,
            transform=ax.get_xaxis_transform(),
            facecolor=colors[group],
            edgecolor="none",
            zorder=Z_BAND,
        )
        mark(band, "backdrop")
        ax.add_patch(band)
    ax.set_xlim(spans[0][1] - 0.06, spans[-1][2] + 0.06)
    if names is not None:
        for group, left, right in spans:
            _name_band(ax, group, left, right, size, names[group])
    return spans


def _name_band(
    ax: Axes, group: str, left: float, right: float, size: float, candidates: Sequence[str]
) -> None:
    fig = ax.get_figure(root=True)
    assert fig is not None, "the axes must belong to a figure"
    ensure_rendered(fig)
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    edges = (ax.transData.transform((x, 0.0))[0] for x in (left, right))
    low, high = edges
    room = high - low - 2 * NAME_MARGIN_PX
    attempts = [(name, size) for name in candidates] + [
        (name, size * BAND_NAME_SHRINK) for name in candidates
    ]
    for name, fontsize in attempts:
        text = ax.text(
            (left + right) / 2,
            0.975,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=fontsize,
            fontstyle="italic",
            color=MUTED_INK,
            zorder=Z_BAND_NAME,
        )
        if text.get_window_extent(renderer).width <= room:
            return
        text.remove()
    raise AssertionError(
        f"no name for the band {group!r} fits its {room:.0f} px: {list(candidates)}"
    )


def _bar_patches(ax: Axes) -> list[Rectangle | FancyBboxPatch]:
    """The bars of a panel in category order: every rectangle that is not a backdrop band."""
    return [
        patch
        for patch in ax.patches
        if isinstance(patch, Rectangle | FancyBboxPatch)
        and patch.get_width()
        and not marked(patch, "backdrop")
    ]


def _clip_to_box(points, x0: float, x1: float, y0: float, y1: float):
    """A convex polygon cut to an axis-aligned box, one edge at a time (Sutherland-Hodgman)."""

    def cut(pts, inside, cross):
        out = []
        for i, current in enumerate(pts):
            previous = pts[i - 1]
            if inside(current):
                if not inside(previous):
                    out.append(cross(previous, current))
                out.append(current)
            elif inside(previous):
                out.append(cross(previous, current))
        return out

    def at_x(x):
        return lambda p, q: (x, p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0]))

    def at_y(y):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]), y)

    for inside, cross in (
        (lambda p: p[0] >= x0, at_x(x0)),
        (lambda p: p[0] <= x1, at_x(x1)),
        (lambda p: p[1] >= y0, at_y(y0)),
        (lambda p: p[1] <= y1, at_y(y1)),
    ):
        points = cut(points, inside, cross)
        if not points:
            return []
    return points


def _over_the_page(color: str, strength: float) -> tuple[float, float, float]:
    """`color` as it looks at `strength` of the way from the page, opaque."""
    page = to_rgb(page_color())
    return tuple(
        c * strength + p * (1.0 - strength) for c, p in zip(to_rgb(color), page, strict=True)
    )  # type: ignore[return-value]


def stripe_bars(ax: Axes, stripes: Mapping[int, str], *, bands: int = STRIPES) -> None:
    """Draw diagonal bands of a second colour over the bars named in `stripes`, by category index.

    The bar keeps its own colour in the other half, so the two colours cover equal areas. Drawn as
    polygons because a matplotlib hatch has a fixed, fine spacing: six thick bands read at a glance
    where a hatch reads as texture. Each band is cut to the bar's rectangle as GEOMETRY, not only
    by a clip path, because the gate measures a patch's unclipped extent and a band reaching above
    the bar sat under its value label.

    The stripe colour is pre-blended with the page and drawn opaque, since a bar is drawn at
    `BAR_ALPHA` and a translucent stripe would show the bar's own colour through itself. Call this
    AFTER `mark_leaders`: a stripe covers the inner half of the bar's outline, so the outline is
    drawn again over the stripes when the bar has one.
    """
    require(
        read_orientation(ax) in (None, "vertical"),
        "stripe_bars draws on upright bars; a horizontal panel has no diagonal that reads",
    )
    bars = _bar_patches(ax)
    for index, color in stripes.items():
        require(0 <= index < len(bars), f"no bar {index} to stripe on a panel of {len(bars)}")
        bar = bars[index]
        left, low = bar.get_x(), bar.get_y()
        right, high = left + bar.get_width(), low + bar.get_height()
        low, high = min(low, high), max(low, high)
        band = (high - low) / bands
        paint = _over_the_page(color, BAR_ALPHA)
        for k in range(-1, bands + 1, 2):
            corners = [
                (left, low + k * band),
                (right, low + (k + 1) * band),
                (right, low + (k + 2) * band),
                (left, low + (k + 1) * band),
            ]
            inside = _clip_to_box(corners, left, right, low, high)
            if len(inside) < 3:
                continue
            stripe = Polygon(inside, closed=True, facecolor=paint, edgecolor="none", zorder=Z_BAR)
            ax.add_patch(stripe)
            stripe.set_clip_path(bar)
        edge = to_rgba(bar.get_edgecolor())
        if bar.get_linewidth() > 0 and edge[3] > 0:
            outline_style = {
                "facecolor": "none",
                "edgecolor": edge,
                "linewidth": bar.get_linewidth(),
                "zorder": Z_BAR,
            }
            at = (left, bar.get_y())
            if isinstance(bar, FancyBboxPatch):
                outline: Rectangle | FancyBboxPatch = type(bar)(
                    at,
                    bar.get_width(),
                    bar.get_height(),
                    boxstyle=bar.get_boxstyle(),
                    **outline_style,
                )
            else:
                outline = Rectangle(at, bar.get_width(), bar.get_height(), **outline_style)
            ax.add_patch(outline)


def mark_leaders(
    ax: Axes, lead: frozenset[int] | set[int], *, label_size: float | None = None
) -> None:
    """Outline each leading bar, and set its name and its interval in ink while the rest stay muted.

    `bar_panel`'s `emphasis` and `value_labels` already do the VALUE half of the signal; this is the
    rest of it, reached through the drawn artists rather than added to `bar_panel` as arguments. A
    panel that grew a knob per decoration would be a figure's style sheet wearing a function's
    clothes. Whiskers live in one `LineCollection` across every bar, so the colour is set per
    segment.
    """
    bars = _bar_patches(ax)
    for index in lead:
        require(0 <= index < len(bars), f"leader {index} is not a bar of this panel ({len(bars)})")
        bars[index].set_edgecolor(INK)
        bars[index].set_linewidth(LEADER_OUTLINE)
    for index, label in enumerate(ax.get_xticklabels()):
        label.set_fontweight("bold" if index in lead else "normal")
        label.set_color(INK if index in lead else MUTED_INK)
        if label_size is not None:
            label.set_fontsize(label_size)
    shades = [INK if index in lead else MUTED_INK for index in range(len(bars))]
    for container in ax.containers:
        parts = getattr(container, "lines", None)
        if not parts or len(parts) < 3:
            continue
        for collection in parts[2]:
            collection.set_colors(shades)
        for cap, shade in zip(parts[1], [s for s in shades for _ in (0, 1)], strict=False):
            cap.set_color(shade)


def panel_letters(
    axes: Sequence[Axes], *, size: float = LETTER_SIZE, letters: str = LETTERS
) -> None:
    """Bold A, B, C in reading order above each panel's top-left corner.

    Each letter is placed against the outermost text on its panel's left, measured, so it clears
    the tick labels and the axis title whatever their width. Call it after the ticks and labels are
    set. The letter is tied to its axes, so a later layout change moves it with the panel.
    """
    require(len(axes) <= len(letters), f"{len(axes)} panels but only {len(letters)} letters")
    for letter, ax in zip(letters, axes, strict=False):
        fig = ax.get_figure(root=True)
        assert fig is not None, "the axes must belong to a figure"
        ensure_rendered(fig)
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        reach = [ax.get_window_extent(renderer).x0]
        reach += [
            tick.label1.get_window_extent(renderer).x0
            for tick in ax.yaxis.get_major_ticks()
            if tick.label1.get_text() and tick.label1.get_visible()
        ]
        if ax.yaxis.label.get_text():
            reach.append(ax.yaxis.label.get_window_extent(renderer).x0)
        left_px = min(reach) - ax.get_window_extent(renderer).x0
        ax.annotate(
            letter,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(units.to_points(left_px, fig=fig) - LETTER_GAP_PT, LETTER_RISE_PT),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=size,
            fontweight="bold",
            color=INK,
        )


def featured_grid(
    *,
    large: int = 2,
    small: tuple[int, int] = (2, 3),
    size: tuple[float, float] = (12.0, 10.5),
    ratios: tuple[float, float] = (1.6, 2.25),
    large_wspace: float = 0.9,
    small_wspace: float = 0.9,
    large_hspace: float = 0.30,
    small_hspace: float = 0.32,
) -> tuple[Figure, list[Axes], list[Axes]]:
    """`large` big panels in one row over a `small` grid of rows by columns, on one shared width.

    The two tiers sit in separate nested grids because one `hspace` is a fraction of the average
    row height, which would give the small rows the gap the large row's two-line names and titles
    need. Both tiers cover the same width: the grid has `lcm(large, columns)` columns and each
    large panel spans its share, so two large panels over three small ones line up at the edges.

    Returns the figure, the large axes left to right, and the small axes in reading order (row by
    row). `ratios` is the height of the large row against the small block.
    """
    rows, columns = small
    require(
        large >= 1 and rows >= 1 and columns >= 1, "featured_grid needs at least one panel per tier"
    )
    width_cells = lcm(large, columns)
    fig = plt.figure(figsize=size)
    outer = fig.add_gridspec(2, 1, height_ratios=list(ratios), hspace=large_hspace)
    top = outer[0].subgridspec(1, width_cells, wspace=large_wspace)
    span = width_cells // large
    big = [fig.add_subplot(top[0, i * span : (i + 1) * span]) for i in range(large)]
    lower = outer[1].subgridspec(rows, width_cells, hspace=small_hspace, wspace=small_wspace)
    each = width_cells // columns
    little = [
        fig.add_subplot(lower[row, col * each : (col + 1) * each])
        for row in range(rows)
        for col in range(columns)
    ]
    return fig, big, little


@dataclass(frozen=True)
class Benchmark:
    """What `benchmark_panel` drew, kept so the labels can be printed once the limits are final."""

    leaders: frozenset[int]
    centres: np.ndarray
    values: np.ndarray
    errors: np.ndarray | None
    value_format: str | None
    knockouts: list[str]
    each: float

    def label(
        self, ax: Axes, *, fontsize: float | None = None, lead_fontsize: float | None = None
    ) -> None:
        """Print every bar's value, the leaders bold and in ink and the rest normal and muted.

        Call after the value limits are final (after `share_value_limits`), because each label's
        height is measured from the limits as they stand. `lead_fontsize` sets the leaders' own
        size, for a panel where the leader should read a step larger than its neighbours.
        """
        common = {
            "errors": self.errors,
            "value_format": self.value_format,
            "knockout_colors": self.knockouts,
            "slot_points": _slot_points(ax, self.each, "vertical"),
        }
        if fontsize is not None:
            common["fontsize"] = fontsize
        before = len(ax.texts)
        value_labels(ax, self.centres, self.values, emphasis=self.leaders, **common)  # type: ignore[arg-type]
        if lead_fontsize is not None:
            printed = ax.texts[before:]
            for index in self.leaders:
                printed[index].set_fontsize(lead_fontsize)


def benchmark_panel(
    ax: Axes,
    arms: Sequence[Arm],
    values: ArrayLike,
    errors: ArrayLike | None = None,
    *,
    lower_is_better: bool = False,
    tied: Callable[[int, int], bool] | None = None,
    labels: Sequence[str] | None = None,
    value_format: str | None = None,
    group_colors: Mapping[str, str] | None = None,
    group_names: Mapping[str, Sequence[str]] | None = None,
    reference_band: tuple[float, float, str] | None = None,
    fade: float = FADE,
    error_width: float = ERROR_WIDTH,
    error_capsize: float = ERROR_CAPSIZE,
) -> Benchmark:
    """One metric across every arm: bars in each arm's hue, the leaders at full strength.

    Draws the bars, the group bands and the ensemble stripes, and marks the leaders. It does NOT
    print the value labels; call `.label(ax)` on the result once the panel's limits are final.
    `labels` overrides the arms' names on the axis, for a small panel where the full name does not
    fit; the colour still says which arm. Value ticks read "0" and "1", never "0.0" and "1.0".
    `errors` is a length-N array (symmetric) or 2 by N (lower, upper), and a NaN means no interval
    for that bar. The whiskers are heavier than a plain panel's (`error_width`, `error_capsize`),
    since a leader's outline and a faded fill both compete with a thin one.
    """
    array = np.asarray(values, dtype=float)
    require(len(arms) == array.size, f"{len(arms)} arms for {array.size} values")
    require(labels is None or len(labels) == len(arms), "pass one label per arm, or none")
    lead = leaders(array, lower_is_better=lower_is_better, tied=tied)
    groups = [arm.group for arm in arms]
    grouped = any(group is not None for group in groups)
    positions = group_positions(groups) if grouped else None
    fills = [
        arm.color if index in lead else tint(arm.color, strength=fade)
        for index, arm in enumerate(arms)
    ]
    bar_panel(
        ax,
        [Series("", array, fills, errors)],  # type: ignore[arg-type]
        list(labels) if labels is not None else [arm.label for arm in arms],
        positions=positions,
        rounded=True,
        show_values=False,
        reference_band=reference_band,
        error_width=error_width,
        error_capsize=error_capsize,
    )
    centres = np.asarray(positions if positions is not None else range(len(arms)), dtype=float)
    knockouts = [page_color()] * len(arms)
    if grouped:
        shades = _band_colors(arms, group_colors)
        group_bands(ax, groups, centres, colors=shades, names=group_names)  # type: ignore[arg-type]
        knockouts = [shades[group] for group in groups]  # type: ignore[index]
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: format_value(value)))
    mark_leaders(ax, lead)
    seconds = {
        index: arm.second_color if index in lead else tint(arm.second_color, strength=fade)
        for index, arm in enumerate(arms)
        if arm.second_color is not None
    }
    if seconds:
        stripe_bars(ax, {i: _hex(c) for i, c in seconds.items()})
    return Benchmark(
        leaders=lead,
        centres=centres,
        values=array,
        errors=None if errors is None else np.asarray(errors, dtype=float),
        value_format=value_format,
        knockouts=knockouts,
        each=BAR_WIDTH,
    )


def _hex(color: object) -> str:
    from matplotlib.colors import to_hex

    return to_hex(color)  # type: ignore[arg-type]


def _band_colors(arms: Sequence[Arm], given: Mapping[str, str] | None) -> dict[str, str]:
    """Each group's band colour: the caller's, or its first arm blended far toward the page."""
    shades: dict[str, str] = {}
    for arm in arms:
        if arm.group is None or arm.group in shades:
            continue
        shades[arm.group] = (
            given[arm.group]
            if given is not None and arm.group in given
            else _hex(tint(arm.color, strength=BAND_FRACTION))
        )
    return shades
