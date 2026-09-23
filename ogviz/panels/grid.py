"""Putting several panels on one scale, and on one line.

A grid of panels is a comparison, and a comparison needs the panels to agree about more than their
data. They have to share a value scale, or a difference of the same size looks different in two
places; and once they do, the rows of printed numbers have to sit at one height, or the gap between
a row and the frame stops meaning anything.

Both live here rather than in `ogviz.layout` because both know what a violin panel is: the row is
found by the `ogviz_mean_row` tag that `group_violins` and `split_violins` set. `layout` is imported
BY the panels, so a panel concept sitting there pointed the dependency the wrong way and put the
rule a long way from the code it governs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout import drawn_value_extent
from ogviz.orientation import is_vertical, value_limits
from ogviz.require import require
from ogviz.tags import mark, marked
from ogviz.units import px_to_value, value_to_px

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes

    from ogviz.orientation import Orientation


def share_value_limits(
    axes: Iterable[Axes], *, orientation: Orientation = "vertical", label_edge: bool = True
) -> tuple[float, float]:
    """Put every panel on one value scale: the union of the limits they each worked out.

    For a grid of comparable panels, which have to share a scale to be read against each other. The
    scale is the union of what the panels ALREADY fitted, not a number chosen in advance — a violin
    panel measures the headroom its bracket stack needs and grows the axis to suit, and a caller who
    then overwrites that with a guess has thrown the measurement away.

    That is the bug this replaces. A grid of one-comparison panels was given headroom sized for a
    three-bracket stack, so every panel carried two brackets' worth of empty page between its stars
    and its title. Ask each panel what it needs and take the widest answer, and a grid of
    single-bracket panels gets exactly one bracket's room.

    `label_edge` prints the value numbers once, on the panels at the grid's edge. Pass False to keep
    them under every panel.

    Each panel is TAGGED with how many it shares the scale with, because the consequence shows up
    somewhere else: a short panel beside a tall one is empty at the top by construction, and
    `dead_space` reported that in the same words as a panel whose limits are merely loose. The two
    want opposite actions — the second wants tightening and the first must not be tightened, or the
    grid stops being comparable — and the note could not tell them apart without being told.

    Returns the shared (low, high).
    """
    panels = list(axes)
    require(
        panels,
        "share_value_limits needs at least one axes",
    )
    reader = (lambda ax: ax.get_ylim()) if is_vertical(orientation) else (lambda ax: ax.get_xlim())
    spans = [reader(ax) for ax in panels]
    low = min(bounds[0] for bounds in spans)
    high = max(bounds[1] for bounds in spans)
    # Measured on each panel's OWN scale, before anything moves: how much room it fitted above its
    # bracket stack. Only consulted if aligning carries a stack past the shared top.
    clearances = [(ax, _clearance_above_stack(ax, orientation)) for ax in panels]
    floor_room = _room_below_the_marks(panels, orientation)
    for ax in panels:
        value_limits(ax, orientation)(low, high)
        mark(ax, "shared_scale", len(panels))
    low = _hold_the_room_below(panels, floor_room, low, high, orientation=orientation)
    # Both ends of the panel. A shared scale that leaves the brackets at six heights and the
    # printed means at six others is a shared scale in name only.
    #
    # WHICHEVER WAY THE PANELS RUN. This block sat under `if is_vertical(orientation)` and passed
    # no orientation, although all three helpers take one — so a horizontal grid got the shared
    # range and kept every panel's own ticks and brackets, and the gate refused it as "different
    # value ticks" on a figure this function had been asked to make agree.
    align_brackets(panels, orientation=orientation)
    high = _raise_over_aligned_stacks(clearances, low, high, orientation=orientation)
    align_ticks(panels, orientation=orientation)
    align_mean_rows(panels, floor=low, orientation=orientation)
    if label_edge:
        label_shared_scale_once(panels, orientation=orientation)
    return low, high


def _room_below_the_marks(panels: list[Axes], orientation: Orientation) -> float | None:
    """The most room, in PIXELS, any panel left between its lowest mark and its floor.

    Only for a grid that prints means, because the room is the mean row's: each panel sized it for
    its own row on its own scale. None where no panel prints means, so a grid without a row keeps
    exactly the limits it had.
    """
    if not any(marked(text, "mean_row") for ax in panels for text in ax.texts):
        return None
    figure = panels[0].get_figure(root=True)
    if figure is not None:
        figure.canvas.draw()
    rooms = []
    for ax in panels:
        extent = drawn_value_extent(ax, orientation=orientation)
        if extent is None:
            continue
        floor = (ax.get_ylim() if is_vertical(orientation) else ax.get_xlim())[0]
        rooms.append(
            abs(
                value_to_px(ax, extent[0], orientation=orientation)
                - value_to_px(ax, floor, orientation=orientation)
            )
        )
    return max(rooms) if rooms else None


def _hold_the_room_below(
    panels: list[Axes],
    room: float | None,
    low: float,
    high: float,
    *,
    orientation: Orientation,
) -> float:
    """Lower the shared floor until the grid's lowest mark keeps the room its row was sized for.

    Each panel set its margin below the data as a fraction of its OWN span. Sharing a scale widens
    the span — a panel of low values beside one of high values triples it — and the margin shrinks
    in pixels by the same factor, so the printed mean that fitted under each violin was squeezed
    onto the dots and refused by the gate. The room is held in pixels, which is what the row needs.

    Lowered and re-measured, because moving the floor rescales the axis: a few rounds settle it.
    A grid whose room already holds is returned untouched.
    """
    if room is None:
        return low
    extents = [drawn_value_extent(ax, orientation=orientation) for ax in panels]
    lowest = min((extent[0] for extent in extents if extent is not None), default=None)
    if lowest is None:
        return low
    reference = panels[0]
    for _attempt in range(8):
        have = abs(
            value_to_px(reference, lowest, orientation=orientation)
            - value_to_px(reference, low, orientation=orientation)
        )
        if have >= room - 0.5:
            return low
        per_px = (high - low) / max(
            abs(
                value_to_px(reference, high, orientation=orientation)
                - value_to_px(reference, low, orientation=orientation)
            ),
            1e-9,
        )
        low -= (room - have) * per_px
        for ax in panels:
            value_limits(ax, orientation)(low, high)
    return low


def _stack_reach(ax: Axes, orientation: Orientation) -> float | None:
    """The highest value this panel's bracket stack reaches — its lines and its stars' boxes."""
    upright = is_vertical(orientation)
    lines, stars = _bracket_artists(ax, upright=upright)
    if not lines:
        return None
    figure = ax.get_figure()
    if figure is not None:
        figure.canvas.draw()
    to_data = ax.transData.inverted()
    reaches = [_bracket_top(line, upright=upright) for line in lines]
    for star in stars:
        box = star.get_window_extent()
        corner = to_data.transform((box.x1, box.y1))
        reaches.append(float(corner[1] if upright else corner[0]))
    return max(reaches)


def _clearance_above_stack(ax: Axes, orientation: Orientation) -> float | None:
    """How much value axis this panel left above its own stack, or None where it has none."""
    reach = _stack_reach(ax, orientation)
    top = ax.get_ylim()[1] if is_vertical(orientation) else ax.get_xlim()[1]
    return None if reach is None else top - reach


def _raise_over_aligned_stacks(
    clearances: list[tuple[Axes, float | None]],
    low: float,
    high: float,
    *,
    orientation: Orientation,
) -> float:
    """The shared top, raised only as far as a stack that aligning lifted past it needs.

    `align_brackets` moves every stack up onto the HIGHEST first bracket in the grid. The shared
    top came from the tallest panel, which sized its headroom for its own stack — so a panel of low
    data with three brackets, beside a panel of high data with one, has its stack lifted onto the
    one-bracket line and its upper two brackets land above an axis sized for one. matplotlib clips
    the lines and not the stars, and the gate refused the figure for lines running past the frame.

    Each overshooting stack gets back the room it fitted above itself on its own scale — the
    measurement `violins._fit_bracket_stack` made — rather than a margin chosen here.

    GROWN AND RE-MEASURED, not computed once, for the reason `_fit_bracket_stack` gives: a star is
    a fixed number of PIXELS, so raising the top makes every pixel worth more data and the star
    reaches higher in data than it did. Computing the top once from the old scale left the star
    0.05 above the axis it had just been given. The stars are re-anchored to their brackets before
    each measurement, as `save` would do, so what is measured is what ships.

    A grid whose stacks already fit is returned untouched, so nothing accepted before moves.
    """
    from ogviz.significance import settle_bracket_labels

    def overshoot(ceiling: float) -> list[float]:
        # A stack counts as reaching past the top only by more than a PIXEL, and its own clearance
        # is never taken as negative. A star is measured here by its text box, which in a wide face
        # stands a fraction of a pixel above the ink `bracket_stack` fits by — so a panel whose
        # stack fits itself exactly has a box poking through by less than the gate can see, a
        # clearance just below zero, and a loop that chased that sliver for every round it had.
        wanted = []
        for ax, clearance in clearances:
            reach = _stack_reach(ax, orientation)
            if reach is None or clearance is None:
                continue
            pixel = abs(
                px_to_value(ax, 1.0, orientation=orientation)
                - px_to_value(ax, 0.0, orientation=orientation)
            )
            if reach > ceiling + pixel:
                wanted.append(reach + max(clearance, 0.0))
        return wanted

    # Each round asks whether anything still reaches past the top it has JUST set, and stops when
    # nothing does. It compared against the original top and stopped only when a round's target
    # came out at or below the last — a fixed point approached from above and never exactly hit,
    # so a wide face (DejaVu, a Linux runner's fallback) ran out of rounds on a two-panel grid
    # with one bracket each and refused a figure that fitted.
    top = high
    for _attempt in range(12):
        wanted = overshoot(top)
        if not wanted:
            return top
        top = max(wanted)
        for ax, _clearance in clearances:
            value_limits(ax, orientation)(low, top)
        figure = clearances[0][0].get_figure(root=True)
        if figure is not None:
            settle_bracket_labels(figure)
    raise AssertionError(
        "the aligned bracket stacks will not fit under one shared top at this figure size. Make "
        "the panels taller along the value axis, or draw fewer comparisons."
    )


def label_shared_scale_once(
    axes: Iterable[Axes], *, orientation: Orientation = "vertical"
) -> list[Axes]:
    """Print the value numbers on the panels at the grid's edge, and drop them from the rest.

    Six panels on one scale repeating the same six numbers say nothing six times, and the inner
    columns' numbers sit against the neighbouring panel's violins.

    Which panels are at the edge comes from each one's own subplotspec, which is why this is not
    left to the caller. The two callers it replaces looped over `axes[:, 1]`, right for the 2-wide
    they were written against; on a 2x3 grid that blanks the MIDDLE column and leaves the right one
    repeating every number. Nothing errors — the figure carries labels in two columns of three.

    Returns the panels that kept their numbers.
    """
    panels = list(axes)
    require(
        panels,
        "label_shared_scale_once needs at least one axes",
    )
    upright = is_vertical(orientation)
    kept: list[Axes] = []
    for ax in panels:
        spec = ax.get_subplotspec()
        if spec is None:  # not in a grid; it has no neighbours to repeat
            kept.append(ax)
            continue
        edge = spec.is_first_col() if upright else spec.is_last_row()
        if edge:
            kept.append(ax)
        elif upright:
            ax.yaxis.set_tick_params(labelleft=False)
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
    return kept


def align_ticks(axes: Iterable[Axes], *, orientation: Orientation = "vertical") -> list[float]:
    """Give every panel the same value ticks, and return them.

    Each panel chose its own before the scale was shared, from its own data, so a grid arrived with
    five rules in one row and eight in the next — measured on a 2x2, and 7/4/4/4/4/7 on a 2x3. On a
    shared scale that is simply wrong: the panels are being compared, the rules are the thing a
    reader compares them with, and different rules in each panel make the same height look like
    different heights.

    The ticks are chosen once, for the shared range, and trimmed against the marks of the WHOLE
    grid so the bracketing tick clears the tallest panel rather than the first one.
    """
    from matplotlib.ticker import AutoLocator

    from ogviz.layout import ticks_over_data

    panels = list(axes)
    require(
        panels,
        "align_ticks needs at least one axes",
    )
    upright = is_vertical(orientation)
    # Along the VALUE axis. Read without `orientation` this measured y, which on a horizontal grid
    # is the category positions, and trimmed the ticks against a number with no meaning.
    extents = (drawn_value_extent(ax, orientation=orientation) for ax in panels)
    reaches = [extent[1] for extent in extents if extent]
    if not reaches:
        return []

    reference = panels[0]
    axis = reference.yaxis if upright else reference.xaxis
    axis.set_major_locator(AutoLocator())
    reference.figure.canvas.draw()
    ticks_over_data(reference, max(reaches), orientation=orientation)
    picked = reference.get_yticks() if upright else reference.get_xticks()
    chosen = [float(tick) for tick in picked]

    for ax in panels:
        # `set_ticks` grows the view to contain fixed ticks, so the limits are put back after.
        limits = ax.get_ylim() if upright else ax.get_xlim()
        if upright:
            ax.set_yticks(chosen)
            ax.set_ylim(*limits)
        else:
            ax.set_xticks(chosen)
            ax.set_xlim(*limits)
    return chosen


def align_brackets(axes: Iterable[Axes], *, orientation: Orientation = "vertical") -> float | None:
    """Put every panel's bracket stack on one line, and return where that line is.

    `orientation` is which way the panels run: a horizontal panel's bracket crossbar lives on x,
    and reading y there — which this did unconditionally — would have shifted every bracket along
    the CATEGORY axis and dragged the stars with it. `share_value_limits` only ever called this on
    a vertical grid, so the defect was latent; the function is public, so it could not stay so.

    The mean-row argument, at the other end of the panel. Each panel anchors its bracket to ITS OWN
    data, which is right for a panel read alone. On a shared scale it is not: the panel with the
    lowest data gets the lowest bracket and then inherits the tallest panel's ceiling, so it wears a
    gap three times the one its neighbour has. Measured on a six-panel grid before this existed,
    the tightest panel had 0.59 of headroom above its bracket and the loosest had 1.84.

    The line is the highest first-bracket in the grid, so no stack moves down onto its own data.
    Each stack shifts as a unit, which keeps the spacing inside a stack of three exactly as
    `bracket_stack` measured it.

    Returns None where no panel has a bracket.
    """
    # Materialised like its four siblings, which all take `Iterable[Axes]` and are called together
    # — often with a single-use `axes.flat`. This one happens to walk the argument exactly once, so
    # it is safe by accident rather than by construction, and `align_mean_rows` is the standing
    # proof of what a second walk costs: given a generator it silently placed nothing at all.
    upright = is_vertical(orientation)
    stacks = []
    for ax in list(axes):
        lines, stars = _bracket_artists(ax, upright=upright)
        if lines:
            stacks.append((lines, stars, _bracket_top(lines[0], upright=upright)))
    if not stacks:
        return None

    line = max(start for _lines, _stars, start in stacks)
    for lines, stars, start in stacks:
        shift = line - start
        if abs(shift) < 1e-12:
            continue
        for bracket in lines:
            if upright:
                bracket.set_ydata(np.asarray(bracket.get_ydata(), dtype=float) + shift)
            else:
                bracket.set_xdata(np.asarray(bracket.get_xdata(), dtype=float) + shift)
        for star in stars:
            x, y = star.get_position()
            star.set_position((x, float(y) + shift) if upright else (float(x) + shift, y))
    return line


def _bracket_top(bracket, *, upright: bool = True) -> float:
    """The crossbar of a bracket — its highest VALUE, the four-point path being down/across/down."""
    values = bracket.get_ydata() if upright else bracket.get_xdata()
    return float(np.asarray(values, dtype=float).max())


def _bracket_artists(ax: Axes, *, upright: bool = True) -> tuple[list, list]:
    """This panel's bracket lines and their stars, lowest bracket first."""
    lines = sorted(
        (line for line in ax.lines if marked(line, "bracket")),
        key=lambda line: _bracket_top(line, upright=upright),
    )
    stars = [text for text in ax.texts if marked(text, "bracket_star")]
    return lines, stars


def align_mean_rows(
    axes: Iterable[Axes], *, floor: float, orientation: Orientation = "vertical"
) -> float | None:
    """Put every panel's printed means on ONE line AND at one size, and return that line.

    A panel places its means in the middle of the margin below its own data. Once the panels share
    a scale that is wrong: the floor is common and the lowest violin is not, so the row sits at a
    different height in each panel and the eye reads four different rows where there is one kind of
    number. The gap from a row to the frame stops meaning anything.

    The line is the midpoint between the floor and the lowest mark ACROSS the panels, so it clears
    the deepest violin in the grid and is identical everywhere. Returns None where no panel prints
    means.

    Measured in DISPLAY space and converted back, not averaged in data units. "Midway between the
    violin and the frame" is a question about the picture, and the two agree only while the axis is
    linear: on a log axis running 1 to 1000, the data-space midpoint of a gap from 1 to 100 lands
    108 px from the middle of a 308 px gap. Every panel here happens to be linear today, which is
    exactly why the error would have sat unnoticed until the first log axis.

    That conversion is `ogviz.units.midpoint`, which exists to be the one place it is written, and
    was written out by hand here instead — the module had no callers at all.

    `orientation` names which way the PANELS run, as everywhere else. It was absent, and every
    step of the answer read y: the extent, the midpoint, and the coordinate the label was moved
    along. On a horizontal panel that measured the CATEGORY axis, took the midpoint of a floor
    from the other axis, and then moved each label vertically — three wrong answers composing
    into one plausible-looking row. `printed_means` has placed a horizontal row correctly the
    whole time, so this was the half of the pair that could not settle it, which is what made
    `show_means=True` on a horizontal panel decline silently.
    """
    from ogviz.units import midpoint

    # Materialised in the first line, because the body walks it three times — the rows, the extents,
    # then the reference axes. Given a generator that silently placed nothing: the first walk found
    # the rows and the second found an exhausted iterator, so the function returned None and moved
    # not one label. `Iterable` in the signature is what invited it.
    panels = list(axes)
    rows = [text for ax in panels for text in ax.texts if marked(text, "mean_row")]
    if not rows:
        return None
    # Along the VALUE axis. Read without `orientation` this measured y, which on a horizontal grid
    # is the category positions, and trimmed the ticks against a number with no meaning.
    extents = (drawn_value_extent(ax, orientation=orientation) for ax in panels)
    measured = [extent[0] for extent in extents if extent is not None]
    if not measured:
        return None
    lowest = min(measured)
    reference = panels[0]
    figure = reference.get_figure()
    if figure is not None:
        figure.canvas.draw()
    line = midpoint(reference, floor, lowest, orientation=orientation)
    # THE SIZE IS RECONCILED HERE TOO, to the smallest any panel settled at. `printed_means`
    # shrinks a row that would collide with itself, which is per-panel by construction — so a grid
    # whose panels are differently crowded ends up with the row at two sizes, and
    # `qc.mean_rows_unaligned` refuses that on purpose, because the row's size is one of the visual
    # constants that make a shared scale read as one comparison. Measured on a two-panel grid with
    # a roomy panel beside a crowded one: 20.0 pt against 11.5 pt. The MINIMUM is the only choice
    # guaranteed to fit — the larger is what the crowded panel already rejected.
    smallest = min(float(text.get_fontsize()) for text in rows)
    for text in rows:
        across, along = text.get_position()
        text.set_position((line, along) if orientation == "horizontal" else (across, line))
        text.set_fontsize(smallest)
    return line
