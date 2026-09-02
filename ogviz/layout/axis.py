"""What the value axis shows, and how far the marks on it actually reach.

Both answer questions a panel asks after it has drawn: which ticks belong to the data rather than to
the room held open above it, and where the marks really start and stop — which is not what a
collection's paths say if the collection is a scatter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout.bounds import panel_prefix
from ogviz.layout.collision import point_offsets
from ogviz.tags import marked

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ogviz.orientation import Orientation


def ticks_over_data(
    ax: Axes, data_high: float | None = None, *, orientation: Orientation = "vertical"
) -> None:
    """Drop value ticks that fall in the room reserved above the data.

    A panel grows its value axis to fit a bracket stack, and the locator then puts ticks up there
    because it sees axis, not meaning. Those ticks and their gridlines say a measurement could sit
    at that height when nothing can — the space is layout, held open for the brackets.

    `data_high` defaults to MEASURING what was drawn, and callers should let it. Passed a value
    BELOW the drawn extent, it drops ticks the marks still need, leaving the top of a shape with
    nothing to read it against.

    The mechanism this docstring used to give for that was wrong, and is corrected here rather than
    quietly deleted: it said a kernel density body reaches past the largest observation. It does
    not. matplotlib evaluates a violin's KDE over the data range, so the body stops exactly at the
    data — measured on `marks.violin` with a normal, a right-skewed and a tight sample, body top
    minus data max is +0.000 in all three. Passing one GROUP's maximum on a panel holding several,
    or one PANEL's maximum on a shared scale, is what actually produces a value below the drawn
    extent. The failure is real; the explanation was not, which is exactly the kind of claim this
    package asks to be measured.

    ON A SHARED SCALE, DO NOT CALL THIS PER PANEL — call `align_ticks` with the axes together.
    `data_high` means the reach across EVERY panel sharing the scale, not this panel's own, and only
    the caller knows which axes are grouped. Measured on two hand-paired panels where the right one
    holds averages of the left one's blocks, so its maximum is necessarily lower: calling this on
    each gives them seven ticks and six, which is the disagreement this function exists to prevent,
    and `panels_disagree_about_ticks` reports it. `align_ticks(axes)` on the same pair gives both
    the same six and the check goes quiet.

    That works for any set of axes, not only a grid — it takes an iterable, and a hand-paired pair
    is a set of two. This docstring used to say such a caller "has to do the same", meaning compute
    the shared maximum themselves; they do not, and doing it by hand is how one panel ends up
    over-bracketed by a tick.

    It also makes panels disagree with each other for no reason a reader can see: one whose stack
    happens to clear a round number carries an extra rule and its neighbour does not. That is the
    inconsistency this removes.

    A CALLER THAT PASSES IT IS WARNED WHEN THE VALUE IS TOO LOW, which is the only way the fix can
    reach the code it was written for. `data_high` was REQUIRED before it gained a default, so every
    caller written against that signature still passes it — positionally, silently, keeping exactly
    the behaviour the default was added to replace. A blanket deprecation on the argument would be
    wrong (`align_ticks` passes it for a real reason: it trims against the whole GRID's marks, not
    one panel's), so what is checked is the actual defect instead — a value BELOW what was drawn,
    which is what passing the data maximum produces and what drops the tick a violin body needs.
    """
    upright = orientation == "vertical"
    if data_high is None:
        extent = drawn_value_extent(ax)
        if extent is None:
            return
        data_high = extent[1]
    else:
        drawn = drawn_value_extent(ax)
        if drawn is not None and data_high < drawn[1] - 1e-9:
            import warnings

            warnings.warn(
                f"ticks_over_data was given data_high={data_high:g}, but the marks on this panel "
                f"reach {drawn[1]:g}. Ticks between the two will be dropped, leaving the top of "
                "the shape with nothing to read against — the case that made this argument "
                "optional. Let it default unless you mean a reach measured across several panels.",
                UserWarning,
                stacklevel=2,
            )
    ticks = ax.get_yticks() if upright else ax.get_xticks()
    # Keep every tick up to the marks AND the first one past them, so the data is BRACKETED. Only
    # dropping what sits above leaves a coarse axis unreadable at the top: a panel with ticks every
    # 2.0 and violins reaching 3.75 kept nothing above 2.0, so the upper 1.75 of every body had no
    # reference beside it. What is worth removing is the LADDER of ticks climbing through the room
    # held open for brackets, not the one that closes the data in.
    below = [float(tick) for tick in ticks if float(tick) <= data_high + 1e-9]
    above = [float(tick) for tick in ticks if float(tick) > data_high + 1e-9]
    kept = below + above[:1]
    if not kept or len(kept) == len(ticks):
        return
    # `set_yticks` FIXES the locator, and matplotlib then grows the view to contain every fixed
    # tick. Dropping the ticks above the data therefore dragged the floor down to the lowest
    # remaining one — on a panel whose ticks ran to zero, the axis reframed itself from zero and
    # the violins ended up in the top third of a panel that had been fitted to them. Restore the
    # limits, which were already correct before the ticks were touched.
    limits = ax.get_ylim() if upright else ax.get_xlim()
    if upright:
        ax.set_yticks(kept)
        ax.set_ylim(*limits)
    else:
        ax.set_xticks(kept)
        ax.set_xlim(*limits)


def _is_furniture(artist) -> bool:
    """Whether this artist is something drawn ABOUT the marks rather than a mark.

    The distinction only started to matter when `drawn_value_extent` learned to read lines and
    patches, and getting it wrong is silent in a way worth spelling out. A BRACKET is drawn in the
    room held open above the data, so counting it makes "how far do the marks reach" answer "to the
    top of the bracket stack" — which is what `ticks_in_the_headroom` subtracts to find the
    headroom, so that check simply stops firing. Measured on a two-group panel with one comparison:
    the extent came back at 3.961, the bracket's own crossbar to the digit, and the check went
    quiet with nine ticks on the axis and two of them above every violin.

    A reference level is the same kind of thing one step along: a threshold at 100% says nothing
    about where the bars got to. A backdrop spans the panel by construction.
    """
    return any(marked(artist, tag) for tag in ("bracket", "reference", "backdrop"))


def drawn_value_extent(
    ax: Axes, *, orientation: Orientation = "vertical", include_furniture: bool = False
) -> tuple[float, float] | None:
    """The lowest and highest value any MARK reaches, in data units, or None if nothing is drawn.

    Reading `collection.get_paths()` is the trap, and it cost a panel its layout. For a filled body
    the path IS the shape in data coordinates. For a scatter it is the MARKER OUTLINE — a unit
    circle about the origin, reused at every offset — so a panel of points reports its extent as
    roughly -0.5 to 0.5 whatever the data says. On values of order one that looks plausible; on
    values of order 0.001 it puts the answer nowhere near the panel.

    So: offsets when a collection has them, path vertices when it does not.

    BARS AND LINES COUNT TOO. This read `ax.collections` alone, which is every mark a violin panel
    draws and none of the marks a bar or line panel draws — measured, a panel of bars plus a line
    returned None. Three callers read that as "nothing is drawn here" and quietly did nothing:
    `ticks_over_data`, so a bar panel kept its ticks in the room held open for brackets;
    `align_mean_rows`; and `rows_outside_their_panel`. Right for the panels it was written against
    and silently absent everywhere else, which is the worst way for a check to be wrong.

    `include_furniture` keeps the brackets, reference levels and backdrops that are otherwise
    excluded. Two different questions are being asked of the same walk: "how far do the MARKS
    reach", which is what a tick trim or a label centring needs, and "is there anything up there at
    all", which is what deciding whether an axis runs past its content needs. A bracket occupies
    the headroom it was given, so for the second question it counts.

    `orientation` picks WHICH coordinate is the value one. It reads y by default, which is what
    every caller written before `settle_axis_labels` wants and what a vertical panel means by "how
    far do the marks reach"; a horizontal panel's value axis is x, and asking about y there returns
    the category positions, which is a number with no meaning.
    """
    axis = 1 if orientation == "vertical" else 0
    lows: list[float] = []
    highs: list[float] = []
    for collection in ax.collections:
        offsets = point_offsets(collection)
        if offsets is not None:
            # An EMPTY cloud reaches nothing. `point_offsets` used to answer None for one, because
            # it identified a cloud by having more than one offset; it answers by type now, so a
            # `scatter` with no data comes back here as an empty array and `.min()` would raise.
            if offsets.size:
                lows.append(float(offsets[:, axis].min()))
                highs.append(float(offsets[:, axis].max()))
            continue
        for path in collection.get_paths():
            vertices = np.asarray(path.vertices, dtype=float)
            if vertices.size:
                lows.append(float(vertices[:, axis].min()))
                highs.append(float(vertices[:, axis].max()))
    for line in ax.lines:
        if _is_furniture(line) and not include_furniture:
            continue
        values = np.asarray(line.get_ydata() if axis else line.get_xdata(), dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            lows.append(float(values.min()))
            highs.append(float(values.max()))
    for patch in ax.patches:
        if _is_furniture(patch) and not include_furniture:
            continue
        vertices = np.asarray(patch.get_path().transformed(patch.get_patch_transform()).vertices)
        finite = vertices[np.isfinite(vertices[:, axis]), axis]
        if finite.size:
            lows.append(float(finite.min()))
            highs.append(float(finite.max()))
    if not lows:
        return None
    return min(lows), max(highs)


def marks_span_px(ax: Axes, *, along: int) -> tuple[float, float] | None:
    """Where the marks reach along one axis, in display pixels, or None if that cannot be said.

    CLAMPED TO THE VIEW before it is transformed, and that is not tidiness. `drawn_value_extent`
    answers in data units and a mark may sit outside the limits — a line reaching zero on a LOG
    axis is the ordinary case, and `transData` maps zero there to minus infinity. Unclamped, a log
    panel in the gallery put its y-label 419443 px off the page, which `text_off_canvas` caught on
    the very next save. Clamping asks the question that was meant anyway: the label names the marks
    a reader can SEE, and one off the bottom of the axis is not among them.
    """
    orientation = "vertical" if along else "horizontal"
    extent = drawn_value_extent(ax, orientation=orientation)
    if extent is None:
        return None
    low, high = sorted((ax.yaxis if along else ax.xaxis).get_view_interval())
    inside = (max(extent[0], low), min(extent[1], high))
    if not inside[0] <= inside[1]:
        return None  # every mark is outside the view; there is nothing on screen to centre on
    ends = [
        ax.transData.transform((0.0, value) if along else (value, 0.0))[along] for value in inside
    ]
    if not all(np.isfinite(end) for end in ends):
        return None
    return min(ends), max(ends)


# Below this the move is not worth making: a couple of pixels is inside the width of the glyphs
# being centred, and re-placing a label every save for it is churn in the rendered file.
LABEL_DRIFT_PX = 2.0


def settle_axis_labels(fig: Figure, *, drift: float = LABEL_DRIFT_PX) -> list[str]:
    """Centre each axis label on the MARKS it names, rather than on the axes box.

    matplotlib centres an axis label on the axes rectangle, which is right for a panel whose data
    fills it and wrong for every panel this package pads asymmetrically — and padding asymmetrically
    is something ogviz does on purpose. A violin panel grows its top to hold a bracket stack;
    `central_clearance` reserves a lane at the bottom for a printed mean. `ticks_over_data` then
    drops the ticks that would sit in the headroom, exactly because nothing can be measured up
    there. So the marks end up low in the box, the label stays at the box's middle, and it names
    reserved space with nothing in it.

    THE REFERENCE IS THE MARKS, and getting that right took two wrong answers worth recording,
    because each looked like it described what a reader sees:

    the SPINE — it spans the whole axes box, so its middle IS the box's middle, and asking whether
    the label is centred on it can only ever answer yes. That check passes on the broken figure;

    the TICK BLOCK — better, and still wrong. A locator that happens to place no tick near the top
    of an ordinary, unpadded axis puts the block's middle well below the box's, so centring on it
    moves a label that was never wrong. Measured across five figure sizes and three tick counts,
    the tick block called an unpadded panel up to 65 px out of true.

    Against `drawn_value_extent` — the reach of the marks, with brackets, references and backdrops
    excluded as furniture — the same panels measure **0.00 px at every size and tick count**, and a
    panel with headroom for a bracket stack measures 154. That is the signal cleanly separated from
    the noise, which is what a threshold needs before it is worth having.

    ONLY THE ALONG-AXIS COORDINATE IS SET, and that is what makes this safe rather than a fight
    with the layout engine. `YAxis._update_label_position` recomputes the label's x from the tick
    boxes on every draw and passes its y through untouched — `x, y = self.label.get_position()`
    and then `set_position((bbox.x0 - pad, y))`. `XAxis` is the mirror. So writing the along-axis
    coordinate leaves the perpendicular placement to matplotlib, which is the half that has to keep
    negotiating with the tick labels' width. `set_label_coords` would freeze both and is the wrong
    tool: it sets `_autolabelpos` False and the label stops following its own ticks.

    Called by `save` alongside the other `settle_*` passes, and for the same reason — nothing is in
    its final place until the figure has been laid out and drawn.
    """
    from ogviz.layout.render import ensure_rendered

    ensure_rendered(fig)
    moved: list[str] = []
    for ax in fig.axes:
        for axis, along in ((ax.xaxis, 0), (ax.yaxis, 1)):
            label = axis.label
            if not label.get_text().strip() or not label.get_visible():
                continue
            span = marks_span_px(ax, along=along)
            if span is None:  # nothing drawn, or nothing visible; no data to centre on
                continue
            box = ax.get_window_extent()
            reach = box.height if along else box.width
            if not reach:
                continue
            edge = box.y0 if along else box.x0
            middle = (span[0] + span[1]) / 2.0
            current = label.get_position()
            at = edge + current[along] * reach
            if abs(at - middle) < drift:
                continue
            target = (middle - edge) / reach
            # The OTHER coordinate is handed back exactly as it was found. matplotlib overwrites it
            # on the next draw anyway; passing it through unchanged is what keeps that true.
            label.set_position((current[0], target) if along else (target, current[1]))
            moved.append(
                f"{panel_prefix(fig, ax)}centred {label.get_text()[:40]!r} on the marks, "
                f"{at - middle:+.0f} px from where the axes box put it"
            )
    return moved
