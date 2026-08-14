"""Whether the parts of a figure ended up where the layout intended.

These are the checks with a memory: each compares what is on the page against something the drawing
code recorded when it knew the answer — a mean row that a grid put on one line, a layout the engine
declined to apply, the headroom a bracket stack reserved. A figure can be perfectly legible and
still have quietly lost one of those, which is why none of them shows up as a collision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ogviz.layout.render import ensure_rendered
from ogviz.qc.reading import (
    bracket_tops_px,
    orientation_of,
)
from ogviz.tags import marked

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _draws_anything(ax: Axes) -> bool:
    """Whether this axes carries marks of any kind, whatever kind they are."""
    return bool(ax.collections or ax.patches or ax.lines)


def mean_rows_unaligned(fig: Figure) -> list[str]:
    """Panels sharing a value scale must print their means on one line, at one size.

    Four rows at four heights read as four different kinds of number. The row's distance from the
    frame is a visual constant a reader uses without noticing, and it only means something if it is
    the same in every panel being compared.

    Only checked where the panels genuinely share a scale — panels on different scales are separate
    figures that happen to share a page, and there is no line for them to share.
    """
    rows = [(ax, text) for ax in fig.axes for text in ax.texts if marked(text, "mean_row")]
    if len({id(ax) for ax, _text in rows}) < 2:
        return []
    # The VALUE axis, whichever screen axis that is. A row of printed means only exists on a
    # vertical panel today, so reading `get_ylim` was right and looked like a fact about y.
    scales = {
        tuple(
            round(v, 9)
            for v in (ax.get_ylim() if orientation_of(ax) == "vertical" else ax.get_xlim())
        )
        for ax, _text in rows
    }
    if len(scales) > 1:
        return []
    heights = {round(float(text.get_position()[1]), 6) for _ax, text in rows}
    sizes = {round(float(text.get_fontsize()), 3) for _ax, text in rows}
    complaints = []
    if len(heights) > 1:
        complaints.append(
            f"printed means sit at {len(heights)} different heights: {sorted(heights)}"
        )
    if len(sizes) > 1:
        complaints.append(f"printed means are set at {len(sizes)} different sizes: {sorted(sizes)}")
    return complaints


def rows_outside_their_panel(fig: Figure) -> list[str]:
    """A printed mean must land between the frame and the marks it belongs to.

    The failure this exists for put the row at -0.25 on an axis running 0.0004 to 0.006 — off the
    panel entirely — because the code measuring "the lowest mark" was reading a scatter's MARKER
    OUTLINE instead of its offsets, and a marker outline is a unit circle about the origin whatever
    the data is. On values of order one that is a small error; on values of order 0.001 the answer
    is not on the page.

    Cheap, and it asks the question directly rather than trusting the measurement that failed.
    """
    from ogviz.layout import drawn_value_extent

    complaints: list[str] = []
    for ax in fig.axes:
        rows = [text for text in ax.texts if marked(text, "mean_row")]
        if not rows:
            continue
        extent = drawn_value_extent(ax)
        if extent is None:
            continue
        floor, _top = ax.get_ylim()
        for text in rows:
            where = float(text.get_position()[1])
            if not floor <= where <= extent[0]:
                complaints.append(
                    f"the printed mean {text.get_text()!r} sits at {where:g}, outside the band "
                    f"between the frame ({floor:g}) and the lowest mark ({extent[0]:g})"
                )
    return complaints


def layout_not_applied(fig: Figure) -> list[str]:
    """A figure whose layout engine declined, and which is therefore on default margins.

    `tight_layout` warns and does nothing when the axis decorations will not fit the rect it was
    given, leaving whatever margins were already set. `fit_under_header` catches that and returns
    whether it ran — and nothing read the answer, so a figure could be laid out by nobody and look
    merely a bit loose. Recorded on the figure so the gate can say it out loud.

    Not fatal on its own: default margins are usually survivable and the rest of the checks still
    measure what was actually drawn. It is here because "nobody laid this out" should be a sentence
    someone reads, not a warning swallowed by a build log.

    A figure whose gridspec PINS its own margins is deliberately NOT reported, and telling the two
    apart is why the tag carries a reason rather than a flag. `panel_row` and `panel_grid` both
    pin, `tight_layout` skips every axes when they do, and that is the arrangement working as
    intended — a complaint would fire on every figure either of them builds. Whether those pinned
    margins are wide ENOUGH is a different question, and `text_off_canvas` and `clipped_artists`
    are the ones that ask it.
    """
    if marked(fig, "layout_refused"):
        return [
            "the layout engine declined — the decorations do not fit, so this figure kept default "
            "margins. Give it more height, or shorten what grows out of the axes"
        ]
    return []


def panels_disagree_about_ticks(fig: Figure) -> list[str]:
    """Panels on one value scale must carry the same value ticks.

    The rules are what a reader compares panels with, so different rules in each panel make the
    same height look like different heights. A grid arrived with five in one row and eight in the
    next, because each panel chose its own from its own data before the scale was shared.

    Only checked where the panels genuinely share a scale — panels on different scales are separate
    figures that happen to share a page.

    It used to skip any axes with no `collections`, which is every panel that is not a violin or a
    scatter: measured, two BAR panels on one 0-2.5 scale carrying three ticks and five produced no
    complaint at all, which is the defect this exists for, on the panel type most likely to have it.
    The filter was a leftover from the violin grid it was written against. What replaces it is a
    question about the panel rather than about its artist types — does it draw anything at all.
    """
    scales: dict[tuple[float, float], set[tuple[float, ...]]] = {}
    for ax in fig.axes:
        if not ax.axison or not _draws_anything(ax):
            continue
        upright = orientation_of(ax) == "vertical"
        low, high = ax.get_ylim() if upright else ax.get_xlim()
        key = (round(low, 9), round(high, 9))
        drawn = ax.get_yticks() if upright else ax.get_xticks()
        ticks = tuple(round(float(t), 9) for t in drawn if low <= t <= high)
        scales.setdefault(key, set()).add(ticks)
    complaints = []
    for (low, high), sets in scales.items():
        if len(sets) > 1:
            counts = sorted(len(one) for one in sets)
            complaints.append(
                f"panels sharing the scale {low:g}..{high:g} carry different value ticks "
                f"({counts} of them) — the rules a reader compares the panels with disagree"
            )
    return complaints


def _data_reach(ax: Axes) -> float | None:
    """The highest value any mark reaches, or None where the panel draws no marks."""
    from ogviz.layout import drawn_value_extent

    extent = drawn_value_extent(ax)
    return extent[1] if extent is not None else None


def ticks_in_the_headroom(fig: Figure) -> list[str]:
    """A value tick above every mark on the panel, in the space reserved for brackets.

    The space above the data is layout, held open so a bracket stack has somewhere to go. A tick
    and its gridline there say a measurement could sit at that height when none can, and they make
    panels disagree for no reason a reader can see — one whose stack happens to clear a round
    number carries an extra rule, its neighbour does not, and the two are meant to be compared.
    """
    ensure_rendered(fig)
    complaints: list[str] = []
    for ax in fig.axes:
        if not bracket_tops_px(ax):
            # No stack, so nothing is being held open. A scatter with a top margin is ordinary
            # breathing room, not reserved space, and ticks in it are the axis doing its job.
            continue
        # Across everything sharing this y axis, not just this panel. Two panels deliberately on
        # ONE scale — a blocks panel beside its condition averages, so the two stay comparable —
        # have a tick that is real on the left and above every mark on the right, and a per-axes
        # reach reported the right one for a tick the shared scale requires. The check survives:
        # a tick above ALL the shared data is still in the headroom.
        reaches = [
            _data_reach(sibling)
            for sibling in ax.get_shared_y_axes().get_siblings(ax)
            if sibling.get_visible()
        ]
        measured = [value for value in reaches if value is not None]
        if not measured:
            continue
        reach = max(measured)
        _low, high = ax.get_ylim()
        stray = [float(tick) for tick in ax.get_yticks() if reach + 1e-9 < float(tick) <= high]
        # One tick above the marks is the axis closing the data in, and is wanted: without it a
        # coarse axis leaves the top of a violin with nothing to read against. Two or more is a
        # ladder climbing through the space held open for brackets.
        if len(stray) > 1:
            complaints.append(
                f"ticks {stray[1:]} climb above the marks into the bracket headroom "
                f"(one bracketing tick at {stray[0]:g} is expected)"
            )
    return complaints


# A header clearing the panels by less than this reads as crowded rather than as deliberate.
# ADVISORY, and the number is this package's own — importing the one it was borrowed from would
# have been wrong. Measured 2026-08-13 across the gallery, header bottom against the highest panel
# text below it: the tightest shipped figure clears by 17.6 px, the next by 18.3, then 26.6 and
# 36.4, with a median of 52. The idea came from a project whose tightest was 66 px and which set
# its floor at 32 — and 32 here would complain about two figures that ship and are correct. So the
# floor sits below everything shipped, as a regression guard rather than a target: a figure that
# drops under 12 px has changed for a reason worth looking at.
CROWDED_HEADER_PX = 12.0


def _visible(texts) -> list:
    return [text for text in texts if text.get_text().strip() and text.get_visible()]


def header_crowds_the_panels(fig: Figure, *, floor: float = CROWDED_HEADER_PX) -> list[str]:
    """A figure header sitting closer to the panels' own text than reads as deliberate.

    The one pair of labels no overlap rule catches, because they are positioned by two mechanisms
    that do not know about each other: a header is laid out against the CANVAS and a panel title
    against its AXES. `text_overlaps` fires when they touch, and a figure is uncomfortable well
    before that — several rounds of "this is too tight" have been caught by eye here and by nothing
    else, which is what makes it worth a number.

    ONLY WHEN THE GAP IS POSITIVE, and that is not a detail. Measured naively — the lowest figure
    text against the highest panel text anywhere — two gallery figures come back hundreds of pixels
    NEGATIVE: a table panel's top row of cells and a coupling grid's upper scatter legitimately sit
    above where the header ends, and neither is crowding anything. A negative gap means either a
    real overlap, which `text_overlaps` owns and reports properly, or text that is not in the
    header's band at all. So this speaks only about the case it can actually judge — clear, but
    not clear enough — and stays silent on the rest rather than inventing a complaint.

    Advisory. It is a judgement about comfort, and `guard` reports it beside `dead_space` rather
    than failing a build on it.
    """
    ensure_rendered(fig)
    header = _visible(fig.texts)
    panel = _visible([text for ax in fig.axes for text in ax.texts])
    panel += [ax.title for ax in fig.axes if ax.get_title().strip()]
    if not header or not panel:
        return []
    lowest = min(text.get_window_extent().y0 for text in header)
    below = [text for text in panel if text.get_window_extent().y1 <= lowest]
    if not below:
        return []
    gap = lowest - max(text.get_window_extent().y1 for text in below)
    if gap >= floor:
        return []
    return [
        f"the header clears the panels' own text by {gap:.0f} px, under the {floor:.0f} px that "
        "reads as deliberate — lower the panels, or shorten the subtitle"
    ]


# A label further than this from the middle of the marks it names is not naming them. Measured
# 2026-08-13: an ordinary unpadded panel sits at 0.00 px against this reference at every figure
# size and tick count tried, and a panel holding bracket headroom at 154 — so signal and noise are
# cleanly separated and the floor only has to sit between them. It is not a comfort judgement like
# `header_crowds_the_panels`: a label centred on reserved headroom names empty space.
LABEL_OFF_ITS_MARKS_PX = 8.0


def value_label_off_its_marks(fig: Figure, *, floor: float = LABEL_OFF_ITS_MARKS_PX) -> list[str]:
    """An axis label centred on the axes box rather than on the marks it names.

    matplotlib centres an axis label on the axes rectangle, which is correct for a panel whose data
    fills it. This package pads panels asymmetrically ON PURPOSE — a violin panel grows its top for
    a bracket stack, `central_clearance` holds a lane at the bottom for a printed mean — and
    `ticks_over_data` then drops the ticks that would sit in that headroom, because nothing can be
    measured there. The marks end up low in the box and the label stays at the box's middle,
    floating in the reserved space with nothing beside it.

    So the panel is mislabelled in a way no overlap or clipping rule can see: nothing collides,
    nothing leaves the canvas, and the label simply names a part of the axis that carries no data.
    It was found by a reader looking at a figure, which is the definition of a defect this package
    should have caught.

    Measured against the MARKS, not the ticks and not the spine — `settle_axis_labels` records what
    each of those two wrong references cost, and both of them pass on the broken figure or fire on
    a correct one.

    `save` runs `settle_axis_labels` and this stays quiet for anything written through it. What it
    catches is a figure saved another way, or a label a caller placed by hand — and `repair` fixes
    it, so `--fix` closes it without a decision.

    `guard()` REPORTS this and does not repair it, which is deliberate: guard polices somebody
    else's `savefig` and has never moved an artist on them. So a project that guards rather than
    saves will see it newly, and the complaint names the one call that closes it rather than
    leaving them to find it.
    """
    from ogviz.layout.axis import marks_span_px

    ensure_rendered(fig)
    complaints: list[str] = []
    for index, ax in enumerate(fig.axes):
        for axis, along, name in ((ax.xaxis, 0, "x"), (ax.yaxis, 1, "y")):
            label = axis.label
            if not label.get_text().strip() or not label.get_visible():
                continue
            span = marks_span_px(ax, along=along)
            if span is None:
                continue
            drawn = label.get_window_extent()
            middle = (drawn.y0 + drawn.y1) / 2 if along else (drawn.x0 + drawn.x1) / 2
            off = middle - (span[0] + span[1]) / 2.0
            if abs(off) < floor:
                continue
            # The direction in the words a reader of THIS axis uses. It said above/below for both,
            # so an x-label was reported as sitting "above" the marks when it had moved sideways.
            if name == "x":
                way = "right" if off > 0 else "left"
            else:
                way = "above" if off > 0 else "below"
            complaints.append(
                f"axes {index}: the {name}-label {label.get_text()[:40]!r} sits {abs(off):.0f} px "
                f"{way} of the middle of the marks it names — it is centred on the axes box, "
                "which is not where the data is. `settle_axis_labels(fig)` moves it; `save` and "
                "`repair` already do"
            )
    return complaints
