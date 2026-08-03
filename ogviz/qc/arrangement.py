"""Whether the parts of a figure ended up where the layout intended.

These are the checks with a memory: each compares what is on the page against something the drawing
code recorded when it knew the answer — a mean row that a grid put on one line, a layout the engine
declined to apply, the headroom a bracket stack reserved. A figure can be perfectly legible and
still have quietly lost one of those, which is why none of them shows up as a collision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ogviz.qc.reading import (
    bracket_tops_px,
)
from ogviz.tags import marked

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
    scales = {tuple(round(v, 9) for v in ax.get_ylim()) for ax, _text in rows}
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
    """
    scales: dict[tuple[float, float], set[tuple[float, ...]]] = {}
    for ax in fig.axes:
        if not ax.axison or not ax.collections:
            continue
        low, high = ax.get_ylim()
        key = (round(low, 9), round(high, 9))
        ticks = tuple(round(float(t), 9) for t in ax.get_yticks() if low <= t <= high)
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
    fig.canvas.draw()
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
