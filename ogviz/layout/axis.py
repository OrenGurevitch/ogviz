"""What the value axis shows, and how far the marks on it actually reach.

Both answer questions a panel asks after it has drawn: which ticks belong to the data rather than to
the room held open above it, and where the marks really start and stop — which is not what a
collection's paths say if the collection is a scatter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout.collision import point_offsets

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ogviz.orientation import Orientation


def ticks_over_data(
    ax: Axes, data_high: float | None = None, *, orientation: Orientation = "vertical"
) -> None:
    """Drop value ticks that fall in the room reserved above the data.

    A panel grows its value axis to fit a bracket stack, and the locator then puts ticks up there
    because it sees axis, not meaning. Those ticks and their gridlines say a measurement could sit
    at that height when nothing can — the space is layout, held open for the brackets.

    `data_high` defaults to MEASURING what was drawn, and callers should let it. Passed the data
    maximum instead, it drops the ticks a violin needs: a kernel density body reaches past the
    largest observation, so the tick above that observation is inside the violin and taking it away
    leaves the top of the shape with nothing to read it against. That is what happened — a panel
    whose bodies reached 3.75 was left with its highest tick at 2.0. The same mistake, of using the
    data extent where the DRAWN extent is meant, put a printed mean off the page a few days ago.

    It also makes panels disagree with each other for no reason a reader can see: one whose stack
    happens to clear a round number carries an extra rule and its neighbour does not. That is the
    inconsistency this removes.
    """
    upright = orientation == "vertical"
    if data_high is None:
        extent = drawn_value_extent(ax)
        if extent is None:
            return
        data_high = extent[1]
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


def drawn_value_extent(ax: Axes) -> tuple[float, float] | None:
    """The lowest and highest value any MARK reaches, in data units, or None if nothing is drawn.

    Reading `collection.get_paths()` is the trap, and it cost a panel its layout. For a filled body
    the path IS the shape in data coordinates. For a scatter it is the MARKER OUTLINE — a unit
    circle about the origin, reused at every offset — so a panel of points reports its extent as
    roughly -0.5 to 0.5 whatever the data says. On values of order one that looks plausible; on
    values of order 0.001 it puts the answer nowhere near the panel.

    So: offsets when a collection has them, path vertices when it does not.
    """
    lows: list[float] = []
    highs: list[float] = []
    for collection in ax.collections:
        offsets = point_offsets(collection)
        if offsets is not None:
            lows.append(float(offsets[:, 1].min()))
            highs.append(float(offsets[:, 1].max()))
            continue
        for path in collection.get_paths():
            vertices = np.asarray(path.vertices, dtype=float)
            if vertices.size:
                lows.append(float(vertices[:, 1].min()))
                highs.append(float(vertices[:, 1].max()))
    if not lows:
        return None
    return min(lows), max(highs)
