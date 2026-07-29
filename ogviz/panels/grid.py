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

from ogviz.layout import drawn_value_extent

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes

    from ogviz.orientation import Orientation


def share_value_limits(
    axes: Iterable[Axes], *, orientation: Orientation = "vertical"
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

    Returns the shared (low, high).
    """
    panels = list(axes)
    assert panels, "share_value_limits needs at least one axes"
    reader = (lambda ax: ax.get_ylim()) if orientation == "vertical" else (lambda ax: ax.get_xlim())
    spans = [reader(ax) for ax in panels]
    low = min(bounds[0] for bounds in spans)
    high = max(bounds[1] for bounds in spans)
    for ax in panels:
        if orientation == "vertical":
            ax.set_ylim(low, high)
        else:
            ax.set_xlim(low, high)
    if orientation == "vertical":
        align_mean_rows(panels, floor=low)
    return low, high


def align_mean_rows(axes: Iterable[Axes], *, floor: float) -> float | None:
    """Put every panel's printed means on ONE line, and return that line.

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
    """
    rows = [text for ax in axes for text in ax.texts if getattr(text, "ogviz_mean_row", False)]
    if not rows:
        return None
    extents = [drawn_value_extent(ax) for ax in axes]
    measured = [extent[0] for extent in extents if extent is not None]
    if not measured:
        return None
    lowest = min(measured)
    reference = next(iter(axes))
    reference.figure.canvas.draw()
    to_pixels, to_data = reference.transData, reference.transData.inverted()
    floor_px = float(to_pixels.transform((0.0, floor))[1])
    lowest_px = float(to_pixels.transform((0.0, lowest))[1])
    line = float(to_data.transform((0.0, (floor_px + lowest_px) / 2.0))[1])
    for text in rows:
        text.set_position((text.get_position()[0], line))
    return line
