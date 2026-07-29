"""Whether a label lands on the data, and where it could go instead.

`text_overlaps` catches labels colliding with other labels. This catches the other half: a label
colliding with the marks. They are different problems and need different geometry — two labels are
two rectangles, but a filled area is nothing like its bounding box, and a rising line only occupies
a thin diagonal of the box that contains it. Testing rectangles against rectangles reports a label
sitting in the empty corner above a curve as a collision, and misses one sitting under it.

So this tests the drawn PATHS. `Path.intersects_bbox` answers exactly the question being asked —
does this outline, filled or not, enter this rectangle — for a line, a violin body, a bar, or a
`fill_between` polygon, without any of them needing a special case.

Two kinds of thing can sit under a label, and they want opposite fixes:

  the marks       a label over the data must MOVE; knocking a hole in the data hides it
  the decoration  a label over a gridline or a reference line may stay, if it knocks the line
                  out behind itself, which is what the value labels already do

`clear_position` does the moving. It is a search, not a formula: try the anchor, then positions
spiralling outward from it, and take the first that touches nothing. A formula would need to model
the free space, and the free space is whatever shape the data leaves behind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text as _Text
from matplotlib.transforms import Bbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.path import Path
    from matplotlib.text import Text
    from numpy.typing import NDArray

# The search walks out in rings until it has covered the panel. Sized from the axes rather than
# fixed in pixels: a fixed 14 x 6 px reach covers 84 px, which is a tenth of a normal panel, so it
# would report "nowhere is clear" while most of the figure sat empty.
SEARCH_RINGS = 40  # rings from the anchor to the far corner
SEARCH_DIRECTIONS = 24  # candidates per ring, one every 15 degrees
PADDING_PX = 3.0  # breathing room between a label's ink and whatever it is avoiding
# Set on labels the library places AGAINST a particular mark on purpose — a significance star over
# its bracket, a value label past its whisker, a strip star beside its row. Each has a dedicated
# check measuring that relationship far more precisely than "do these two boxes touch", so the
# general "is this label on the data" test must leave them alone or it would drag them off the
# thing they label. Only for deliberate placement: a label a caller drops onto a curve has no
# such claim, and is exactly what this module exists to catch.
#
# It excuses a label from the DATA check, and from nothing else. Reading it as a blanket pardon is
# what let "league average" sit on a bar unreported: the label is fixed to its line vertically and
# free to slide along it, so it was exempt from the one axis it was actually free on.
ANCHORED = "ogviz_anchored"


def decoration_ids(ax: Axes) -> set[int]:
    """Artists that are the frame rather than the finding, identified by id.

    A gridline is not data. Neither is the spine. A label may cross either, provided it knocks it
    out; treating them as marks would send every label on a gridded panel off to hunt for space
    that does not exist.
    """
    lines = [*ax.get_xgridlines(), *ax.get_ygridlines()]
    return {id(line) for line in lines}


def data_paths(ax: Axes) -> list[tuple[Path, bool]]:
    """Every mark in the axes as a display-space path, paired with whether it is filled.

    Offsets are resolved rather than ignored: a scatter's paths are one marker outline at the
    origin, repeated at every offset, so reading `get_paths` alone would test one point near (0, 0)
    and pronounce the whole cloud clear.
    """
    skip = decoration_ids(ax)
    found: list[tuple[Path, bool]] = []
    for artist in [*ax.lines, *ax.collections, *ax.patches, *ax.images]:
        if id(artist) in skip or not artist.get_visible():
            continue
        found.extend(_paths_of(artist))
    return found


def _paths_of(artist: Artist) -> Iterator[tuple[Path, bool]]:
    if isinstance(artist, Line2D):
        yield artist.get_path().transformed(artist.get_transform()), False
    elif isinstance(artist, Patch):
        yield artist.get_path().transformed(artist.get_transform()), True
    elif isinstance(artist, AxesImage):
        corners = artist.get_extent()
        box = Bbox.from_extents(corners[0], corners[2], corners[1], corners[3])
        yield _bbox_path(box.transformed(artist.get_transform())), True
    elif isinstance(artist, Collection):
        yield from _collection_paths(artist)


def point_offsets(collection: Collection) -> NDArray[np.float64] | None:
    """The per-point positions if this collection is a point cloud, else None.

    The one question that has to be asked the same way everywhere. A scatter carries ONE path — the
    marker outline, a small shape about the origin — repeated at every offset; a filled body carries
    a path that is the shape itself and no meaningful offsets. Read the wrong one and a point cloud
    reports its extent as roughly -0.5 to 0.5 whatever the data says, which is how a panel in ppm
    ended up with its printed means off the page.

    Two callers asked it with two different conditions before this existed, which is worse than
    duplication: they could disagree about the same collection.
    """
    offsets = np.asarray(collection.get_offsets(), dtype=float)
    if offsets.shape[0] > 1 and len(collection.get_paths()) <= 1:
        return offsets
    return None


def _collection_paths(collection: Collection) -> Iterator[tuple[Path, bool]]:
    from matplotlib.path import Path as MplPath

    offsets = point_offsets(collection)
    if offsets is not None:
        # The marker outline is sized in points, so the honest cheap test is the offset points
        # themselves, as a single unclosed path.
        yield MplPath(collection.get_offset_transform().transform(offsets)), False
        return
    transform = collection.get_transform()
    for path in collection.get_paths():
        yield path.transformed(transform), True


def _bbox_path(box: Bbox):
    from matplotlib.path import Path as MplPath

    return MplPath(
        [
            (box.x0, box.y0),
            (box.x1, box.y0),
            (box.x1, box.y1),
            (box.x0, box.y1),
            (box.x0, box.y0),
        ]
    )


def _padded(box: Bbox, padding: float) -> Bbox:
    return Bbox.from_extents(box.x0 - padding, box.y0 - padding, box.x1 + padding, box.y1 + padding)


def hits_data(ax: Axes, box: Bbox, *, padding: float = PADDING_PX) -> int:
    """How many marks enter `box`, which is in display pixels."""
    target = _padded(box, padding)
    return sum(1 for path, filled in data_paths(ax) if path.intersects_bbox(target, filled=filled))


def hits_decoration(ax: Axes, box: Bbox, *, padding: float = 0.0) -> int:
    """How many gridlines enter `box` — the things a label may cross if it knocks them out."""
    target = _padded(box, padding)
    lines = [*ax.get_xgridlines(), *ax.get_ygridlines()]
    return sum(
        1
        for line in lines
        if line.get_visible()
        and line.get_path().transformed(line.get_transform()).intersects_bbox(target, filled=False)
    )


def text_box(text: Text) -> Bbox:
    """The label's own extent, excluding any arrow it carries.

    `Annotation.get_window_extent` spans the text AND its arrow, and the arrow is drawn to touch
    the data on purpose. Measuring the pair would report every annotation as sitting on the marks,
    then move it, and the arrow would grow to follow — a search that can never succeed. The base
    `Text` method measures the string alone, which is the thing that has to be legible.
    """
    return _Text.get_window_extent(text)


def _knocked_out(text: Text) -> bool:
    """Whether the label carries an opaque box, which is what makes crossing a line acceptable."""
    patch = text.get_bbox_patch()
    if patch is None:
        return False
    # get_facecolor may hand back a name, a hex string or an RGBA tuple; to_rgba normalises all
    # three and folds in the patch's own alpha, so opacity is read once from one representation.
    red, green, blue, alpha = to_rgba(patch.get_facecolor(), patch.get_alpha())
    del red, green, blue
    return bool(alpha >= 1.0 and patch.get_visible())


def clear_position(
    ax: Axes,
    box: Bbox,
    *,
    rings: int = SEARCH_RINGS,
    directions: int = SEARCH_DIRECTIONS,
    padding: float = PADDING_PX,
) -> tuple[float, float] | None:
    """A display-space offset that lifts `box` clear of the marks, or None if there is none.

    Searched outward in rings so the label ends up as near its anchor as the data allows: the first
    ring is a few pixels away in sixteen directions, and each ring after that is one step further.
    Candidates that leave the axes are rejected — a label outside the panel it belongs to is not an
    improvement on one over a curve.
    """
    if not hits_data(ax, box, padding=padding):
        return (0.0, 0.0)
    inside = ax.get_window_extent()
    reach = float(np.hypot(inside.width, inside.height))
    step = reach / rings
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    for ring in range(1, rings + 1):
        radius = ring * step
        for angle in angles:
            offset = (float(radius * np.cos(angle)), float(radius * np.sin(angle)))
            moved = Bbox.from_extents(
                box.x0 + offset[0],
                box.y0 + offset[1],
                box.x1 + offset[0],
                box.y1 + offset[1],
            )
            if not inside.containsx(moved.x0) or not inside.containsx(moved.x1):
                continue
            if not inside.containsy(moved.y0) or not inside.containsy(moved.y1):
                continue
            if not hits_data(ax, moved, padding=padding):
                return offset
    return None


def text_over_data(fig: Figure) -> list[str]:
    """Labels sitting on the marks, and labels crossing a gridline with nothing behind them.

    Both are the same visual failure — a string competing with ink for the same pixels — but they
    want opposite fixes, so they are reported as different complaints. A label on the DATA has to
    move, because a knockout box there would punch a hole in the finding. A label merely crossing a
    GRIDLINE can stay where it is and knock the line out behind itself.
    """
    fig.canvas.draw()
    complaints: list[str] = []
    for ax in fig.axes:
        for text in ax.texts:
            content = text.get_text().strip()
            if not content or not text.get_visible():
                continue
            if getattr(text, ANCHORED, False):
                continue
            box = text_box(text)
            struck = hits_data(ax, box)
            if struck:
                complaints.append(f"{content[:40]!r} sits on {struck} mark(s) — it has to move")
            elif hits_decoration(ax, box) and not _knocked_out(text):
                complaints.append(
                    f"{content[:40]!r} crosses a gridline with nothing behind it — knock it out"
                )
    return complaints


def annotate_clear(
    ax: Axes,
    text: str,
    xy: tuple[float, float],
    *,
    prefer: tuple[float, float] | None = None,
    color: str | None = None,
    fontsize: float | None = None,
    fontweight: str = "bold",
    arrow: bool = True,
    knockout: bool = False,
    **kwargs: object,
):
    """Annotate `xy` with a label placed where it does not land on the data.

    The arrow keeps pointing at `xy`; only the text moves. That is the whole trick — the anchor is
    a claim about the data and must not move, while the text is just a string that has to live
    somewhere legible.

    `prefer` is the starting position in DATA coordinates. Given none, the label starts a little up
    and to the right of its anchor and the search takes it from there. If nothing in the axes is
    clear the label stays at its preferred spot rather than being flung outside the panel: a
    crowded figure is the caller's to fix, and silently relocating a label to a corner is worse
    than leaving it where the caller asked and failing the check.
    """
    start = prefer if prefer is not None else _default_prefer(ax, xy)
    annotation = ax.annotate(
        text,
        xy=xy,
        xytext=start,
        textcoords="data",
        color=color,
        fontsize=fontsize,
        fontweight=fontweight,
        arrowprops=(
            {"arrowstyle": "-", "color": color or "0.3", "linewidth": 1.4, "shrinkB": 4.0}
            if arrow
            else None
        ),
        **kwargs,  # type: ignore[arg-type]
    )
    if knockout:
        from ogviz.theme import KNOCKOUT_PAD, page_color

        annotation.set_bbox(
            {
                "facecolor": page_color(),
                "edgecolor": "none",
                "pad": KNOCKOUT_PAD,
                "boxstyle": "square",
            }
        )
    ax.figure.canvas.draw()
    offset = clear_position(ax, text_box(annotation))
    if offset is None or offset == (0.0, 0.0):
        return annotation
    moved = ax.transData.inverted().transform(
        ax.transData.transform(start) + np.asarray(offset, dtype=float)
    )
    annotation.set_position((float(moved[0]), float(moved[1])))
    return annotation


def _default_prefer(ax: Axes, xy: tuple[float, float]) -> tuple[float, float]:
    """A starting spot up and right of the anchor, sized to the axes rather than to the units."""
    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    return (xy[0] + 0.06 * (high_x - low_x), xy[1] + 0.10 * (high_y - low_y))
