"""Whether a label lands on the data, and where it could go instead.

`text_overlaps` catches labels colliding with other labels. This catches the other half: a label
colliding with the marks. They are different problems and need different geometry — two labels are
two rectangles, but a filled area is nothing like its bounding box, and a rising line only occupies
a thin diagonal of the box that contains it. Testing rectangles against rectangles reports a label
sitting in the empty corner above a curve as a collision, and misses one sitting under it.

So this tests what was DRAWN, in the two shapes drawn things come in.

Anything continuous is a path, and `Path.intersects_bbox` answers exactly the question being asked —
does this outline, filled or not, enter this rectangle — for a line, a violin body, a bar or a
`fill_between` polygon alike.

A cloud of separate marks is not a path and cannot be made into one. Joining the marks claims ink
along every segment between them, and a compound path of one square per mark does not help either,
since `intersects_bbox` answers True for a box lying BETWEEN two subpaths. Those go through
`data_points` as positions with a measured footprint, and `hits_data` asks both.

Two kinds of thing can sit under a label, and they want opposite fixes:

  the marks       a label over the data must MOVE; knocking a hole in the data hides it
  the decoration  a label over a gridline or a reference line may stay, if it knocks the line
                  out behind itself, which is what the value labels already do

`clear_position` does the moving. It is a search, not a formula: try the anchor, then positions
spiralling outward from it, and take the first that touches nothing. A formula would need to model
the free space, and the free space is whatever shape the data leaves behind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text as _Text
from matplotlib.transforms import Bbox

from ogviz import units
from ogviz.layout.render import ensure_rendered
from ogviz.tags import marked

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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
# The paragraph that used to sit here described a constant that no longer exists — the flag for a
# label placed against a mark on purpose, which became the `"anchored"` tag in `ogviz.tags` and is
# documented there, where the vocabulary is. Left behind, it ran straight into the comment below
# with no blank line between them, so the two read as one paragraph about the wrong constant.
#
# How much of a label a complaint quotes. ONE number, because a report groups complaints by the
# string they name: two checks quoting the same label at 32 and 40 characters describe it as two
# different labels, and the reader is shown one problem twice.
QUOTED_LENGTH = 40


def quoted(text: str) -> str:
    """A label as a complaint should name it — trimmed, and trimmed the same way everywhere.

    A truncated label carries its SHAPE, because otherwise two versions of one label read as the
    same string. Reflowing a long caption from two lines to three leaves the first forty characters
    untouched, so the complaint before and after the edit was identical and only the pixel count
    moved — a reader reasonably took that as evidence the edit had not applied when it had, and had
    made things worse. The shape is what changed, so the shape is what is shown.
    """
    stripped = text.strip()
    if len(stripped) <= QUOTED_LENGTH and "\n" not in stripped:
        return stripped
    lines = stripped.count("\n") + 1
    shape = f"{lines} lines, {len(stripped)} chars" if lines > 1 else f"{len(stripped)} chars"
    return f"{stripped[:QUOTED_LENGTH]}… ({shape})"


def decoration_ids(ax: Axes) -> set[int]:
    """Artists that are the frame rather than the finding, identified by id.

    A gridline is not data. Neither is the spine. A label may cross either, provided it knocks it
    out; treating them as marks would send every label on a gridded panel off to hunt for space
    that does not exist.

    A BACKDROP is not data either — a highlighted column, a reference band, a shaded window a label
    names. `colliding_ink` has always known that and this did not, so the same figure was told its
    label was fine on the pixels and had to move off the marks. One tag, two checks, two answers.
    """
    lines = [*ax.get_xgridlines(), *ax.get_ygridlines()]
    shaded = [
        artist for artist in [*ax.patches, *ax.collections, *ax.lines] if marked(artist, "backdrop")
    ]
    return {id(artist) for artist in (*lines, *shaded)}


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
        yield from _line_paths(artist)
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


NO_LINE = ("None", "none", " ", "")


def _stroked(line: Line2D) -> bool:
    return line.get_linestyle() not in NO_LINE and float(line.get_linewidth()) > 0.0


def _marked(line: Line2D) -> bool:
    return line.get_marker() not in (*NO_LINE, None)


def _px_per_point(artist: Artist) -> float:
    figure = artist.get_figure()
    return units.px_per_point(figure) if figure is not None else 1.0


class MarkCloud(NamedTuple):
    """Marks drawn separately: where each one is, and how big it is drawn — width and height apart.

    Two numbers rather than one, because a mark is not square and the difference is not cosmetic. An
    error-bar cap is matplotlib's `"_"` marker: at markersize 12 it is 12 points wide and ZERO high,
    ink only from its own linewidth. Testing it as a 12x12 square claimed 8 px of height it does not
    have, and reported every value label in a real figure as "sits on 1 mark(s)" — fourteen of them,
    each 0.5 px clear of the cap it was accused of touching.
    """

    positions: NDArray[np.float64]
    width_px: float
    height_px: float


def _stroke_px(artist: Artist, width: float) -> float:
    """A stroked mark's thinnest dimension is still its linewidth of ink."""
    return max(width * _px_per_point(artist), 1.0)


def _marker_footprint_px(line: Line2D) -> tuple[float, float]:
    """How wide and how high one of this line's markers is drawn, in display pixels.

    From the marker's own path, which is the shape matplotlib will stroke, scaled by the markersize
    it will be stroked at. A round marker measures square, a cap measures wide and flat, and neither
    needs a special case here.
    """
    from matplotlib.markers import MarkerStyle

    style = MarkerStyle(line.get_marker())
    # The marker path is a unit shape; the markersize scales it. Multiplying the extents is the same
    # arithmetic as scaling the transform and does not need the transform to be an affine.
    unit = style.get_path().transformed(style.get_transform()).get_extents()
    scale = float(line.get_markersize()) * _px_per_point(line)
    thinnest = _stroke_px(line, float(line.get_markeredgewidth()))
    return (
        max(float(unit.width) * scale, thinnest),
        max(float(unit.height) * scale, thinnest),
    )


def _cloud_footprint_px(collection: Collection) -> tuple[float, float]:
    """The same measurement for a collection, through its own per-element transform.

    That transform is where matplotlib keeps the size, and it already folds in the dpi. Reading
    `get_sizes` instead works only for the collections that define it, and describes an AREA in
    points squared, which then has to be rooted and converted — two conversions to get wrong for a
    number matplotlib will hand over directly.
    """
    from matplotlib.transforms import Affine2D

    paths = collection.get_paths()
    transforms = collection.get_transforms()
    if not len(paths) or not len(transforms):
        return (1.0, 1.0)
    footprints = [paths[0].transformed(Affine2D(matrix)).get_extents() for matrix in transforms]
    widest = max(float(box.width) for box in footprints)
    highest = max(float(box.height) for box in footprints)
    return (max(widest, 1.0), max(highest, 1.0))


def data_points(ax: Axes) -> list[MarkCloud]:
    """Every cloud of separately drawn marks in the axes, in display pixels.

    Separate from `data_paths` because a cloud cannot be tested as a path. Joining the marks into
    one polyline claims ink along every segment between them, and a compound path of one square per
    mark does not help — matplotlib's `intersects_bbox` answers True for a box lying BETWEEN two
    subpaths, so the disjointness is lost either way. Positions, tested individually, are the only
    faithful representation.

    The defect this fixes: `errorbar` draws its caps as ONE marker-only Line2D per side, so the
    polyline ran diagonally across the whole panel and every value label in a bar panel came back
    "sits on 2 mark(s)" — five confident complaints about a figure with nothing overlapping. The
    same flaw was latent for every scatter, where a dense jitter cloud hid it: the zigzag through
    the dots happens to cover roughly the area the dots do.
    """
    skip = decoration_ids(ax)
    clouds: list[MarkCloud] = []
    for line in ax.lines:
        if id(line) in skip or not line.get_visible() or _stroked(line) or not _marked(line):
            continue
        points = np.column_stack(
            [np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)]
        )
        if not len(points):
            continue
        width, height = _marker_footprint_px(line)
        clouds.append(MarkCloud(line.get_transform().transform(points), width, height))
    for collection in ax.collections:
        if id(collection) in skip or not collection.get_visible():
            continue
        offsets = point_offsets(collection)
        if offsets is None:
            continue
        width, height = _cloud_footprint_px(collection)
        clouds.append(
            MarkCloud(collection.get_offset_transform().transform(offsets), width, height)
        )
    return clouds


def _marks_in(cloud: MarkCloud, box: Bbox) -> int:
    """How many of the cloud's marks reach into `box`, each as the rectangle it is drawn as."""
    half_wide = cloud.width_px / 2.0
    half_high = cloud.height_px / 2.0
    inside = (
        (cloud.positions[:, 0] >= box.x0 - half_wide)
        & (cloud.positions[:, 0] <= box.x1 + half_wide)
        & (cloud.positions[:, 1] >= box.y0 - half_high)
        & (cloud.positions[:, 1] <= box.y1 + half_high)
    )
    return int(np.count_nonzero(inside))


def _line_paths(line: Line2D) -> Iterator[tuple[Path, bool]]:
    """A line's stroke, when it has one. Its markers are a cloud and go through `data_points`.

    `get_path` is the polyline through the points whether or not that polyline is drawn, and a
    marker-only line draws none of it.
    """
    if _stroked(line):
        yield line.get_path().transformed(line.get_transform()), False


def _collection_paths(collection: Collection) -> Iterator[tuple[Path, bool]]:
    if point_offsets(collection) is not None:
        return  # a cloud; `data_points` has it
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


class PanelMarks(NamedTuple):
    """Everything in one axes that a label has to avoid, resolved to display space once.

    A snapshot, and it has to be treated as one: it is valid until something is drawn, moved, or
    the axes rescaled. `clear_position` takes it because the geometry cannot change during a
    search — the search only moves a rectangle around.
    """

    paths: list[tuple[Path, bool]]
    clouds: list[MarkCloud]

    @classmethod
    def of(cls, ax: Axes) -> PanelMarks:
        return cls(data_paths(ax), data_points(ax))


def hits_data(
    ax: Axes, box: Bbox, *, padding: float = PADDING_PX, marks: PanelMarks | None = None
) -> int:
    """How many marks enter `box`, which is in display pixels.

    `marks` is the panel resolved once. Without it every call rebuilds the whole panel — walking
    each artist, transforming each path, re-deriving each marker footprint — and `clear_position`
    makes up to `rings x directions` of those, which is 960 at the defaults. Measured on six
    300-point scatters and six lines: 1.4 ms a call, so 0.31 s for one successful search and 1.4 s
    for a failed one, per label, and `repair` runs it on every label in the figure. The geometry is
    identical at every candidate; only the rectangle moves.
    """
    target = _padded(box, padding)
    panel = marks if marks is not None else PanelMarks.of(ax)
    struck = sum(1 for path, filled in panel.paths if path.intersects_bbox(target, filled=filled))
    return struck + sum(_marks_in(cloud, target) for cloud in panel.clouds)


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
    """Whether the label hides what runs behind it, which is what makes crossing a line acceptable.

    TWO ways do that, and only the box was recognised. A `withStroke` halo in the page colour knocks
    a gridline out just as effectively and hugs the glyphs instead of boxing them, which is the
    better typography — so a figure using it was told its labels crossed a gridline "with nothing
    behind it" and had to exclude the complaint by matching on its text.
    """
    return _boxed(text) or _haloed(text)


def _boxed(text: Text) -> bool:
    patch = text.get_bbox_patch()
    if patch is None:
        return False
    # get_facecolor may hand back a name, a hex string or an RGBA tuple; to_rgba normalises all
    # three and folds in the patch's own alpha, so opacity is read once from one representation.
    red, green, blue, alpha = to_rgba(patch.get_facecolor(), patch.get_alpha())
    del red, green, blue
    return bool(alpha >= 1.0 and patch.get_visible())


def _haloed(text: Text) -> bool:
    """A stroke drawn behind the glyphs in the page colour, wide enough to cover a hairline.

    `withStroke` keeps its settings in a private `_gc` dict and offers no accessor, so this reads
    one. That is a dependency on a matplotlib internal, and the failure mode if it is renamed is the
    quiet one: `getattr(..., {})` would find nothing, every haloed label would start being reported
    as crossing a gridline with nothing behind it, and the message would be about the label rather
    than about this function. `test_a_halo_knocks_a_gridline_out_as_well_as_a_box_does` in
    `qc/test.py` is what turns that into a failing test instead — if it ever breaks on a matplotlib
    upgrade, this private read is the first place to look.
    """
    from ogviz.theme import page_color

    page = to_rgba(page_color())
    for effect in text.get_path_effects():
        settings = getattr(effect, "_gc", None)
        if not isinstance(settings, dict):
            continue
        foreground = settings.get("foreground")
        if foreground is None or not settings.get("linewidth", 0.0):
            continue
        red, green, blue, alpha = to_rgba(foreground)
        if alpha >= 1.0 and (red, green, blue) == page[:3]:
            return True
    return False


def clear_position(
    ax: Axes,
    box: Bbox,
    *,
    rings: int = SEARCH_RINGS,
    directions: int = SEARCH_DIRECTIONS,
    padding: float = PADDING_PX,
    avoid: Sequence[Bbox] = (),
) -> tuple[float, float] | None:
    """A display-space offset that lifts `box` clear of the marks, or None if there is none.

    Searched outward in rings so the label ends up as near its anchor as the data allows: the first
    ring is one step out in `directions` directions, and each ring after that is one step further.
    Candidates that leave the axes are rejected — a label outside the panel it belongs to is not an
    improvement on one over a curve.

    The panel is resolved ONCE, before the search: nothing is drawn or moved while it runs, so the
    marks cannot change between candidates. See `hits_data` for what rebuilding them each time cost.

    `avoid` is the OTHER LABELS, and it exists because this searched with `hits_data` alone — which
    walks lines, collections, patches and images, and no text at all. So the repair could lift a
    label off the data and set it down on top of another label, trading a collision the gate
    reports for one it also reports. Passing them keeps the search honest about what "clear" means;
    the default is empty so a caller asking only about the marks still gets that.
    """
    marks = PanelMarks.of(ax)

    def free(candidate: Bbox) -> bool:
        if hits_data(ax, candidate, padding=padding, marks=marks):
            return False
        target = _padded(candidate, padding)
        return not any(target.overlaps(other) for other in avoid)

    if free(box):
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
            if free(moved):
                return offset
    return None


def _spans_the_panel(ax: Axes, box: Bbox) -> bool:
    """Whether an untagged patch under `box` reaches across the panel, as a band or column does."""
    frame = ax.get_window_extent()
    for patch in ax.patches:
        if marked(patch, "backdrop") or not patch.get_visible():
            continue
        extent = patch.get_window_extent()
        if not extent.overlaps(box):
            continue
        if extent.width >= frame.width * 0.98 or extent.height >= frame.height * 0.98:
            return True
    return False


def labels_on_the_marks(fig: Figure) -> list[tuple[Axes, Text, int]]:
    """Every free-standing label sitting on the data, with how many marks it covers.

    The ARTISTS, not sentences about them, because both the complaint and the repair need this and
    only one of them can be written from prose. `repair` used to recover the label from the
    complaint string with `complaint.split("'")[1]`, which fails the moment a label contains an
    apostrophe: `repr` then quotes it with double quotes, the split returns a fragment matching
    nothing, and the repair silently declines — no fix and no word about it. Measured with the label
    `won't fit`, which the gate reported and `repair` returned nothing for.
    """
    ensure_rendered(fig)
    found: list[tuple[Axes, Text, int]] = []
    for ax in fig.axes:
        marks = PanelMarks.of(ax)  # once per panel, not once per label
        for text in ax.texts:
            if not text.get_text().strip() or not text.get_visible():
                continue
            if marked(text, "anchored"):
                continue
            struck = hits_data(ax, text_box(text), marks=marks)
            if struck:
                found.append((ax, text, struck))
    return found


def labels_crossing_a_rule(
    fig: Figure, *, on_the_marks: set[int] | None = None
) -> list[tuple[Axes, Text]]:
    """Every label crossing a gridline with nothing painted behind it. Artists, for the same reason.

    A label ON the marks is excluded: the two want opposite fixes, and a knockout over the data
    punches a hole in the finding rather than fixing anything.

    `on_the_marks` lets a caller that has ALREADY done that sweep hand the answer over. `repair`
    runs both halves back to back, and without this it paid for `labels_on_the_marks` twice — a
    per-label `hits_data` probe across every panel, which is the measured expensive path.
    """
    ensure_rendered(fig)
    if on_the_marks is None:
        on_the_marks = {id(text) for _ax, text, _struck in labels_on_the_marks(fig)}
    found: list[tuple[Axes, Text]] = []
    for ax in fig.axes:
        for text in ax.texts:
            if not text.get_text().strip() or not text.get_visible():
                continue
            if marked(text, "anchored") or id(text) in on_the_marks:
                continue
            if hits_decoration(ax, text_box(text)) and not _knocked_out(text):
                found.append((ax, text))
    return found


def text_over_data(fig: Figure) -> list[str]:
    """Labels sitting on the marks, and labels crossing a gridline with nothing behind them.

    Both are the same visual failure — a string competing with ink for the same pixels — but they
    want opposite fixes, so they are reported as different complaints. A label on the DATA has to
    move, because a knockout box there would punch a hole in the finding. A label merely crossing a
    GRIDLINE can stay where it is and knock the line out behind itself.
    """
    complaints: list[str] = []
    for ax, text, struck in labels_on_the_marks(fig):
        # The remedy depends on WHAT it sits on, and a consumer's first move is a knockout box,
        # which does nothing here: a box hides the artist, and this check is about a label standing
        # on the data rather than about what shows through it. When the thing underneath spans the
        # panel it is almost always a shaded region the label NAMES, and the fix is to say so.
        remedy = (
            'tag the shaded region with `mark(span, "backdrop")` if the label names it'
            if _spans_the_panel(ax, text_box(text))
            else "it has to move"
        )
        complaints.append(
            f"{quoted(text.get_text().strip())!r} sits on {struck} mark(s) — {remedy}"
        )
    complaints += [
        f"{quoted(text.get_text().strip())!r} crosses a gridline with nothing behind it — "
        "knock it out"
        for _ax, text in labels_crossing_a_rule(fig)
    ]
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
