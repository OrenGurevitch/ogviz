"""Overlap decided on rendered pixels, with bounding boxes demoted to a first-pass filter.

A bounding box is not what the reader sees. `"Ay"` and `"1.42"` have boxes that intersect while no
glyph does; a dashed line's box covers the gaps between its dashes; a filled area's box is a
rectangle around a shape that is mostly empty. So a box test is wrong in both directions — it
invents collisions that are not there and misses ones that are — and every threshold added to it is
a guess at how wrong it is on average.

The literature on automatic label placement treats this as a two-part problem, and so does this
module. `adjustText` and `ggrepel` iterate on bounding-box overlap because they must move labels
thousands of times and can only afford a cheap test; Theophil and Schödl's scatter-chart labelling
(2006) pairs a cheap candidate step with an exact one, and Kakoulis and Tollis's labelling chapter
in the Handbook of Graph Drawing surveys the same split. The cheap test decides what MIGHT collide;
something exact decides what DOES.

Here the exact step is the renderer itself. `artist_ink` draws the figure twice — once with an
artist and once without — and the difference is exactly the pixels that artist contributes, through
antialiasing, alpha, clipping, dash patterns and font hinting, none of which a geometric model
reproduces faithfully. Two artists collide when those pixel sets intersect. There is nothing to
tune, and the answer cannot disagree with the image.

It costs one render per artist, so it is not run over everything: `exact_overlaps` uses boxes to
find candidate pairs and pays for ink only on those. On a figure where nothing is close, that is
one render. On a figure with a real collision, it is a handful.

A companion check — "this artist is drawn and contributes no pixel, so something covers it" — is
the obvious next use of the same measurement and is NOT here. Written, it disagreed with
`artist_ink` called directly on the same artists, and the cause is not yet understood; a check
whose failures cannot be explained is worse than no check, because it teaches people to ignore
output. FIXME has it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

INK_TOLERANCE = 10  # per channel, against the page colour
MIN_SHARED_PX = 2  # one shared pixel is antialiasing; two is contact


def _render(fig: Figure) -> NDArray[np.bool_]:
    fig.canvas.draw()
    read_back = getattr(fig.canvas, "buffer_rgba", None)
    assert read_back is not None, (
        "ink comparison needs a raster canvas — run under Agg, which is what the builders do; "
        f"this figure has a {type(fig.canvas).__name__}"
    )
    frame = np.asarray(read_back(), dtype=np.int16)[:, :, :3]
    return np.any(np.abs(frame - frame[0, 0, :]) > INK_TOLERANCE, axis=2)


def artist_ink(fig: Figure, artist: Artist, *, others: list[Artist] | None = None):
    """The pixels this artist puts down when nothing else is drawn — its footprint.

    Rendered ALONE rather than by difference, and the distinction is the whole correctness of this
    module. Difference — render with it, render without it — gives an artist's MARGINAL
    contribution, which is not its footprint: two identical strings in one place each contribute
    nothing, because removing either leaves the pixels unchanged. A difference-based overlap check
    therefore reports the most complete overlap possible as no overlap at all, which is the failure
    mode a test caught here before this shipped.

    Drawing it alone answers what it would put on the page, which is the thing that has to be
    compared. Everything the artist really does is included — antialiasing, alpha, clipping, dash
    pattern, glyph hinting — because the renderer does it, not a geometric model of it.
    """
    everything = others if others is not None else _every_artist(fig)
    hidden = [other for other in everything if other is not artist and other.get_visible()]
    for other in hidden:
        other.set_visible(False)
    was_visible = artist.get_visible()
    try:
        artist.set_visible(False)
        # The frame is drawn on every render — spines, ticks, the background — and it is not an
        # artist in this list, so without subtracting it every "alone" mask would contain it and
        # every pair would appear to share the axes furniture.
        bare = _render(fig)
        artist.set_visible(True)
        alone = _render(fig)
    finally:
        for other in hidden:
            other.set_visible(True)
        artist.set_visible(was_visible)
    return alone & ~bare


def _every_artist(fig: Figure) -> list[Artist]:
    """Everything drawable on the figure, so rendering one alone really is alone."""
    found: list[Artist] = list(fig.texts)
    for ax in fig.axes:
        found += [*ax.texts, *ax.lines, *ax.patches, *ax.collections, *ax.images]
        legend = ax.get_legend()
        if legend is not None:
            found.append(legend)
    return found


def _candidates(artists: list[Artist]) -> list[tuple[int, int]]:
    """Index pairs whose boxes touch — everything that COULD collide, cheaply."""
    boxes = [artist.get_window_extent() for artist in artists]
    return [
        (first, second)
        for first in range(len(artists))
        for second in range(first + 1, len(artists))
        if boxes[first].overlaps(boxes[second])
    ]


def exact_overlaps(
    fig: Figure, artists: list[Artist], *, min_shared: int = MIN_SHARED_PX
) -> list[tuple[int, int, int]]:
    """Which of `artists` genuinely share pixels, as (first, second, shared count).

    Boxes narrow the field and the renderer settles it. A pair whose boxes miss cannot share a
    pixel, so it is never rendered for; a pair whose boxes touch is checked properly rather than
    assumed guilty, which is what a threshold on "fraction of the smaller box covered" was doing.
    """
    fig.canvas.draw()
    pairs = _candidates(artists)
    if not pairs:
        return []
    everything = _every_artist(fig)
    needed = sorted({index for pair in pairs for index in pair})
    masks = {index: artist_ink(fig, artists[index], others=everything) for index in needed}
    found = []
    for first, second in pairs:
        shared = int(np.count_nonzero(masks[first] & masks[second]))
        if shared >= min_shared:
            found.append((first, second, shared))
    return found
