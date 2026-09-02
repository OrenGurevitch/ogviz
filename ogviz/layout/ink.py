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

Here the exact step is the renderer itself. `artist_ink` renders the artist ALONE on the bare
figure — its footprint, not its marginal contribution — through antialiasing, alpha, clipping, dash
patterns and font hinting, none of which a geometric model reproduces faithfully. Two artists
collide when those footprints intersect. There is nothing to tune, and the answer cannot disagree
with the image. (This paragraph described the difference method — with and without — for some time
after `artist_ink` had stopped using it; its own docstring says why a difference reports the most
complete overlap as no overlap at all.)

It costs one render per artist, so it is not run over everything: `exact_overlaps` uses boxes to
find candidate pairs and pays for ink only on those. On a figure where nothing is close, that is
one render. On a figure with a real collision, it is a handful.

`hidden_artists` is the same measurement asking a different question, and it took two attempts to
get right. An artist is hidden when it would put ink on the page and decides the colour of almost
none of it.

Both halves matter and the first version had neither. It measured with a boolean INK mask — "does
this pixel differ from the page" — and green drawn over blue is ink either way, so removing an
artist that sits on top of another changed no mask pixel and every such artist measured as
contributing nothing. That is why it once reported nearly every label as buried. The contribution
has to be measured in COLOUR: which pixels change value when the artist is taken away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout.raster import INK_TOLERANCE, frame_rgb, ink_of

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

MIN_SHARED_PX = 2  # one shared pixel is antialiasing; two is contact
HIDDEN_FRACTION = 0.05  # below this share of its own footprint, an artist is covered


def _render(fig: Figure) -> NDArray[np.bool_]:
    return ink_of(frame_rgb(fig))


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


def visible_contribution(fig: Figure, artist: Artist) -> NDArray[np.bool_]:
    """Pixels whose COLOUR this artist decides — what a reader would lose without it.

    Not the same as its footprint, and not the same as its ink. A line drawn over a filled area has
    a full footprint, and contributes nothing to a boolean ink mask because the pixels were already
    ink; in colour it contributes everything, because they were blue and are now green. Measuring
    this in ink rather than colour is the mistake that made the first version of `hidden_artists`
    report almost every artist as buried.
    """
    was_visible = artist.get_visible()
    try:
        artist.set_visible(True)
        with_it = _frame(fig)
        artist.set_visible(False)
        without = _frame(fig)
    finally:
        artist.set_visible(was_visible)
    # Two RENDERS compared with each other rather than a render compared with the page, which is why
    # this cannot go through `ink_of` — but it takes the same tolerance from the same place, or an
    # artist could contribute a pixel by one measure and not by the other.
    return np.any(np.abs(with_it - without) > INK_TOLERANCE, axis=2)


def _frame(fig: Figure) -> NDArray[np.int16]:
    return frame_rgb(fig)


def hidden_artists(
    fig: Figure, artists: list[Artist], *, showing: float = HIDDEN_FRACTION
) -> list[int]:
    """Indices of artists that would draw something and decide the colour of almost none of it.

    A geometric check cannot ask this: the artist is exactly where it was asked to be, with a
    perfectly good bounding box, and something is on top of it. `showing` is the share of an
    artist's own footprint it must still decide to count as visible — a line poking out at the ends
    of the shape covering it is not a visible line.
    """
    fig.canvas.draw()
    everything = _every_artist(fig)
    found: list[int] = []
    for index, artist in enumerate(artists):
        if not artist.get_visible() or not _could_be_covered(artist, everything):
            continue
        footprint = int(artist_ink(fig, artist, others=everything).sum())
        if footprint == 0:
            continue
        contributes = int(visible_contribution(fig, artist).sum())
        if contributes / footprint < showing:
            found.append(index)
    return found


def _could_be_covered(artist: Artist, everything: list[Artist]) -> bool:
    """Whether anything is drawn above this artist and overlaps its box.

    A cheap gate in front of an expensive answer, which is the same shape as `exact_overlaps`: two
    renders per artist is affordable for the few that something sits on and ruinous for all of
    them. Nothing above it means nothing can be hiding it, and no render is needed to know that —
    on a 78-artist panel this is the difference between twelve seconds and one.
    """
    box = artist.get_window_extent()
    order = artist.get_zorder()
    return any(
        other is not artist
        and other.get_visible()
        and other.get_zorder() > order
        and other.get_window_extent().overlaps(box)
        for other in everything
    )
