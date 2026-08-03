"""Whether two artists collide in the pixels, decided by rendering rather than by their boxes.

The exact half of the collision question. A bounding box reports a label sitting in the empty corner
above a curve and misses one tucked under it, so boxes are used only to choose the pairs worth
rendering for — a cheap test for what MIGHT collide, an exact one for what does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ogviz.layout.ink import exact_overlaps, hidden_artists
from ogviz.qc.reading import (
    artist_name,
    drawn_artists,
    is_backdrop,
    is_excused,
    knocked_out_over,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def drawn_but_invisible(fig: Figure) -> list[str]:
    """Marks that would draw something and are covered by something else.

    The case no geometric check can see: the artist is where it was asked to be, its bounding box is
    fine, and a reader cannot see it. Measured in colour — which pixels change value when it is
    taken away — because in a boolean ink mask a line drawn over a filled area contributes nothing
    whether it is visible or not.

    Backdrops and the marks sitting on them are exempt: a shaded column exists to be covered.
    """
    complaints: list[str] = []
    for ax in fig.axes:
        artists = [
            artist
            for artist in [*ax.lines, *ax.collections, *ax.patches]
            if not is_backdrop(artist)
        ]
        for index in hidden_artists(fig, artists):
            complaints.append(
                f"a {type(artists[index]).__name__} is drawn and almost entirely covered — "
                "either it is redundant or something is on top of it"
            )
    return complaints


def colliding_ink(fig: Figure) -> list[str]:
    """Artists that genuinely share pixels, decided by the renderer rather than by their boxes.

    The box test cannot settle this. It reports a collision between "Ay" and "1.42" whose boxes
    intersect while no glyph does, and misses one between a label and a curve whose box is mostly
    empty. Boxes are used here only to pick the pairs worth rendering for, which is the split
    Theophil and Schodl's scatter-chart labelling uses and Kakoulis and Tollis survey: a cheap
    test for what MIGHT collide, an exact one for what DOES.

    Deliberately-placed labels are exempt, as everywhere else — a value label sits on its bar and
    a star sits on its bracket by design, and each has a check measuring that relationship
    properly.
    """
    complaints: list[str] = []
    for ax in fig.axes:
        artists = drawn_artists(ax)
        text_index = {index for index, a in enumerate(artists) if hasattr(a, "get_text")}
        for first, second, shared in exact_overlaps(fig, artists):
            if first not in text_index and second not in text_index:
                continue  # two marks may overlap; that is a chart, not a defect
            if is_excused(artists[first], artists[second]) or is_excused(
                artists[second], artists[first]
            ):
                continue
            if is_backdrop(artists[first]) or is_backdrop(artists[second]):
                continue
            if knocked_out_over(artists[first], artists[second]) or knocked_out_over(
                artists[second], artists[first]
            ):
                continue
            complaints.append(
                f"{artist_name(artists[first])} and {artist_name(artists[second])} "
                f"share {shared} px of ink"
            )
    return complaints
