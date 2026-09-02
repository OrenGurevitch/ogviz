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
            # `artist_name`, the same namer `colliding_ink` uses below. This said only
            # "a Line2D is drawn and almost entirely covered", which on a panel carrying thirty
            # lines names nothing a reader can go and look at.
            complaints.append(
                f"{artist_name(artists[index])} is drawn and almost entirely covered — "
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
            # THE TEXT FIRST, and "shares … with …" rather than "A and B share …". Both are
            # about `group_by_subject`, which folds every complaint about one label under that
            # label and strips the label out of each line. On the old shape that left a dangling
            # conjunction — "'0.341 units': and Rectangle share 412 px of ink" — and, where the
            # text happened to be the second of the pair, an "and" stranded mid-sentence. It was
            # visible in this package's own gallery plate of the gate's output. This shape reads
            # correctly both grouped and alone, and at least one of the pair is always text
            # because the loop above skips two marks.
            subject, other = (first, second) if first in text_index else (second, first)
            complaints.append(
                f"{artist_name(artists[subject])} shares {shared} px of ink "
                f"with {artist_name(artists[other])}"
            )
    return complaints
