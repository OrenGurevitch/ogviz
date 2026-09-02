"""What a mark MEANS, recorded on the artist that carries it.

The checks measure a finished figure, so they see rectangles and lines and have no way to know that
one rectangle is a bar and another is a shaded column behind it. These tags are how the panel that
drew them says so, and they are the whole cross-module contract between `panels` and `qc`.

They used to be written as `artist.ogviz_backdrop = True  # type: ignore[attr-defined]` in one file
and read as `getattr(artist, "ogviz_backdrop", False)` in another — a bare string at each end, in
thirty-odd places. A typo in either spelling disables a rule silently: no error, no failing test
unless one happens to cover that path, and a check that quietly stops checking. Nothing listed them,
so two ways of drawing one mark came to disagree about whether a label may sit on it.

One module, one spelling each, and a reader can see the whole vocabulary at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.figure import Figure

Tag = Literal[
    # A shaded region a label is MEANT to sit on: a highlighted column, a reference band. Not a
    # mark — it carries no value of its own and is there to tint the marks that do.
    "backdrop",
    # Placed against a particular artist on purpose. Excuses the label from the general "is this
    # label on the data" rule, and from nothing else.
    "anchored",
    # The one artist an anchored label names, so the excuse covers that pair and no other.
    "anchor",
    # A significance bracket, and the label sitting over one.
    "bracket",
    "bracket_star",
    # A star marking a whole ROW of a strip rather than a comparison between two positions, so it
    # has no bracket and must not be measured against one.
    "column_star",
    # A printed mean, which a grid puts on one line across its panels.
    "mean_row",
    # The caption `panel_row` reserved a row for, which `settle_caption` re-places.
    "caption_row",
    # A header line hung off the panels' left edge, which `settle_header` re-anchors.
    "header_left",
    # A reference level or band: context the reader measures against, which must stay legible.
    "reference",
    # The jitter lane a point cloud was actually placed against, and the category position it was
    # placed around. Recorded because a `tight_layout` afterwards changes what a lane is worth.
    "lane",
    "position",
    # Which way a panel runs, recorded by the panel that was told.
    "orientation",
    # How many panels this one shares a value scale with, set by `share_value_limits`. A panel is
    # often empty at one end BECAUSE a taller neighbour set the scale, and that wants a different
    # action from a panel whose limits are simply loose — `dead_space` reported both in the same
    # words, so the note read as "tighten this" on the one figure where tightening is wrong.
    "shared_scale",
    # Set on the FIGURE when `tight_layout` declined to lay it out; the value is its reason.
    "layout_refused",
    # Set on the FIGURE when some or all of its axes pin their own layout, so `tight_layout` skipped
    # them; the value says how many. Separate from a refusal because the gate reports one and not
    # the other: a pinned grid is the arrangement working as intended.
    "layout_pinned",
    # Set on the FIGURE by `panel_grid`: the shape it built and the page it was sized for, so
    # `grid_warnings` can say what that shape will cost without being told again.
    "grid_page",
    # Set by `panel_grid` on each axes it created. `grid_warnings` counted every axes carrying a
    # subplotspec instead, which a colourbar also does — so one colourbar made the empty-slot
    # arithmetic go NEGATIVE and the warning read "leaves -1 slot(s) empty".
    "grid_cell",
]

PREFIX = "ogviz_"


def _attribute(tag: Tag) -> str:
    return f"{PREFIX}{tag}"


def mark(artist: Artist | Figure, tag: Tag, value: Any = True) -> None:
    """Record what this artist is. `value` defaults to True, the usual case for a flag."""
    setattr(artist, _attribute(tag), value)


def marked(artist: Artist | Figure, tag: Tag) -> bool:
    """Whether this artist carries `tag` at all — the question every boolean tag is asked."""
    return bool(getattr(artist, _attribute(tag), False))


def value_of(artist: Artist | Figure, tag: Tag, default: Any = None) -> Any:
    """What a tag CARRIES, for the ones holding a number or another artist rather than a flag."""
    return getattr(artist, _attribute(tag), default)
