"""Fix the figure problems that have one obvious fix, and report the ones that do not.

`audit` says what is wrong. This changes it, for the subset where "what should it be instead" has a
single defensible answer:

  a label sitting on the marks          move it to the nearest empty spot, arrow anchored
  a label crossing a gridline           knock the line out behind it, rather than moving it
  a spine buried under the marks        raise it above them
  a threshold buried under the marks    raise it above them

Everything else is reported and left alone, on purpose. Two series that merge under colour-vision
deficiency need a marker, a dash or a different palette, and which one is a design decision. A
caption with a word too long to wrap needs the word shortened or the figure widened. Guessing at
either would produce a figure the author did not choose and cannot easily undo.

The distinction worth keeping: this repairs PRESENTATION, never data. Nothing here moves a mark,
changes a limit, or alters a value — a label moves, a line's z-order changes, a knockout appears.
If a fix would change what the figure says, it is not in here.

Works on any matplotlib figure. It reads what was drawn, so a project that has never imported ogviz
gets the same treatment as one built with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ogviz.layout.collision import (
    clear_position,
    labels_crossing_a_rule,
    labels_on_the_marks,
    quoted,
    text_box,
)
from ogviz.layout.render import ensure_rendered
from ogviz.qc.reading import filled_marks_over
from ogviz.tags import marked
from ogviz.theme import KNOCKOUT_PAD

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text


def _page_color(fig: Figure) -> str:
    """This FIGURE's page colour, which is what a knockout painted on it has to match.

    `theme.page_color()` answers the same question from rcParams, and the two disagree for a figure
    whose facecolor was set per-figure — which is a real case, since `use_house_style(PAPER_WHITE)`
    and a warm default can both be in one process. A knockout is paint going onto THIS canvas, so
    it takes the canvas's own colour; the rcParams answer is right for a mark being drawn under the
    style currently in force, and that is why both exist.
    """
    from matplotlib.colors import to_hex

    return to_hex(fig.get_facecolor())


def move_labels_off_the_marks(
    fig: Figure, *, on_the_marks: list[tuple[Axes, Text, int]] | None = None
) -> list[str]:
    """Shift every free-standing label that sits on the data to the nearest clear spot.

    Only labels the figure has not pinned to something: a value printed against its own bar and a
    star over its own bracket are placed deliberately, and moving them would break the thing they
    are there to say. An `ogviz_anchored` tag marks those; a foreign figure has none, so every one
    of its labels is eligible, which is the right default — nothing else knows they were meant to
    be where they are.
    """
    moved: list[str] = []
    for ax, text, _struck in on_the_marks if on_the_marks is not None else labels_on_the_marks(fig):
        content = text.get_text().strip()
        # Every OTHER label in the panel, so the search does not set this one down on one of them.
        # Recomputed per label rather than once, on purpose: a label moved earlier in this loop is
        # somewhere new, and a snapshot taken before the loop would route this one around where it
        # used to be.
        others = [
            text_box(other)
            for other in ax.texts
            if other is not text and other.get_visible() and other.get_text().strip()
        ]
        offset = clear_position(ax, text_box(text), avoid=others)
        if offset is None or offset == (0.0, 0.0):
            moved.append(f"{quoted(content)!r} sits on the marks and nowhere in the panel is free")
            continue
        start = ax.transData.transform(text.get_position())
        shifted = ax.transData.inverted().transform((start[0] + offset[0], start[1] + offset[1]))
        text.set_position((float(shifted[0]), float(shifted[1])))
        moved.append(f"moved {quoted(content)!r} clear of the marks")
    return moved


def knock_out_labels_over_rules(fig: Figure, *, on_the_marks: set[int] | None = None) -> list[str]:
    """Put an opaque box behind any label crossing a gridline, so the rule stops running through it.

    The right fix for a gridline and the wrong one for data: a knockout over the marks punches a
    hole in the finding. `labels_crossing_a_rule` separates the two cases, and only the gridline one
    is repaired here.

    It works from the ARTISTS that check found. It used to re-derive them by splitting the complaint
    STRING on apostrophes, which silently repaired nothing whenever a label contained one — `repr`
    quotes such a string with double quotes, so the split returned a fragment that matched no label
    on the figure. A repair that declines without saying so is worse than one that refuses out loud.
    """
    changed: list[str] = []
    for _ax, text in labels_crossing_a_rule(fig, on_the_marks=on_the_marks):
        if text.get_bbox_patch() is not None:
            continue
        text.set_bbox(
            {
                "facecolor": _page_color(fig),
                "edgecolor": "none",
                "pad": KNOCKOUT_PAD,
                "boxstyle": "square",
            }
        )
        changed.append(f"knocked out the rule behind {quoted(text.get_text())!r}")
    return changed


def raise_buried_lines(fig: Figure) -> list[str]:
    """Lift a spine or a threshold above the marks drawn over it.

    A line a reader measures against — the category axis a bar stands on, a reference level the
    bars are compared to — survives only in the gaps when it is behind them, and reads as broken.
    Raising it is the whole fix; nothing moves.

    BOTH LOOPS TEST OVERLAP, and only the spine one did. A low z-order is not by itself a defect:
    a threshold drawn above every bar in the panel is behind them in paint order and in front of
    nothing, because nothing reaches it. Measured on a threshold at 9.0 over bars reaching 2.0 —
    `buried_baselines` correctly said only the spine was covered, and this returned "raised a
    reference line above the marks it is read against" for a line no mark touched. The z-order
    change was harmless, presentation being all this module moves; the REPORT was not. What
    `repair` returns is a caller's list of what was wrong with their figure.

    A FILLED COLLECTION COUNTS AS A MARK HERE, through the same pre-filter `buried_baselines` uses.
    The two have to agree about what buries a line, or the gate reports a covered spine that the
    repair then declines to lift — the check/repair disagreement fixed just above, one artist type
    along.
    """
    ensure_rendered(fig)
    changed: list[str] = []
    for ax in fig.axes:
        if not ax.axison:
            continue

        def marks_over(artist, axes=ax) -> list:
            """Everything drawn over this artist that is genuinely standing on it."""
            box = artist.get_window_extent()
            patches = [
                patch
                for patch in axes.patches
                if patch.get_zorder() > artist.get_zorder()
                and patch.get_window_extent().overlaps(box)
            ]
            return patches + filled_marks_over(axes, box, artist.get_zorder())

        # The ceiling to lift to, taken from the same two artist types the burial test looks at.
        # Read from patches alone, a spine buried under a raised band was lifted to just above the
        # bars and left under the band that was covering it.
        highest = max(
            (
                artist.get_zorder()
                for artist in (*ax.patches, *ax.collections)
                if artist.get_visible()
            ),
            default=0.0,
        )

        for side, spine in ax.spines.items():
            if not spine.get_visible() or spine.get_zorder() > highest:
                continue
            if marks_over(spine):
                spine.set_zorder(highest + 0.5)
                changed.append(f"raised the {side} spine above the marks standing on it")
        for line in ax.lines:
            if not marked(line, "reference") or line.get_zorder() > highest:
                continue
            if marks_over(line):
                line.set_zorder(highest + 0.75)
                changed.append("raised a reference line above the marks it is read against")
    return changed


def repair(fig: Figure) -> list[str]:
    """Apply every repair that has one obvious answer. Returns what changed, in order.

    Run `audit` afterwards: what remains is what needs a person. That pairing is the point — this
    is not meant to make a figure pass, it is meant to leave only the decisions on the desk.

    The "which labels sit on the marks" sweep is done ONCE and handed to both halves. It is the
    expensive one — a `hits_data` probe per label per panel — and the two repairs that need it ran
    it independently, so every call paid for it twice.
    """
    from ogviz.layout.collision import labels_on_the_marks

    struck = labels_on_the_marks(fig)
    ids = {id(text) for _ax, text, _n in struck}
    return [
        *move_labels_off_the_marks(fig, on_the_marks=struck),
        *knock_out_labels_over_rules(fig, on_the_marks=ids),
        *raise_buried_lines(fig),
    ]
