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
    ANCHORED,
    clear_position,
    hits_data,
    quoted,
    text_box,
    text_over_data,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

KNOCKOUT_PAD = 0.18  # in font-size units, which is what matplotlib's boxstyle pad means


def _page_color(fig: Figure) -> str:
    from matplotlib.colors import to_hex

    return to_hex(fig.get_facecolor())


def move_labels_off_the_marks(fig: Figure) -> list[str]:
    """Shift every free-standing label that sits on the data to the nearest clear spot.

    Only labels the figure has not pinned to something: a value printed against its own bar and a
    star over its own bracket are placed deliberately, and moving them would break the thing they
    are there to say. An `ogviz_anchored` tag marks those; a foreign figure has none, so every one
    of its labels is eligible, which is the right default — nothing else knows they were meant to
    be where they are.
    """
    fig.canvas.draw()
    moved: list[str] = []
    for ax in fig.axes:
        for text in ax.texts:
            content = text.get_text().strip()
            if not content or not text.get_visible() or getattr(text, ANCHORED, False):
                continue
            if not hits_data(ax, text_box(text)):
                continue
            offset = clear_position(ax, text_box(text))
            if offset is None or offset == (0.0, 0.0):
                moved.append(
                    f"{quoted(content)!r} sits on the marks and nowhere in the panel is free"
                )
                continue
            start = ax.transData.transform(text.get_position())
            shifted = ax.transData.inverted().transform(
                (start[0] + offset[0], start[1] + offset[1])
            )
            text.set_position((float(shifted[0]), float(shifted[1])))
            moved.append(f"moved {quoted(content)!r} clear of the marks")
    return moved


def knock_out_labels_over_rules(fig: Figure) -> list[str]:
    """Put an opaque box behind any label crossing a gridline, so the rule stops running through it.

    The right fix for a gridline and the wrong one for data: a knockout over the marks punches a
    hole in the finding. `text_over_data` separates the two cases, and only the gridline one is
    repaired here.
    """
    fig.canvas.draw()
    changed: list[str] = []
    wanted = {
        complaint.split("'")[1] for complaint in text_over_data(fig) if "knock it out" in complaint
    }
    if not wanted:
        return changed
    for ax in fig.axes:
        for text in ax.texts:
            if quoted(text.get_text()) not in wanted or text.get_bbox_patch() is not None:
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
    """
    fig.canvas.draw()
    changed: list[str] = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        highest = max((patch.get_zorder() for patch in ax.patches), default=0.0)
        for side, spine in ax.spines.items():
            if not spine.get_visible() or spine.get_zorder() > highest:
                continue
            covered = any(
                patch.get_zorder() > spine.get_zorder()
                and patch.get_window_extent().overlaps(spine.get_window_extent())
                for patch in ax.patches
            )
            if covered:
                spine.set_zorder(highest + 0.5)
                changed.append(f"raised the {side} spine above the marks standing on it")
        for line in ax.lines:
            if not getattr(line, "ogviz_reference", False) or line.get_zorder() > highest:
                continue
            line.set_zorder(highest + 0.75)
            changed.append("raised a reference line above the marks it is read against")
    return changed


REPAIRS = (move_labels_off_the_marks, knock_out_labels_over_rules, raise_buried_lines)


def repair(fig: Figure) -> list[str]:
    """Apply every repair that has one obvious answer. Returns what changed, in order.

    Run `audit` afterwards: what remains is what needs a person. That pairing is the point — this
    is not meant to make a figure pass, it is meant to leave only the decisions on the desk.
    """
    return [change for fix in REPAIRS for change in fix(fig)]
