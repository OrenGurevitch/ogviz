"""Whether the numbers in a figure are set the way the house sets numbers.

Two rules, both about a reader reading a value rather than deciphering it: one glyph for the minus
sign throughout, and a separator from a thousand up. Both are defaults everywhere this library
prints, so what these catch is the numbers a CALLER formatted itself.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ogviz.layout.bounds import figure_text
from ogviz.layout.collision import quoted
from ogviz.layout.render import ensure_rendered
from ogviz.require import require

if TYPE_CHECKING:
    from matplotlib.figure import Figure


CANDIDATE_MINUS = re.compile(r"(?<![^\s(\[])-(?=\d)")


def one_minus_sign(fig: Figure) -> list[str]:
    """Every negative number in a figure must use the same glyph for its sign.

    matplotlib typesets its own tick labels with U+2212 while `"{:.2f}".format` writes an ASCII
    hyphen, so a panel that prints its values lands both in one figure — different glyphs, at
    different widths, for the same sign.
    """
    hyphen: set[str] = set()
    minus: set[str] = set()
    for ax in fig.axes:
        for text in [*ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            content = text.get_text().strip()
            if not content or not any(character.isdigit() for character in content):
                continue
            signed = bool(CANDIDATE_MINUS.search(content))
            (hyphen if signed else minus if "\u2212" in content else set()).add(content)
    if hyphen and minus:
        return [
            f"two different minus signs in one figure: {sorted(hyphen)[:3]} use a hyphen, "
            f"{sorted(minus)[:3]} use \u2212"
        ]
    return []


UNGROUPED = re.compile(r"(?<![\d,.])\d{4,}(?![\d,]*\.?\d*%)(?!\.\d)")

YEARS = range(1800, 2200)


# A figure whose smallest type would stack more than this many times up its short side is dense
# enough to be worth a second look. MEASURED, not chosen: across the seventeen shipped examples the
# densest is the comparison table, at 1.37% of its short side — about 73 lines stacked up the page.
# The floor sits below that, so the gallery is silent, and a many-row table on a tall canvas — the
# shape that gets reported as unreadable — falls under it.
#
# ADVISORY, and that is the finding rather than a hedge. The reported shape and the densest figure
# here are about 1.3x apart on this measurement, which is not a separation a build should be failed
# on. The same table is comfortable transposed to a handful of rows and cramped at twenty, with no
# change of type size at all, so what the number really tracks is the ASPECT RATIO. A gate needs a
# rule that is right; this is a rule that is worth reading.
CRAMPED_SHARE = 0.012
LINES_UP_THE_PAGE = "would stack {count:.0f} times up the figure's short side"

# The floor a journal states, in POINTS AT FINAL PUBLISHED SIZE. Nature asks for 5-7 pt and takes
# 5 pt as the minimum; Science and ACS also 5 pt; IEEE 6 pt. So 5 is the permissive one — a figure
# under it is under everybody's floor, and a caller submitting to IEEE passes 6.0.
#
# This is the number the short-side ratio below could never be. That ratio is a property of the
# figure ALONE, which is what makes it answerable with no argument — but legibility is decided
# by how far the figure is scaled when placed, and the ratio can only stand in for that. Told
# the width it will be placed at, the question stops being a proxy and becomes the journal's.
JOURNAL_MINIMUM_PT = 5.0


def _smallest_text(fig: Figure) -> tuple[float, float, str] | None:
    """(height px, point size, content) of the smallest visible label, or None if there is none."""
    ensure_rendered(fig)
    smallest: tuple[float, float, str] | None = None
    for text, _owner in figure_text(fig, ticks=True, legend=True):
        height = float(text.get_window_extent().height)
        if height <= 0:
            continue
        if smallest is None or height < smallest[0]:
            smallest = (height, float(text.get_fontsize()), text.get_text())
    return smallest


def type_too_small(fig: Figure, *, column_width: float | None = None) -> list[str]:
    """Text too small to read once the figure is placed. Advisory.

    The gap `assert_clean` structurally cannot see: every check it runs is about COLLISION, and
    cramped type collides with nothing. A table can pass the whole gate at every size and be
    unreadable, which is what happened — the report was "hard to read", the cause was the aspect
    ratio, and nothing in the package had a word to say about it.

    TWO WAYS TO ASK IT, and the second is much the sharper.

    With no argument, measured as a share of the figure's SHORT SIDE, which is a property of the
    figure alone. That is its virtue and its limit: legibility is decided by how far the figure is
    scaled when it is placed, the figure cannot know that, and a ratio can only stand in for it. The
    threshold is correspondingly soft — the shape that gets reported as unreadable and the densest
    figure shipped here are only about 1.3x apart on it, which is why this is advisory.

    Given `column_width` — the inches the figure will be PLACED at — it stops being a proxy. A
    figure authored 18 in wide and placed in a 6.5 in column is scaled by 0.35, and its 10 pt type
    reads as 3.6. That is the number a journal specifies: Nature 5 pt, Science and ACS 5 pt, IEEE
    6 pt, all at final published size. Pass the width you will place at and the answer is checkable
    against a published standard instead of against a number this package chose.

    Three of the seventeen figures shipped here fall under 5 pt at a 6.5 in single column. They are
    not defective — all three are wide and would run double-column or full-width, where they pass.
    That is exactly why the width is an argument: it is the one thing only the caller knows.
    """
    smallest = _smallest_text(fig)
    if smallest is None:
        return []
    height_px, point_size, content = smallest

    if column_width is not None:
        require(column_width > 0, f"a column width is inches on the page, got {column_width}")
        placed = point_size * (column_width / float(fig.get_figwidth()))
        if placed >= JOURNAL_MINIMUM_PT:
            return []
        return [
            f"the smallest type ({quoted(content)!r}, {point_size:g} pt) reads as {placed:.1f} pt "
            f"when this {fig.get_figwidth():g} in figure is placed at {column_width:g} in — under "
            f"the {JOURNAL_MINIMUM_PT:g} pt most journals set as their floor. Place it wider, or "
            "set the type larger"
        ]

    short = min(float(fig.bbox.width), float(fig.bbox.height))
    if short <= 0:
        return []
    share = height_px / short
    if share >= CRAMPED_SHARE:
        return []
    stacked = LINES_UP_THE_PAGE.format(count=1.0 / share)
    return [
        f"the smallest type on this figure ({quoted(content)!r}) {stacked} — it is the SHAPE "
        "of the canvas rather than the point size that usually does this, so check the aspect "
        "ratio before reaching for larger type. Pass `column_width` to ask the sharper question: "
        "whether it clears the point size a journal requires at final size"
    ]


def ungrouped_thousands(fig: Figure) -> list[str]:
    """Numbers of a thousand and up printed without a separator.

    The house rule: from 1000 a number is grouped, because an ungrouped "1200000" is counted digit
    by digit while "1,200,000" is read at a glance. Everything this library prints does it by
    default; this catches the numbers a caller formatted itself, which is where the rule is
    forgotten — a custom `value_format`, a hand-placed annotation, a title carrying a total.

    A four-digit number that could be a year is left alone. A year is an identifier, not a quantity,
    and no figure carries the fact of which it is.
    """
    ensure_rendered(fig)
    complaints: list[str] = []
    for text, _owner in figure_text(fig, ticks=True):
        for run in UNGROUPED.findall(text.get_text()):
            if len(run) == 4 and int(run) in YEARS:
                continue
            complaints.append(
                f"{quoted(text.get_text())!r} prints {run} without a thousands separator"
            )
    return complaints
