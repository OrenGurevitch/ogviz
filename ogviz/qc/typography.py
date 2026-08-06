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


def type_too_small(fig: Figure) -> list[str]:
    """Text so small against its own figure that no zoom makes it comfortable. Advisory.

    The gap `assert_clean` structurally cannot see: every check it runs is about COLLISION, and
    cramped type collides with nothing. A table can pass the whole gate at every size and be
    unreadable, which is what happened — the report was "hard to read", the cause was the aspect
    ratio, and nothing in the package had a word to say about it.

    Measured as a share of the figure's SHORT SIDE, which makes it a property of the figure alone.
    Points are not: 12 pt is comfortable on a 5-inch panel and vanishes on a 20-inch canvas, because
    what decides legibility is how far the figure is scaled when it is placed, and the figure cannot
    know that. The ratio survives the scaling, so it can be asked here.
    """
    fig.canvas.draw()
    short = min(float(fig.bbox.width), float(fig.bbox.height))
    if short <= 0:
        return []
    smallest: tuple[float, str] | None = None
    for text, _owner in figure_text(fig, ticks=True, legend=True):
        height = float(text.get_window_extent().height)
        if height <= 0:
            continue
        share = height / short
        if smallest is None or share < smallest[0]:
            smallest = (share, text.get_text())
    if smallest is None or smallest[0] >= CRAMPED_SHARE:
        return []
    share, what = smallest
    stacked = LINES_UP_THE_PAGE.format(count=1.0 / share)
    return [
        f"the smallest type on this figure ({quoted(what)!r}) {stacked} — it is the SHAPE of the "
        "canvas rather than the point size that usually does this, so check the aspect ratio "
        "before reaching for larger type"
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
    fig.canvas.draw()
    complaints: list[str] = []
    for text, _owner in figure_text(fig, ticks=True):
        for run in UNGROUPED.findall(text.get_text()):
            if len(run) == 4 and int(run) in YEARS:
                continue
            complaints.append(
                f"{quoted(text.get_text())!r} prints {run} without a thousands separator"
            )
    return complaints
