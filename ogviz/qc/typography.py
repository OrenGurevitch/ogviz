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
