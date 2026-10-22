"""That a figure prints the numbers it was given, in the spelling they were formatted in.

Every other check here reads a figure and asks whether it is well made. This one asks whether it is
TRUE to its source, which no rendering check can see: a figure can be clean and still print a mean
from an older run. Two projects built the same guard independently. One formats each printed number
from a committed table and asserts that string is on the canvas; the other asserts that the means,
intervals and p-values it is about to draw equal the table the manuscript prints. The principle is
the same: a figure draws the table's numbers and does not compute a second version of them.

A string in `expected` matches when it occurs INSIDE some visible text, tick labels and legends
included, so a bar's "0.74" matches a label that reads "0.74" and not one that reads "0.7".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ogviz.layout.bounds import figure_text
from ogviz.require import require

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.figure import Figure

MINUS = chr(0x2212)  # the true minus sign, which this module normalises away


def _plain(text: str) -> str:
    """The text with one minus sign, so "-0.12" matches what a tick formatter drew as U+2212.

    The gate refuses a figure that carries both spellings, so a caller formatting with an f-string
    and a figure drawn by matplotlib legitimately differ in this one character and no other.
    """
    return text.replace(MINUS, "-")


def unprinted(fig: Figure, expected: Iterable[str]) -> list[str]:
    """The strings in `expected` that occur in no visible text on the figure, in order."""
    drawn = [_plain(text.get_text()) for text, _ax in figure_text(fig, ticks=True, legend=True)]
    return [wanted for wanted in expected if not any(_plain(wanted) in text for text in drawn)]


def assert_printed(fig: Figure, expected: Iterable[str]) -> None:
    """Raise unless every string in `expected` is drawn on `fig`, naming those that are not."""
    from ogviz.layout.render import ensure_rendered

    ensure_rendered(fig)
    missing = unprinted(fig, expected)
    require(
        not missing,
        f"{len(missing)} string(s) formatted from the data are not on the figure: {missing}",
    )
