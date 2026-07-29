"""The house look: canvas, ink, type, grid, and the guard that stops a figure shipping broken.

One `use_house_style()` call sets every rcParam a figure should not restate. Everything a
project varies — group colours, series colours — is a value here, not a decision re-made in
each builder.

Font stacks hold only families that register a real regular AND bold face. macOS ships Avenir
Next and Helvetica Neue as `.ttc` collections from which matplotlib resolves a single weight
(Avenir Next resolves to 700), so choosing them renders every string bold and erases the weight
contrast the style is built on. `DejaVu` closes each stack because matplotlib bundles it — the
figures render the same on a machine with no system fonts.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import matplotlib as mpl

if TYPE_CHECKING:
    from collections.abc import Iterator

WARM_CANVAS = "#FCFCFA"  # the house page: less glare, and saturated marks sit forward on it
PAPER_WHITE = "#FFFFFF"  # for manuscript submission only — see `use_house_style`
CANVAS = WARM_CANVAS
INK = "#141413"
MUTED_INK = "#5C5B57"
GRID = "#E7E5DD"
PANEL_FILL = "#F3F2EC"

DISPLAY_FONTS: tuple[str, ...] = ("Arial", "Verdana", "DejaVu Sans")
SERIF_FONTS: tuple[str, ...] = ("Georgia", "DejaVu Serif")

TITLE_SIZE = 27
SUBTITLE_SIZE = 19
AXIS_LABEL_SIZE = 18
TICK_SIZE = 16
VALUE_LABEL_SIZE = 17  # value labels printed against a bar
# The printed mean under a violin IS the number the panel exists to report, so it is set larger
# than a label that merely annotates a mark.
MEAN_LABEL_SIZE = 20
STAR_SIZE = 20
# "n.s." is a WORD where "***" is three glyphs, and a word set at glyph size reads as roughly twice
# the mark: same point size, several times the ink. Sized down so the two carry equal weight on a
# panel where some comparisons clear the threshold and some do not.
WORD_LABEL_SIZE = 13
# Padding around a value label's opaque knockout box. matplotlib measures a boxstyle `pad` in
# FONT-SIZE units, not points: 1.5 here meant 1.5 x 17 pt of padding on every side, a rectangle
# large enough to erase the tick labels below the axes and punch holes in the spine.
KNOCKOUT_PAD = 0.18

# Categorical series with no intrinsic meaning. Ordered for maximum separation at n=2 and n=3.
# The fifth is a plum rather than the violet it used to be: violet and the first blue read as
# distinct here and land 0.10 apart under deuteranopia, which is a single colour to a reader.
# Checked by `ogviz.color.indistinguishable_series`, which found it.
SERIES: tuple[str, ...] = ("#2E7CE0", "#EFA607", "#14A97C", "#ED6B3B", "#9B3B8F")
REFERENCE = "#D9D7CE"  # a de-emphasised comparison series


def page_color() -> str:
    """The page colour in force RIGHT NOW, read from rcParams rather than the constant.

    Halos and the median dot are the page showing through, so they have to follow
    `use_house_style(canvas=...)`. Reading a module constant would freeze them at import and
    leave white marks on a coloured page.
    """
    color = mpl.rcParams.get("figure.facecolor", CANVAS)
    return str(color)


def use_house_style(canvas: str = CANVAS) -> None:
    """Set the rcParams every house figure shares. Call once, before creating figures.

    `canvas` defaults to the warm page, which is what these figures are read on: it cuts glare and
    keeps saturated marks forward. Pass `PAPER_WHITE` when exporting for a manuscript — a journal
    typesets on white, and there the warm page becomes a visible grey rectangle.

    Which colour is right depends on the destination; what matters is that ONE call decides it for
    every figure in a project, since three projects each picking their own is how a set of figures
    stops looking like a set.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": canvas,
            "savefig.facecolor": canvas,
            "axes.facecolor": canvas,
            "font.family": "sans-serif",
            "font.sans-serif": list(DISPLAY_FONTS),
            "font.serif": list(SERIF_FONTS),
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": "bold",
            "axes.edgecolor": INK,
            "axes.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.major.size": 0.0,
            "ytick.major.size": 4.0,
            "ytick.major.width": 1.2,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "legend.frameon": False,
            "legend.fontsize": TICK_SIZE,
            "svg.fonttype": "none",  # keep SVG text as text
        }
    )


@contextmanager
def house_style(canvas: str = CANVAS) -> Iterator[None]:
    """`use_house_style()` for the duration of a `with` block, then put rcParams back.

    `use_house_style()` writes global state, so a process that renders one figure in this style
    and another in a project's own cannot use it twice. This scopes it.
    """
    with mpl.rc_context():
        use_house_style(canvas)
        yield


@contextmanager
def glyphs_must_render() -> Iterator[None]:
    """Turn matplotlib's missing-glyph warning into a failure for the wrapped save.

    A character absent from the resolved font is drawn as a tofu box and the figure ships
    looking broken — "R2* (s⁻¹)" did, because Arial has no U+207B. Matplotlib only warns, and a
    warning in a figure build scrolls past. Write the exponent as mathtext instead.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    missing = sorted({str(w.message) for w in caught if "missing from font" in str(w.message)})
    assert not missing, "figure text has no glyph in the house font: " + "; ".join(missing)
    for other in (w for w in caught if "missing from font" not in str(w.message)):
        warnings.warn_explicit(other.message, other.category, other.filename, other.lineno)
