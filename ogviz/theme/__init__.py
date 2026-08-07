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


# A conditions table wants a tick and a cross, and the display stack cannot draw them. Verified
# against the font `findfont` actually returns on this machine: Arial has NO U+2713 ✓, U+2717 ✗,
# U+25C6 ◆ or U+2605 ★, and matplotlib renders a missing glyph as a tofu box. Present and safe in
# Arial: U+00D7 multiply, U+25CF/U+25CB circles, U+25A0/U+25A1 squares, U+25B2/U+25BC triangles,
# U+2212 minus, U+2022 bullet, U+2020 dagger, U+2014 em dash.
#
# The answer is not "use a filled circle instead", which is what a project reduced to when it hit
# this — ●/○ is strictly worse at saying has/lacks than ✓/✗. It is to draw those two characters in a
# font that has them. `DejaVu Sans` is the one font this package can promise, because matplotlib
# BUNDLES it, so it is on every machine that can run any of this.
YES = "✓"  # ✓
NO = "✗"  # ✗
GLYPH_FAMILY = "DejaVu Sans"  # matplotlib bundles it; unlike the display stack it has the marks
# Semantic, not raw colour: a caller says what a cell MEANS and the palette stays this package's
# decision. Both are dark enough to read as ink on the page rather than as highlighting.
GOOD = "#2E7D4F"
BAD = "#B3261E"


def _display_face():
    """The font file the display stack actually resolves to, opened once."""
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.ft2font import FT2Font

    path = findfont(FontProperties(family=list(mpl.rcParams["font.sans-serif"])))
    if path not in _FACES:
        _FACES[path] = FT2Font(path)
    return _FACES[path]


_FACES: dict[str, object] = {}


def family_for(text: str) -> str | None:
    """The family `text` has to be set in, or None when the display font can draw it all.

    The PROACTIVE half of `glyphs_must_render`, which only fails at save time — by then the figure
    is built and the caller is reading an assertion instead of getting a readable mark. Asked
    through matplotlib's own `FT2Font.get_char_index`, which returns 0 for a character the face has
    no glyph for, so it needs no dependency this package does not already have.

    Returns a family rather than a boolean because the useful answer is what to do about it.
    """
    face = _display_face()
    if all(face.get_char_index(ord(character)) for character in text):  # type: ignore[attr-defined]
        return None
    return GLYPH_FAMILY


def page_color() -> str:
    """The page colour in force RIGHT NOW, read from rcParams rather than the constant.

    Halos and the median dot are the page showing through, so they have to follow
    `use_house_style(canvas=...)`. Reading a module constant would freeze them at import and
    leave white marks on a coloured page.
    """
    # Indexed, not `.get(..., CANVAS)`: `figure.facecolor` is always present in rcParams, so the
    # default could never fire, and if it ever could this would quietly return a colour that
    # disagrees with the figure instead of saying so.
    return str(mpl.rcParams["figure.facecolor"])


def use_reproducible_svg() -> None:
    """Make an SVG re-render the same bytes: fix the salt matplotlib randomises its ids with.

    Its own call rather than a line inside one of the halves. It landed in `use_house_ink`, which is
    a colour decision, and the project most likely to need it is the one that takes ink ALONE
    because it must pin its own font to matplotlib's bundled DejaVu — for byte-identical SVGs across
    machines, which is this same requirement, one layer up. A project that pins its own COLOURS and
    takes the house type got nothing, and had no way to find out why its gallery still churned.

    Both halves call it, so `use_house_style()` and either half on its own all still set it; a
    project wanting a salt of its own sets `svg.hashsalt` after.
    """
    # A fixed salt makes matplotlib's clip-path ids a function of the figure instead of the run, so
    # re-rendering an unchanged figure rewrites the same bytes and `git diff` on a committed gallery
    # shows only what actually changed. `save` drops the date stamp, which is the other half.
    mpl.rcParams["svg.hashsalt"] = "ogviz"
    # And text stays TEXT rather than being traced into outlines. This sat in `use_house_type` until
    # 2026-08-07, which was the same misfiling the salt had: it is a SERIALISATION decision, not a
    # typographic one, and the project most likely to need it is the one that calls neither type
    # function because it pins its own family for byte-identical output. That project got
    # matplotlib's default instead — every glyph in every SVG traced to a `<path>`, not one `<text>`
    # element in the figure set — and with it a whole class of verification that silently passes:
    # diffing the text of two SVGs to see what a bump changed compares nothing against nothing.
    mpl.rcParams["svg.fonttype"] = "none"


def use_house_type() -> None:
    """Set the TYPOGRAPHY every house figure shares: families, sizes, weight.

    Separate from the ink because a project can be unable to take it. One that stacks its
    significance brackets by measuring rendered ink extents is calibrated against a particular
    font's metrics — change the family and its stars collide — and one that needs byte-identical
    SVGs across machines must resolve to matplotlib's bundled DejaVu rather than to a font that may
    or may not be installed. Either has to pin its own family.

    Bundled with the colours, that meant such a project could call NOTHING, and one did: it
    hand-rolled the ink instead, in one of its two renderers, so its two-panel figures used the
    house near-black and its condition grids inherited matplotlib's pure #000000. Two blacks in one
    paper, side by side, undetected for months because each renderer was internally consistent.

    `svg.fonttype` used to be set here, and moved to `use_reproducible_svg` on 2026-08-07 — see
    there for why keeping it with the type was exactly backwards.
    """
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(DISPLAY_FONTS),
            "font.serif": list(SERIF_FONTS),
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": TICK_SIZE,
        }
    )
    use_reproducible_svg()


def use_house_ink(canvas: str = CANVAS) -> None:
    """Set the COLOURS and the weights every house figure shares — everything but the type.

    `canvas` defaults to the warm page, which is what these figures are read on: it cuts glare and
    keeps saturated marks forward. Pass `PAPER_WHITE` when exporting for a manuscript — a journal
    typesets on white, and there the warm page becomes a visible grey rectangle.

    Which colour is right depends on the destination; what matters is that ONE call decides it for
    every figure in a project, since three projects each picking their own is how a set of figures
    stops looking like a set. That argument is the reason this half is callable on its own: a
    project pinning its own font still gets one source of truth for ink instead of five hex values
    copied into its own module.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": canvas,
            "savefig.facecolor": canvas,
            "axes.facecolor": canvas,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 0.0,
            "ytick.major.size": 4.0,
            "ytick.major.width": 1.2,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "legend.frameon": False,
        }
    )
    use_reproducible_svg()


def use_house_style(canvas: str = CANVAS) -> None:
    """Both halves, which is what a project with no constraint on its font wants. Call once.

    Unchanged in what it sets — `use_house_ink` plus `use_house_type` is exactly the set this always
    wrote, and a test holds that so the split cannot drift into a difference.
    """
    use_house_ink(canvas)
    use_house_type()


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
    if missing:
        # Raised, not asserted: `python -O` deletes an `assert` and would let the tofu through.
        raise AssertionError("figure text has no glyph in the house font: " + "; ".join(missing))
    for other in (w for w in caught if "missing from font" not in str(w.message)):
        warnings.warn_explicit(other.message, other.category, other.filename, other.lineno)
