from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.theme import (
    CANVAS,
    DISPLAY_FONTS,
    PAPER_WHITE,
    SERIES,
    SERIF_FONTS,
    WARM_CANVAS,
    glyphs_must_render,
    use_house_style,
)


def test_style_sets_the_canvas_and_keeps_svg_text_selectable() -> None:
    assert matplotlib.rcParams["figure.facecolor"] == CANVAS
    assert matplotlib.rcParams["svg.fonttype"] == "none"


def test_every_stacked_family_resolves_a_regular_and_a_bold_face() -> None:
    # A family registering one weight (macOS .ttc collections) renders the whole figure at that
    # weight and silently erases the style's weight contrast.
    from matplotlib import font_manager as fm

    weights: dict[str, set[int]] = {}
    for font in fm.fontManager.ttflist:
        weights.setdefault(font.name, set()).add(font.weight)
    for family in (*DISPLAY_FONTS, *SERIF_FONTS):
        if family in weights:
            assert {400, 700} <= weights[family], f"{family} lacks a regular/bold pair"


def test_series_colours_are_distinct() -> None:
    assert len(set(SERIES)) == len(SERIES)


# Bundled with matplotlib and WITHOUT U+207B, verified against the resolved file with `FT2Font`.
# The font has to be pinned to something that lacks the glyph, or the test asks a question whose
# answer depends on the machine: Arial has no U+207B, so on macOS the guard fires — and a Linux
# runner has no Arial at all, falls through to DejaVu, which covers it, and the guard correctly does
# nothing. This test used to SKIP in that case, which meant the half that proves the guard FIRES
# never ran on CI: it was only ever shown not to false-positive. `STIXGeneral` ships with
# matplotlib, so the question is posed identically everywhere.
WITHOUT_SUPERSCRIPT_MINUS = "STIXGeneral"


def test_glyph_guard_fails_on_a_character_the_font_cannot_draw() -> None:
    """The guard has to FIRE on a missing glyph, not merely refrain from false alarms."""
    import io

    mpl.rcParams["font.sans-serif"] = [WITHOUT_SUPERSCRIPT_MINUS]
    fig, ax = plt.subplots()
    ax.set_ylabel("R2* (s⁻¹)")  # U+207B
    with pytest.raises(AssertionError, match="no glyph"), glyphs_must_render():
        fig.savefig(io.BytesIO(), format="png")


def test_the_pinned_font_really_lacks_the_glyph() -> None:
    """Guards the test above: if STIXGeneral ever gained U+207B it would pass by proving nothing."""
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.ft2font import FT2Font

    face = FT2Font(findfont(FontProperties(family=WITHOUT_SUPERSCRIPT_MINUS)))
    assert face.get_char_index(0x207B) == 0, (
        "STIXGeneral now covers U+207B; pick another bundled font"
    )


def test_glyph_guard_passes_on_mathtext() -> None:
    import io

    fig, ax = plt.subplots()
    ax.set_ylabel(r"Relaxation rate $\mathregular{R_2^*}$ ($\mathregular{s^{-1}}$)")
    with glyphs_must_render():
        fig.savefig(io.BytesIO(), format="png")
    plt.close("all")


def test_the_house_page_is_warm_and_white_is_one_argument_away():
    """Warm by default because that is what these are read on; white for submission, where a
    journal typesets on white and the warm page becomes a visible grey rectangle."""
    use_house_style()
    assert mpl.rcParams["figure.facecolor"] == WARM_CANVAS
    use_house_style(PAPER_WHITE)
    for key in ("figure.facecolor", "savefig.facecolor", "axes.facecolor"):
        assert mpl.rcParams[key] == PAPER_WHITE


def test_halos_follow_the_canvas_instead_of_freezing_at_import():
    """A halo is the page showing through. Reading the module constant would leave white marks
    on a coloured page, which is the inconsistency this parameter exists to end."""
    import matplotlib.pyplot as plt

    from ogviz.marks import iqr_box, mean_line

    values = np.linspace(0.0, 1.0, 20)
    for canvas in (PAPER_WHITE, WARM_CANVAS, "#EEDDCC"):
        use_house_style(canvas)
        _fig, ax = plt.subplots()
        mean_line(ax, values, 0.0)
        iqr_box(ax, values, 0.0)
        halo = ax.lines[0].get_path_effects()[0]._gc["foreground"]
        assert mcolors.same_color(halo, canvas), f"mean-line halo ignored canvas {canvas}"
        median_dot = ax.lines[-1]
        assert mcolors.same_color(median_dot.get_markerfacecolor(), canvas)
        plt.close("all")


def test_a_caller_can_still_override_the_halo():
    import matplotlib.pyplot as plt

    from ogviz.marks import mean_line

    use_house_style()
    _fig, ax = plt.subplots()
    mean_line(ax, np.linspace(0, 1, 10), 0.0, halo="none")
    assert ax.lines[0].get_path_effects()[0]._gc["foreground"] == "none"
    plt.close("all")


def test_the_two_halves_are_exactly_the_whole() -> None:
    """`use_house_style` must set what it always set, so the split cannot drift into a difference.

    A project that can take only one half — one pinning its own font for byte-identical SVGs, say —
    still needs the other half to be the same ink everyone else gets.
    """
    import matplotlib as mpl

    from ogviz.theme import CANVAS, PAPER_WHITE, use_house_ink, use_house_style, use_house_type

    def written(apply) -> dict[str, str]:
        with mpl.rc_context():
            before = {key: repr(value) for key, value in mpl.rcParams.items()}
            apply()
            return {
                key: repr(value)
                for key, value in mpl.rcParams.items()
                if repr(value) != before.get(key)
            }

    for canvas in (CANVAS, PAPER_WHITE):
        both = written(lambda canvas=canvas: (use_house_ink(canvas), use_house_type()))
        whole = written(lambda canvas=canvas: use_house_style(canvas))
        assert whole == both, {
            "only in use_house_style": {k: v for k, v in whole.items() if k not in both},
            "only in the halves": {k: v for k, v in both.items() if k not in whole},
        }


def test_the_halves_do_not_overlap() -> None:
    """Ink and type are separable only while neither writes the other's keys.

    Measured from MATPLOTLIB'S DEFAULTS, not from the rcParams in force. The autouse `house_style`
    fixture has already applied both halves before this test runs, so a snapshot taken then already
    holds every house value and a key a half re-writes to the same value never registers as
    changed. Under that harness the test passed while the halves overlapped on two keys — the exact
    trap CLAUDE.md records — and it could not have caught a third.

    The two keys they DO share are `use_reproducible_svg`'s, which both halves call on purpose (its
    docstring says why). So the invariant is not "no overlap" but "the overlap is exactly that
    call", asserted as a set so a new shared key fails by name.
    """
    import matplotlib as mpl

    from ogviz.theme import use_house_ink, use_house_type

    def keys(apply) -> set[str]:
        with mpl.rc_context():
            mpl.rcdefaults()
            before = {key: repr(value) for key, value in mpl.rcParams.items()}
            apply()
            return {key for key, value in mpl.rcParams.items() if repr(value) != before.get(key)}

    assert keys(use_house_ink) & keys(use_house_type) == {"svg.hashsalt", "svg.fonttype"}


def test_a_project_can_take_the_ink_and_keep_its_own_font() -> None:
    """The case this split exists for: pin DejaVu, and still get the house ink."""
    import matplotlib as mpl

    from ogviz.theme import INK, use_house_ink

    with mpl.rc_context():
        mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        use_house_ink()
        assert mpl.rcParams["font.sans-serif"] == ["DejaVu Sans"], "its font is untouched"
        assert mpl.rcParams["text.color"] == INK, "and it gets the house ink"


def test_glyphs_must_render_leaves_the_callers_warning_filters_in_force() -> None:
    """A `simplefilter("always")` inside the block replaced `-W error::DeprecationWarning`, so a
    deprecation raised during a save did not raise where it happened — and was lost outright when a
    glyph was also missing. Only the glyph message is forced now."""
    import warnings

    from ogviz.theme import glyphs_must_render

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning), glyphs_must_render():
            warnings.warn("renamed", DeprecationWarning, stacklevel=1)


def test_identity_colors_refuses_zero_as_series_colors_does() -> None:
    """It returned `()`, which reads as a palette rather than as a mistake.

    A caller looping over the result draws nothing and nothing says why. The two functions answer
    the same shape of question and disagreed about the empty case.
    """
    import pytest

    from ogviz.panels.lines import series_colors
    from ogviz.theme import identity_colors

    with pytest.raises(AssertionError, match="at least one"):
        identity_colors(0)
    with pytest.raises(AssertionError, match="at least one"):
        series_colors(0)
    assert len(identity_colors(1)) == 1
