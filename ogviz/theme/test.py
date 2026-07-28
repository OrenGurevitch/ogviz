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


@pytest.fixture(autouse=True)
def _style() -> None:
    use_house_style()


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


def test_glyph_guard_fails_on_a_character_the_font_cannot_draw() -> None:
    """`font.sans-serif` is the list we ASK for, not what matplotlib resolves. On a machine with
    no Arial it falls through to DejaVu, which does cover U+207B — so the skip has to test the
    resolved file, not the first name in the list. CI on Linux caught the difference."""
    import io
    from pathlib import Path

    from matplotlib.font_manager import FontProperties, findfont

    resolved = Path(findfont(FontProperties())).stem  # bare: follows rcParams font.family
    if not resolved.lower().startswith(("arial", "verdana")):
        pytest.skip(f"resolved to {resolved}, which covers U+207B")

    fig, ax = plt.subplots()
    ax.set_ylabel("R2* (s⁻¹)")  # U+207B is absent from Arial
    with pytest.raises(AssertionError, match="no glyph"), glyphs_must_render():
        fig.savefig(io.BytesIO(), format="png")
    plt.close("all")


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
