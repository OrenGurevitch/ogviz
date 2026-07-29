"""Colour-vision checks: the defect thirteen geometric checks cannot see."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import use_house_style
from ogviz.color import indistinguishable_series, separation, simulate
from ogviz.panels.lines import series_colors
from ogviz.theme import SERIES


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def test_red_and_green_converge_for_a_deuteranope() -> None:
    """The textbook pair, and the reason the check exists."""
    red, green = "#E8552D", "#14A97C"
    assert separation(red, green) > 0.8, "obviously different to normal vision"
    assert separation(red, green, "deuteranopia") < 0.4, "and much closer without the M cone"
    # Still above the confusable threshold: they keep a lightness difference, and reporting them
    # would fire on a very large share of real figures.
    assert not indistinguishable_series({"red": red, "green": green})


def test_a_grey_is_unchanged_by_any_deficiency() -> None:
    """A colour on the neutral axis has no chroma to lose; a simulation that moves it is wrong."""
    for deficiency in ("deuteranopia", "protanopia", "tritanopia"):
        seen = simulate("#808080", deficiency)
        assert seen == pytest.approx((0.5019, 0.5019, 0.5019), abs=0.02)


@pytest.mark.parametrize("palette", [SERIES, series_colors(5)], ids=["theme", "lines"])
def test_the_shipped_palettes_survive_colour_vision_deficiency(palette) -> None:
    """Both had a violet that collapsed onto their blue under deuteranopia, at 0.10 apart."""
    named = {f"[{index}]": color for index, color in enumerate(palette)}
    assert not indistinguishable_series(named)


def test_a_pair_that_is_already_close_is_not_reported() -> None:
    """Two near-identical colours are the caller's own choice and plainly visible to them.

    The check is for the pair that looks well separated on the author's screen and merges for a
    reader — not for a palette they can see is subtle.
    """
    assert not indistinguishable_series({"a": "#2E7CE0", "b": "#2F7DE1"})


def test_the_gate_reads_the_legend_and_reports_a_confusable_pair() -> None:
    """The pair that actually merges — the blue and violet this package used to ship.

    Not red and green: those drop from 0.95 to 0.35 apart, which is a large loss and still above
    the threshold. A strong red and a strong green keep a lightness difference, and calling them
    confusable would cry wolf on half the figures in the world.
    """
    from ogviz.qc import series_confusable_under_cvd

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="#2E7CE0", label="blue")
    ax.plot([0, 1], [1, 0], color="#8A63D2", label="violet")
    ax.legend()
    fig.canvas.draw()
    complaints = series_confusable_under_cvd(fig)
    assert complaints and "deuteranopia" in complaints[0]


def test_a_figure_with_no_legend_says_nothing() -> None:
    """Marks with no legend entry are not being told apart by colour in the first place."""
    from ogviz.qc import series_confusable_under_cvd

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="#E8552D")
    ax.plot([0, 1], [1, 0], color="#14A97C")
    fig.canvas.draw()
    assert not series_confusable_under_cvd(fig)
