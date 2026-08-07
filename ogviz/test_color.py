"""Colour-vision checks: the defect none of the geometric checks can see."""

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
    """The textbook pair, and the reason the check exists.

    The numbers here moved on 2026-08-06, when `simulate` was corrected to work in light rather than
    in gamma-encoded values: this pair measured 0.35 apart under deuteranopia before and 0.43 after.
    The bound is stated against the corrected space and left loose, because what the test is for is
    the DIRECTION — a pair that is unmistakable to normal vision losing more than half its
    separation — not either endpoint.
    """
    red, green = "#E8552D", "#14A97C"
    assert separation(red, green) > 0.8, "obviously different to normal vision"
    assert separation(red, green, "deuteranopia") < 0.5, "and much closer without the M cone"
    # Still above the confusable threshold: they keep a lightness difference, and reporting them
    # would fire on a very large share of real figures.
    assert not indistinguishable_series({"red": red, "green": green})


def test_a_simulation_comes_back_in_the_space_it_was_given() -> None:
    """The transfer function has to be undone AND redone, or the answer is in the wrong space.

    Without the second half a mid-grey simulates as 0.216 — its own linear value — which looks like
    a plausible colour and is a third of the lightness it should be. Locked because the failure is
    invisible: the simulation still collapses reds onto greens, just by the wrong amount, and every
    distance the check compares against its threshold is wrong with it.
    """
    for deficiency in ("deuteranopia", "protanopia", "tritanopia"):
        assert simulate("#FFFFFF", deficiency) == pytest.approx((1.0, 1.0, 1.0), abs=0.01)
        assert simulate("#000000", deficiency) == pytest.approx((0.0, 0.0, 0.0), abs=0.01)


def test_the_matplotlib_red_and_green_are_reported() -> None:
    """The most cited confusable pair in scientific figures must not pass the check.

    It did until 2026-08-06, at 0.145 apart against a threshold of 0.12 — and 0.145 was itself the
    wrong number, measured in the wrong space. Corrected, the pair is 0.165 apart, and the threshold
    is 0.18, chosen with room on both sides: the tightest pair in either shipped palette is 0.216.
    """
    complaints = indistinguishable_series({"red": "#D62728", "green": "#2CA02C"})
    assert complaints and "deuteranopia" in complaints[0]


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

    Not this package's own red and green: those drop from 0.95 to 0.43 apart, which is a large loss
    and still well above the threshold. A strong red and a strong green keep a lightness difference,
    and calling THOSE confusable would cry wolf on half the figures in the world — which is a
    different pair from matplotlib's `tab10` red and green, whose lightnesses nearly match and which
    the check does report.
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
