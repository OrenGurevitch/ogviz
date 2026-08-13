"""Colour-vision checks: the defect none of the geometric checks can see."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.color import indistinguishable_series, separation, simulate
from ogviz.panels.lines import series_colors
from ogviz.theme import LINE_SERIES, SERIES


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


@pytest.mark.parametrize("palette", [SERIES, LINE_SERIES], ids=["theme", "lines"])
def test_the_shipped_palettes_survive_colour_vision_deficiency(palette) -> None:
    """Both had a violet that collapsed onto their blue under deuteranopia, at 0.10 apart.

    Over the WHOLE palette, not `series_colors(5)`, which is what this used to take. That call
    could not exhibit the defect it was guarding: the palette had five colours and the helper
    wrapped modulo its length, so five was the one count at which no colour could repeat.
    """
    named = {f"[{index}]": color for index, color in enumerate(palette)}
    assert not indistinguishable_series(named)


def test_no_two_series_are_handed_the_same_colour() -> None:
    """`series_colors` indexed the palette modulo its length, so the sixth series repeated the
    first.

    The colour check cannot see this and is not going to: it reports pairs that separate for normal
    vision and converge under a deficiency, and two IDENTICAL colours never separated. So the
    palette helper could produce the defect and the palette checker would decline to mention it —
    which is why the guarantee is asserted here, over every count a caller may ask for.
    """
    for count in range(1, len(LINE_SERIES) + 1):
        chosen = series_colors(count)
        assert len(chosen) == count
        assert len(set(chosen)) == count, f"series_colors({count}) repeats a colour: {chosen}"


def test_a_count_the_palette_cannot_serve_is_refused() -> None:
    """Eight is a real ceiling, and wrapping past it drew two series as one."""
    with pytest.raises(AssertionError, match="were asked for"):
        series_colors(len(LINE_SERIES) + 1)
    with pytest.raises(AssertionError, match="at least one"):
        series_colors(0)


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


def test_a_chosen_colour_clears_the_check_that_asked_for_it() -> None:
    """The round trip that makes `separated_from` worth having at all.

    Reporting a confusable pair and choosing a replacement are two halves of one job, and they have
    to agree: a colour this returns must be one `indistinguishable_series` then passes.
    """
    from ogviz.color import indistinguishable_series, separated_from

    taken = ["#2E7CE0", "#EFA607", "#14A97C"]
    picked = separated_from(taken)
    named = {str(index): color for index, color in enumerate([*taken, picked])}
    assert indistinguishable_series(named) == [], named


def test_it_beats_the_obvious_wrong_answer() -> None:
    """The premise: a hue-wheel step is what a caller reaches for, and it is not good enough.

    Without this the test above would pass on a function that returned any old colour, since three
    arbitrary colours often happen to clear the threshold.
    """
    from ogviz.color import separated_from, worst_separation

    taken = ["#2E7CE0", "#EFA607", "#14A97C", "#ED6B3B"]
    naive = "#9B3B8F"  # a fifth hue, evenly spaced, chosen the way a person would
    assert worst_separation(separated_from(taken), taken) > worst_separation(naive, taken)


def test_a_full_palette_is_refused_rather_than_fudged() -> None:
    """Returning the best-but-failing colour would hand back something the gate then refuses."""
    import pytest

    from ogviz.color import separated_from

    # The whole RGB cube, coarsely: nothing can be far from all of it.
    crowded = [
        f"#{r:02x}{g:02x}{b:02x}"
        for r in range(0, 256, 51)
        for g in range(0, 256, 51)
        for b in range(0, 256, 51)
    ]
    with pytest.raises(AssertionError, match="cannot be fixed by choosing better colours"):
        separated_from(crowded)


def test_the_answer_does_not_move_between_runs() -> None:
    """A palette helper returning a different colour each call makes a figure irreproducible."""
    from ogviz.color import separated_from

    taken = ["#2E7CE0", "#EFA607"]
    assert separated_from(taken) == separated_from(taken)


def test_near_orders_the_winners_and_does_not_override_them() -> None:
    """A tiebreak, never a constraint — separation still decides."""
    from ogviz.color import separated_from, separation, worst_separation

    taken = ["#2E7CE0", "#EFA607"]
    plain, biased = separated_from(taken), separated_from(taken, near="#c00000")
    assert biased != plain, "the preference reached a different colour, so it is doing something"
    assert separation(biased, "#c00000") < separation(plain, "#c00000"), "and a nearer one"
    assert worst_separation(biased, taken) >= 0.18, "while still clearing the threshold"


def test_worst_separation_reads_the_deficiencies_and_not_only_normal_vision() -> None:
    """A pair that separates on screen and merges for a reader must score as merged."""
    from ogviz.color import separation, worst_separation

    red, green = "#D62728", "#2CA02C"  # matplotlib's own, the textbook confusable pair
    assert separation(red, green) > 0.18, "the premise: they look distinct to normal vision"
    assert worst_separation(red, [green]) < 0.18


def test_the_vectorised_search_agrees_with_the_one_at_a_time_metric() -> None:
    """`separated_from` scores by broadcasting; `worst_separation` scores one pair at a time.

    Two implementations of one question is exactly how they come to disagree, so the fast one is
    held to the slow one on the answer it actually returns.
    """
    from ogviz.color import separated_from, worst_separation

    for taken in (["#2E7CE0"], ["#2E7CE0", "#EFA607", "#14A97C"]):
        picked = separated_from(taken)
        assert worst_separation(picked, taken) >= 0.18, (taken, picked)
