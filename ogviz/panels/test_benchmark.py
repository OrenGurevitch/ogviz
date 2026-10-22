import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon

from ogviz import assert_clean
from ogviz.layout.bounds import figure_text
from ogviz.panels import (
    Arm,
    benchmark_panel,
    featured_grid,
    group_bands,
    group_positions,
    leaders,
    panel_letters,
)
from ogviz.panels.benchmark import _bar_patches
from ogviz.theme import SERIES, page_color

pytestmark = pytest.mark.usefixtures("pinned_font")

BLUE, AMBER, GREEN, _ORANGE, PURPLE = SERIES
ARMS = [
    Arm("Ensemble", GREEN, "ours", second_color=BLUE),
    Arm("A", GREEN, "ours"),
    Arm("B", BLUE, "ours"),
    Arm("Base", AMBER, "baselines"),
]


def test_the_best_value_leads_in_the_direction_of_the_metric() -> None:
    assert leaders([0.2, 0.9, 0.5]) == {1}
    assert leaders([0.2, 0.9, 0.5], lower_is_better=True) == {0}


def test_a_tie_leads_together_by_the_callers_rule_and_only_that_rule() -> None:
    values = [0.80, 0.79, 0.60]
    assert leaders(values) == {0}, "no rule, no tie"
    assert leaders(values, tied=lambda best, other: abs(values[best] - values[other]) < 0.02) == {
        0,
        1,
    }


def test_a_missing_arm_cannot_lead() -> None:
    with pytest.raises(AssertionError, match="finite"):
        leaders([0.5, float("nan")])


def test_groups_get_a_gap_and_must_be_contiguous() -> None:
    assert group_positions(["a", "a", "b"], gap=0.5) == [0.0, 1.0, 2.5]
    with pytest.raises(AssertionError, match="contiguous"):
        group_positions(["a", "b", "a"])


def test_featured_grid_lines_the_two_tiers_up_at_the_edges() -> None:
    fig, large, small = featured_grid(size=(12.0, 9.0))
    assert (len(large), len(small)) == (2, 6)
    fig.canvas.draw()
    left = min(ax.get_position().x0 for ax in large)
    right = max(ax.get_position().x1 for ax in large)
    assert min(ax.get_position().x0 for ax in small) == pytest.approx(left)
    assert max(ax.get_position().x1 for ax in small) == pytest.approx(right)
    assert large[0].get_position().y0 > small[0].get_position().y1, (
        "large row above the small block"
    )


def _panel(values, *, lower: bool = False, tied=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    drawn = benchmark_panel(
        ax,
        ARMS,
        values,
        [0.02] * len(ARMS),
        lower_is_better=lower,
        tied=tied,
        value_format="{:.2f}",
    )
    ax.set_xlabel("Method")
    fig.subplots_adjust(bottom=0.2)
    return fig, ax, drawn


def test_the_leader_is_outlined_and_saturated_and_the_rest_are_faded_without_changing_hue() -> None:
    _fig, ax, drawn = _panel([0.6, 0.9, 0.5, 0.4])
    bars = _bar_patches(ax)
    assert drawn.leaders == {1}
    outlined = [bar.get_edgecolor()[3] > 0 and bar.get_linewidth() > 0 for bar in bars]
    assert outlined == [False, True, False, False], "only the leader is outlined"
    full = np.array(to_rgb(GREEN))
    faded = np.array(bars[0].get_facecolor()[:3])
    page = np.array(to_rgb(page_color()))
    # Faded means on the line from the colour to the page, so the hue is the arm's own.
    toward_page = faded - full
    assert np.allclose(
        toward_page,
        (page - full) * np.dot(toward_page, page - full) / np.dot(page - full, page - full),
        atol=0.05,
    )
    assert not np.allclose(faded, full)


def test_an_ensemble_is_striped_inside_its_own_bar_in_the_second_colour() -> None:
    _fig, ax, _ = _panel([0.9, 0.6, 0.5, 0.4])
    bar = _bar_patches(ax)[0]
    stripes = [p for p in ax.patches if isinstance(p, Polygon)]
    assert len(stripes) >= 3, (
        "six bands alternate with the bar's own colour, so about three are drawn"
    )
    left, right = bar.get_x(), bar.get_x() + bar.get_width()
    top = bar.get_y() + bar.get_height()
    for stripe in stripes:
        xs, ys = stripe.get_xy()[:, 0], stripe.get_xy()[:, 1]
        assert xs.min() >= left - 1e-9 and xs.max() <= right + 1e-9
        assert ys.min() >= -1e-9 and ys.max() <= top + 1e-9, "cut to the bar, not only clipped"


def test_a_striped_leader_keeps_its_outline_above_the_stripes() -> None:
    _fig, ax, _ = _panel([0.9, 0.6, 0.5, 0.4])
    outlines = [p for p in _bar_patches(ax) if p.get_facecolor()[3] == 0 and p.get_linewidth() > 0]
    assert len(outlines) == 1, "the outline is drawn again over the stripes"


def test_a_clean_panel_passes_the_gate_with_labels_printed_last() -> None:
    fig, ax, drawn = _panel([0.6, 0.9, 0.5, 0.4])
    ax.set_ylim(0, 1.05)
    drawn.label(ax)
    printed = {text.get_text() for text, _ in figure_text(fig)}
    assert {"0.60", "0.90", "0.50", "0.40"} <= printed
    assert_clean(fig)


def test_a_band_name_that_fits_nowhere_raises() -> None:
    _fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.set_xlim(0, 2)
    with pytest.raises(AssertionError, match="no name for the band"):
        group_bands(
            ax,
            ["g", "g"],
            [0.0, 1.0],
            colors={"g": "#EEEEEE"},
            names={"g": ("An absurdly long band name",)},
        )


def test_a_band_name_falls_back_to_a_shorter_candidate() -> None:
    _fig, ax = plt.subplots(figsize=(4.0, 3.0))
    group_bands(
        ax,
        ["g"],
        [0.0],
        colors={"g": "#EEEEEE"},
        names={"g": ("An absurdly long band name", "Ours")},
    )
    assert [t.get_text() for t in ax.texts] == ["Ours"]


def test_a_panel_letter_sits_clear_of_the_tick_labels_and_above_the_axes() -> None:
    fig, ax = plt.subplots()
    ax.set_yticks([0, 1000, 2000], ["0", "1,000", "2,000"])
    panel_letters([ax])
    fig.canvas.draw()
    letter = next(t for t in ax.texts if t.get_text() == "A").get_window_extent()
    ticks = [t.get_window_extent() for t in ax.get_yticklabels()]
    assert letter.y0 >= ax.get_window_extent().y1, "above the top edge of the axes"
    assert not any(letter.overlaps(box) for box in ticks)
    assert letter.x0 <= min(box.x0 for box in ticks) + 1.0, "no further right than the tick labels"
