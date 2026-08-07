import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.layout import text_overlaps
from ogviz.layout.ticks import MINUS
from ogviz.panels import Series, bar_panel
from ogviz.tags import marked
from ogviz.theme import SERIES


def _label_positions(ax) -> dict[str, float]:
    """Printed value label -> its y in data units."""
    return {t.get_text(): t.get_position()[1] for t in ax.texts}


def test_negative_bar_labels_sit_below_the_bar_not_inside_it():
    """A label placed above every bar lands inside the negative ones, hiding the number."""
    _fig, ax = plt.subplots()
    values = np.array([-1.4, 0.9])
    bar_panel(ax, [Series("z", values, SERIES[0])], ["down", "up"])
    labels = _label_positions(ax)
    assert labels[f"{MINUS}1.40"] < -1.4, "negative bar's label must clear its free end downward"
    assert labels["+0.90"] > 0.9, "positive bar's label must clear its free end upward"


def test_label_clears_the_whisker_cap():
    """Placing the label at the bar top buries it under the error bar whenever one is drawn."""
    _fig, ax = plt.subplots()
    values, errors = np.array([2.0]), np.array([0.6])
    bar_panel(ax, [Series("z", values, SERIES[0], errors)], ["a"])
    assert _label_positions(ax)["2.00"] > 2.6


def test_asymmetric_errors_use_the_matching_cap_on_each_side():
    """A CI is not symmetric; the label must clear the UPPER bound, not the mean +- lower."""
    _fig, ax = plt.subplots()
    values = np.array([1.0])
    bar_panel(ax, [Series("z", values, SERIES[0], np.array([[0.1], [0.9]]))], ["a"])
    assert _label_positions(ax)["1.00"] > 1.9


def test_grouped_series_stay_within_one_series_width():
    """Two series must divide the slot, not each take a full-width bar and overlap."""
    _fig, ax = plt.subplots()
    bar_panel(
        ax,
        [
            Series("ctrl", np.array([1.0, 2.0]), SERIES[0]),
            Series("lc", np.array([1.5, 2.5]), SERIES[1]),
        ],
        ["one", "two"],
        width=0.6,
    )
    lefts = sorted(patch.get_x() for patch in ax.patches)
    widths = {round(patch.get_width(), 6) for patch in ax.patches}
    assert len(widths) == 1
    # Both bars for category 0 lie inside [-0.3, +0.3] around the tick.
    assert lefts[0] >= -0.301
    assert lefts[1] + max(widths) <= 0.301


def test_value_labels_do_not_collide_in_a_grouped_panel():
    """The regression that keeps recurring: labels colliding once bars get close."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    rng = np.random.default_rng(0)
    bar_panel(
        ax,
        [
            Series("control", rng.normal(-0.4, 0.2, 5), SERIES[0], np.full(5, 0.12)),
            Series("treated", rng.normal(0.6, 0.2, 5), SERIES[1], np.full(5, 0.12)),
        ],
        ["fatigue", "sleep", "pain", "cognition", "autonomic"],
    )
    assert text_overlaps(fig) == []


def test_series_length_must_match_the_categories():
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="values for"):
        bar_panel(ax, [Series("z", np.array([1.0, 2.0]), SERIES[0])], ["only-one"])


def _decimals(ax) -> int:
    text = next(iter(_label_positions(ax)))
    return len(text.split(".")[1]) if "." in text else 0


def test_decimals_scale_with_magnitude():
    """Auto formatting: small values need more places than large ones to stay informative.

    A fixed format is what makes a shared helper unusable — ",.2f" turns 0.004 into "+0.00".
    """
    _fig, ax = plt.subplots()
    bar_panel(ax, [Series("z", np.array([0.004, 0.008]), SERIES[0])], ["a", "b"])
    small = _decimals(ax)
    plt.close(_fig)
    _fig, ax = plt.subplots()
    bar_panel(ax, [Series("z", np.array([4100.0, 8200.0]), SERIES[0])], ["a", "b"])
    assert small > _decimals(ax)
    assert small >= 3, "a value of 0.004 must not round to zero in its own label"


def test_reference_line_is_drawn_and_labelled():
    _fig, ax = plt.subplots()
    bar_panel(
        ax,
        [Series("z", np.array([0.5, 0.7]), SERIES[0])],
        ["a", "b"],
        reference=(0.80, "reference level"),
    )
    assert any(round(line.get_ydata()[0], 6) == 0.80 for line in ax.lines)
    assert "reference level" in {t.get_text() for t in ax.texts}


def test_a_bar_series_refuses_non_finite_values():
    """A non-finite bar draws nothing and the gap reads as a zero."""
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match=r"1 non-finite value\(s\) of 3"):
        bar_panel(ax, [Series("z", np.array([1.0, np.nan, 3.0]), SERIES[0])], list("abc"))


def test_a_reference_line_is_drawn_over_the_bars_it_is_read_against() -> None:
    """Behind them a threshold survives only in the gaps, which loses the comparison it was for."""
    from ogviz.panels.bars import Z_BAR, reference_line

    _fig, ax = plt.subplots()
    bar_panel(ax, [Series("s", np.array([0.3, 0.9, 1.4]), ["#8FA9C9"] * 3)], ["a", "b", "c"])
    reference_line(ax, 0.8, "threshold")
    drawn = [line for line in ax.lines if marked(line, "reference")]
    assert len(drawn) == 1
    assert drawn[0].get_zorder() > Z_BAR, "the threshold must sit above the bars"


def test_a_buried_threshold_is_reported() -> None:
    from ogviz.panels.bars import reference_line
    from ogviz.qc import buried_baselines

    fig, ax = plt.subplots()
    bar_panel(ax, [Series("s", np.array([0.3, 0.9, 1.4]), ["#8FA9C9"] * 3)], ["a", "b", "c"])
    reference_line(ax, 0.8, "threshold")
    fig.canvas.draw()
    assert not buried_baselines(fig)

    [line] = [line for line in ax.lines if marked(line, "reference")]
    line.set_zorder(1)
    fig.canvas.draw()
    assert any("reference line" in complaint for complaint in buried_baselines(fig))


def test_a_rounded_corner_is_the_same_size_on_any_axis() -> None:
    """The radius was a fraction of the tallest VALUE, and `rounding_size` applies to BOTH axes.

    So it was only sensible where the value axis happened to be order-1: measured on a counts axis,
    the corner came out at 394,460% of the bar's own width — a lozenge rather than a bar.
    """
    from matplotlib.patches import FancyBboxPatch

    from ogviz import bar_panel
    from ogviz.panels.bars import Series

    def corner_px(values: list[float]) -> float:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        bar_panel(ax, [Series("s", values, "#7C9A6E")], ["A", "B", "C"], rounded=True)
        fig.canvas.draw()
        patch = next(p for p in ax.patches if isinstance(p, FancyBboxPatch))
        low, high = ax.get_xlim()
        return patch.get_boxstyle().rounding_size / (high - low) * ax.get_window_extent().width

    near_one = corner_px([0.42, 0.61, 0.70])
    near_thousands = corner_px([26000.0, 38000.0, 45000.0])
    assert near_one == pytest.approx(near_thousands, rel=0.01), (near_one, near_thousands)
