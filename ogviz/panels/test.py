from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.layout.ticks import MINUS
from ogviz.panels import group_violins

CONTROL, TREATED = ("#E8A838", "#B97C10"), ("#7C9A6E", "#4A6136")


def _two_groups(rng: np.random.Generator) -> list[tuple[float, np.ndarray, str, str]]:
    return [
        (0.0, rng.normal(-0.7, 0.4, 30), *CONTROL),
        (1.0, rng.normal(0.6, 0.5, 48), *TREATED),
    ]


def test_bracket_clears_the_highest_observation() -> None:
    # The failure this panel exists to prevent: a bracket drawn across a violin's tail.
    rng = np.random.default_rng(0)
    groups = _two_groups(rng)
    _fig, ax = plt.subplots()
    group_violins(ax, groups, comparisons=[(0.0, 1.0, 0.001)])
    data_max = max(float(v.max()) for _p, v, _f, _e in groups)
    span = data_max - min(float(v.min()) for _p, v, _f, _e in groups)
    bracket = next(ln.get_ydata()[1] for ln in ax.lines if len(ln.get_ydata()) == 4)
    assert bracket > data_max + 0.10 * span
    assert bracket < ax.get_ylim()[1]
    plt.close("all")


def test_an_annotated_panel_reserves_more_headroom_than_a_plain_one() -> None:
    rng = np.random.default_rng(1)
    tops = []
    for comparisons in ((), [(0.0, 1.0, 0.01)]):
        _fig, ax = plt.subplots()
        group_violins(ax, _two_groups(np.random.default_rng(1)), comparisons=comparisons)
        tops.append(ax.get_ylim()[1])
        plt.close("all")
    assert tops[1] > tops[0]
    assert rng is not None


def test_printed_means_land_below_every_observation() -> None:
    rng = np.random.default_rng(2)
    groups = _two_groups(rng)
    _fig, ax = plt.subplots()
    group_violins(ax, groups)
    data_min = min(float(v.min()) for _p, v, _f, _e in groups)
    printed = [t for t in ax.texts if t.get_position()[1] < data_min]
    assert len(printed) == 2
    # Decimals adapt to magnitude, so assert the value rather than a format string: each label
    # must read back to its group's mean within the precision it was printed at.
    # printed with a typographic minus, as matplotlib sets its ticks — undo it to parse
    labels = sorted(float(t.get_text().replace(",", "").replace(MINUS, "-")) for t in printed)
    means = sorted(float(np.mean(v)) for _p, v, _f, _e in groups)
    assert labels == pytest.approx(means, abs=5e-4)
    plt.close("all")


def test_printed_means_gain_decimals_as_the_values_get_smaller() -> None:
    # A group mean of 0.0032 printed to two decimals is "0.00" — the rule exists so a small
    # measure does not lose every digit that matters.
    rng = np.random.default_rng(9)
    labels = {}
    for scale in (0.003, 3000.0):
        _fig, ax = plt.subplots()
        group_violins(ax, [(0.0, rng.normal(scale, scale / 20, 30), *TREATED)])
        labels[scale] = ax.texts[0].get_text()
        plt.close("all")

    def decimals(label: str) -> int:
        return len(label.split(".")[1]) if "." in label else 0

    assert decimals(labels[0.003]) > decimals(labels[3000.0])
    assert decimals(labels[3000.0]) == 0, "a value in the thousands needs no decimal places"


def test_anchor_value_is_pulled_into_the_range() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(50.0, 1.0, 40)
    _fig, ax = plt.subplots()
    group_violins(ax, [(0.0, values, *TREATED)], anchor_value=0.0)
    assert ax.get_ylim()[0] < 0.0
    plt.close("all")


def test_empty_groups_are_rejected_loudly() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="at least one non-empty"):
        group_violins(ax, [(0.0, np.array([]), *TREATED)])
    plt.close("all")


def test_stacked_brackets_do_not_overlap_each_other() -> None:
    rng = np.random.default_rng(4)
    _fig, ax = plt.subplots()
    group_violins(
        ax,
        [
            (0.0, rng.normal(0, 1, 25), *CONTROL),
            (1.0, rng.normal(1, 1, 25), *TREATED),
            (2.0, rng.normal(2, 1, 25), *CONTROL),
        ],
        comparisons=[(0.0, 1.0, 0.02), (0.0, 2.0, 0.0005)],
    )
    heights = sorted(ln.get_ydata()[1] for ln in ax.lines if len(ln.get_ydata()) == 4)
    assert len(heights) == 2
    assert heights[1] > heights[0]
    plt.close("all")


def test_a_row_of_printed_means_shares_one_format():
    """Per-value decimals give a ragged "0.0887 / 0.545 / 1.57", which reads as three
    different measurements rather than one row."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(9)
    _fig, ax = plt.subplots()
    group_violins(
        ax,
        [(float(i), rng.normal(i * 0.8, 0.9, 30), "#2E7CE0", "#333333") for i in range(3)],
    )
    decimals = {len(t.get_text().split(".")[1]) for t in ax.texts if "." in t.get_text()}
    assert len(decimals) == 1, f"the row must share one format, got {decimals}"


def test_headroom_follows_the_number_of_brackets() -> None:
    """One comparison must not be given room for three.

    The panel measures what its own stack needs. A caller that overwrites the measured limit with a
    guess sized for the worst case leaves two brackets' worth of empty page on every panel that has
    one, which is what a grid of single-comparison panels was carrying.
    """
    import numpy as np

    from ogviz import group_violins

    def headroom(comparisons):
        _fig, ax = plt.subplots(figsize=(5.0, 6.0))
        rng = np.random.default_rng(3)
        groups = [(float(i), rng.normal(0.0, 1.0, 30), "#E8A838", "#B97C10") for i in range(3)]
        group_violins(ax, groups, comparisons=comparisons)
        top = max(float(v.max()) for _p, v, _f, _e in groups)
        return ax.get_ylim()[1] - top

    one = headroom([(0.0, 1.0, 0.001)])
    three = headroom([(0.0, 1.0, 0.001), (0.0, 2.0, 0.01), (1.0, 2.0, 0.04)])
    assert three > one * 1.5, "three stacked brackets need appreciably more room than one"


def test_the_printed_mean_sits_in_the_middle_of_the_margin_below_the_data() -> None:
    """Centred by construction, so changing the margin cannot leave the row crowding the violins."""
    import numpy as np

    from ogviz import group_violins
    from ogviz.panels.violins import BOTTOM_PAD, MEAN_ROW_OFFSET

    assert pytest.approx(BOTTOM_PAD / 2.0) == MEAN_ROW_OFFSET

    _fig, ax = plt.subplots(figsize=(5.0, 6.0))
    rng = np.random.default_rng(4)
    values = rng.normal(0.0, 1.0, 40)
    group_violins(ax, [(0.0, values, "#E8A838", "#B97C10")])
    printed = [t for t in ax.texts if t.get_text().strip()]
    assert printed, "the mean should be printed"
    row = printed[0].get_position()[1]
    floor, lowest = ax.get_ylim()[0], float(values.min())
    assert floor < row < lowest, "the row belongs between the frame and the lowest observation"
    assert row == pytest.approx((floor + lowest) / 2, rel=0.25), "and near the middle of that gap"


def test_a_word_label_is_set_smaller_than_a_row_of_stars() -> None:
    """ "n.s." is a word where "***" is three glyphs; at one size the non-result shouts."""
    from ogviz.significance import label_size
    from ogviz.theme import STAR_SIZE

    thin = "\u2009"  # the spacer `spaced_stars` puts between glyphs

    assert label_size("***", STAR_SIZE) == STAR_SIZE
    assert label_size(f"*{thin}{thin}*", STAR_SIZE) == STAR_SIZE  # spaced stars
    assert label_size("n.s.", STAR_SIZE) < STAR_SIZE
    assert label_size("p=0.06", STAR_SIZE) < STAR_SIZE


def test_a_shared_scale_puts_every_printed_mean_on_one_line() -> None:
    """Four rows at four heights read as four kinds of number, not one measured four times.

    Each panel places its row in the middle of the margin under its OWN data, which is right for a
    panel on its own and wrong the moment panels share a floor: the floor is common and the lowest
    violin is not.
    """
    import numpy as np

    from ogviz import group_violins, share_value_limits
    from ogviz.qc import mean_rows_unaligned

    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0))
    for index, ax in enumerate(axes):
        values = rng.normal(index * 0.8, 1.0, 30)
        group_violins(ax, [(0.0, values, "#E8A838", "#B97C10")])
    rows = [t for ax in axes for t in ax.texts if getattr(t, "ogviz_mean_row", False)]
    assert len({round(t.get_position()[1], 6) for t in rows}) > 1, "they start out unaligned"

    share_value_limits(axes)
    fig.canvas.draw()
    assert not mean_rows_unaligned(fig)
    aligned = {round(t.get_position()[1], 6) for t in rows}
    assert len(aligned) == 1, "one line for the whole grid"
    floor = axes[0].get_ylim()[0]
    lowest = min(
        float(np.asarray(path.vertices, dtype=float)[:, 1].min())
        for ax in axes
        for collection in ax.collections
        for path in collection.get_paths()
    )
    assert aligned.pop() == pytest.approx((floor + lowest) / 2), "midway between floor and deepest"
