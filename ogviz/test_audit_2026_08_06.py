"""The defects found in the 2026-08-06 read-through, each held so it cannot come back.

Every one of these passed the tests, the linter and the type checker on the day it was found. That
is what they have in common and why they are worth locking: none of them was a crash, and each was
either silent (a check that quietly stopped checking) or plausible (a number that looked right).
"""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import ogviz
from ogviz.layout.axis import drawn_value_extent
from ogviz.layout.panels import rows_that_fit
from ogviz.layout.ticks import format_value
from ogviz.marks import central_clearance
from ogviz.panels.bars import HIGHLIGHT_FILL
from ogviz.panels.grid import align_mean_rows
from ogviz.qc.arrangement import panels_disagree_about_ticks
from ogviz.qc.marks import buried_baselines
from ogviz.qc.repair import knock_out_labels_over_rules


def _sample(count: int = 60, shift: float = 0.0) -> np.ndarray:
    return np.random.default_rng(0).normal(size=count) + shift


def test_the_gates_survive_python_dash_oh() -> None:
    """`assert` is deleted by `-O`, and every contract in this package was written as one.

    Measured before the fix: a figure whose two labels shared 524 px of ink passed `assert_clean`
    under `-O` without a word, and `stars(-5.0)` returned "***" — the most significant result on the
    figure, from a value that is not a probability.

    Run in a subprocess because `-O` is an interpreter flag; there is no other way to test it.
    """
    program = (
        "import matplotlib; matplotlib.use('Agg');"
        "import matplotlib.pyplot as plt, ogviz;"
        "fig, ax = plt.subplots(figsize=(3, 2));"
        "ax.text(0.5, 0.5, 'aaaaaaaaaaaaaaaa'); ax.text(0.5, 0.5, 'bbbbbbbbbbbbbbbb');"
        "ok = True\n"
        "try:\n"
        "    ogviz.assert_clean(fig)\n"
        "except AssertionError:\n"
        "    ok = False\n"
        "try:\n"
        "    ogviz.stars(-5.0); starred = True\n"
        "except AssertionError:\n"
        "    starred = False\n"
        "print('clean', ok, 'starred', starred)"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program], capture_output=True, text=True, check=True
    )
    assert "clean False starred False" in result.stdout, result.stdout


def test_a_caller_who_states_a_tick_precision_gets_it() -> None:
    """`value_ticks` passed `strip_trailing_zeros=True` over the top of its own `decimals`."""
    _fig, ax = plt.subplots()
    ax.set_ylim(0.0, 3.0)
    ogviz.value_ticks(ax, count=4, decimals=2)
    assert [text.get_text() for text in ax.get_yticklabels()] == ["0.60", "1.20", "1.80", "2.40"]


def test_a_small_negative_keeps_its_sign() -> None:
    """Stripping turned "-0.00" into "-0", and the signed-zero rule then printed it as "0"."""
    from ogviz.layout.ticks import MINUS

    assert format_value(-0.0014, decimals=2, strip_trailing_zeros=True) == f"{MINUS}0.00"
    # A value that IS zero still loses the sign, and an auto-chosen precision still strips.
    assert format_value(-0.0) == "0"
    assert format_value(0.0, decimals=2, strip_trailing_zeros=False) == "0.00"


def test_the_figures_own_width_cannot_change_the_row_count() -> None:
    """Not a bug in the arithmetic, which cancels — but it was REQUIRED and could do nothing."""
    assert rows_that_fit(2, width=4.0) == rows_that_fit(2) == rows_that_fit(2, width=40.0)


def test_a_dot_in_the_tail_clears_the_whisker_not_the_box() -> None:
    """The branch that was supposed to narrow the lane outside Q1-Q3 had two identical arms."""
    _fig, ax = plt.subplots()
    values = _sample(200)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(float(values.min()), float(values.max()))
    lane = central_clearance(ax, values)

    q1, q3 = (float(v) for v in np.percentile(values, [25, 75]))
    median, mean = float(np.median(values)), float(values.mean())
    in_the_box = (values >= q1) & (values <= q3)
    # Out in a tail, and clear of the median dot and the mean line, which have lanes of their own.
    in_a_tail = ~in_the_box & (np.abs(values - median) > 0.5) & (np.abs(values - mean) > 0.5)
    assert in_a_tail.any() and in_the_box.any(), "the sample has to cover both cases"
    assert lane[in_a_tail].max() < lane[in_the_box].min(), (
        "a dot beside the thin whisker must reserve less room than one beside the IQR bar"
    )


def test_a_label_over_a_highlighted_range_knocks_out_to_the_shade() -> None:
    """`position == highlight` compared an int against a tuple, so a range never matched."""
    from matplotlib.colors import to_hex

    _fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ogviz.bar_panel(
        ax,
        [ogviz.Series("s", [1.0, 2.0, 3.0], "#2E7CE0")],
        ["a", "b", "c"],
        highlight=(0, 1),
    )
    boxes = [
        to_hex(text.get_bbox_patch().get_facecolor())
        for text in ax.texts
        if text.get_bbox_patch() is not None
    ]
    assert boxes[:2] == [HIGHLIGHT_FILL.lower()] * 2, "the two shaded categories"
    assert boxes[2] != HIGHLIGHT_FILL.lower(), "and the one outside the range"


def test_repair_fixes_a_label_that_contains_an_apostrophe() -> None:
    """It recovered the label by splitting the complaint on `'`, which such a label breaks."""
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.grid(visible=True, axis="y")
    ax.set_ylim(0.0, 1.0)
    fig.canvas.draw()
    tick = next(float(t) for t in ax.get_yticks() if 0.0 < float(t) < 1.0)
    label = ax.text(0.5, tick, "won't fit", ha="center", va="center")
    assert knock_out_labels_over_rules(fig), "it has to report what it did"
    assert label.get_bbox_patch() is not None, "and it has to have actually done it"


def test_align_mean_rows_takes_a_generator() -> None:
    """It walked its `Iterable` three times, so a generator placed nothing and said nothing."""
    _fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ogviz.group_violins(ax, [(0.0, _sample(40), "#E8A838", "#B97C10")])
    assert align_mean_rows((ax for ax in axes), floor=0.0) is not None


def test_a_buried_threshold_is_named_by_its_own_value_on_a_horizontal_panel() -> None:
    """It read `get_ydata`, which on a horizontal panel is the category coordinate."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ogviz.bar_panel(
        ax,
        [ogviz.Series("s", [1.0, 2.0, 3.0], "#2E7CE0")],
        ["a", "b", "c"],
        orientation="horizontal",
        reference=(2.5, "target"),
    )
    for line in ax.lines:
        if ogviz.marked(line, "reference"):
            line.set_zorder(0)  # bury it, so the check has something to report
    complaints = [c for c in buried_baselines(fig) if "reference line" in c]
    assert complaints and "2.5" in complaints[0], complaints


def test_bar_panels_on_one_scale_must_agree_about_their_ticks() -> None:
    """The check skipped any panel with no `collections`, which is every bar panel."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for ax, ticks in zip(axes, ([0.0, 1.0, 2.0], [0.0, 0.5, 1.0, 1.5, 2.0]), strict=True):
        ogviz.bar_panel(ax, [ogviz.Series("s", [1.0, 2.0], "#2E7CE0")], ["a", "b"])
        ax.set_ylim(0.0, 2.5)
        ax.set_yticks(ticks)
    assert panels_disagree_about_ticks(fig)


def test_the_drawn_extent_sees_bars_and_lines_but_not_the_furniture() -> None:
    """It read `ax.collections` alone, so a bar panel reported that nothing was drawn on it.

    And the correction has its own trap, which is why both halves are held here: counting a BRACKET
    as a mark makes "how far do the marks reach" answer "to the top of the bracket stack", and
    `ticks_in_the_headroom` — which subtracts exactly that — silently stops firing.
    """
    _fig, ax = plt.subplots()
    ax.bar([0, 1], [1.0, 2.0])
    ax.plot([0, 1], [3.0, 4.0])
    extent = drawn_value_extent(ax)
    assert extent is not None and extent == pytest.approx((0.0, 4.0))

    _fig2, ax2 = plt.subplots(figsize=(7.0, 8.0))
    ogviz.group_violins(
        ax2,
        [(0.0, _sample(60), "#E8A838", "#B97C10"), (1.0, _sample(60, 1.0), "#7C9A6E", "#4A6136")],
        comparisons=[(0.0, 1.0, 0.001)],
    )
    reach = drawn_value_extent(ax2)
    crossbars = [
        float(np.max(line.get_ydata())) for line in ax2.lines if ogviz.marked(line, "bracket")
    ]
    assert reach is not None and crossbars
    assert reach[1] < min(crossbars), "a bracket is drawn ABOUT the marks, not among them"


def test_guarded_leaves_is_guarded_telling_the_truth() -> None:
    """The block restored `savefig` and not the handle `is_guarded` compares it against."""
    ogviz.guard()
    try:
        assert ogviz.is_guarded()
        with ogviz.guarded(mode="warn"):
            assert ogviz.is_guarded()
        assert ogviz.is_guarded(), "the outer guard is still installed and must still be reported"
    finally:
        ogviz.unguard()
    assert not ogviz.is_guarded()


def test_the_lane_follows_the_widths_the_marks_are_actually_drawn_at() -> None:
    """Defaults agreeing is not the same as the widths matching.

    `central_clearance` reserves a lane and `iqr_box` draws the bar, and until the two were wired
    together a caller who widened one got dots placed against the other's default — sitting on the
    bar. Found while fixing the whisker lane, which is the same failure one step in.
    """
    from ogviz.tags import value_of

    values = np.random.default_rng(0).normal(size=200)
    q1, q3 = (float(v) for v in np.percentile(values, [25, 75]))
    median, mean = float(np.median(values)), float(values.mean())
    at_the_bar = (
        (values >= q1)
        & (values <= q3)
        & (np.abs(values - median) > 0.3)
        & (np.abs(values - mean) > 0.3)
    )
    in_the_tail = (values < q1) | (values > q3)

    def lane_for(**box_kwargs) -> np.ndarray:
        plt.close("all")
        _fig, ax = plt.subplots(figsize=(7.0, 8.0))
        ogviz.group_violins(ax, [(0.0, values, "#E8A838", "#B97C10")], box_kwargs=box_kwargs)
        return next(value_of(c, "lane") for c in ax.collections if value_of(c, "lane") is not None)

    narrow, wide = lane_for(box_width=5.5), lane_for(box_width=12.0)
    assert wide[at_the_bar].max() > narrow[at_the_bar].max(), "a wider bar needs a wider lane"

    thin, thick = lane_for(whisker_width=1.5), lane_for(whisker_width=6.0)
    assert thick[in_the_tail].min() > thin[in_the_tail].min(), "and so does a thicker whisker"


def test_a_caller_can_still_override_the_lane_widths() -> None:
    """`setdefault`, so an explicit `mark_widths` beats what the box kwargs imply."""
    from ogviz.marks import widths_of

    assert widths_of({"box_width": 9.0, "color": "#000000"}) == {"box_linewidth": 9.0}
    assert widths_of({"half_width": 0.3}, {"linewidth": 4.0}) == {
        "mean_half_width": 0.3,
        "mean_linewidth": 4.0,
    }
    assert widths_of(None, {}) == {}
