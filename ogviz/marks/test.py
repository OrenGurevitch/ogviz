from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.marks import (
    CENTER_GAP,
    MEAN_HALF_WIDTH,
    VIOLIN_WIDTH,
    Z_IQR,
    Z_MEAN_LINE,
    Z_MEDIAN_DOT,
    Z_POINTS,
    Z_VIOLIN,
    iqr_box,
    jitter_x,
    mean_line,
    points,
    violin,
)


def test_the_stacking_order_is_the_one_the_figure_needs() -> None:
    # A mean line under the IQR bar reads as passing behind it; a median dot under the mean
    # line disappears. The order is the contract.
    assert Z_VIOLIN < Z_POINTS < Z_IQR < Z_MEAN_LINE < Z_MEDIAN_DOT


def test_jitter_keeps_every_dot_inside_the_violin_half_width() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    xs = jitter_x(values, 3.0, rng)
    assert np.all(np.abs(xs - 3.0) <= VIOLIN_WIDTH / 2)


def test_jitter_leaves_the_centre_clear_for_the_box_and_mean() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=400)
    offsets = np.abs(jitter_x(values, 0.0, rng))
    # Points in the dense middle of the distribution must clear the central marks.
    dense = np.abs(values) < 0.5
    assert offsets[dense].min() >= CENTER_GAP * 0.85


def test_jitter_is_degenerate_safe() -> None:
    rng = np.random.default_rng(2)
    assert jitter_x(np.array([4.2]), 1.0, rng).tolist() == [1.0]
    assert jitter_x(np.full(9, 4.2), 1.0, rng).tolist() == [1.0] * 9


def test_iqr_box_puts_the_median_dot_at_the_median() -> None:
    _fig, ax = plt.subplots()
    values = np.arange(101, dtype=float)
    iqr_box(ax, values, 0.0)
    dot = next(line for line in ax.lines if line.get_marker() == "o")
    assert dot.get_ydata()[0] == pytest.approx(50.0)
    assert dot.get_zorder() == Z_MEDIAN_DOT
    plt.close("all")


def test_mean_line_sits_at_the_mean_and_carries_a_canvas_halo() -> None:
    # The halo must be the canvas colour: a white halo punches a pale gap through the dark IQR
    # bar it crosses, which is what made the mean line look like it passed behind.
    _fig, ax = plt.subplots()
    values = np.array([0.0, 1.0, 2.0, 97.0])
    mean_line(ax, values, 0.0)
    line = ax.lines[0]
    assert line.get_ydata()[0] == pytest.approx(values.mean())
    assert line.get_zorder() == Z_MEAN_LINE
    assert line.get_path_effects(), "the mean line needs a halo to stay legible over dots"
    plt.close("all")


def test_violin_and_points_take_their_declared_layers() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=50)
    _fig, ax = plt.subplots()
    violin(ax, values, 0.0, "#7C9A6E")
    points(ax, values, 0.0, "#7C9A6E", "#4A6136", rng)
    body = ax.collections[0]
    scatter = ax.collections[-1]
    assert body.get_zorder() == Z_VIOLIN
    assert scatter.get_zorder() == Z_POINTS
    plt.close("all")


def test_a_points_y_is_always_the_datum_itself() -> None:
    """The jitter is horizontal only. A vertical nudge would misplace a value on its own axis."""
    values = np.random.default_rng(11).normal(-2.1, 1.3, 40)
    _fig, ax = plt.subplots()
    points(ax, values, 1.0, "#7C9A6E", "#4A6136", np.random.default_rng(0))
    np.testing.assert_array_equal(np.asarray(ax.collections[0].get_offsets())[:, 1], values)
    plt.close("all")


def test_a_narrower_violin_narrows_its_own_jitter() -> None:
    """width is passed to both or the dots spread outside the body they belong to."""
    values = np.random.default_rng(5).normal(0.0, 1.0, 60)
    wide = jitter_x(values, 0.0, np.random.default_rng(1), width=0.8)
    narrow = jitter_x(values, 0.0, np.random.default_rng(1), width=0.62)
    assert np.abs(narrow).max() < np.abs(wide).max()
    assert np.abs(wide).max() <= 0.8 / 2


def test_a_violin_can_carry_a_coloured_outline() -> None:
    """A lighter fill with a matching edge is a legitimate house variant, not a fork."""
    values = np.random.default_rng(2).normal(0.0, 1.0, 30)
    _fig, ax = plt.subplots()
    violin(ax, values, 0.0, "#3FA372", alpha=0.28, edge_color="#3FA372", edge_width=1.3)
    body = ax.collections[0]
    assert body.get_alpha() == pytest.approx(0.28)
    assert body.get_edgecolor().shape[0] == 1
    plt.close("all")


def _lane_violations(ax, values, position: float) -> int:
    """Dots whose centre sits inside the lane reserved for the central marks."""
    from ogviz.marks import central_clearance

    offsets = np.asarray(ax.collections[0].get_offsets())
    lane = central_clearance(ax, values)
    return int(np.count_nonzero(np.abs(offsets[:, 0] - position) < lane * 0.999))


def test_no_dot_lands_on_the_central_marks() -> None:
    """The reported defect: dots sat on the mean line, whose reach (0.15) is more than twice the
    old single gap (0.06), and on the median dot wherever the body was too narrow to hold them."""
    rng = np.random.default_rng(0)
    for values in (
        rng.normal(0.0, 1.0, 60),  # ordinary
        rng.gamma(1.5, 0.5, 60),  # long tail: the body is narrow where dots still exist
        np.concatenate([rng.normal(0, 0.05, 40), rng.normal(3, 0.05, 20)]),  # bimodal
    ):
        _fig, ax = plt.subplots()
        ax.set_ylim(values.min() - 1, values.max() + 1)
        points(ax, values, 1.0, "#7C9A6E", "#4A6136", np.random.default_rng(1))
        assert _lane_violations(ax, values, 1.0) == 0
        plt.close("all")


def test_the_lane_is_widest_where_the_mean_line_is() -> None:
    """One number could not do this: the mean line reaches far wider than the whisker, but only
    over its own linewidth in y, so a blanket lane would shove every dot out for nothing."""
    from ogviz.marks import central_clearance

    values = np.linspace(-3.0, 3.0, 61)  # mean 0, and points either side of it
    _fig, ax = plt.subplots()
    ax.set_ylim(-4, 4)
    lane = central_clearance(ax, values)
    at_mean = lane[np.argmin(np.abs(values - values.mean()))]
    far = lane[0]
    assert at_mean >= MEAN_HALF_WIDTH
    assert far < at_mean / 3, "a dot out in the tail should not be pushed as far as one at the mean"
    plt.close("all")
