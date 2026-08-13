"""What the coupling panel promises: one scale for every strip, and no invented statistics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.panels.coupling import (
    Cloud,
    Estimate,
    Leg,
    coupling_panels,
    estimate_strip,
    shared_limits,
    trend_line,
)


def _cloud(seed: int = 0, size: int = 30) -> Cloud:
    rng = np.random.default_rng(seed)
    return Cloud(rng.normal(0, 1, size), rng.normal(0, 1, size), "#2E7CE0", "#1B4E8F", "a group")


def _leg(*estimates: Estimate) -> Leg:
    return Leg("x", "y", (_cloud(),), estimates)


def test_an_estimate_outside_its_interval_is_rejected() -> None:
    """A dot drawn outside its own bar is a wiring mistake, not a result worth rendering."""
    with pytest.raises(AssertionError, match="outside its interval"):
        Estimate("row", 0.9, (0.1, 0.5), "#000000")


def test_a_backwards_interval_is_rejected() -> None:
    with pytest.raises(AssertionError, match="runs backwards"):
        Estimate("row", 0.3, (0.5, 0.1), "#000000")


def test_every_strip_gets_the_same_scale() -> None:
    """The comparison across columns is the point; a per-panel scale would silently break it."""
    legs = (
        _leg(Estimate("r", 0.1, (0.0, 0.2), "#000000")),
        _leg(Estimate("r", 0.6, (0.4, 0.9), "#000000")),
        _leg(Estimate("r", -0.3, (-0.5, -0.1), "#000000")),
    )
    fig = plt.figure(figsize=(12, 6))
    coupling_panels(fig, legs)
    strips = [ax for ax in fig.axes if ax.get_ylim()[0] == pytest.approx(-0.7)]
    assert len(strips) == len(legs)
    assert len({tuple(round(v, 9) for v in ax.get_xlim()) for ax in strips}) == 1


def test_the_shared_scale_reaches_the_widest_interval() -> None:
    legs = (_leg(Estimate("r", 0.6, (0.4, 0.9), "#000000")),)
    low, high = shared_limits(legs)
    assert low < -0.9 <= 0.9 < high
    assert low == pytest.approx(-high), "a strip centred off zero reads as a lean before it is read"


def test_only_the_first_strip_names_its_rows() -> None:
    """Repeating the row names under every column spends width and tells the reader nothing."""
    rows = (Estimate("Pooled", 0.2, (0.1, 0.3), "#000000"),)
    fig = plt.figure(figsize=(12, 6))
    coupling_panels(fig, (_leg(*rows), _leg(*rows)))
    strips = [ax for ax in fig.axes if ax.get_ylim()[0] == pytest.approx(-0.7)]
    named = [[t.get_text() for t in ax.get_yticklabels() if t.get_text()] for ax in strips]
    assert named[0] == ["Pooled"]
    assert named[1] == []


def test_two_points_get_no_trend_line() -> None:
    """A line through two points describes those two points and nothing else."""
    _fig, ax = plt.subplots()
    before = len(ax.lines)
    assert trend_line(ax, np.array([0.0, 1.0]), np.array([0.0, 1.0]), color="#000000") is None
    assert len(ax.lines) == before


def test_a_trend_line_spans_only_the_data() -> None:
    """Extrapolating past the observations asserts something that was never measured."""
    _fig, ax = plt.subplots()
    x = np.array([2.0, 3.0, 4.0, 5.0])
    trend_line(ax, x, 2 * x + 1, color="#000000")
    drawn = ax.lines[-1].get_xdata()
    assert (float(min(drawn)), float(max(drawn))) == pytest.approx((2.0, 5.0))


def test_a_strip_with_no_estimates_is_refused() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="no estimates"):
        estimate_strip(ax, (), limits=(-1.0, 1.0))


def test_points_stacked_on_one_x_get_no_trend_line() -> None:
    """`polyfit` returns a meaningless slope for them, with a warning `strict` does not catch.

    So the panel drew a confident guide through a column of points and the gate passed it.
    """
    _fig, ax = plt.subplots()
    before = len(ax.lines)
    stacked = np.array([2.0, 2.0, 2.0, 2.0])
    assert trend_line(ax, stacked, np.array([1.0, 2.0, 3.0, 4.0]), color="#000000") is None
    assert len(ax.lines) == before, "and nothing was drawn on the way to deciding that"


def test_a_ratio_scaled_figure_can_say_where_its_null_is() -> None:
    """`estimate_strip` took a `reference` and `coupling_panels` had no way to pass one.

    The shared scale was centred on zero regardless, so a quantity whose null is 1.0 — an odds
    ratio, a hazard ratio — was drawn with its rule in the wrong place and its scale off-centre.
    """
    rows = (Estimate("pooled", 1.4, (1.1, 1.7), "#000000"),)
    low, high = shared_limits((_leg(*rows),), reference=1.0)
    assert (low + high) / 2 == pytest.approx(1.0), "the scale is centred on the null"
    assert low < 1.1 and high > 1.7, "and still holds the interval"

    fig = plt.figure(figsize=(9, 6))
    coupling_panels(fig, (_leg(*rows),), reference=1.0)
    (strip,) = [ax for ax in fig.axes if ax.get_ylim()[0] == pytest.approx(-0.7)]
    rules = [line.get_xdata()[0] for line in strip.lines if len(set(line.get_xdata())) == 1]
    assert any(value == pytest.approx(1.0) for value in rules), rules


def test_a_leg_without_estimates_leaves_no_hole_under_it() -> None:
    """An unfilled gridspec cell in a finished figure reads as a rendering failure."""
    rows = (Estimate("pooled", 0.2, (0.1, 0.3), "#000000"),)
    fig = plt.figure(figsize=(12, 6))
    with_rows, without = _leg(*rows), _leg()
    coupling_panels(fig, (with_rows, without))
    fig.canvas.draw()

    scatters = [ax for ax in fig.axes if ax.get_ylim()[0] != pytest.approx(-0.7)]
    tall = max(scatters, key=lambda ax: ax.get_window_extent().height)
    short = min(scatters, key=lambda ax: ax.get_window_extent().height)
    assert tall.get_window_extent().height > short.get_window_extent().height * 1.5, (
        "the scatter with no strip under it takes the whole column"
    )
