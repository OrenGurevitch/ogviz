"""Measuring how much of the page a figure actually uses."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import use_house_style
from ogviz.layout.density import dead_space, measure, panel_emptiness, trim_margins


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def _curve(ax):
    x = np.linspace(0.0, 10.0, 200)
    ax.plot(x, np.sin(x) * 0.1 + 0.5, lw=2.0)


def test_a_generous_limit_shows_up_as_an_empty_band() -> None:
    """The number that says "your y-limit is too big" — invisible to any bounding box."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    ax.set_ylim(0.0, 5.0)
    assert panel_emptiness(fig, ax)["top"] > 0.5


def test_a_fitted_panel_reports_almost_no_empty_band() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    ax.set_ylim(0.38, 0.62)
    ax.set_xlim(0.0, 10.0)
    panel = panel_emptiness(fig, ax)
    assert panel["top"] < 0.12 and panel["bottom"] < 0.12


def test_gridlines_do_not_make_an_empty_panel_look_full() -> None:
    """A grid inks every edge, so the raw render calls every panel full however empty it is."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    ax.set_ylim(0.0, 5.0)
    ax.grid(True)
    assert panel_emptiness(fig, ax)["top"] > 0.5


def test_trim_margins_reclaims_outer_space_and_changes_no_data() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    fig.subplots_adjust(left=0.35, right=0.65, top=0.65, bottom=0.35)
    limits = (ax.get_xlim(), ax.get_ylim())
    before = measure(fig)
    assert trim_margins(fig) is True
    after = measure(fig)
    assert after.wasted_margin() < before.wasted_margin()
    assert after.coverage > before.coverage
    assert (ax.get_xlim(), ax.get_ylim()) == limits, "trimming must not touch the data limits"


def test_a_figure_that_already_fills_its_page_is_left_alone() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    fig.tight_layout(pad=0.05)
    assert trim_margins(fig) is False


def test_dead_space_is_quiet_on_a_well_packed_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    ax.set_ylim(0.38, 0.62)
    fig.tight_layout(pad=0.05)
    assert not dead_space(fig)
