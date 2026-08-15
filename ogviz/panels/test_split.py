"""Two half-violins back to back: what each half owns, and what it refuses.

The module had no test file, and writing one found two inputs that reached numpy and matplotlib
before anything here spoke — including the one the docstring explicitly invites, since it promises
the two sides need not be the same length.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import split_violins
from ogviz.panels.split import PAIR_WIDTH
from ogviz.tags import marked

pytestmark = pytest.mark.usefixtures("pinned_font")

COLORS = {"left_color": "#E8A838", "right_color": "#7C9A6E"}


def _pair(categories=("sensor A", "sensor B"), seed: int = 0):
    rng = np.random.default_rng(seed)
    left = [rng.normal(10.0, 1.0, 40) for _ in categories]
    right = [rng.normal(10.4, 1.0, 40) for _ in categories]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    split_violins(ax, list(categories), left, right, **COLORS)
    fig.canvas.draw()
    return fig, ax


def test_each_category_gets_two_halves() -> None:
    fig, ax = _pair()
    assert len(ax.collections) >= 4, "two halves for each of two categories"
    plt.close(fig)


def test_the_halves_sit_on_opposite_sides_of_their_shared_spine() -> None:
    """The whole point: edge to edge, so a difference of a fraction of a percent is visible."""
    fig, ax = _pair(categories=("only",))
    spans = [body.get_paths()[0].get_extents() for body in ax.collections if body.get_paths()]
    lefts = [box for box in spans if box.x1 <= 0.0 + 1e-6]
    rights = [box for box in spans if box.x0 >= 0.0 - 1e-6]
    assert lefts and rights, "one body each side of the category position"
    assert max(box.x1 for box in lefts) <= min(box.x0 for box in rights) + 1e-6
    plt.close(fig)


def test_a_pair_stays_inside_its_own_slot() -> None:
    """Neighbours must stay clear, so a PAIR is `width` wide and each half is half of it."""
    fig, ax = _pair(categories=("a", "b"))
    for body in ax.collections:
        if not body.get_paths():
            continue
        box = body.get_paths()[0].get_extents()
        nearest = round(float((box.x0 + box.x1) / 2))
        assert abs(box.x0 - nearest) <= PAIR_WIDTH / 2 + 1e-6
        assert abs(box.x1 - nearest) <= PAIR_WIDTH / 2 + 1e-6
    plt.close(fig)


def test_each_half_prints_its_own_mean_and_they_share_one_row() -> None:
    """Neither half borrows the other's summary, and a row at two heights reads as two kinds."""
    fig, ax = _pair(categories=("only",))
    rows = [text for text in ax.texts if marked(text, "mean_row")]
    assert len(rows) == 2, [text.get_text() for text in rows]
    assert rows[0].get_text() != rows[1].get_text(), "two distributions, two means"
    heights = {round(float(text.get_position()[1]), 6) for text in rows}
    assert len(heights) == 1, f"the printed means sit at {len(heights)} heights"
    plt.close(fig)


def test_the_means_can_be_turned_off() -> None:
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    split_violins(
        ax, ["only"], [rng.normal(0, 1, 30)], [rng.normal(1, 1, 30)], show_means=False, **COLORS
    )
    fig.canvas.draw()
    assert [text for text in ax.texts if marked(text, "mean_row")] == []
    plt.close(fig)


def test_the_two_sides_may_differ_in_length() -> None:
    """A paired measurement can drop samples on one side, which the docstring promises."""
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    split_violins(ax, ["only"], [rng.normal(0, 1, 12)], [rng.normal(1, 1, 40)], **COLORS)
    fig.canvas.draw()
    plt.close(fig)


def test_an_empty_side_is_refused_by_name() -> None:
    """Dropping SOME samples is promised; dropping all of them is a different thing.

    It used to surface as `IndexError: list index out of range` from inside the violin — an error
    naming nothing the caller wrote, on the input the docstring invites.
    """
    rng = np.random.default_rng(3)
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="is empty"):
        split_violins(ax, ["only"], [np.array([])], [rng.normal(0, 1, 20)], **COLORS)


def test_no_categories_is_refused_by_name() -> None:
    """It reached numpy first: "need at least one array to concatenate"."""
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="at least one category"):
        split_violins(ax, [], [], [], **COLORS)


def test_mismatched_lengths_are_refused() -> None:
    rng = np.random.default_rng(4)
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="categories"):
        split_violins(ax, ["a", "b"], [rng.normal(0, 1, 10)], [rng.normal(0, 1, 10)], **COLORS)


def test_a_non_finite_value_is_refused_and_says_which_side() -> None:
    """A gap that reads as a zero is the caller's choice to make, where it is visible."""
    rng = np.random.default_rng(5)
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="right series for 'only'"):
        split_violins(
            ax, ["only"], [rng.normal(0, 1, 10)], [np.array([1.0, np.nan, 3.0])], **COLORS
        )


def test_a_single_observation_still_draws() -> None:
    """n=1 has no shape, but refusing it would refuse a legitimate paired design."""
    rng = np.random.default_rng(6)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    split_violins(ax, ["only"], [np.array([2.0])], [rng.normal(0, 1, 20)], **COLORS)
    fig.canvas.draw()
    plt.close(fig)


def test_the_panel_passes_its_own_gate() -> None:
    """The claim the package makes about every panel it draws."""
    from ogviz.qc import audit

    fig, _ax = _pair()
    assert audit(fig) == []
    plt.close(fig)
