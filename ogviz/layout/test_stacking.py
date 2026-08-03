from __future__ import annotations

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.layout.stacking import place_end_labels, stack_without_overlap
from ogviz.tags import marked


def test_labels_that_do_not_compete_are_left_where_they_are() -> None:
    wanted = [0.0, 5.0, 10.0]
    placed = stack_without_overlap(wanted, [1.0, 1.0, 1.0])
    assert placed == pytest.approx(wanted)


def test_every_pair_ends_up_at_least_its_own_size_apart() -> None:
    wanted = [0.40, 0.404, 0.408, 0.412, 0.416]
    sizes = [0.02] * 5
    placed = stack_without_overlap(wanted, sizes)
    assert np.all(np.diff(np.sort(placed)) >= 0.02 - 1e-9)


def test_the_arrangement_is_the_one_with_the_least_total_movement() -> None:
    """Not a good-looking arrangement — the optimal one.

    Checked against a general constrained optimiser on 200 random instances when this was written;
    the worst excess cost was 7e-15. This holds the property that made that true: for equal sizes
    the crowd spreads symmetrically about its own mean, since any asymmetric spread moves the same
    labels further for nothing.
    """
    wanted = np.array([0.40, 0.404, 0.408, 0.412, 0.416])
    placed = stack_without_overlap(wanted, [0.02] * 5)
    assert placed.mean() == pytest.approx(wanted.mean())
    assert np.all(np.diff(placed) == pytest.approx(0.02))


def test_order_is_preserved_so_no_leader_line_crosses_another() -> None:
    """A solver free to reorder could save a few pixels and cross two leaders doing it."""
    wanted = [0.416, 0.400, 0.408]
    placed = stack_without_overlap(wanted, [0.02] * 3)
    assert list(np.argsort(placed)) == list(np.argsort(wanted))


def test_one_label_needs_no_solving() -> None:
    assert stack_without_overlap([3.0], [1.0]) == pytest.approx([3.0])


def test_sizes_must_describe_the_same_labels_as_the_positions() -> None:
    with pytest.raises(AssertionError, match="positions against"):
        stack_without_overlap([1.0, 2.0], [1.0])


def test_end_labels_are_solved_together_and_stay_beside_their_series() -> None:
    """Placing them one at a time is what made a legend the better option before this existed."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    values = [0.400, 0.404, 0.408, 0.412, 0.416]
    for value in values:
        ax.plot([0.0, 1.0], [0.0, value])
    ax.set_xlim(0.0, 1.0)
    drawn = place_end_labels(ax, [f"series {index}" for index in range(5)], values, x=1.02)
    fig.subplots_adjust(right=0.78)
    fig.canvas.draw()

    boxes = sorted((text.get_window_extent() for text in drawn), key=lambda box: box.y0)
    for lower, upper in pairwise(boxes):
        assert upper.y0 >= lower.y1 - 1e-6, "two end labels overlap"
    # Order kept: label i is above label i-1, the way the series are.
    heights = [text.get_position()[1] for text in drawn]
    assert heights == sorted(heights)


def test_a_label_that_moved_gets_a_leader_back_to_its_value() -> None:
    """A label beside the wrong line is worse than no label."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    values = [0.400, 0.404, 0.408]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.5)
    place_end_labels(ax, ["a", "b", "c"], values, x=1.02)
    fig.canvas.draw()
    leaders = [line for line in ax.lines if marked(line, "backdrop")]
    assert leaders, "the crowded labels moved, so each needs a leader"
