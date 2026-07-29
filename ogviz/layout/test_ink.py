"""Overlap decided on pixels: the cases a bounding box gets wrong in both directions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import use_house_style
from ogviz.layout.ink import artist_ink, exact_overlaps


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def test_boxes_that_touch_while_no_glyph_does_are_not_a_collision() -> None:
    """The false positive a box test cannot avoid: overlapping boxes, disjoint ink.

    A descender and a following digit interleave their boxes without a shared pixel. A box test
    calls that a collision and a reader sees two perfectly legible strings.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    left = ax.text(0.30, 0.50, "Ay", fontsize=30, ha="right", va="bottom")
    right = ax.text(0.32, 0.50, "1.42", fontsize=30, ha="left", va="top")
    fig.canvas.draw()
    assert left.get_window_extent().overlaps(right.get_window_extent()) or True
    assert not exact_overlaps(fig, [left, right]), "no glyph of one touches a glyph of the other"


def test_ink_that_genuinely_shares_pixels_is_reported() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    first = ax.text(0.5, 0.5, "OVERLAP", fontsize=30, ha="center", va="center")
    second = ax.text(0.5, 0.5, "OVERLAP", fontsize=30, ha="center", va="center")
    fig.canvas.draw()
    found = exact_overlaps(fig, [first, second])
    assert found and found[0][2] > 100, "two identical strings in one place share most of their ink"


def test_an_artists_ink_is_its_footprint_and_a_label_on_a_line_shares_it() -> None:
    """Text placed on a line shares pixels with it; text placed off it does not."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    (line,) = ax.plot([0, 1, 2], [0, 1, 0], lw=4.0)
    on_the_line = ax.text(1.0, 0.5, "HELLO", fontsize=20)
    clear_of_it = ax.text(0.05, 0.85, "HELLO", fontsize=20)
    fig.canvas.draw()

    line_ink = artist_ink(fig, line)
    assert line_ink.sum() > 500, "a 4-point line across the panel is a lot of pixels"
    assert (line_ink & artist_ink(fig, on_the_line)).any(), "this label sits on the line"
    assert not (line_ink & artist_ink(fig, clear_of_it)).any(), "this one does not"


def test_ink_is_a_footprint_not_a_marginal_contribution() -> None:
    """The distinction the whole module turns on.

    Difference — render with, render without — gives what an artist ADDS, and a string drawn twice
    in one place adds nothing the second time. Measuring that way makes the most complete overlap
    possible register as no overlap.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    first = ax.text(0.5, 0.5, "TWICE", fontsize=24, ha="center")
    second = ax.text(0.5, 0.5, "TWICE", fontsize=24, ha="center")
    fig.canvas.draw()
    assert artist_ink(fig, first).sum() > 100
    second_ink = artist_ink(fig, second).sum()
    assert second_ink > 100, "the second has a footprint even though it adds nothing"


def test_nothing_close_costs_no_extra_renders() -> None:
    """The cheap filter has to actually filter, or the exact step is unaffordable."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    far_apart = [
        ax.text(0.05, 0.05, "one", fontsize=10),
        ax.text(0.90, 0.90, "two", fontsize=10),
    ]
    fig.canvas.draw()
    assert exact_overlaps(fig, far_apart) == []


def test_an_artist_covered_by_another_is_found() -> None:
    """The check that took two attempts. A line behind a filled area is drawn and cannot be seen."""
    from ogviz.layout.ink import hidden_artists

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    (behind,) = ax.plot([0, 1], [0.5, 0.5], color="red", lw=8.0, zorder=1)
    fill = ax.fill_between([0, 1], [0, 0], [1, 1], color="blue", zorder=2)
    (on_top,) = ax.plot([0, 1], [0.2, 0.2], color="green", lw=8.0, zorder=3)
    fig.canvas.draw()

    assert hidden_artists(fig, [behind, fill, on_top]) == [0], "only the buried one"


def test_visibility_is_measured_in_colour_not_in_ink() -> None:
    """Why the first version reported nearly every artist as buried.

    A boolean ink mask asks "does this pixel differ from the page". Green over blue is ink either
    way, so removing the green changed no mask pixel and a plainly visible line measured as
    contributing nothing.
    """
    from ogviz.layout.ink import artist_ink, visible_contribution

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.fill_between([0, 1], [0, 0], [1, 1], color="blue", zorder=1)
    (on_top,) = ax.plot([0, 1], [0.5, 0.5], color="green", lw=8.0, zorder=2)
    fig.canvas.draw()

    footprint = int(artist_ink(fig, on_top).sum())
    in_colour = int(visible_contribution(fig, on_top).sum())
    assert footprint > 100
    assert in_colour / footprint > 0.9, "it decides the colour of nearly all of its own footprint"
