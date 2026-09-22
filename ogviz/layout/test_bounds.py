from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.layout.bounds import text_off_canvas, text_wider_than_its_panel

# Rendered-text assertions: measured under the font every machine has (see conftest.py).
pytestmark = pytest.mark.usefixtures("pinned_font")


def test_a_label_running_off_the_page_is_caught() -> None:
    """The defect `clipped_artists` cannot see: it tests LINES, and matplotlib never clips text."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.text(1.0, 0.9, "a right-hand label that runs off the page entirely", ha="left")
    fig.canvas.draw()
    assert any("off the page" in complaint for complaint in text_off_canvas(fig))


def test_an_ordinary_panel_is_not_reported_for_its_tick_labels() -> None:
    """End tick labels sit a few pixels past the canvas on purpose — a tight save absorbs them.

    Counting them reported four complaints on a figure with nothing wrong, which is how a gate
    stops being believed.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_title("a short title")
    fig.canvas.draw()
    assert not text_off_canvas(fig)


def test_a_label_wider_than_its_own_panel_is_caught() -> None:
    """It sits exactly where it belongs and reaches across the panel next door."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax in axes:
        ax.plot([0.0, 1.0], [0.0, 1.0])
    axes[0].set_title("a sub-line long enough to reach across the panel beside it, easily")
    fig.canvas.draw()
    complaints = text_wider_than_its_panel(fig)
    assert any("wider than" in complaint for complaint in complaints), complaints
    # The panel is named so the reader does not have to grep for the string. Here the offender IS
    # the title, so naming the panel by its title would just repeat it — the index is used instead.
    assert any("panel 0 of 2" in complaint for complaint in complaints), complaints


def test_a_label_that_fits_its_panel_is_left_alone() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax in axes:
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.set_title("short")
    fig.canvas.draw()
    assert not text_wider_than_its_panel(fig)


def test_a_single_panel_figure_is_not_told_about_a_neighbour_it_has_not_got() -> None:
    """Two things were said unconditionally, and both are false when there is only one panel.

    The complaint named "panel 0 of 1 (in reading order)" — an identification that was never in
    doubt — and ended "so it reaches across the one beside it", asserting a collision that cannot
    happen and sending a reader to look for it. What actually happens is that the label runs into
    the margin, which is a different edit.
    """
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_title("a horizontal title far too long for this narrow panel indeed")
    fig.canvas.draw()

    (complaint,) = text_wider_than_its_panel(fig)
    assert "of 1" not in complaint, complaint
    assert "beside it" not in complaint, complaint
    assert "margin" in complaint, complaint


def test_a_rotated_label_is_told_which_way_to_edit() -> None:
    """For rotated text the line COUNT is the width, so reflowing is exactly the wrong remedy."""
    from ogviz.layout.bounds import _rotation_hint

    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    upright = ax.set_title("a title")
    rotated = ax.set_ylabel("a label", rotation=90)
    fig.canvas.draw()

    assert "ROTATED" in _rotation_hint(rotated)
    assert "shorten it" in _rotation_hint(rotated)
    assert _rotation_hint(upright) == "", "a horizontal label gets the opposite advice, correctly"


def test_a_panel_is_identified_one_way_across_every_check() -> None:
    """Three conventions existed, so one report could open three ways about the same panel.

    A title where the panel has one, because that is what a person reads off the picture; a
    reading-order index only where it does not; and NOTHING on a single-panel figure, where
    `axes 0:` was a prefix identifying nothing on the commonest kind of figure there is.
    """
    from ogviz.layout.bounds import panel_prefix

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    assert panel_prefix(fig, ax) == ""
    ax.set_title("Latency")
    assert panel_prefix(fig, ax) == "panel 'Latency': "

    grid, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    assert panel_prefix(grid, axes[1]) == "panel 1 of 3: "
    axes[1].set_title("Throughput")
    assert panel_prefix(grid, axes[1]) == "panel 'Throughput': "
    plt.close("all")


def test_a_subfigure_title_is_read_like_any_other_label() -> None:
    """`fig.axes` reaches into subfigures, but `fig.texts` does not, and nor did this walker.

    A subfigure's `suptitle` lives in that subfigure's own `texts`, so a title running off the page
    or into a panel was outside every check built on `figure_text` — spacing, canvas, overflow.
    """
    from ogviz.layout.bounds import figure_text

    fig = plt.figure(figsize=(8.0, 4.0))
    left, right = fig.subfigures(1, 2)
    left.subplots().plot([0.0, 1.0], [0.0, 1.0])
    right.subplots().plot([0.0, 1.0], [1.0, 0.0])
    left.suptitle("the left half")
    nested = right.subfigures(2, 1)[0]
    nested.text(0.5, 0.5, "a nested note")
    fig.canvas.draw()
    assert "the left half" not in [text.get_text() for text in fig.texts], "premise"

    read = [text.get_text() for text, _owner in figure_text(fig)]
    assert "the left half" in read
    assert "a nested note" in read
    assert len(read) == len(set(read)), "each label once: the subfigures' axes are not re-walked"
    plt.close(fig)
