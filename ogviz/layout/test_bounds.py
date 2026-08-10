from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ogviz.layout.bounds import text_off_canvas, text_wider_than_its_panel


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
