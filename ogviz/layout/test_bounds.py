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
