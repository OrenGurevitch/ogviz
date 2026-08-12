"""Things a consumer had to reach past the public API for, and now does not.

Each test here started as a project doing by hand what this package should have offered: spelling a
tag name as a bare string, taking room for a caption out of the plot when it wanted it out of the
page, inventing a rule for how wide a panel of N bars must be, guessing a character count to wrap
to, knocking a label out with a stroke where only a box was supported.

They are grouped by that origin rather than by which module they touch, because the gap is the
thing they have in common — an API that was adequate from inside the package and not from outside
it, which is a defect no test written from inside would have found.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import mark, marked, room_below, width_for_bars, wrap_to_panel


def test_a_consumer_can_name_a_tag_without_spelling_it() -> None:
    """Inside the package a typo is a typecheck error; outside it was a bare attribute.

    A consumer placing its own label against its own mark has a legitimate reason to say so, and was
    setting the attribute by hand with the prefix spelled out — the silent-failure footing the
    package fixed for itself and left every caller standing on.
    """
    _fig, ax = plt.subplots()
    label = ax.text(0.5, 0.5, "1.42x")
    assert not marked(label, "anchored")
    mark(label, "anchored")
    assert marked(label, "anchored")


def test_room_below_grows_the_page_and_not_the_plot() -> None:
    """Text under an axis has to come from somewhere, and `subplots_adjust` takes it from the plot.

    In a figure whose whole job is comparing bar heights, that silently shortens every bar.
    """
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    fig.subplots_adjust(top=0.97, bottom=0.10)
    fig.canvas.draw()
    before = ax.get_window_extent().height / fig.dpi

    height = room_below(fig, 0.33)
    fig.canvas.draw()
    after = ax.get_window_extent().height / fig.dpi

    assert after == pytest.approx(before, abs=0.01), "the panel keeps its height"
    assert height > 6.0, "and the page grew to pay for the margin"
    assert fig.subplotpars.bottom == pytest.approx(0.33)


def test_room_below_can_be_told_to_take_it_from_the_plot_instead() -> None:
    """Right when the panels have height to spare and the page does not."""
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    fig.subplots_adjust(top=0.97, bottom=0.10)
    fig.canvas.draw()
    before = ax.get_window_extent().height / fig.dpi

    room_below(fig, 0.33, keep_panels=False)
    fig.canvas.draw()
    assert fig.get_figheight() == pytest.approx(6.0)
    assert ax.get_window_extent().height / fig.dpi < before


def test_a_wider_panel_is_asked_for_by_the_bar_count() -> None:
    """Twelve inches held six bars and not eight. The gate said so; the caller invented the rule."""
    assert width_for_bars(6) == 12.0
    assert width_for_bars(8) == pytest.approx(15.2)
    assert width_for_bars(2) == 12.0, "a small panel keeps the house minimum"


def test_text_wraps_to_the_panel_a_caller_actually_has() -> None:
    """The width is the panel's, so the same words take more lines in half of it."""
    note = "a note long enough that it has to break somewhere sensible rather than run on"
    _fig, ax = plt.subplots(figsize=(6.0, 4.0))
    whole = wrap_to_panel(ax, note, 9.0)
    half = wrap_to_panel(ax, note, 9.0, fraction=0.5)
    assert len(half) > len(whole), (whole, half)
    assert " ".join(" ".join(half).split()) == note, "wrapping loses no words"


def test_a_value_label_can_knock_out_with_a_stroke_instead_of_a_box() -> None:
    """A box punches a hole in a reference band; a stroke clears only the digits.

    The gate has accepted a stroke since 2026-08-01 — this is the half that draws one.
    """
    import numpy as np

    from ogviz.panels.bars import value_labels

    _fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_ylim(0.0, 1.0)
    styles = (("box", True, False), ("stroke", False, True), ("none", False, False))
    for style, boxed, stroked in styles:
        value_labels(ax, np.array([0.0]), np.array([0.5]), halo=style)
        label = ax.texts[-1]
        assert (label.get_bbox_patch() is not None) is boxed, style
        assert bool(label.get_path_effects()) is stroked, style


def test_the_old_boolean_halo_still_means_a_box() -> None:
    import numpy as np

    from ogviz.panels.bars import value_labels

    _fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_ylim(0.0, 1.0)
    value_labels(ax, np.array([0.0]), np.array([0.5]), halo=True)
    assert ax.texts[-1].get_bbox_patch() is not None
    value_labels(ax, np.array([1.0]), np.array([0.5]), halo=False)
    assert ax.texts[-1].get_bbox_patch() is None
