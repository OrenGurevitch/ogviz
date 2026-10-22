import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import Series, bar_panel
from ogviz.layout import LARGE_AXIS_TEXT, TEMPLATE_AXIS_TEXT, style_axis_text
from ogviz.qc import category_axis_untitled

pytestmark = pytest.mark.usefixtures("pinned_font")


def _bars(ax, **kwargs):
    bar_panel(ax, [Series("s", np.array([1.0, 2.0]), "#2E7CE0")], ["a", "b"], **kwargs)


def test_the_category_title_is_bold_and_the_ticks_take_the_scale() -> None:
    _fig, ax = plt.subplots()
    _bars(ax)
    style_axis_text(ax, "Method", text=LARGE_AXIS_TEXT)
    assert ax.xaxis.label.get_text() == "Method"
    assert ax.xaxis.label.get_fontweight() == "bold"
    assert ax.xaxis.label.get_fontsize() == LARGE_AXIS_TEXT.category_title
    assert ax.get_xticklabels()[0].get_fontsize() == LARGE_AXIS_TEXT.category_tick
    assert ax.get_yticklabels()[0].get_fontsize() == LARGE_AXIS_TEXT.value_tick


def test_a_horizontal_panel_titles_its_y_axis_because_the_names_are_there() -> None:
    _fig, ax = plt.subplots()
    _bars(ax, orientation="horizontal")
    style_axis_text(ax, "Method")
    assert ax.yaxis.label.get_text() == "Method"
    assert ax.xaxis.label.get_text() == ""
    assert ax.get_yticklabels()[0].get_fontsize() == TEMPLATE_AXIS_TEXT.category_tick


def test_a_bar_panel_without_a_title_is_reported_and_with_one_is_not() -> None:
    fig, ax = plt.subplots()
    _bars(ax)
    fig.canvas.draw()
    assert len(category_axis_untitled(fig)) == 1, "the premise: an untitled panel is reported"
    style_axis_text(ax, "Method")
    assert category_axis_untitled(fig) == []


def test_a_grid_is_asked_for_one_title_per_column_on_its_bottom_row() -> None:
    fig, axes = plt.subplots(2, 1)
    for ax in axes:
        _bars(ax)
    fig.canvas.draw()
    found = category_axis_untitled(fig)
    assert len(found) == 1, "only the bottom panel, which is where the title goes"
    style_axis_text(axes[1], "Method")
    assert category_axis_untitled(fig) == []


def test_a_panel_nothing_stamped_is_never_asked_for_a_category_title() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()
    assert category_axis_untitled(fig) == []


def test_the_gate_refuses_to_write_a_bar_panel_whose_names_have_no_title() -> None:
    """Gating, not advice: `assert_clean` raises, and a one-word title is enough to pass."""
    from ogviz import assert_clean

    fig, ax = plt.subplots()
    _bars(ax)
    with pytest.raises(AssertionError, match="category names and no title"):
        assert_clean(fig)
    style_axis_text(ax, "Method")
    assert_clean(fig)
