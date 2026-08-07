"""Placing a label off the data, and noticing when it is on it."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

from ogviz.layout.collision import (
    annotate_clear,
    clear_position,
    hits_data,
    text_box,
    text_over_data,
)
from ogviz.tags import mark


def _filled_curve():
    """A filled area under a curve — the shape a bounding box describes worst."""
    x = np.linspace(0.0, 10.0, 300)
    y = np.exp(-x / 3.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.fill_between(x, 0.0, y, color="#D9B84C")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 1.4)
    fig.canvas.draw()
    return fig, ax


def test_a_label_under_the_curve_is_caught_and_one_above_it_is_not() -> None:
    """The case a bounding box cannot judge: both labels are inside the artist's bbox.

    The fill spans the whole panel by bounding box, so a bbox test calls both of these collisions.
    Only one of them actually touches ink.
    """
    fig, ax = _filled_curve()
    ax.text(1.0, 0.2, "under the fill")  # inside the shaded area
    ax.text(7.0, 1.2, "in the empty corner")  # inside the bbox, on nothing
    fig.canvas.draw()
    complaints = text_over_data(fig)
    assert len(complaints) == 1, complaints
    assert "under the fill" in complaints[0]


def test_clear_position_reports_no_move_needed_when_nothing_is_hit() -> None:
    fig, ax = _filled_curve()
    empty = ax.text(7.0, 1.2, "clear already")
    fig.canvas.draw()
    assert clear_position(ax, text_box(empty)) == (0.0, 0.0)


def test_annotate_clear_moves_the_text_and_leaves_the_arrow_anchored() -> None:
    """The anchor is a claim about the data; only the string is free to move."""
    fig, ax = _filled_curve()
    annotation = annotate_clear(ax, "the peak", xy=(0.2, 0.93), prefer=(0.6, 0.30))
    fig.canvas.draw()
    assert annotation.xy == (0.2, 0.93), "the arrow must keep pointing where it was told to"
    assert annotation.get_position() != (0.6, 0.30), "the text should have left the fill"
    assert not hits_data(ax, text_box(annotation)), "and should now touch nothing"


def test_an_annotation_is_measured_without_its_arrow() -> None:
    """The arrow touches the data on purpose; measuring it would make the search unsatisfiable."""
    fig, ax = _filled_curve()
    annotation = ax.annotate(
        "far away", xy=(0.2, 0.93), xytext=(8.0, 1.25), arrowprops={"arrowstyle": "-"}
    )
    fig.canvas.draw()
    whole = annotation.get_window_extent()
    assert hits_data(ax, whole), "the arrow does cross the fill, as intended"
    assert not hits_data(ax, text_box(annotation)), "but the string itself is clear"
    assert not text_over_data(fig)


def test_a_label_crossing_a_gridline_wants_a_knockout_not_a_move() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")
    fig.canvas.draw()
    gridline = ax.get_ygridlines()[2]
    height = float(gridline.get_ydata()[0])
    bare = ax.text(0.5, height, "crosses a rule", ha="center", va="center")
    fig.canvas.draw()
    complaints = text_over_data(fig)
    assert complaints and "knock it out" in complaints[0]

    bare.set_bbox({"facecolor": "white", "edgecolor": "none"})
    fig.canvas.draw()
    assert not text_over_data(fig)


def test_a_deliberately_anchored_label_is_left_where_the_panel_put_it() -> None:
    """Stars and value labels are placed against a mark on purpose and have their own checks."""
    fig, ax = _filled_curve()
    placed = ax.text(1.0, 0.2, "on the fill by design")
    fig.canvas.draw()
    assert text_over_data(fig)
    mark(placed, "anchored")
    assert not text_over_data(fig)


def test_the_search_covers_the_panel_not_a_fixed_pixel_reach() -> None:
    """A fixed 84px reach found nothing on an 800px panel and called the figure hopeless."""
    fig, ax = _filled_curve()
    wide = ax.text(0.2, 0.15, "a fairly long label that needs real room", fontsize=11)
    fig.canvas.draw()
    offset = clear_position(ax, text_box(wide))
    assert offset is not None, "there is empty space in this panel; the search must find it"
    box = text_box(wide)
    moved = Bbox.from_extents(
        box.x0 + offset[0], box.y0 + offset[1], box.x1 + offset[0], box.y1 + offset[1]
    )
    assert not hits_data(ax, moved)


def test_a_thin_marker_is_not_measured_as_a_square() -> None:
    """An error-bar cap is `"_"`: wide, and ink only as thick as its own stroke.

    Measuring it as markersize-by-markersize claimed 8 px of height it does not have and reported
    fourteen value labels on a real figure as sitting on a mark each was 0.5 px clear of.
    """
    import numpy as np

    from ogviz.layout.collision import data_points

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(np.arange(6.0), np.arange(1.0, 7.0), yerr=0.15, fmt="none", capsize=6)
    fig.canvas.draw()
    clouds = [cloud for cloud in data_points(ax) if cloud.width_px > 2.0]
    assert clouds, "the caps are a cloud of marks"
    for cloud in clouds:
        assert cloud.height_px < cloud.width_px / 4.0, (
            f"a cap is flat: {cloud.width_px:.1f} x {cloud.height_px:.1f} px"
        )


def test_a_round_marker_still_measures_square() -> None:
    """The other direction: the shape comes from the marker, so a dot must not be flattened."""
    from ogviz.layout.collision import data_points

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], linestyle="none", marker="o", markersize=12)
    fig.canvas.draw()
    (cloud,) = data_points(ax)
    assert abs(cloud.width_px - cloud.height_px) < 0.01, (cloud.width_px, cloud.height_px)
