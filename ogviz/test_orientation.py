"""Horizontal must be the same figure with the axes swapped, not a lookalike.

Each test compares the vertical and horizontal renders of identical data and asserts the drawn
coordinates are transposes of each other. That is the only claim worth making: a horizontal mode
that merely runs without error can still place a mean line at the wrong value.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import Series, bar_panel, group_violins, house_style
from ogviz.layout.ticks import MINUS
from ogviz.marks import iqr_box, mean_line, points, violin
from ogviz.theme import CANVAS, SERIES, use_house_style

VALUES = np.random.default_rng(4).normal(2.0, 1.0, 40)
FILL, EDGE = SERIES[0], "#1A4F94"


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def _both(draw):
    """Run `draw(ax, orientation)` twice and hand back the two axes."""
    _fv, vertical = plt.subplots()
    draw(vertical, "vertical")
    _fh, horizontal = plt.subplots()
    draw(horizontal, "horizontal")
    return vertical, horizontal


def test_violin_body_is_the_transpose():
    vertical, horizontal = _both(
        lambda ax, o: violin(ax, VALUES, 1.0, FILL, orientation=o)  # type: ignore[arg-type]
    )
    up = vertical.collections[0].get_paths()[0].vertices
    across = horizontal.collections[0].get_paths()[0].vertices
    np.testing.assert_allclose(np.sort(up[:, 0]), np.sort(across[:, 1]))
    np.testing.assert_allclose(np.sort(up[:, 1]), np.sort(across[:, 0]))


def test_points_keep_their_datum_on_the_value_axis():
    """The failure a smoke test misses: dots drawn at the right spread but the wrong value."""
    vertical, horizontal = _both(
        lambda ax, o: points(  # type: ignore[arg-type]
            ax, VALUES, 1.0, FILL, EDGE, np.random.default_rng(0), orientation=o
        )
    )
    up = np.asarray(vertical.collections[0].get_offsets())
    across = np.asarray(horizontal.collections[0].get_offsets())
    np.testing.assert_array_equal(up[:, 1], VALUES)
    np.testing.assert_array_equal(across[:, 0], VALUES)
    # The SPREAD is deliberately not a transpose: the central lane is sized in points of ink, and
    # two differently-shaped axes convert those to different distances in data units. What must
    # hold is that both stay inside the violin's own width.
    assert np.abs(up[:, 0] - 1.0).max() <= 0.62 / 2
    assert np.abs(across[:, 1] - 1.0).max() <= 0.62 / 2


def test_iqr_box_and_median_dot_transpose():
    vertical, horizontal = _both(
        lambda ax, o: iqr_box(ax, VALUES, 1.0, orientation=o)  # type: ignore[arg-type]
    )
    for up, across in zip(vertical.lines, horizontal.lines, strict=True):
        np.testing.assert_allclose(up.get_xdata(), across.get_ydata())
        np.testing.assert_allclose(up.get_ydata(), across.get_xdata())


def test_mean_line_sits_at_the_mean_either_way():
    vertical, horizontal = _both(
        lambda ax, o: mean_line(ax, VALUES, 1.0, orientation=o)  # type: ignore[arg-type]
    )
    assert vertical.lines[0].get_ydata()[0] == pytest.approx(float(VALUES.mean()))
    assert horizontal.lines[0].get_xdata()[0] == pytest.approx(float(VALUES.mean()))


def test_group_violins_sets_the_limit_on_the_value_axis():
    """Vertical grows the y range; horizontal must grow x, or the panel clips its own data."""
    groups = [(0.0, VALUES, FILL, EDGE), (1.0, VALUES + 3.0, SERIES[1], "#8A5E00")]
    vertical, horizontal = _both(
        lambda ax, o: group_violins(ax, groups, orientation=o)  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(vertical.get_ylim(), horizontal.get_xlim())


def test_a_horizontal_bar_panel_labels_its_categories_down_the_side():
    values = np.array([1.0, -2.0, 3.0])
    names = ["autonomic burden", "sleep disturbance", "cognitive complaint"]
    _fig, ax = plt.subplots(figsize=(7, 4))
    bar_panel(ax, [Series("z", values, SERIES[0])], names, orientation="horizontal")
    assert [t.get_text() for t in ax.get_yticklabels()] == names
    assert [patch.get_width() for patch in ax.patches] == pytest.approx(list(values))


def test_a_horizontal_negative_bar_labels_outward_too():
    """The sign-aware rule has to follow the axis, or a negative bar buries its own number."""
    _fig, ax = plt.subplots()
    bar_panel(
        ax, [Series("z", np.array([-2.0, 3.0]), SERIES[0])], ["a", "b"], orientation="horizontal"
    )
    placed = {t.get_text(): t.get_position()[0] for t in ax.texts}
    assert placed[f"{MINUS}2.00"] < -2.0
    assert placed["+3.00"] > 3.0


def test_a_horizontal_reference_line_is_vertical_on_screen():
    _fig, ax = plt.subplots()
    bar_panel(
        ax,
        [Series("z", np.array([1.0, 2.0]), SERIES[0])],
        ["a", "b"],
        orientation="horizontal",
        reference=(2.5, "ceiling"),
    )
    assert any(round(line.get_xdata()[0], 6) == 2.5 for line in ax.lines)


def test_an_unknown_orientation_is_refused_rather_than_guessed():
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="unknown orientation"):
        violin(ax, VALUES, 0.0, FILL, orientation="sideways")  # type: ignore[arg-type]


def test_house_style_can_be_scoped_so_two_styles_coexist():
    """A process that renders one house figure and one in a project's own style needs this."""
    with mpl.rc_context({"figure.facecolor": "papayawhip"}):
        with house_style():
            assert mpl.rcParams["figure.facecolor"] == CANVAS
        assert mpl.rcParams["figure.facecolor"] == "papayawhip"


def test_a_horizontal_bracket_spans_the_category_axis():
    """The defect a render caught: the bracket kept spanning value, so it compared nothing."""
    groups = [(0.0, VALUES, FILL, EDGE), (1.0, VALUES + 3.0, SERIES[1], "#8A5E00")]
    _fig, ax = plt.subplots()
    group_violins(ax, groups, comparisons=[(0.0, 1.0, 0.004)], orientation="horizontal")
    bracket = ax.lines[-1]
    assert sorted(set(np.round(bracket.get_ydata(), 6))) == [0.0, 1.0], (
        "the bracket must run between the two GROUPS, which lie on y when horizontal"
    )
    assert len(set(np.round(bracket.get_xdata(), 6))) == 2, "and step out along the value axis"


def test_printed_means_default_off_when_horizontal():
    """Their row would land exactly where the category tick labels are."""
    groups = [(0.0, VALUES, FILL, EDGE)]
    _fig, sideways = plt.subplots()
    group_violins(sideways, groups, orientation="horizontal")
    _fig2, upright = plt.subplots()
    group_violins(upright, groups)
    assert not sideways.texts
    assert upright.texts


def test_a_horizontal_star_clears_its_bracket_by_width_not_height():
    """Using the vertical ink on a horizontal bracket offsets the star by its own height,
    which puts it back on the line it is supposed to sit beside."""
    from ogviz.significance import ink_extents_points

    wide, tall = ink_extents_points("* * *", 18, axis=0), ink_extents_points("* * *", 18, axis=1)
    assert wide[1] - wide[0] > tall[1] - tall[0], "the label is wider than it is tall"

    groups = [(0.0, VALUES, FILL, EDGE), (1.0, VALUES + 3.0, SERIES[1], "#8A5E00")]
    _fig, ax = plt.subplots()
    group_violins(ax, groups, comparisons=[(0.0, 1.0, 0.004)], orientation="horizontal")
    bracket_x = max(ax.lines[-1].get_xdata())
    star_x = ax.texts[-1].get_position()[0]
    assert star_x > bracket_x, "the star must sit beyond the bracket, on the outward side"
