"""One test per defect from the 2026-07-27 audit, so none of them can come back.

Each name says the failure, not the fix. See FIXME.md for how each was found.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import ogviz
from ogviz import (
    Series,
    bar_panel,
    bracket_stack,
    clipped_artists,
    group_violins,
    stars,
    use_house_style,
    value_ticks,
)
from ogviz.layout import round_ticks


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def _three_groups(seed: int = 9):
    rng = np.random.default_rng(seed)
    return [(float(i), rng.normal(i * 0.5, 0.9, 25), "#2E7CE0", "#333333") for i in range(3)]


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6])
def test_a_stack_of_brackets_never_loses_its_lines(count: int) -> None:
    """§1. Headroom was a fixed fraction, so from the THIRD bracket the stack ran past the axis.
    matplotlib clips Line2D and not Text, so the line vanished and its star stayed — a star
    floating over nothing, with nothing raised."""
    fig, ax = plt.subplots(figsize=(8, 8))
    group_violins(ax, _three_groups(), comparisons=[(0.0, 1.0, 0.01)] * count)
    top = ax.get_ylim()[1]
    assert not [ln for ln in ax.lines[-count:] if max(ln.get_ydata()) > top]
    assert clipped_artists(fig) == []


def test_a_log_value_axis_is_refused_rather_than_computed_wrongly() -> None:
    """§2/§3. The data-to-pixel ratio does not exist on a log scale: the clearance came back 0.0
    and the ticks came back linear (200/400/600/800 where a log axis wants 1/10/100/1000)."""
    _fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1, 1000)
    with pytest.raises(AssertionError, match="linear value axis"):
        value_ticks(ax, count=4)
    with pytest.raises(AssertionError, match="linear value axis"):
        group_violins(
            ax, [(0.0, np.abs(np.random.default_rng(0).normal(100, 30, 30)), "#111111", "#222222")]
        )


def test_an_empty_range_does_not_yield_identical_ticks() -> None:
    """§4. round_ticks(5, 5, 4) returned [5, 5, 5, 5]."""
    with pytest.raises(AssertionError, match="empty range"):
        round_ticks(5.0, 5.0, 4)


def test_two_groups_cannot_share_a_position() -> None:
    """§5. They were drawn on top of each other with no complaint."""
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="share a position"):
        group_violins(
            ax,
            [
                (0.0, np.arange(9.0), "#111111", "#222222"),
                (0.0, np.arange(9.0) + 1, "#333333", "#444444"),
            ],
        )


@pytest.mark.parametrize("bad", [-0.5, 1.5, -1e-9])
def test_an_impossible_p_is_refused_not_turned_into_three_stars(bad: float) -> None:
    """§6. stars() maps everything under 0.001 to ***, so a sign error upstream printed as the
    most significant result on the figure."""
    with pytest.raises(AssertionError, match=r"p must be in \[0, 1\]"):
        stars(bad)


def test_the_panel_row_helper_is_callable_from_the_top_level() -> None:
    """§9. `ogviz.panels` resolved to the SUBPACKAGE, so the documented call raised
    TypeError: 'module' object is not callable."""
    assert callable(ogviz.panel_row)
    assert "panels" not in ogviz.__all__
    fig, axes = ogviz.panel_row(2, caption="a caption")
    assert len(axes) == 2
    assert len(fig.axes) == 3


def test_a_colour_per_bar_must_match_the_number_of_bars() -> None:
    """§10. Three bars and two colours rendered without complaint."""
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="colours for"):
        bar_panel(ax, [Series("z", [1.0, 2.0, 3.0], ["#111111", "#222222"])], list("abc"))


def test_zero_categories_says_so_instead_of_an_index_error() -> None:
    """§11. It raised a raw IndexError from inside the limit arithmetic."""
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="at least one category"):
        bar_panel(ax, [Series("z", [], "#111111")], [])


def test_content_clipped_out_of_the_axes_is_caught() -> None:
    """§12. The overlap guard measures where text IS and knows nothing about clipping, so the §1
    figure — a bracket clipped out of existence — passed it with zero problems."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_ylim(0, 10)
    bracket_stack(ax, [(0.0, 1.0, 0.01)], start=12.0, span=10.0)
    assert any("past the top" in hit for hit in clipped_artists(fig))


def test_a_value_label_knocks_out_what_runs_behind_it() -> None:
    """§13. The stroke halo followed the glyph CONTOURS, so a dashed reference line still showed
    through the gaps between and inside the digits."""
    _fig, ax = plt.subplots()
    bar_panel(
        ax,
        [Series("z", [0.74], "#2E7CE0")],
        ["only"],
        reference=(0.74, "ceiling"),
    )
    label = next(t for t in ax.texts if t.get_text().startswith("0.7"))
    box = label.get_bbox_patch()
    assert box is not None, "the label needs an opaque box, not a contour-following stroke"
    assert box.get_edgecolor()[3] == 0 or box.get_edgecolor() == "none"
