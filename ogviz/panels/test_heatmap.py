from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

from ogviz.panels.heatmap import MISSING_MARK, effect_heatmap
from ogviz.theme import page_color


def _fills(ax) -> list[tuple[float, float, float]]:
    return [to_rgb(patch.get_facecolor()) for patch in ax.patches]


def _distance(first, second) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def test_the_scale_is_symmetric_about_the_neutral_value() -> None:
    """Equal effects in opposite directions must get equal colour.

    An automatic scale over the data lands the neutral value off-centre, and the reader then sees a
    coloured cell where the number says nothing is happening.
    """
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    effect_heatmap(ax, np.array([[-0.8, 0.0, 0.8]]), row_labels=["r"], column_labels=list("abc"))
    fig.canvas.draw()
    from ogviz.panels.heatmap import diverging_map

    low, middle, high = _fills(ax)
    colormap = diverging_map()
    # Symmetry is a property of the SCALE, not of the two hues: the equal-and-opposite values land
    # on the two ends of the map. Comparing their RGB distance from the middle instead would test
    # whether blue and amber happen to be equally far from the page colour, which they are not and
    # need not be.
    assert _distance(low, to_rgb(colormap(0.0))) < 0.01, (low, colormap(0.0))
    assert _distance(high, to_rgb(colormap(1.0))) < 0.01, (high, colormap(1.0))
    # The midpoint stop IS the page colour; the colormap quantises to a 256-entry table, so this is
    # equality within that and not exact.
    assert _distance(middle, to_rgb(page_color())) < 0.02, (middle, page_color())


def test_an_off_centre_range_still_centres_on_the_neutral_value() -> None:
    """The case the symmetry exists for: every value on one side of neutral."""
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    effect_heatmap(ax, np.array([[0.1, 0.5, 0.9]]), row_labels=["r"], column_labels=list("abc"))
    fig.canvas.draw()
    fills = _fills(ax)
    page = to_rgb(page_color())
    # All three sit on the warm side, and the smallest is nearest the page colour.
    distances = [_distance(fill, page) for fill in fills]
    assert distances[0] < distances[1] < distances[2]


def test_a_number_takes_its_colour_from_the_cell_behind_it() -> None:
    """One ink colour is unreadable at the dark end of the map, white unreadable in the middle."""
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    effect_heatmap(ax, np.array([[-1.2, 0.05]]), row_labels=["r"], column_labels=["dark", "pale"])
    fig.canvas.draw()
    printed = {text.get_text(): to_rgb(text.get_color()) for text in ax.texts}
    dark_cell = next(color for label, color in printed.items() if label.endswith("1.20"))
    pale_cell = next(color for label, color in printed.items() if label.endswith("0.05"))
    assert _distance(dark_cell, to_rgb(page_color())) < 0.02, "light ink on the dark fill"
    assert _distance(pale_cell, to_rgb(page_color())) > 0.5, "and dark ink on the pale one"


def test_a_missing_cell_is_drawn_as_missing_and_not_as_neutral() -> None:
    """`nan` shaded like zero reads as a measured null, which is the one thing it is not."""
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    effect_heatmap(ax, np.array([[np.nan, 0.0]]), row_labels=["r"], column_labels=["nan", "zero"])
    fig.canvas.draw()
    missing, neutral = _fills(ax)
    assert _distance(missing, neutral) > 0.02, (missing, neutral)
    assert MISSING_MARK in [text.get_text() for text in ax.texts]


def test_a_missing_cell_carries_no_star_even_when_given_a_p() -> None:
    p = np.array([[0.0001, 0.0001]])
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    effect_heatmap(
        ax,
        np.array([[np.nan, 0.9]]),
        row_labels=["r"],
        column_labels=["nan", "real"],
        p_values=p,
    )
    fig.canvas.draw()
    stars = [t for t in ax.texts if getattr(t, "ogviz_column_star", False)]
    assert len(stars) == 1, [t.get_text() for t in stars]


def test_the_panel_passes_its_own_gate() -> None:
    """Every number sits on its own cell, which is the design: 46 complaints before the cells were
    tagged as backdrops."""
    from ogviz.qc import audit

    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.5, (6, 4))
    values[2, 3] = np.nan
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    effect_heatmap(
        ax,
        values,
        row_labels=[f"row {index}" for index in range(6)],
        column_labels=[f"c{index}" for index in range(4)],
        p_values=rng.uniform(0.0, 0.3, (6, 4)),
        row_dividers=[4],
    )
    fig.tight_layout()
    fig.canvas.draw()
    assert not audit(fig)


def test_a_matrix_and_its_labels_must_agree() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="against 1 rows and 3 columns"):
        effect_heatmap(ax, np.zeros((1, 2)), row_labels=["r"], column_labels=list("abc"))
