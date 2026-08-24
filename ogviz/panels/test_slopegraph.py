from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.panels.slopegraph import Strand, null_distance, slopegraph
from ogviz.theme import SERIES

STAGES = ("Baseline", "3 months", "6 months", "12 months")


def _strands(last: tuple[float, float, float]) -> list[Strand]:
    return [
        Strand("Signal A", [0.02, 0.05, 0.09, last[0]], SERIES[0]),
        Strand("Signal B", [0.03, 0.04, 0.05, last[1]], SERIES[1]),
        Strand("Signal C", [0.01, -0.01, -0.04, last[2]], SERIES[2]),
    ]


def test_the_panel_says_when_its_end_labels_cannot_be_placed_separately() -> None:
    """Placing them one at a time is only correct when they do not compete for the same space.

    Reported before the figure is written rather than discovered in the rendered PNG, because the
    set-wise placement this would need does not exist yet and a caller whose series converge should
    reach for the legend knowingly.
    """
    _fig, ax = plt.subplots(figsize=(8.0, 5.0))
    assert not slopegraph(ax, _strands((0.12, 0.05, -0.07)), STAGES), "these separate"

    _fig2, ax2 = plt.subplots(figsize=(8.0, 5.0))
    crowding = slopegraph(ax2, _strands((0.100, 0.1005, 0.101)), STAGES)
    assert len(crowding) == 2, crowding
    assert all("apart" in complaint for complaint in crowding)


def test_the_stages_are_evenly_spaced_whatever_they_represent() -> None:
    """Uneven spacing would make a slope mean something the data did not say."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    slopegraph(ax, _strands((0.12, 0.05, -0.07)), STAGES)
    fig.canvas.draw()
    ticks = np.asarray(ax.get_xticks(), dtype=float)
    assert np.allclose(np.diff(ticks), 1.0)


def test_a_strand_must_have_one_value_per_stage() -> None:
    _fig, ax = plt.subplots()
    short = [Strand("Signal A", [0.1, 0.2], SERIES[0])]
    with pytest.raises(AssertionError, match="2 values across 4 stages"):
        slopegraph(ax, short, STAGES)


def test_a_band_is_bounds_per_stage_and_says_so_when_it_is_not() -> None:
    _fig, ax = plt.subplots(figsize=(8.0, 5.0))
    banded = [
        Strand(
            "Signal A",
            [0.02, 0.05, 0.09, 0.12],
            SERIES[0],
            [(0.0, 0.04), (0.03, 0.08), (0.06, 0.12), (0.08, 0.16)],
        )
    ]
    slopegraph(ax, banded, STAGES)
    assert ax.collections, "the band is drawn"

    with pytest.raises(AssertionError, match="spreads for 4 values"):
        Strand("Signal A", [0.02, 0.05, 0.09, 0.12], SERIES[0], [(0.0, 0.04)])


def test_the_shipped_slopegraph_passes_its_own_gate() -> None:
    from ogviz.qc import audit

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    slopegraph(ax, _strands((0.12, 0.05, -0.07)), STAGES)
    ax.set_ylabel("Effect (d)")
    fig.tight_layout()
    fig.canvas.draw()
    assert not audit(fig)


def test_each_metric_is_measured_against_its_own_null() -> None:
    """An OOS correlation is at chance around 0 and an AUC around 0.5."""
    assert null_distance([0.31, 0.72], [0.0, 0.5]) == pytest.approx([0.31, 0.22])

    with pytest.raises(AssertionError, match="every metric needs its own null"):
        null_distance([0.31, 0.72], [0.0])
