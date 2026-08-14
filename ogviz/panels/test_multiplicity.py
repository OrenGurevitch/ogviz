from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.panels.multiplicity import (
    benjamini_hochberg_rank,
    bonferroni_threshold,
    multiplicity_ladder,
)


def test_bh_declares_the_largest_clearing_rank_not_the_first_failure() -> None:
    """The step that makes BH more powerful than a fixed cutoff, and the one a table hides.

    Rank 2 sits ABOVE the ramp and rank 3 below it, so all three are declared. A rule that stopped
    at the first failure would say one. Cross-checked against `statsmodels.multipletests` on this
    family and six others when it was written.
    """
    p = np.array([0.001, 0.040, 0.045])
    assert benjamini_hochberg_rank(p) == 3


def test_bh_declares_nothing_when_nothing_clears() -> None:
    assert benjamini_hochberg_rank(np.array([0.2, 0.3, 0.4])) == 0


def test_bonferroni_is_alpha_over_the_family_size() -> None:
    assert bonferroni_threshold(10) == pytest.approx(0.005)
    assert bonferroni_threshold(4, alpha=0.10) == pytest.approx(0.025)


def test_the_ladder_colours_by_rank_and_not_by_each_point() -> None:
    """Whether a point is declared is a property of its RANK, since rank 3 can carry rank 2."""
    from matplotlib.colors import to_hex

    from ogviz.panels.multiplicity import DECLARED_COLOR

    # Rank 2 sits ABOVE the ramp and is declared anyway, because rank 3 clears it. Colouring each
    # point by its own value against the ramp would draw the middle point as rejected.
    p = [0.001, 0.040, 0.045]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    declared = multiplicity_ladder(ax, p)
    fig.canvas.draw()
    colors = [to_hex(c) for c in ax.collections[-1].get_facecolors()]
    ramp = [0.05 * rank / 3 for rank in (1, 2, 3)]
    assert p[1] > ramp[1], "the premise: the middle point is above the line"
    assert declared == 3
    assert colors == [to_hex(DECLARED_COLOR)] * 3

    # And a family where the same values are joined by a fourth: the ramp changes, and only the
    # first survives. Same points, different family, different answer — which is the whole subject.
    _fig2, ax2 = plt.subplots(figsize=(7.0, 4.0))
    assert multiplicity_ladder(ax2, [*p, 0.9]) == 1


def test_the_ladder_refuses_a_value_that_is_not_a_p_value() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match=r"p-values must be in \[0, 1\]"):
        multiplicity_ladder(ax, [0.1, 1.4])


def test_a_labelled_family_does_not_collide_with_itself() -> None:
    """Twelve ordinary names at 45 degrees produced eight complaints on the first render."""
    from ogviz.layout.axis import settle_axis_labels
    from ogviz.qc import audit

    names = [f"trial {index}" for index in range(12)]
    p = [0.0004, 0.006, 0.011, 0.019, 0.028, 0.033, 0.041, 0.049, 0.08, 0.12, 0.19, 0.24]
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    multiplicity_ladder(ax, p, labels=names)
    fig.tight_layout()
    # What `save` does, and this figure is never saved: the ladder reserves room above its marks
    # for the rotated names, so the axis labels start centred on that headroom.
    settle_axis_labels(fig)
    fig.canvas.draw()
    assert not audit(fig)


def test_a_log_ladder_shows_the_two_rules_apart() -> None:
    """The reason `log` exists: on a linear axis both thresholds sit under alpha and merge.

    The premise is asserted first — that the linear panel really does put the Bonferroni line and
    the BH ramp's first rung within a pixel of each other — because without it this test would pass
    against a family whose thresholds were never close, and prove nothing about the fix.
    """
    p = [
        0.0004,
        0.006,
        0.011,
        0.019,
        0.028,
        0.033,
        0.041,
        0.049,
        0.08,
        0.12,
        0.19,
        0.24,
        0.41,
        0.63,
        0.88,
    ]
    figsize = (10.5, 6.6)

    def gap_px(**kwargs) -> float:
        fig, ax = plt.subplots(figsize=figsize)
        multiplicity_ladder(ax, p, **kwargs)
        fig.canvas.draw()
        first_rung = 0.05 * 1 / len(p)
        bonferroni = 0.05 / len(p)
        both = ax.transData.transform([(1, first_rung), (1, bonferroni)])
        apart = abs(both[0][1] - both[1][1])
        plt.close(fig)
        return apart

    # They are the SAME NUMBER for rank 1 — 0.05/15 either way — so the pixel gap is zero on both
    # scales. What differs is the room around them: the distance from that shared point to alpha.
    def alpha_room_px(**kwargs) -> float:
        fig, ax = plt.subplots(figsize=figsize)
        multiplicity_ladder(ax, p, **kwargs)
        fig.canvas.draw()
        edges = ax.transData.transform([(1, 0.05 / len(p)), (1, 0.05)])
        room = abs(edges[0][1] - edges[1][1])
        plt.close(fig)
        return room

    assert gap_px() == pytest.approx(gap_px(log=True), abs=0.5)
    linear, logged = alpha_room_px(), alpha_room_px(log=True)
    assert linear < 30.0, f"the premise: linear crushes threshold to alpha into {linear:.0f} px"
    assert logged > 3 * linear, f"log spreads it to {logged:.0f} px, from {linear:.0f}"


def test_a_log_ladder_refuses_a_p_of_zero() -> None:
    """Clamping would draw it at the axis floor, where it reads as the strongest result there is."""
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="nowhere to put them"):
        multiplicity_ladder(ax, [0.0, 0.02, 0.4], log=True)


def test_a_linear_ladder_still_accepts_a_p_of_zero() -> None:
    """The refusal belongs to `log`, not to the panel — a linear axis has a place for zero."""
    _fig, ax = plt.subplots()
    assert multiplicity_ladder(ax, [0.0, 0.02, 0.4]) >= 1
    assert ax.get_yscale() == "linear"
