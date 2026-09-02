"""A level and a range drawn as context, and what each refuses.

Neither had a test file. Writing one found two inputs that drew nothing and said nothing — a
reference the figure is read against, silently absent.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import Series, bar_panel
from ogviz.panels.reference import (
    Z_BAND_FILL,
    reference_band,
    reference_line,
    slide_label_clear,
)
from ogviz.tags import marked

pytestmark = pytest.mark.usefixtures("pinned_font")


def _bars():
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bar_panel(ax, [Series("arm", np.array([0.62, 0.71, 0.78]), "#7C9A6E")], ["A", "B", "C"])
    return fig, ax


def test_a_level_is_drawn_over_the_bars_it_is_read_against() -> None:
    """Behind them it survives only in the gaps, which loses the comparison it exists for."""
    from ogviz.panels.bars import Z_BAR

    fig, ax = _bars()
    reference_line(ax, 0.80, "target")
    (drawn,) = [line for line in ax.lines if marked(line, "reference")]
    assert drawn.get_zorder() > Z_BAR
    plt.close(fig)


def test_a_bands_fill_stays_under_the_frame_and_its_edges_do_not() -> None:
    """At the reference z-order the fill washed over the left spine — a two-tone frame."""
    fig, ax = _bars()
    reference_band(ax, 0.85, 0.95, "human agreement")
    fig.canvas.draw()
    (fill,) = [patch for patch in ax.patches if marked(patch, "reference")]
    edges = [line for line in ax.lines if marked(line, "reference")]
    assert fill.get_zorder() == Z_BAND_FILL < ax.spines["left"].get_zorder()
    assert len(edges) == 2, "two dashed edges carry the band's precision"
    assert all(edge.get_zorder() > fill.get_zorder() for edge in edges)
    plt.close(fig)


def test_a_band_is_also_a_backdrop_so_a_label_may_sit_on_it() -> None:
    """Untagged, every value label reaching into the band is reported as sitting on the marks."""
    fig, ax = _bars()
    reference_band(ax, 0.85, 0.95, "human agreement")
    (fill,) = [patch for patch in ax.patches if marked(patch, "reference")]
    assert marked(fill, "backdrop")
    plt.close(fig)


def test_the_bands_label_sits_above_its_upper_edge_and_never_inside() -> None:
    """A bar tall enough to reach there is inside the band, not above it.

    That is what stops the design degrading as the bars improve — the failure it replaced.
    """
    fig, ax = _bars()
    reference_band(ax, 0.85, 0.95, "human agreement")
    fig.canvas.draw()
    (label,) = [text for text in ax.texts if "agreement" in text.get_text()]
    bottom = ax.transData.inverted().transform((0.0, label.get_window_extent().y0))[1]
    assert bottom >= 0.95, f"the label starts at {bottom:.3f}, inside the band"
    plt.close(fig)


def test_a_span_limits_the_level_to_the_arms_it_speaks_about() -> None:
    """Across the whole axis it claims a comparison the figure does not make."""
    fig, ax = _bars()
    reference_line(ax, 0.80, "target", span=(-0.4, 1.4))
    (drawn,) = [line for line in ax.lines if marked(line, "reference")]
    xs = np.asarray(drawn.get_xdata(), dtype=float)
    assert float(xs.min()) == pytest.approx(-0.4) and float(xs.max()) == pytest.approx(1.4)
    plt.close(fig)


def test_a_non_finite_level_is_refused() -> None:
    """It drew nothing, and the figure then made a comparison against a line that is not there."""
    _fig, ax = _bars()
    with pytest.raises(AssertionError, match="draws nothing"):
        reference_line(ax, float("nan"), "target")


def test_a_band_whose_bounds_run_backwards_is_refused() -> None:
    """`Estimate` refuses the same mistake on an interval, for the same reason."""
    _fig, ax = _bars()
    with pytest.raises(AssertionError, match="runs backwards"):
        reference_band(ax, 0.95, 0.85, "human agreement")


def test_a_band_with_a_non_finite_bound_is_refused() -> None:
    _fig, ax = _bars()
    with pytest.raises(AssertionError, match="draws nothing"):
        reference_band(ax, 0.85, float("inf"), "human agreement")


def test_a_zero_width_band_is_allowed() -> None:
    """A tolerance can legitimately collapse to a level; only backwards is a mistake."""
    fig, ax = _bars()
    reference_band(ax, 0.90, 0.90, "exact")
    fig.canvas.draw()
    plt.close(fig)


def test_the_panel_passes_its_own_gate_with_both() -> None:
    from ogviz.qc import audit

    fig, ax = _bars()
    reference_band(ax, 0.85, 0.95, "human agreement")
    fig.canvas.draw()
    assert audit(fig) == []
    plt.close(fig)


def test_a_backdrop_is_not_something_a_threshold_label_must_avoid() -> None:
    """A backdrop is what a label may sit ON, and it was counted as something to slide away from.

    Tested against `slide_label_clear` directly, with a tint covering the panel and nothing else
    on it, because on a whole `bar_panel` the answer also depends on where the bars and the value
    labels fall — which is a different question and one the font changes. Here the tint is the
    only artist, so the two runs differ in exactly the rule under test: excused, the label takes
    the first slot; counted, no slot is clear and it goes to the 1.012 fallback outside the axes.
    """
    import ogviz.panels.reference as reference_module
    from ogviz.tags import mark

    def label_fraction(*, count_backdrops: bool) -> float:
        real = reference_module.is_backdrop
        if count_backdrops:
            reference_module.is_backdrop = lambda _artist: False
        try:
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            tint = ax.axhspan(0.0, 1.0, color="#EFEFEF", zorder=0)
            mark(tint, "backdrop")
            label = reference_line(ax, 0.5, "target")
            fig.canvas.draw()
            slide_label_clear(ax, label)
            fraction = float(label.get_position()[0])
        finally:
            reference_module.is_backdrop = real
            plt.close("all")
        return fraction

    assert label_fraction(count_backdrops=True) > 1.0, "premise: counting the tint exiles the label"
    assert label_fraction(count_backdrops=False) < 1.0
