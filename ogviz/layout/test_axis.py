"""`layout/axis.py`: what the value axis shows, and where its label belongs.

Named for the module rather than for the defect, so the "modules with no test file" predictor
counts it — it was `test_axis_labels.py`, which covers the same ground and matched nothing.

Whether an axis label is centred on the marks it names, or on the box that contains them.

The defect this module exists for was found by a reader looking at a figure, not by a check — and
the first attempt to verify it asked whether the label was centred on the SPINE, which for a padded
panel spans the whole axes box and so passed tautologically. Every test here measures against the
reach of the MARKS, which is the reference `settle_axis_labels` records two wrong answers for.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.layout.axis import settle_axis_labels
from ogviz.qc.arrangement import value_label_off_its_marks

pytestmark = pytest.mark.usefixtures("pinned_font")


def _padded(headroom: float = 3.0):
    """A panel with room reserved above its data, which is what a bracket stack needs."""
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_ylim(0.0, headroom)
    ax.set_yticks([0.0, 0.5, 1.0])  # `ticks_over_data` leaves none in the headroom
    ax.set_ylabel("concentration (ppm)")
    fig.canvas.draw()
    return fig, ax


def _offset(ax) -> float:
    """Label centre minus the centre of the marks, in display pixels."""
    from ogviz.layout.axis import drawn_value_extent

    extent = drawn_value_extent(ax)
    assert extent is not None, "the panel draws nothing to centre on"
    ends = [ax.transData.transform((0.0, value))[1] for value in extent]
    label = ax.yaxis.label.get_window_extent()
    return (label.y0 + label.y1) / 2 - (min(ends) + max(ends)) / 2


def test_a_label_on_a_padded_panel_starts_off_its_ticks() -> None:
    """The premise. Without it every test below could pass on a panel that was never wrong."""
    fig, ax = _padded()
    assert _offset(ax) > 20.0, f"only {_offset(ax):.1f} px off — this panel is not padded enough"
    plt.close(fig)


def test_centring_on_the_spine_is_the_check_that_passes_tautologically() -> None:
    """Why the first verification of this was wrong, kept so it cannot be made again.

    The spine spans the whole axes box, so its middle IS the box's middle — which is exactly where
    matplotlib already put the label. Asking that question can only ever answer yes.
    """
    fig, ax = _padded()
    spine = ax.spines["left"].get_window_extent()
    label = ax.yaxis.label.get_window_extent()
    assert (label.y0 + label.y1) / 2 == pytest.approx((spine.y0 + spine.y1) / 2, abs=1.0)
    assert _offset(ax) > 20.0, "and yet it is nowhere near the marks"
    plt.close(fig)


def test_settling_puts_it_on_the_ticks() -> None:
    fig, ax = _padded()
    moved = settle_axis_labels(fig)
    assert moved and "centred" in moved[0], moved
    fig.canvas.draw()
    assert abs(_offset(ax)) < 2.0, _offset(ax)
    plt.close(fig)


def test_the_horizontal_placement_is_still_matplotlib_s_to_decide() -> None:
    """Only the along-axis coordinate is set, so the label keeps following its own tick widths.

    `set_label_coords` would freeze both and stop that; this writes the label's position directly,
    which matplotlib recomputes the perpendicular half of on every draw.
    """
    fig, ax = _padded()
    settle_axis_labels(fig)
    fig.canvas.draw()
    before = ax.yaxis.label.get_window_extent().x0
    ax.set_yticklabels(["a much wider tick label"] * len(ax.get_yticks()))
    fig.canvas.draw()
    assert ax.yaxis.label.get_window_extent().x0 < before, (
        "the label did not move left for wider ticks — automatic placement was frozen"
    )
    plt.close(fig)


def test_it_survives_a_later_draw() -> None:
    """matplotlib passes the along-axis coordinate through untouched; this proves it does."""
    fig, ax = _padded()
    settle_axis_labels(fig)
    for _ in range(3):
        fig.canvas.draw()
    assert abs(_offset(ax)) < 2.0, _offset(ax)
    plt.close(fig)


def test_an_x_label_is_settled_the_same_way() -> None:
    """The mirror. `XAxis` preserves x and recomputes y, so the same seam works both ways."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_xlim(0.0, 4.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("time (s)")
    fig.canvas.draw()

    def off() -> float:
        boxes = [t.get_window_extent() for t in ax.get_xticklabels() if t.get_text().strip()]
        block = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
        label = ax.xaxis.label.get_window_extent()
        return (label.x0 + label.x1) / 2 - block

    assert abs(off()) > 20.0, "the premise: this x-label starts off its ticks"
    settle_axis_labels(fig)
    fig.canvas.draw()
    assert abs(off()) < 2.0, off()
    plt.close(fig)


def test_a_panel_that_fills_its_box_is_left_alone() -> None:
    """Nothing to fix, so nothing is moved — and no churn in the saved file."""
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_ylabel("y")
    fig.canvas.draw()
    assert settle_axis_labels(fig) == []
    plt.close(fig)


def test_the_gate_reports_it_and_goes_quiet_once_repaired() -> None:
    fig, _ax = _padded()
    assert any("marks it names" in one for one in value_label_off_its_marks(fig))
    settle_axis_labels(fig)
    fig.canvas.draw()
    assert value_label_off_its_marks(fig) == []
    plt.close(fig)


def test_repair_closes_it() -> None:
    """So `--fix` needs no decision from a caller."""
    from ogviz.qc.repair import repair

    fig, _ax = _padded()
    assert any("centred" in one for one in repair(fig))
    fig.canvas.draw()
    assert value_label_off_its_marks(fig) == []
    plt.close(fig)


def test_a_real_violin_panel_is_settled_by_save() -> None:
    """End to end, on the panel type whose bracket headroom causes this in the first place."""
    import tempfile
    from pathlib import Path

    from ogviz import group_violins, save, value_ticks

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.0, 7.0))
    group_violins(
        ax,
        [
            (0.0, rng.normal(10.0, 1.0, 40), "#E8A838", "#B97C10"),
            (1.0, rng.normal(12.0, 1.0, 40), "#7C9A6E", "#4A6136"),
        ],
        comparisons=[(0.0, 1.0, 0.001)],
    )
    value_ticks(ax, count=4)
    ax.set_ylabel("concentration (ppm)")
    fig.canvas.draw()
    assert abs(_offset(ax)) > 8.0, "the premise: a bracket stack pushes the label off its ticks"

    with tempfile.TemporaryDirectory() as directory:
        save(fig, Path(directory), "violins", close=False)
    assert abs(_offset(ax)) < 2.0, _offset(ax)
    plt.close(fig)
