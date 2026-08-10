"""Repairs, on figures drawn by projects that never imported this package."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.qc import audit
from ogviz.qc.repair import repair


@pytest.fixture
def foreign_figure():
    """Plain matplotlib: no house style, no tags, a label on the curve and one over a rule."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.linspace(0.0, 10.0, 200)
    ax.fill_between(x, 0.0, np.exp(-x / 3.0), color="#88aacc")
    ax.plot(x, np.exp(-x / 3.0), color="#224466", lw=2.0)
    ax.grid(visible=True, axis="y")
    ax.set_ylim(0.0, 1.2)
    ax.text(1.5, 0.35, "this label is on the curve", fontsize=12)
    ax.text(6.0, 0.6, "this one crosses a gridline", fontsize=12)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def test_repair_clears_a_foreign_figure(foreign_figure) -> None:
    assert audit(foreign_figure), "the fixture is meant to be broken"
    changes = repair(foreign_figure)
    assert any("moved" in change for change in changes)
    assert any("knocked out" in change for change in changes)
    assert not audit(foreign_figure), "and repairable defects should leave nothing behind"


def test_repair_moves_the_label_and_not_the_marks(foreign_figure) -> None:
    """Presentation only. A repair that moved a curve would change what the figure says."""
    ax = foreign_figure.axes[0]
    before = [line.get_ydata().copy() for line in ax.lines]
    limits = (ax.get_xlim(), ax.get_ylim())
    repair(foreign_figure)
    after = [line.get_ydata() for line in ax.lines]
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert (ax.get_xlim(), ax.get_ylim()) == limits


def test_a_colour_pair_is_reported_and_never_silently_changed() -> None:
    """Which of marker, dash or palette to change is a decision, so the repair does not make it."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="#2E7CE0", label="blue")
    ax.plot([0, 1], [1, 0], color="#8A63D2", label="violet")
    ax.legend()
    fig.canvas.draw()

    colours = [line.get_color() for line in ax.lines]
    changes = repair(fig)
    assert not any("colour" in change for change in changes)
    assert [line.get_color() for line in ax.lines] == colours
    assert any("deuteranopia" in complaint for complaint in audit(fig))
    plt.close(fig)


def test_repairing_a_clean_figure_changes_nothing() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0], color="#2E7CE0")
    fig.canvas.draw()
    assert repair(fig) == []
    plt.close(fig)


def _threshold_panel(level: float):
    """Bars reaching 2.0, and a tagged reference line at `level` drawn UNDER them."""
    from ogviz.tags import mark

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar([0, 1, 2], [1.0, 2.0, 1.5], zorder=3)
    line = ax.axhline(level, color="#B3261E", zorder=1)
    mark(line, "reference", True)
    ax.set_ylim(0.0, 10.0)
    fig.canvas.draw()
    return fig, line


def test_a_threshold_nothing_covers_is_left_alone_and_not_reported() -> None:
    """A low z-order is not a defect: nothing reaches a threshold drawn above every bar.

    The spine loop tested overlap and the reference-line loop did not, so this returned "raised a
    reference line above the marks it is read against" while `buried_baselines` — correctly — said
    the line was fine. The z-order change was harmless; a repair claiming to have fixed a figure
    that was not broken is not, because that list is what a caller reads as the defect report.
    """
    from ogviz.qc.marks import buried_baselines
    from ogviz.qc.repair import raise_buried_lines

    fig, line = _threshold_panel(9.0)
    before = line.get_zorder()

    assert not [c for c in buried_baselines(fig) if "reference" in c], "the check sees nothing"
    assert not [c for c in raise_buried_lines(fig) if "reference" in c], (
        "so the repair says nothing"
    )
    assert line.get_zorder() == before, "and changes nothing"
    plt.close(fig)


def test_a_threshold_the_bars_run_through_is_still_raised() -> None:
    """The other direction, so the overlap test cannot be satisfied by never firing."""
    from ogviz.qc.marks import buried_baselines
    from ogviz.qc.repair import raise_buried_lines

    fig, line = _threshold_panel(1.2)
    before = line.get_zorder()

    assert [c for c in buried_baselines(fig) if "reference" in c], "the bars really do cover it"
    assert [c for c in raise_buried_lines(fig) if "reference" in c]
    assert line.get_zorder() > before, "and it comes forward"
    plt.close(fig)


def test_a_repaired_label_is_not_set_down_on_another_label() -> None:
    """`clear_position` searched with `hits_data`, which walks marks and no TEXT at all.

    So the repair could lift a label off the data and put it on top of another label — trading a
    collision the gate reports for one the gate also reports. Measured on this figure: without the
    labels in the search, the two repaired annotations overlap; with them, they do not.
    """
    from ogviz.layout.collision import text_box

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.linspace(0.0, 10.0, 200)
    ax.fill_between(x, 0.0, np.exp(-x / 3.0), color="#88aacc")
    first = ax.text(4.0, 0.28, "first annotation")
    second = ax.text(4.2, 0.24, "second annotation")
    fig.canvas.draw()

    assert len(repair(fig)) == 2, "both sit on the band and both are moved"
    fig.canvas.draw()
    assert not text_box(first).overlaps(text_box(second))
    assert audit(fig) == [], "and the figure comes out clean"
    plt.close(fig)


def _band_over_the_spine():
    """A filled collection raised above the frame — the case a BOX test cannot judge."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.linspace(0.0, 10.0, 100)
    ax.fill_between(x, -1.0, 3.0, color="#88aacc", zorder=5)
    ax.set_ylim(0.0, 10.0)
    fig.canvas.draw()
    return fig, ax


def test_a_filled_band_over_the_spine_is_seen_and_lifted() -> None:
    """`buried_baselines` walked `ax.patches` only, so a `fill_between` band covering the category
    axis was invisible to the check AND to the repair.

    Widening the box test to every collection is the wrong fix and was measured as such — 17 tests
    fail, because a scatter's box spans its cloud and overlaps a spine its dots never reach.
    `filled_marks_over` excludes point clouds and tests the rest as paths.
    """
    from ogviz.qc.marks import buried_baselines
    from ogviz.qc.repair import raise_buried_lines

    fig, ax = _band_over_the_spine()
    assert buried_baselines(fig), "the band really does cover the bottom spine"

    assert raise_buried_lines(fig), "and the repair lifts it"
    assert ax.spines["bottom"].get_zorder() > 5.0, "above the band, not merely above the patches"
    assert buried_baselines(fig) == [], "check and repair agree once it is lifted"
    plt.close(fig)


def test_a_band_left_under_the_frame_is_not_reported() -> None:
    """matplotlib gives a collection zorder 1 and a spine 2.5, so a default band is UNDER the
    frame — which is where a band belongs, and must stay silent."""
    from ogviz.qc.marks import buried_baselines

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.linspace(0.0, 10.0, 100)
    ax.fill_between(x, -1.0, 3.0, color="#88aacc")
    ax.set_ylim(0.0, 10.0)
    fig.canvas.draw()
    assert buried_baselines(fig) == []
    plt.close(fig)


def test_a_point_cloud_over_the_spine_is_not_reported() -> None:
    """The false positive that makes the box test unusable for collections: a scatter's bounding
    box spans the whole cloud and overlaps a spine none of its dots come near."""
    from ogviz.qc.marks import buried_baselines

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter(np.linspace(1, 9, 40), np.linspace(4, 8, 40), zorder=5)
    ax.set_ylim(0.0, 10.0)
    fig.canvas.draw()
    assert buried_baselines(fig) == []
    plt.close(fig)
