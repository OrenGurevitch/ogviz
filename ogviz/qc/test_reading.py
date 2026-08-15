"""The vocabulary every other check in `qc/` is written against.

It had no test file of its own, which is why it was the one the tracker named to take next: a wrong
answer here is wrong in several checks at once, and silently — `ticks_in_the_headroom`,
`significance_gaps`, `stack_spacing` and `unused_value_headroom` all decide what a bracket is by
asking this module, and none of them would report that the answer came back wrong.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import group_violins
from ogviz.qc.reading import (
    artist_name,
    bracket_spans_px,
    bracket_tops_px,
    brackets_of,
    drawn_artists,
    filled_marks_over,
    is_backdrop,
    is_excused,
    knocked_out_over,
    orientation_of,
)
from ogviz.tags import mark, marked

pytestmark = pytest.mark.usefixtures("pinned_font")


def _bracketed(comparisons=((0.0, 1.0, 0.001),)):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.0, 7.0))
    group_violins(
        ax,
        [
            (0.0, rng.normal(10.0, 1.0, 30), "#E8A838", "#B97C10"),
            (1.0, rng.normal(12.0, 1.0, 30), "#7C9A6E", "#4A6136"),
        ],
        comparisons=list(comparisons),
    )
    fig.canvas.draw()
    return fig, ax


def test_a_hidden_bracket_is_not_measured() -> None:
    """Neither reader filtered on visibility, though the other two readers here both do.

    So every check built on them measured ink that is not drawn.
    """
    fig, ax = _bracketed()
    assert len(bracket_tops_px(ax)) == 1, "the premise: one bracket, and it is found"
    for line in [one for one in ax.lines if marked(one, "bracket")]:
        line.set_visible(False)
    fig.canvas.draw()
    assert bracket_tops_px(ax) == []
    assert bracket_spans_px(ax) == []
    plt.close(fig)


def test_the_tag_wins_over_the_shape() -> None:
    """`layout/axis.py` excludes a bracket from the data extent BY TAG.

    So a bracket-shaped line nobody tagged was furniture to one file and data to the other — one
    package, one artist, two answers, which is what this module exists to prevent.
    """
    from ogviz.layout.axis import _is_furniture

    fig, ax = _bracketed()
    (impostor,) = ax.plot([0.2, 0.2, 0.8, 0.8], [13.5, 14.0, 14.0, 13.5])
    fig.canvas.draw()
    assert not _is_furniture(impostor), "the premise: axis.py does not call it a bracket"
    assert len(bracket_tops_px(ax)) == 1, "and neither does this, now the tag decides"
    plt.close(fig)


def test_the_shape_fallback_still_serves_a_figure_this_package_did_not_draw() -> None:
    """The tag cannot be required: `python -m ogviz.qc` runs on anyone's figure."""
    fig, ax = plt.subplots()
    ax.plot([0.2, 0.2, 0.8, 0.8], [1.0, 2.0, 2.0, 1.0])
    fig.canvas.draw()
    assert len(brackets_of(ax)) == 1
    plt.close(fig)


def test_an_error_bar_cap_is_not_a_bracket() -> None:
    """Four points alone is not the test — a cap has four too, and counting them reported a whole
    bar panel as an uneven stack with brackets 0 px apart."""
    fig, ax = plt.subplots()
    ax.errorbar(np.arange(5.0), np.arange(1.0, 6.0), yerr=0.2, fmt="none", capsize=6)
    fig.canvas.draw()
    assert brackets_of(ax) == []
    plt.close(fig)


def test_a_recorded_orientation_beats_what_the_marks_suggest() -> None:
    """Inferring where the answer was known produced a confident complaint about a good figure."""
    from ogviz.orientation import stamp_orientation

    fig, ax = plt.subplots()
    ax.plot([0.0, 0.0], [1.0, 2.0])  # a vertical two-point line: the vote says "vertical"
    stamp_orientation(ax, "horizontal")
    assert orientation_of(ax) == "horizontal"
    plt.close(fig)


def test_the_vote_is_the_fallback_and_reads_the_whiskers() -> None:
    fig, ax = plt.subplots()
    for centre in (0.0, 1.0):
        ax.plot([centre, centre], [1.0, 2.0])  # an IQR whisker: constant x
    assert orientation_of(ax) == "vertical"
    plt.close(fig)


def test_bracket_spans_report_the_ends_so_disjoint_pairs_can_be_told_apart() -> None:
    """Two comparisons side by side cannot collide however close their heights are."""
    fig, ax = _bracketed(comparisons=())
    for near, far, height in ((0.0, 0.4, 13.5), (0.6, 1.0, 13.5)):
        line = ax.plot([near, near, far, far], [height - 0.3, height, height, height - 0.3])[0]
        mark(line, "bracket")
    fig.canvas.draw()
    spans = bracket_spans_px(ax)
    assert len(spans) == 2
    (_top_a, _near_a, far_a), (_top_b, near_b, _far_b) = spans
    assert far_a < near_b, "they do not overlap along the category axis"
    plt.close(fig)


def test_an_anchored_label_is_excused_only_against_its_own_anchor() -> None:
    """A blanket exemption let a reference-line label sit on a bar without a word."""
    fig, ax = plt.subplots()
    label = ax.text(0.5, 0.5, "1.42x")
    (anchor,) = ax.plot([0.0, 1.0], [0.5, 0.5])
    (other,) = ax.plot([0.0, 1.0], [0.2, 0.2])
    mark(label, "anchored")
    mark(label, "anchor", anchor)
    assert is_excused(label, anchor)
    assert not is_excused(label, other)
    plt.close(fig)


def test_a_knockout_excuses_only_what_it_is_painted_over() -> None:
    """Paint order decides it, so a box UNDER the other artist excuses nothing."""
    fig, ax = plt.subplots()
    label = ax.text(0.5, 0.5, "0.42", zorder=5, bbox={"facecolor": "#FCFCFA", "edgecolor": "none"})
    (under,) = ax.plot([0.0, 1.0], [0.5, 0.5], zorder=1)
    (over,) = ax.plot([0.0, 1.0], [0.5, 0.5], zorder=9)
    fig.canvas.draw()
    assert knocked_out_over(label, under)
    assert not knocked_out_over(label, over), "an artist painted after the box is not hidden by it"
    plt.close(fig)


def test_a_point_cloud_never_counts_as_burying_a_line() -> None:
    """Its bounding box spans the cloud while every dot is somewhere else."""
    fig, ax = plt.subplots()
    rng = np.random.default_rng(1)
    cloud = ax.scatter(rng.uniform(0, 1, 200), rng.uniform(0.6, 1.0, 200), zorder=9)
    fig.canvas.draw()
    assert cloud.get_zorder() > 2.5
    assert filled_marks_over(ax, ax.spines["bottom"].get_window_extent(), 2.5) == []
    plt.close(fig)


def test_a_raised_filled_band_does_count() -> None:
    """The defect the other half exists for: a band deliberately drawn over the frame."""
    fig, ax = plt.subplots()
    band = ax.fill_between([0.0, 1.0], [-1.0, -1.0], [1.0, 1.0], zorder=9)
    # The limits are pinned INSIDE the band, or matplotlib's margins leave the bottom spine below
    # it and there is nothing to bury — which is what the first version of this test measured.
    ax.set_ylim(-0.5, 0.5)
    fig.canvas.draw()
    assert filled_marks_over(ax, ax.spines["bottom"].get_window_extent(), 2.5) == [band]
    plt.close(fig)


def test_drawn_artists_leaves_out_what_is_hidden() -> None:
    fig, ax = plt.subplots()
    shown = ax.text(0.5, 0.5, "shown")
    hidden = ax.text(0.5, 0.6, "hidden")
    hidden.set_visible(False)
    fig.canvas.draw()
    found = drawn_artists(ax)
    assert shown in found and hidden not in found
    plt.close(fig)


def test_a_backdrop_is_named_by_its_tag_and_nothing_else() -> None:
    fig, ax = plt.subplots()
    band = ax.axhspan(0.2, 0.4)
    assert not is_backdrop(band)
    mark(band, "backdrop")
    assert is_backdrop(band)
    plt.close(fig)


def test_artist_name_quotes_text_and_falls_back_to_the_type() -> None:
    """Both halves feed complaint strings, which `group_by_subject` then keys on."""
    fig, ax = plt.subplots()
    assert artist_name(ax.text(0.5, 0.5, "a label")) == "'a label'"
    assert artist_name(ax.plot([0, 1], [0, 1])[0]) == "Line2D"
    plt.close(fig)
