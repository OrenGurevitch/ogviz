from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.marks import (
    CENTER_GAP,
    MEAN_HALF_WIDTH,
    VIOLIN_WIDTH,
    Z_IQR,
    Z_MEAN_LINE,
    Z_MEDIAN_DOT,
    Z_POINTS,
    Z_VIOLIN,
    iqr_box,
    jitter_x,
    mean_line,
    points,
    violin,
)


def test_the_stacking_order_is_the_one_the_figure_needs() -> None:
    # A mean line under the IQR bar reads as passing behind it; a median dot under the mean
    # line disappears. The order is the contract.
    assert Z_VIOLIN < Z_POINTS < Z_IQR < Z_MEAN_LINE < Z_MEDIAN_DOT


def test_jitter_keeps_every_dot_inside_the_violin_half_width() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    xs = jitter_x(values, 3.0, rng)
    assert np.all(np.abs(xs - 3.0) <= VIOLIN_WIDTH / 2)


def test_jitter_leaves_the_centre_clear_for_the_box_and_mean() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=400)
    offsets = np.abs(jitter_x(values, 0.0, rng))
    # Points in the dense middle of the distribution must clear the central marks.
    dense = np.abs(values) < 0.5
    assert offsets[dense].min() >= CENTER_GAP * 0.85


def test_jitter_is_degenerate_safe() -> None:
    rng = np.random.default_rng(2)
    assert jitter_x(np.array([4.2]), 1.0, rng).tolist() == [1.0]
    assert jitter_x(np.full(9, 4.2), 1.0, rng).tolist() == [1.0] * 9


def test_iqr_box_puts_the_median_dot_at_the_median() -> None:
    _fig, ax = plt.subplots()
    values = np.arange(101, dtype=float)
    iqr_box(ax, values, 0.0)
    dot = next(line for line in ax.lines if line.get_marker() == "o")
    assert dot.get_ydata()[0] == pytest.approx(50.0)
    assert dot.get_zorder() == Z_MEDIAN_DOT
    plt.close("all")


def test_mean_line_sits_at_the_mean_and_carries_a_canvas_halo() -> None:
    # The halo must be the canvas colour: a white halo punches a pale gap through the dark IQR
    # bar it crosses, which is what made the mean line look like it passed behind.
    _fig, ax = plt.subplots()
    values = np.array([0.0, 1.0, 2.0, 97.0])
    mean_line(ax, values, 0.0)
    line = ax.lines[0]
    assert line.get_ydata()[0] == pytest.approx(values.mean())
    assert line.get_zorder() == Z_MEAN_LINE
    assert line.get_path_effects(), "the mean line needs a halo to stay legible over dots"
    plt.close("all")


def test_violin_and_points_take_their_declared_layers() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=50)
    _fig, ax = plt.subplots()
    violin(ax, values, 0.0, "#7C9A6E")
    points(ax, values, 0.0, "#7C9A6E", "#4A6136", rng)
    body = ax.collections[0]
    scatter = ax.collections[-1]
    assert body.get_zorder() == Z_VIOLIN
    assert scatter.get_zorder() == Z_POINTS
    plt.close("all")


def test_a_points_y_is_always_the_datum_itself() -> None:
    """The jitter is horizontal only. A vertical nudge would misplace a value on its own axis."""
    values = np.random.default_rng(11).normal(-2.1, 1.3, 40)
    _fig, ax = plt.subplots()
    points(ax, values, 1.0, "#7C9A6E", "#4A6136", np.random.default_rng(0))
    np.testing.assert_array_equal(np.asarray(ax.collections[0].get_offsets())[:, 1], values)
    plt.close("all")


def test_a_narrower_violin_narrows_its_own_jitter() -> None:
    """width is passed to both or the dots spread outside the body they belong to."""
    values = np.random.default_rng(5).normal(0.0, 1.0, 60)
    wide = jitter_x(values, 0.0, np.random.default_rng(1), width=0.8)
    narrow = jitter_x(values, 0.0, np.random.default_rng(1), width=0.62)
    assert np.abs(narrow).max() < np.abs(wide).max()
    assert np.abs(wide).max() <= 0.8 / 2


def test_a_violin_can_carry_a_coloured_outline() -> None:
    """A lighter fill with a matching edge is a legitimate house variant, not a fork."""
    values = np.random.default_rng(2).normal(0.0, 1.0, 30)
    _fig, ax = plt.subplots()
    violin(ax, values, 0.0, "#3FA372", alpha=0.28, edge_color="#3FA372", edge_width=1.3)
    body = ax.collections[0]
    assert body.get_alpha() == pytest.approx(0.28)
    assert body.get_edgecolor().shape[0] == 1
    plt.close("all")


def _lane_violations(ax, values, position: float) -> int:
    """Dots whose centre sits inside the lane reserved for the central marks."""
    from ogviz.marks import central_clearance

    offsets = np.asarray(ax.collections[0].get_offsets())
    lane = central_clearance(ax, values)
    return int(np.count_nonzero(np.abs(offsets[:, 0] - position) < lane * 0.999))


def test_no_dot_lands_on_the_central_marks() -> None:
    """The reported defect: dots sat on the mean line, whose reach (0.15) is more than twice the
    old single gap (0.06), and on the median dot wherever the body was too narrow to hold them."""
    rng = np.random.default_rng(0)
    for values in (
        rng.normal(0.0, 1.0, 60),  # ordinary
        rng.gamma(1.5, 0.5, 60),  # long tail: the body is narrow where dots still exist
        np.concatenate([rng.normal(0, 0.05, 40), rng.normal(3, 0.05, 20)]),  # bimodal
    ):
        _fig, ax = plt.subplots()
        ax.set_ylim(values.min() - 1, values.max() + 1)
        points(ax, values, 1.0, "#7C9A6E", "#4A6136", np.random.default_rng(1))
        assert _lane_violations(ax, values, 1.0) == 0
        plt.close("all")


def test_the_lane_is_widest_where_the_mean_line_is() -> None:
    """One number could not do this: the mean line reaches far wider than the whisker, but only
    over its own linewidth in y, so a blanket lane would shove every dot out for nothing."""
    from ogviz.marks import central_clearance

    values = np.linspace(-3.0, 3.0, 61)  # mean 0, and points either side of it
    _fig, ax = plt.subplots()
    ax.set_ylim(-4, 4)
    lane = central_clearance(ax, values)
    at_mean = lane[np.argmin(np.abs(values - values.mean()))]
    far = lane[0]
    assert at_mean >= MEAN_HALF_WIDTH
    assert far < at_mean / 3, "a dot out in the tail should not be pushed as far as one at the mean"
    plt.close("all")


def test_a_lane_is_not_reserved_for_a_mark_that_was_never_drawn() -> None:
    """`central_clearance` sized the lane for the FULL mark set, whatever the panel actually drew.

    A panel hand-assembling `violin` + `points` + `mean_line` and deliberately drawing no IQR box
    still held room for the box, the whisker and the median dot — dots pushed wide of a bar that is
    not on the figure. Two consumers assemble exactly that trio. The failure is invisible in the
    sense that matters: the figure looks fine, and the jitter stops following the shape it exists
    to describe.
    """
    from ogviz.marks import central_clearance

    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.set_ylim(1.0, 9.0)
    ax.set_xlim(-0.5, 0.5)

    everything = central_clearance(ax, values)
    mean_only = central_clearance(ax, values, drawn={"mean"})
    nothing = central_clearance(ax, values, drawn=())

    assert (mean_only <= everything).all(), "dropping a mark can only free room, never claim it"
    assert mean_only.mean() < everything.mean(), "and it frees some"
    assert mean_only.max() == pytest.approx(everything.max()), (
        "the mean line still reserves its own"
    )
    assert (nothing > 0.0).all(), (
        "a dot still reserves its own radius, so it never straddles centre"
    )
    plt.close(fig)


def test_the_default_is_every_mark_so_no_existing_caller_moves() -> None:
    from ogviz.marks import MARK_NAMES, central_clearance

    rng = np.random.default_rng(1)
    values = rng.normal(5.0, 1.0, 120)
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.set_ylim(1.0, 9.0)

    assert np.array_equal(
        central_clearance(ax, values), central_clearance(ax, values, drawn=MARK_NAMES)
    )
    plt.close(fig)


def test_a_mark_name_it_does_not_know_is_refused() -> None:
    """A typo must not silently reserve nothing — that is a lane quietly set to zero."""
    from ogviz.marks import central_clearance

    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.set_ylim(1.0, 9.0)
    with pytest.raises(AssertionError, match="does not know the mark"):
        central_clearance(ax, np.array([1.0, 2.0, 3.0]), drawn={"whisker"})
    plt.close(fig)


def test_a_width_that_is_not_a_number_is_refused_by_widths_of() -> None:
    """`bool` is an `int`, so `box_width=True` reserved a 1.0 pt lane. An unknown KEY is still
    passed over, which is the documented cost of accepting a whole kwargs mapping."""
    from ogviz.marks import widths_of

    with pytest.raises(AssertionError, match="box_width is a width"):
        widths_of({"box_width": True})
    assert widths_of({"box_wdith": 4.0, "color": "#000"}) == {}


def test_iqr_box_accepts_a_plain_sequence_like_every_other_mark() -> None:
    """It half-accepted one: `np.percentile` took a list and `values.min()` two lines later did not.

    The premise is the sibling: `violin`, `points` and `central_clearance` all coerce, so a caller
    handing a list to one mark and to the next got a mark and an `AttributeError`.
    """
    _fig, ax = plt.subplots()
    iqr_box(ax, [1.0, 2.0, 3.0, 4.0, 9.0], 0.0)  # type: ignore[arg-type]
    assert len(ax.lines) == 3  # the whisker, the box, and the median dot
    with pytest.raises(AssertionError, match="at least one value"):
        iqr_box(ax, [], 0.0)  # type: ignore[arg-type]


def test_the_category_axis_is_pinned_rather_than_padded_by_the_body() -> None:
    """It was `min - body .. max + body`, so the pad tracked the violin instead of the cell.

    Which means a wider body pushed the axis out with it and the body's share of its cell never
    moved — a `width` argument that could not change the picture, only the units it was measured
    in. That is the assertion below, and it is the one that would have failed before.

    0.54 is held rather than a round number because it is matplotlib's own three-slot answer,
    measured here alongside it.
    """
    from ogviz import group_violins
    from ogviz.marks import CATEGORY_HALF_SLOT

    def autoscale_pad(count: int) -> float:
        _fig, ax = plt.subplots()
        rng = np.random.default_rng(0)
        ax.violinplot(
            [rng.normal(0.0, 1.0, 30) for _ in range(count)],
            positions=list(range(count)),
            widths=VIOLIN_WIDTH,
            showextrema=False,
        )
        low, _high = ax.get_xlim()
        return abs(float(low))

    pads = {count: autoscale_pad(count) for count in (2, 3, 4)}
    assert pads[2] < pads[3] < pads[4], f"premise: autoscale drifts with the count — {pads}"
    assert pads[3] == pytest.approx(CATEGORY_HALF_SLOT, abs=1e-9), "0.54 is the three-slot answer"

    def panel_pad(count: int) -> float:
        _fig, ax = plt.subplots()
        rng = np.random.default_rng(1)
        group_violins(
            ax,
            [(float(i), rng.normal(5.0, 1.0, 30), "#E8A838", "#B97C10") for i in range(count)],
            show_means=False,
        )
        return abs(float(ax.get_xlim()[0]))

    # PINNED: the same pad whatever the count, and — the part that changed — whatever the body.
    assert [panel_pad(count) for count in (2, 3, 4)] == pytest.approx([CATEGORY_HALF_SLOT] * 3)

    def share_of_the_cell(width: float) -> float:
        """What fraction of the panel one body occupies at a given asked-for width."""
        _fig, ax = plt.subplots()
        rng = np.random.default_rng(3)
        group_violins(
            ax,
            [(float(i), rng.normal(5.0, 1.0, 30), "#E8A838", "#B97C10") for i in range(3)],
            show_means=False,
            violin_kwargs={"width": width},
            point_kwargs={"width": width},
        )
        low, high = ax.get_xlim()
        return width / (high - low)

    narrow, wide = share_of_the_cell(VIOLIN_WIDTH * 0.6), share_of_the_cell(VIOLIN_WIDTH)
    assert wide > narrow * 1.5, (
        "a width argument has to change the body's share of its cell; padding by the body made "
        f"that share constant, and it is {narrow:.3f} against {wide:.3f}"
    )


def test_a_caller_can_still_have_the_retired_padding() -> None:
    """The old look was one whole violin width either side, and it stays reachable by name."""
    from ogviz import group_violins

    _fig, ax = plt.subplots()
    rng = np.random.default_rng(2)
    group_violins(
        ax,
        [(float(i), rng.normal(5.0, 1.0, 30), "#E8A838", "#B97C10") for i in range(2)],
        show_means=False,
        category_pad=VIOLIN_WIDTH,
    )
    assert abs(float(ax.get_xlim()[0])) == pytest.approx(VIOLIN_WIDTH)
