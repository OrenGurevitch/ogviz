"""Measuring how much of the page a figure actually uses."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ogviz.layout.density import dead_space, measure, panel_emptiness, trim_margins


def _curve(ax):
    x = np.linspace(0.0, 10.0, 200)
    ax.plot(x, np.sin(x) * 0.1 + 0.5, lw=2.0)


def test_a_generous_limit_shows_up_as_an_empty_band() -> None:
    """The number that says "your y-limit is too big" — invisible to any bounding box."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    ax.set_ylim(0.0, 5.0)
    assert panel_emptiness(fig, ax)["top"] > 0.5


def test_a_fitted_panel_reports_almost_no_empty_band() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    ax.set_ylim(0.38, 0.62)
    ax.set_xlim(0.0, 10.0)
    panel = panel_emptiness(fig, ax)
    assert panel["top"] < 0.12 and panel["bottom"] < 0.12


def test_gridlines_do_not_make_an_empty_panel_look_full() -> None:
    """A grid inks every edge, so the raw render calls every panel full however empty it is."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    ax.set_ylim(0.0, 5.0)
    ax.grid(True)
    assert panel_emptiness(fig, ax)["top"] > 0.5


def test_trim_margins_reclaims_outer_space_and_changes_no_data() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    fig.subplots_adjust(left=0.35, right=0.65, top=0.65, bottom=0.35)
    limits = (ax.get_xlim(), ax.get_ylim())
    before = measure(fig)
    assert trim_margins(fig) is True
    after = measure(fig)
    assert after.wasted_margin() < before.wasted_margin()
    assert after.coverage > before.coverage
    assert (ax.get_xlim(), ax.get_ylim()) == limits, "trimming must not touch the data limits"


def test_a_figure_that_already_fills_its_page_is_left_alone() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    fig.tight_layout(pad=0.05)
    assert trim_margins(fig) is False


def test_dead_space_is_quiet_on_a_well_packed_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    ax.set_ylim(0.38, 0.62)
    fig.tight_layout(pad=0.05)
    assert not dead_space(fig)


def test_an_axes_off_the_canvas_is_not_reported_as_a_full_panel() -> None:
    """The silent default: no slice to measure used to come back as every edge 0% empty.

    Which is the answer a perfectly packed panel gives, so the one panel that is actually broken
    was the one `dead_space` had nothing to say about.
    """
    fig = plt.figure(figsize=(6.0, 4.0))
    off = fig.add_axes((1.6, 0.1, 0.3, 0.3))
    panel = panel_emptiness(fig, off)
    assert panel["top"] == 1.0 and panel["left"] == 1.0
    assert any("of the panel is empty" in note for note in dead_space(fig))


def test_a_horizontal_panel_is_told_which_limit_is_generous() -> None:
    """Top and bottom are the CATEGORY axis when the panel runs horizontally.

    The note hardcoded the vertical convention, so it sent a reader to tighten the axis carrying
    the comparison. Asserted against a vertical panel too, or the mapping could be inverted and
    only the wording would look right.
    """
    from ogviz.panels.bars import Series, bar_panel

    def named_axis(orientation: str, side: str) -> str:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        bar_panel(
            ax,
            [Series("only", [0.30, 0.34, 0.31], "#7C9A6E")],
            ["a", "b", "c"],
            orientation=orientation,  # type: ignore[arg-type]
            show_values=False,
        )
        span = ax.set_xlim if orientation == "horizontal" else ax.set_ylim
        span(0.0, 4.0)  # a generous VALUE limit, whichever way the panel runs
        for note in dead_space(fig):
            if f"the {side} " in note and "of the panel is empty" in note:
                return note.rsplit("— the ", 1)[1]
        raise AssertionError(f"no note about the {side} of a {orientation} panel")

    # The same generous VALUE limit, twice, on the edge it leaves empty each way round.
    assert named_axis("vertical", "top") == "value limit is generous"
    assert named_axis("horizontal", "right") == "value limit is generous"


def test_trimming_one_wasteful_side_does_not_shrink_a_tight_one() -> None:
    """`trim_margins` grows; the per-side arithmetic could also un-grow.

    A side whose ink already reaches within `pad_px` of the edge contributed a negative gain, and
    the gate tests only the widest side — so one wasteful side licensed a shrink on the rest.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _curve(ax)
    fig.subplots_adjust(left=0.45, right=0.999, top=0.999, bottom=0.999 - 0.55)
    fig.canvas.draw()
    tight = {side: getattr(fig.subplotpars, side) for side in ("right", "top")}
    assert trim_margins(fig) is True
    assert fig.subplotpars.left < 0.45, "the wasteful side is what it was asked to reclaim"
    for side, was in tight.items():
        assert getattr(fig.subplotpars, side) >= was, f"{side} was pulled in"


def test_a_panel_on_a_shared_scale_is_not_told_to_tighten_it() -> None:
    """The one figure where tightening the limit is the wrong action.

    A short panel beside a tall one is empty at the top BY CONSTRUCTION — the tall one set the
    scale — and acting on the note is what stops the grid being comparable. Both premises are
    asserted: the same panel drawn alone still gets the plain "generous limit" advice.
    """
    from ogviz.panels.grid import share_value_limits

    def note_for(share: bool) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
        for ax, top in zip(axes, (1.0, 8.0), strict=True):
            _curve(ax)
            ax.set_ylim(0.0, top)
        if share:
            share_value_limits(axes, label_edge=False)
        # "panel 0 of 2: " — untitled panels on a two-panel figure, which is what
        # `panel_prefix` falls back to. It was `axes 0:`, one of the three conventions that
        # function replaced.
        notes = [n for n in dead_space(fig) if n.startswith("panel 0 of 2: ") and " top " in n]
        assert notes, "the premise: the short panel reports an empty top either way"
        return notes[0]

    assert "limit is generous" in note_for(share=False)
    shared = note_for(share=True)
    assert "shares a value scale with 2 panels" in shared
    assert "limit is generous" not in shared


def test_a_blank_figure_is_one_note_and_not_four_margins() -> None:
    """Each band runs to the edge of the ink, and with no ink each is the whole canvas.

    So the four notes described "the canvas past the left edge of the ink" on a page with no such
    edge, and left and right each claimed the full width — the same emptiness counted twice per
    axis. The premise is the measurement itself.
    """
    fig = plt.figure(figsize=(6.0, 4.0))
    overall = measure(fig)
    assert overall.coverage == 0.0
    assert overall.left == overall.right, "premise: both sides claim the whole width"
    assert dead_space(fig) == ["the figure has no ink on it"]
    plt.close(fig)
