import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.layout import text_overlaps
from ogviz.layout.ticks import MINUS
from ogviz.panels import Series, bar_panel
from ogviz.tags import marked
from ogviz.theme import SERIES


def _label_positions(ax) -> dict[str, float]:
    """Printed value label -> its y in data units."""
    return {t.get_text(): t.get_position()[1] for t in ax.texts}


def test_negative_bar_labels_sit_below_the_bar_not_inside_it():
    """A label placed above every bar lands inside the negative ones, hiding the number."""
    _fig, ax = plt.subplots()
    values = np.array([-1.4, 0.9])
    bar_panel(ax, [Series("z", values, SERIES[0])], ["down", "up"])
    labels = _label_positions(ax)
    assert labels[f"{MINUS}1.40"] < -1.4, "negative bar's label must clear its free end downward"
    assert labels["+0.90"] > 0.9, "positive bar's label must clear its free end upward"


def test_label_clears_the_whisker_cap():
    """Placing the label at the bar top buries it under the error bar whenever one is drawn."""
    _fig, ax = plt.subplots()
    values, errors = np.array([2.0]), np.array([0.6])
    bar_panel(ax, [Series("z", values, SERIES[0], errors)], ["a"])
    assert _label_positions(ax)["2.00"] > 2.6


def test_asymmetric_errors_use_the_matching_cap_on_each_side():
    """A CI is not symmetric; the label must clear the UPPER bound, not the mean +- lower."""
    _fig, ax = plt.subplots()
    values = np.array([1.0])
    bar_panel(ax, [Series("z", values, SERIES[0], np.array([[0.1], [0.9]]))], ["a"])
    assert _label_positions(ax)["1.00"] > 1.9


def test_grouped_series_stay_within_one_series_width():
    """Two series must divide the slot, not each take a full-width bar and overlap."""
    _fig, ax = plt.subplots()
    bar_panel(
        ax,
        [
            Series("ctrl", np.array([1.0, 2.0]), SERIES[0]),
            Series("lc", np.array([1.5, 2.5]), SERIES[1]),
        ],
        ["one", "two"],
        width=0.6,
    )
    lefts = sorted(patch.get_x() for patch in ax.patches)
    widths = {round(patch.get_width(), 6) for patch in ax.patches}
    assert len(widths) == 1
    # Both bars for category 0 lie inside [-0.3, +0.3] around the tick.
    assert lefts[0] >= -0.301
    assert lefts[1] + max(widths) <= 0.301


def test_value_labels_do_not_collide_in_a_grouped_panel():
    """The regression that keeps recurring: labels colliding once bars get close."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    rng = np.random.default_rng(0)
    bar_panel(
        ax,
        [
            Series("control", rng.normal(-0.4, 0.2, 5), SERIES[0], np.full(5, 0.12)),
            Series("treated", rng.normal(0.6, 0.2, 5), SERIES[1], np.full(5, 0.12)),
        ],
        ["fatigue", "sleep", "pain", "cognition", "autonomic"],
    )
    assert text_overlaps(fig) == []


def test_series_length_must_match_the_categories():
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="values for"):
        bar_panel(ax, [Series("z", np.array([1.0, 2.0]), SERIES[0])], ["only-one"])


def _decimals(ax) -> int:
    text = next(iter(_label_positions(ax)))
    return len(text.split(".")[1]) if "." in text else 0


def test_decimals_scale_with_magnitude():
    """Auto formatting: small values need more places than large ones to stay informative.

    A fixed format is what makes a shared helper unusable — ",.2f" turns 0.004 into "+0.00".
    """
    _fig, ax = plt.subplots()
    bar_panel(ax, [Series("z", np.array([0.004, 0.008]), SERIES[0])], ["a", "b"])
    small = _decimals(ax)
    plt.close(_fig)
    _fig, ax = plt.subplots()
    bar_panel(ax, [Series("z", np.array([4100.0, 8200.0]), SERIES[0])], ["a", "b"])
    assert small > _decimals(ax)
    assert small >= 3, "a value of 0.004 must not round to zero in its own label"


def test_reference_line_is_drawn_and_labelled():
    _fig, ax = plt.subplots()
    bar_panel(
        ax,
        [Series("z", np.array([0.5, 0.7]), SERIES[0])],
        ["a", "b"],
        reference=(0.80, "reference level"),
    )
    assert any(round(line.get_ydata()[0], 6) == 0.80 for line in ax.lines)
    assert "reference level" in {t.get_text() for t in ax.texts}


def test_a_bar_series_refuses_non_finite_values():
    """A non-finite bar draws nothing and the gap reads as a zero."""
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match=r"1 non-finite value\(s\) of 3"):
        bar_panel(ax, [Series("z", np.array([1.0, np.nan, 3.0]), SERIES[0])], list("abc"))


def test_a_reference_line_is_drawn_over_the_bars_it_is_read_against() -> None:
    """Behind them a threshold survives only in the gaps, which loses the comparison it was for."""
    from ogviz.panels.bars import Z_BAR, reference_line

    _fig, ax = plt.subplots()
    bar_panel(ax, [Series("s", np.array([0.3, 0.9, 1.4]), ["#8FA9C9"] * 3)], ["a", "b", "c"])
    reference_line(ax, 0.8, "threshold")
    drawn = [line for line in ax.lines if marked(line, "reference")]
    assert len(drawn) == 1
    assert drawn[0].get_zorder() > Z_BAR, "the threshold must sit above the bars"


def test_a_buried_threshold_is_reported() -> None:
    from ogviz.panels.bars import reference_line
    from ogviz.qc import buried_baselines

    fig, ax = plt.subplots()
    bar_panel(ax, [Series("s", np.array([0.3, 0.9, 1.4]), ["#8FA9C9"] * 3)], ["a", "b", "c"])
    reference_line(ax, 0.8, "threshold")
    fig.canvas.draw()
    assert not buried_baselines(fig)

    [line] = [line for line in ax.lines if marked(line, "reference")]
    line.set_zorder(1)
    fig.canvas.draw()
    assert any("reference line" in complaint for complaint in buried_baselines(fig))


def test_a_rounded_corner_is_the_same_size_on_any_axis() -> None:
    """The radius was a fraction of the tallest VALUE, and a rounding applies to BOTH axes.

    So it was only sensible where the value axis happened to be order-1: measured on a counts axis,
    the corner came out at 394,460% of the bar's own width — a lozenge rather than a bar.

    MEASURED IN RENDERED PIXELS. This used to read `rounding_size` off the boxstyle, which stopped
    existing when the boxstyle became a callable that rounds only the free end — and an attribute is
    the input rather than the result anyway. The corner is now taken from the ink: how much narrower
    the bar is at its top than at its middle is twice the radius, whatever produced it.
    """
    import numpy as np

    from ogviz import bar_panel
    from ogviz.layout.raster import frame_rgb, ink_of
    from ogviz.panels.bars import Series

    def corner_px(values: list[float]) -> float:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        bar_panel(ax, [Series("s", np.array(values), "#7C9A6E")], ["A"], rounded=True)
        fig.canvas.draw()
        ink = ink_of(frame_rgb(fig), tolerance=8)
        rows = np.flatnonzero(ink.sum(axis=1) > 0)
        top, bottom = rows.min(), rows.max()
        at_top = int(ink[top + 2].sum())  # clear of the antialiased first row
        at_middle = int(ink[(top + bottom) // 2].sum())
        plt.close(fig)
        return (at_middle - at_top) / 2.0

    near_one = corner_px([0.61])
    near_thousands = corner_px([38000.0])
    assert near_one > 1.0, "the premise: there is a corner to measure at all"
    assert near_one == pytest.approx(near_thousands, abs=1.5), (near_one, near_thousands)


def test_a_rounded_bar_still_meets_its_baseline_square() -> None:
    """A bar encodes its value as the distance from zero, so the corner AT zero must be square.

    `boxstyle="round"` rounds all four, and for as long as `rounded=True` existed the foot curved
    away from the axis: a visible gap where the bar should meet zero, and a base narrower than the
    bar. Measured on the RENDERED pixels rather than on the path, because the path is mutated by
    `mutation_aspect` on the way to the page and the vertices alone do not say what landed.
    """
    import numpy as np

    from ogviz.layout.raster import frame_rgb, ink_of

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bar_panel(ax, [Series("v", np.array([0.9, 0.6]), "#5B7FB9")], ["a", "b"], rounded=True)
    fig.canvas.draw()
    ink = ink_of(frame_rgb(fig), tolerance=8)

    base_y = round(ax.transData.transform((0.0, 0.0))[1])
    row_at_base = ink.shape[0] - base_y - 3  # a few px above zero, inside the bar
    mid_y = ink.shape[0] - round(ax.transData.transform((0.0, 0.45))[1])
    widths = [int(ink[row].sum()) for row in (row_at_base, mid_y)]
    assert widths[0] >= widths[1], (
        f"narrower at the baseline ({widths[0]} px) than at mid height ({widths[1]} px) — "
        "the feet are rounded and the bar does not meet zero"
    )
    plt.close(fig)


def test_a_horizontal_bar_is_rounded_too_and_at_its_free_end() -> None:
    """`rounded` was tested together with `upright`, so a sideways panel silently got square bars.

    The same two measurements as the upright case, transposed: the bar is narrower along its
    thickness at the free END than at mid length, and NOT narrower at the foot.
    """
    import numpy as np

    from ogviz.layout.raster import frame_rgb, ink_of

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bar_panel(
        ax,
        [Series("v", np.array([0.9]), "#5B7FB9")],
        ["a"],
        rounded=True,
        orientation="horizontal",
        show_values=False,
    )
    fig.canvas.draw()
    ink = ink_of(frame_rgb(fig), tolerance=8)

    def column_at(value: float) -> int:
        return int(ink[:, round(ax.transData.transform((value, 0.0))[0])].sum())

    at_foot, at_middle, at_tip = column_at(0.02), column_at(0.45), column_at(0.885)
    assert at_middle > 0, "the premise: there is a bar in the column being measured"
    assert at_tip < at_middle, f"the free end is not softened ({at_tip} px vs {at_middle} px)"
    assert at_foot >= at_middle, f"the foot is rounded ({at_foot} px vs {at_middle} px)"
    plt.close(fig)


def test_a_rounded_bar_carries_the_same_opacity_as_a_plain_one() -> None:
    """`ax.bar` has always been passed BAR_ALPHA and the FancyBboxPatch path was not.

    So the same hex rendered at two opacities depending on `rounded` — a shape option quietly
    deciding a colour. Measured as the RGB the page ends up with, which is what a reader and the
    colour-vision check both see.
    """
    import numpy as np

    from ogviz.layout.raster import frame_rgb

    def bar_rgb(rounded: bool) -> tuple[int, int, int]:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        bar_panel(
            ax,
            [Series("v", np.array([0.9]), "#5B7FB9")],
            ["a"],
            rounded=rounded,
            show_values=False,
        )
        fig.canvas.draw()
        frame = frame_rgb(fig)
        x = round(ax.transData.transform((0.0, 0.0))[0])
        y = frame.shape[0] - round(ax.transData.transform((0.0, 0.45))[1])
        plt.close(fig)
        return tuple(int(channel) for channel in frame[y, x])  # type: ignore[return-value]

    assert bar_rgb(rounded=True) == bar_rgb(rounded=False)


def test_emphasis_names_one_bar_on_a_grouped_panel() -> None:
    """It was passed into each series unchanged, so it emphasised one bar PER series.

    And the series it was not in had to be told so explicitly — passing None there means "every
    label is one to read", which would have left that whole series bold beside the muted one.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    bar_panel(
        ax,
        [
            Series("first", np.array([0.4, 0.5, 0.6]), "#7C9A6E"),
            Series("second", np.array([0.3, 0.7, 0.5]), "#E8A838"),
        ],
        ["a", "b", "c"],
        emphasis=(1, 2),
    )
    bold = [text for text in ax.texts if text.get_fontweight() == "bold"]
    assert len(bold) == 1, [text.get_text() for text in bold]
    assert bold[0].get_text() == "0.500", bold[0].get_text()
    plt.close(fig)


def test_a_category_emphasis_still_marks_that_category_in_every_series() -> None:
    """The reading "look at this one" for a grouped panel, and what the int has always done."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    bar_panel(
        ax,
        [
            Series("first", np.array([0.4, 0.5, 0.6]), "#7C9A6E"),
            Series("second", np.array([0.3, 0.7, 0.5]), "#E8A838"),
        ],
        ["a", "b", "c"],
        emphasis=1,
    )
    bold = sorted(text.get_text() for text in ax.texts if text.get_fontweight() == "bold")
    assert bold == ["0.500", "0.700"], bold
    plt.close(fig)


def test_a_bar_panel_and_a_direct_call_draw_one_and_the_same_band() -> None:
    """`bar_panel(reference_band=...)` drew its own until 2026-07-31, and drew the OLD design.

    Solid fill with the label centred inside it, which is what `reference_band` was written to
    replace — and it stayed reachable through the argument, so a caller got the rejected design by
    picking the other door.
    """
    from ogviz import bar_panel, reference_band
    from ogviz.panels.bars import Series

    def band_of(fig, ax) -> tuple[float, bool, float]:
        fig.canvas.draw()
        patch = next(p for p in ax.patches if marked(p, "reference"))
        label = next(t for t in ax.texts if "agreement" in t.get_text())
        bottom = ax.transData.inverted().transform((0.0, label.get_window_extent().y0))[1]
        return patch.get_zorder(), marked(patch, "backdrop"), float(bottom)

    series = [Series("arm", [0.62, 0.71, 0.78], "#7C9A6E")]
    direct_fig, direct_ax = plt.subplots(figsize=(8.0, 5.0))
    bar_panel(direct_ax, series, ["A", "B", "C"])
    reference_band(direct_ax, 0.80, 0.88, "human agreement")

    panel_fig, panel_ax = plt.subplots(figsize=(8.0, 5.0))
    bar_panel(panel_ax, series, ["A", "B", "C"], reference_band=(0.80, 0.88, "human agreement"))

    assert band_of(direct_fig, direct_ax) == band_of(panel_fig, panel_ax)
    _zorder, _backdrop, label_bottom = band_of(panel_fig, panel_ax)
    assert label_bottom >= 0.88, "the label sits above the band, never inside it"


def test_a_bands_fill_stays_under_the_frame() -> None:
    """At `Z_REFERENCE` it washed over the left spine — a frame that changes colour halfway up."""
    from ogviz import bar_panel, reference_band
    from ogviz.panels.bars import Series
    from ogviz.qc import buried_baselines

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bar_panel(ax, [Series("arm", [0.62, 0.71, 0.78], "#7C9A6E")], ["A", "B", "C"])
    reference_band(ax, 0.80, 0.88, "human agreement")
    fig.canvas.draw()
    patch = next(p for p in ax.patches if marked(p, "reference"))
    assert patch.get_zorder() < ax.spines["left"].get_zorder()
    assert not buried_baselines(fig)


def test_a_reference_band_runs_the_right_way_on_a_horizontal_panel() -> None:
    """The axis nothing was exercising. `rounded` was silently dropped there for the same reason.

    On a horizontal panel the value axis is x, so the band is a vertical span and its label is
    lifted along x — checked by where the ink actually landed, not by which branch was taken.
    """
    from ogviz import bar_panel, reference_band
    from ogviz.panels.bars import Series

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bar_panel(
        ax,
        [Series("arm", np.array([0.62, 0.71, 0.78]), "#7C9A6E")],
        ["A", "B", "C"],
        orientation="horizontal",
        show_values=False,
    )
    reference_band(ax, 0.80, 0.88, "human agreement", orientation="horizontal")
    fig.canvas.draw()

    patch = next(p for p in ax.patches if marked(p, "reference"))
    box = patch.get_window_extent()
    panel = ax.get_window_extent()
    assert box.height >= panel.height * 0.99, "a horizontal panel's band spans the CATEGORY axis"
    assert box.width < panel.width * 0.5, "and is bounded on the value axis"

    label = next(t for t in ax.texts if "agreement" in t.get_text())
    left = ax.transData.inverted().transform((label.get_window_extent().x0, 0.0))[0]
    assert left >= 0.88, "the label sits beyond the band's far edge, never inside it"
    plt.close(fig)


def test_a_band_a_label_names_is_not_reported_as_data_under_the_label() -> None:
    """Untagged, every value label reaching into the band is reported as sitting on the marks.

    The band is a region a reader looks THROUGH, so it carries `backdrop` as well as `reference` —
    and `text_over_data` has to agree, since it is the check that would otherwise fire.
    """
    from ogviz import bar_panel, reference_band
    from ogviz.layout.collision import text_over_data
    from ogviz.panels.bars import Series

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bar_panel(ax, [Series("arm", np.array([0.78, 0.82, 0.86]), "#7C9A6E")], ["A", "B", "C"])
    reference_band(ax, 0.80, 0.92, "human agreement")
    fig.canvas.draw()
    on_the_band = [one for one in text_over_data(fig) if "sits on" in one]
    assert on_the_band == [], on_the_band
    plt.close(fig)
