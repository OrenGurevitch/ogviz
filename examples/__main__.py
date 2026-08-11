"""Render every example into `examples/out/`. Run with `uv run python -m examples`.

Each function below is one figure and is meant to be read as much as run: it is the shortest
call that produces the thing, so a reader can see what the library does and what stays theirs.
Colours, statistics and units are always the caller's.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Pinned BEFORE pyplot is imported, and this is not a formality. The committed gallery is a
# reproducibility claim — the whole point of `svg.hashsalt` and of dropping the date stamp — and a
# backend is part of what a figure is rendered by. On a Mac with a display, matplotlib picks
# `macosx`, which lays text out differently from `Agg`: the violin panel that stood at slot 06 came
# out with SEVEN y-ticks under `macosx` and FOUR under `Agg`, from identical code and identical
# data, because the text metrics moved the axis limits enough to change what the locator chose.
#
# Every test runs under `Agg`. So without this line the gallery was rendered by something the suite
# never exercises, and could not be reproduced on a headless machine or in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from examples.data import (
    ARM_LABELS,
    BENDING_LEVELS,
    BLOCKS,
    COHORT,
    CONDITION_TINTS,
    CONTROL,
    CONTROL_EDGE,
    DOMAINS,
    FORMATIONS,
    LEAGUE_AVERAGE_XG,
    NATIONS,
    SPECTRAL_BANDS,
    STAGES_OF_THE_ROAD,
    TREATED,
    TREATED_EDGE,
    bending_by_nation,
    concentration_ppm,
    correlation_with_interval,
    coupled_variables,
    domain_profile,
    effort_ladders,
    expected_goals,
    headline_arms,
    heart_rate_spectra,
    journey_measures,
    large_magnitudes,
    paired_sensors,
    skewed_groups,
    two_groups,
)
from ogviz import (
    INK,
    MUTED_INK,
    Cell,
    Cloud,
    Estimate,
    Leg,
    Line,
    Row,
    Series,
    Strand,
    bar_panel,
    baseline,
    broken_zero,
    caption,
    coupling_panels,
    effect_heatmap,
    error_bars,
    fit_under_header,
    group_violins,
    hairline_grid,
    identity_colors,
    label_rows,
    legend_pill,
    line_panel,
    page_color,
    reference_line,
    reproducible_metadata,
    save,
    series_colors,
    slopegraph,
    spectrogram,
    split_violins,
    table_panel,
    ticks_over_data,
    tint,
    titled,
    use_house_style,
    value_floor,
    value_ticks,
    zero_baseline,
)
from ogviz.layout.ticks import typeset
from ogviz.panels.lines import MUTED_SERIES
from ogviz.tags import mark

OUT = Path(__file__).parent / "out"


def render(fig, name: str) -> None:
    """Write the example as both a raster and a vector.

    The README links the PNG. GitHub renders a clicked SVG at its intrinsic size — 1188 pt across
    for these — which lands smaller and softer than the same figure at 200 dpi, where the PNG is
    3300 px across. The SVG is kept beside it for anyone who wants to zoom without limit or read
    the coordinates.

    They cannot drift apart: one call writes both, so a regenerated example regenerates the pair.
    """
    save(fig, OUT, name, formats=("png", "svg"), dpi=200)


def _groups(data: dict[str, np.ndarray]) -> list[tuple[float, np.ndarray, str, str]]:
    return [
        (0.0, data["control"], CONTROL, CONTROL_EDGE),
        (1.0, data["treated"], TREATED, TREATED_EDGE),
    ]


ALPHA = 0.05
# Big enough that a star reads as a mark rather than as punctuation. The house default is sized for
# a single panel; a grid cell is a fraction of the page, and the same point size in a cell that is
# half the width is half the apparent size to the reader.
GRID_STAR_SIZE = 26.0


def _significant(tests, seat) -> list[tuple[float, float, float]]:
    """Only the comparisons that cleared `ALPHA`, positioned for `group_violins`.

    A BRACKET IS DRAWN WHEN THERE IS SOMETHING TO SAY. `bracket_stack` will happily label a null
    result "n.s.", and that is the right behaviour for a caller who wants it — a two-group panel
    reporting one planned comparison should say the comparison was made and failed. But a grid
    stacks three of them per cell, and a cell whose three brackets all read "n.s." spends its whole
    headroom announcing an absence: the reader gets three lines, three labels and a taller axis in
    exchange for nothing, and the panels that DO have a finding are pushed down to match.

    The cost is that a missing bracket is ambiguous between "tested and null" and "never tested",
    so the figure has to say which — both grids state the rule in their subtitle. That is one line
    for the whole figure against three labels per cell.
    """
    return [(seat[a], seat[b], p) for a, b, p in tests if p < ALPHA]


def _ticks(ax, data: dict[str, np.ndarray], count: int = 2) -> None:
    ax.set_xticks(range(count))
    ax.set_xticklabels(
        [f"Control\nn={len(data['control'])}", f"Treated\nn={len(data['treated'])}"], fontsize=15
    )
    ax.tick_params(axis="x", length=0)
    baseline(ax)


def two_group_violin() -> None:
    """The whole panel in one call: bodies, dots, IQR, mean line, median dot, means, bracket."""
    data = two_groups()
    fig, ax = plt.subplots(figsize=(7.6, 8.0))
    group_violins(ax, _groups(data), comparisons=[(0.0, 1.0, 0.004)])
    ax.set_ylabel("Measurement (units)", fontsize=18, fontweight="bold", labelpad=8)
    _ticks(ax, data)
    header_bottom = titled(
        fig,
        "Two groups",
        subtitle="one call: marks, limits, printed means, bracket",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "01_two_group_violin")


def display_units() -> None:
    """A stored unit that is not the printed one — the axis and the means must agree."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.0))
    for ax, (scale, thousands, label, decimals, values) in zip(
        axes,
        [
            (1e3, False, "Concentration (ppb)", 2, concentration_ppm()),
            (1.0, True, "Relaxation time (ms)", 0, large_magnitudes()),
        ],
        strict=True,
    ):
        group_violins(
            ax,
            _groups(values),
            comparisons=[(0.0, 1.0, 0.02)],
            display_scale=scale,
            thousands_separator=thousands,
            mean_decimals=decimals,
        )
        value_ticks(ax, count=4, scale=scale, thousands_separator=thousands)
        # Re-applied after choosing ticks: `value_ticks` places them across the whole axis, and the
        # top of this one is room held open for the bracket, not values anything can take. No
        # argument — it measures what was drawn, which is not the same as the largest observation.
        ticks_over_data(ax)
        ax.set_ylabel(label, fontsize=17, fontweight="bold", labelpad=8)
        _ticks(ax, values)
    header_bottom = titled(
        fig,
        "Stored units are not printed units",
        subtitle="left: values held in ppm, drawn in ppb — the axis and the means agree",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "02_display_units")


def power_spectrum() -> None:
    """A power spectral density: three conditions on log-log, with a 95% CI around each.

    The companion to the spectrogram at the end of the gallery — the same measurement with time
    integrated out, which is the form most spectra are actually reported in.

    THE SERIES PALETTE, NOT THE CONDITION TINTS, and the difference is the whole colour argument in
    one figure. A condition grid may keep a pair that collapses under a deficiency, because every
    violin sits over its own labelled tick and nothing is identified by colour alone. This panel
    names its conditions in a LEGEND, so colour is the only key a reader has — and the gate refused
    the condition tints here for exactly that reason. `series_colors` is the palette that was
    checked with `indistinguishable_series` for this case.

    The ribbon is a 95% CI OF THE MEAN, and the subtitle says so because the alternative is a
    different claim, not a different style: a 95% interval across subjects is wider by the root of
    n — 4.9 times, here — and a reader who takes one for the other draws the opposite conclusion
    about whether the conditions differ. Computed in log power, since the spectra are log-normal
    and an interval built on the raw values would run negative at the quiet end.
    """
    frequencies, spectra = heart_rate_spectra()
    palette = series_colors(3)
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    for label, low, high, tone in SPECTRAL_BANDS:
        # Tagged as a backdrop, or the gate reads a band's own name as a label sitting on a mark.
        band = ax.axvspan(low, high, color=tone, alpha=0.085, lw=0, zorder=0)
        mark(band, "backdrop")
        edge = ax.axvline(low, color=tone, alpha=0.35, lw=1.0, zorder=1)
        mark(edge, "backdrop")
        ax.text(
            np.sqrt(low * high),  # the midpoint of a log axis is the GEOMETRIC mean
            0.965,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            color=tone,
        )

    for (label, rows), color in zip(spectra.items(), palette, strict=True):
        logs = np.log(rows)
        middle = np.exp(logs.mean(axis=0))
        half = 2.069 * logs.std(axis=0, ddof=1) / np.sqrt(rows.shape[0])  # t, 23 d.f.
        # UNDER the frame: a ribbon that reaches the left limit is drawn over the spine, and the
        # gate reports a buried axis. The line stays above it.
        ax.fill_between(
            frequencies,
            middle * np.exp(-half),
            middle * np.exp(half),
            color=color,
            alpha=0.25,
            lw=0,
            zorder=2,
        )
        ax.plot(frequencies, middle, color=color, lw=2.8, label=label, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(frequencies[0], frequencies[-1])

    # THE AXIS HAS TO LOOK LOGARITHMIC, and two decades of bare powers of ten do not: with only
    # 10⁻² and 10⁻¹ labelled, the reader has two marks and no evidence of what happens between
    # them. So the decade ticks are joined by the 2x and 5x steps, LABELLED — 0.005, 0.01, 0.02,
    # 0.05, 0.1, 0.2, 0.5 — whose visibly UNEVEN spacing is the log structure made legible, and by
    # unlabelled minor ticks at every remaining multiple, which say the same thing at finer grain.
    # Plain decimals rather than scientific, because 5e-3 beside 10⁻² asks the reader to compare
    # a mantissa and an exponent at once. The axis label carries the word as well: a figure should
    # not depend on the reader inferring its scale from the tick spacing alone.
    decades_and_halves = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    ax.xaxis.set_major_locator(FixedLocator(decades_and_halves))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: typeset(f"{value:g}")))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    # The value axis is logarithmic too, and saying so on one axis and not the other reads as a
    # claim that they differ. Marks only — labelling both sets would crowd a panel this size.
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="minor", length=3.5, width=0.9, color=MUTED_INK)
    ax.tick_params(axis="both", which="major", length=6.0, width=1.2)

    ax.set_xlabel("Frequency (Hz), log scale", fontsize=16, fontweight="bold", labelpad=8)
    ax.set_ylabel("Power (ms² per Hz), log scale", fontsize=16, fontweight="bold", labelpad=8)
    hairline_grid(ax, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    legend_pill(ax, loc="lower left", fontsize=14)
    header_bottom = titled(
        fig,
        "Where the power sits",
        subtitle="invented spectra, 24 subjects; the ribbon is a 95% CI of the geometric mean",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "06_power_spectrum")


def grouped_bars() -> None:
    """Signed bars: a negative one grows downward, so its label goes below it."""
    profile = domain_profile()
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    bar_panel(
        ax,
        [
            Series("Control", profile["control"][0], CONTROL, profile["control"][1]),
            Series("Treated", profile["treated"][0], TREATED, profile["treated"][1]),
        ],
        list(DOMAINS),
    )
    ax.set_ylabel("Domain score (z)", fontsize=17, fontweight="bold")
    zero_baseline(ax)
    legend_pill(ax, loc="upper left", ncol=2)
    header_bottom = titled(
        fig,
        "Grouped bars",
        subtitle="value labels clear the whisker cap, on the correct side",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "08_grouped_bars")


def bars_with_reference() -> None:
    """One series, per-bar colour, an asymmetric CI, and a level to compare against."""
    means, errors = expected_goals()
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    bar_panel(
        ax,
        [Series("xG", means, ["#C9C6BB", "#8FA9C9", "#5C82B5", "#2E7CE0"], errors)],
        list(FORMATIONS),
        reference=(LEAGUE_AVERAGE_XG, "league average"),
        reference_side="right",
    )
    ax.set_ylabel("Expected goals per match", fontsize=17, fontweight="bold")
    ax.set_ylim(0, 2.15)
    header_bottom = titled(
        fig,
        "Expected goals by formation",
        subtitle="invented numbers; the label masks the reference line it crosses",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "07_bars_with_reference")


def horizontal() -> None:
    """Category names too long to sit under a tick — the case horizontal exists for."""
    profile = domain_profile()
    names = [
        "autonomic burden",
        "sleep disturbance",
        "cognitive complaint",
        "post-exertional malaise",
        "pain interference",
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(18.5, 6.4))
    group_violins(
        left, _groups(skewed_groups()), comparisons=[(0.0, 1.0, 0.004)], orientation="horizontal"
    )
    left.set_yticks([0, 1])
    left.set_yticklabels(["Control", "Treated"], fontsize=15)
    left.set_xlabel("Measurement (units)", fontsize=16, fontweight="bold")
    bar_panel(
        right,
        [
            Series("Control", profile["control"][0], CONTROL, profile["control"][1]),
            Series("Treated", profile["treated"][0], TREATED, profile["treated"][1]),
        ],
        names,
        orientation="horizontal",
    )
    # Room on the left: a negative bar's label sits at its left end, which is where the category
    # labels are. Without the margin they touch, and the QC gate says so.
    right.set_xlim(right.get_xlim()[0] * 1.45, right.get_xlim()[1])
    right.set_xlabel("Domain score (z)", fontsize=16, fontweight="bold")
    right.axvline(0.0, color=INK, lw=2.0, zorder=4)  # the same value, said by name
    right.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, -0.26), ncol=2)
    titled(fig, "Laid on their side", subtitle="orientation='horizontal' on any mark or panel")
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.09, right=0.985, wspace=0.72)
    render(fig, "10_horizontal")


def split_violin_pair() -> None:
    """Two measurements of the same thing per category, back to back on a shared spine."""
    first, second = paired_sensors()
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    split_violins(
        ax,
        list(BLOCKS),
        first,
        second,
        left_color="#C1685C",
        right_color="#5C8C7A",
        mean_decimals=1,
    )
    ax.set_ylabel("Interval (ms)", fontsize=17, fontweight="bold")
    baseline(ax)
    header_bottom = titled(
        fig,
        "The same quantity, measured two ways",
        subtitle="a shared spine, so a pair that agrees looks like it agrees",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "05_split_violins")


def headline_bars() -> None:
    """The headline comparison: paired metrics, a highlighted arm, a reference band."""
    first, second = headline_arms()
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    bar_panel(
        ax,
        [
            Series("metric A", first, "#5C82B5", np.full(4, 0.035)),
            Series("metric B", second, "#C9C6BB", np.full(4, 0.030)),
        ],
        list(ARM_LABELS),
        rounded=True,
        highlight=3,
        emphasis=3,
        value_format="{:.3f}",
    )
    ax.set_ylabel("Score", fontsize=17, fontweight="bold")
    ax.set_ylim(0, 0.86)
    baseline(ax)
    legend_pill(ax, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    header_bottom = titled(
        fig,
        "Headline comparison",
        subtitle="rounded bars, the arm in question shaded, the reference as a band not a line",
    )
    fit_under_header(fig, header_bottom, bottom=0.06)
    render(fig, "09_headline_bars")


def coupling_triangle() -> None:
    """Three variables taken two at a time, each pair's correlation on one shared scale.

    The figure the strip is for: pooled, every leg trends up; within either group, none of them
    does. A reader given only the pooled number would call these three variables one thing.
    """
    data = coupled_variables()
    axis = {
        "marker": "Marker (ppb)",
        "mass": "Mass index",
        "burden": "Burden score (z)",
    }
    rows = (
        ("control", CONTROL, CONTROL_EDGE, "Control"),
        ("treated", TREATED, TREATED_EDGE, "Treated"),
    )

    def leg(first: str, second: str) -> Leg:
        estimates = []
        for key, fill, edge, label in rows:
            value, interval, p = correlation_with_interval(data[key][first], data[key][second])
            estimates.append(Estimate(label, value, interval, edge, fill=fill, p=p))
        pooled_x = np.concatenate([data[key][first] for key, *_rest in rows])
        pooled_y = np.concatenate([data[key][second] for key, *_rest in rows])
        value, interval, p = correlation_with_interval(pooled_x, pooled_y)
        estimates.append(Estimate("Pooled", value, interval, INK, p=p))
        return Leg(
            x_label=axis[first],
            y_label=axis[second],
            clouds=tuple(
                Cloud(data[key][first], data[key][second], fill, edge, label)
                for key, fill, edge, label in rows
            ),
            estimates=tuple(estimates),
        )

    legs = (leg("marker", "mass"), leg("mass", "burden"), leg("marker", "burden"))
    fig = plt.figure(figsize=(16.5, 8.2))
    coupling_panels(
        fig,
        legs,
        estimate_axis_label="Rank correlation (dot) with bootstrap 95% CI (bar)",
    )
    header_bottom = titled(
        fig,
        "Does one marker stand in for the others?",
        subtitle="Three made-up variables, taken two at a time, pooled and within each group",
    )
    figure_points = fig.get_figheight() * 72.0
    fig.subplots_adjust(
        left=0.085,
        # The star column sits just outside each panel, so the rightmost panel needs a lane at the
        # page edge to put it in. At 0.985 the stars of the third leg ran 17 px off the canvas.
        right=0.975,
        top=header_bottom - 34.0 / figure_points,
        bottom=62.0 / figure_points,
    )
    render(fig, "12_coupling_panels")


def the_gate() -> None:
    """The one claim the README leads on, and the only figure here that is not saved by `save`.

    A bar panel with three ordinary defects — the value labels planted on the bars, a threshold
    label lying across its own rule — beside the same panel after `repair` has been over it. The
    complaint counts in the subtitle come from `audit`, so they cannot drift from what is drawn.

    THE LEFT PANEL IS SUPPOSED TO FAIL, which is why this uses `fig.savefig` where every other
    example uses `save`. `save` runs the gate and refuses, correctly: it cannot know that half the
    figure is an exhibit. It is the one bypass in the gallery and it is asserted rather than
    trusted — `_assert_shows_the_defect` below checks the left panel is still refused on its own,
    so this cannot quietly become a picture of two clean panels.

    BOTH panels are built broken and `repair` is run on the whole figure; the left panel's labels
    are then put back where they started. Repairing one axes of a two-axes figure is not a thing
    `repair` offers — it takes a figure — and building the right panel "already correct" by hand
    would make the caption a claim rather than a demonstration. Here the right panel's labels are
    where `repair` actually moved them.
    """
    from ogviz.qc import audit
    from ogviz.qc.repair import repair

    values = np.array([0.34, 0.47, 0.58, 0.64])
    target = 0.52

    def build(ax) -> None:
        ax.bar(range(len(values)), values, color=series_colors(3)[0], width=0.62, zorder=2)
        ax.axhline(target, color=MUTED_INK, lw=2.0, zorder=3)
        for index, value in enumerate(values):
            ax.text(
                index,
                value * 0.55,  # planted ON the bar, which is the defect
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=INK,
                zorder=4,
            )
        ax.text(1.5, target, "target", ha="center", va="center", fontsize=14, color=MUTED_INK)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(list(ARM_LABELS), fontsize=13)
        ax.set_ylim(0.0, 0.78)
        hairline_grid(ax, axis="y")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for ax in axes:
        build(ax)
    fig.canvas.draw()
    before = audit(fig)
    as_drawn = [(label, label.get_position()) for label in axes[0].texts]
    resolved = repair(fig)
    for label, where in as_drawn:  # the left panel goes back to being the "before"
        label.set_position(where)

    axes[0].set_title("as drawn", fontsize=17, fontweight="bold", pad=12)
    axes[1].set_title("after repair()", fontsize=17, fontweight="bold", pad=12)
    header_bottom = titled(
        fig,
        "The gate reads the pixels, then moves the ink",
        subtitle=(
            f"{len(before) // 2} complaints about the panel on the left; "
            f"repair() resolved {len(resolved) // 2}"
        ),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, header_bottom))
    _assert_shows_the_defect(build)
    # `reproducible_metadata`, because `save` is what usually drops the write date and this is the
    # one figure that does not go through it. Without it the committed SVG carries a live timestamp
    # and changes on every render, which is precisely the un-diffable gallery that helper exists to
    # prevent — caught here by `git status` after a no-op re-render.
    for suffix, kwargs in ((".png", {"dpi": 200}), (".svg", {})):
        path = OUT / f"13_the_gate{suffix}"
        fig.savefig(path, metadata=reproducible_metadata(path), **kwargs)
    plt.close(fig)


def _assert_shows_the_defect(build) -> None:
    """The premise: a panel built by `build` really is refused on its own.

    Without this the figure would keep rendering after a change that made the "before" panel clean —
    two identical panels, a subtitle counting zero, and nothing to see. The house style is autouse
    in the tests and moves layout, so a figure written to be broken is exactly the thing that
    quietly stops being broken.
    """
    from ogviz.qc import audit

    probe, ax = plt.subplots(figsize=(6.6, 5.4))
    build(ax)
    probe.canvas.draw()
    complaints = audit(probe)
    plt.close(probe)
    if not complaints:
        message = "the 'as drawn' panel no longer fails the gate — the figure shows nothing"
        raise AssertionError(message)


def effect_matrix() -> None:
    """Every act against every judge, coloured by sign and magnitude, with the number in the cell.

    Three things a hand-rolled version of this gets wrong, all visible here: the scale is symmetric
    about zero so equal effects in opposite directions colour equally; each number takes its colour
    from the cell behind it, which is the only way one is legible at both ends of the map; and the
    unjudged cell is drawn as missing rather than shaded like a measured zero.
    """
    from examples.data import ACTS, JUDGES, act_scores

    effects, p_values = act_scores()
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    effect_heatmap(
        ax,
        effects,
        row_labels=ACTS,
        column_labels=JUDGES,
        p_values=p_values,
        row_dividers=[4],
    )
    header_bottom = titled(
        fig,
        "Who wins on what",
        subtitle="invented effects; the dash is an act that criterion never judged",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, header_bottom))
    render(fig, "14_effect_heatmap")


def stft_spectrogram() -> None:
    """Time against frequency, coloured by power — and the key that makes the colour readable.

    The template for a Short-Time Fourier Transform figure. Three readings the panel exists to
    support are all present: the horizontal band is a constant tone that STOPS partway through, the
    diagonal is a sweep crossing it, and the vertical line is one broadband instant.

    The colour scale is not decoration here the way it can be argued to be on `effect_heatmap`,
    which prints its number in every cell. A spectrogram prints no numbers at all, so without the
    bar a reader cannot tell 70 dB of dynamic range from 20, and the figure stops being a
    measurement. That is why it is drawn by default and why the floor is stated in the caption:
    everything below 70 dB down is one colour, and saying so is the difference between a noise floor
    and an empty one.
    """
    from examples.data import short_time_spectrum

    power_db, times, frequencies = short_time_spectrum()
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    spectrogram(ax, power_db, times=times, frequencies=frequencies)
    header_bottom = titled(
        fig,
        "What is present, and when",
        subtitle="an invented signal; a 32 ms Hann window, floored 70 dB below the peak",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, header_bottom))
    render(fig, "18_stft_spectrogram")


def rounds_slopegraph() -> None:
    """Three disciplines across four rounds: the SHAPE of each line is the comparison.

    A slopegraph rather than a line panel because the x axis is a sequence of stages, not a
    quantity — so the stages are evenly spaced whatever they represent, and every series carries a
    marked point at every stage so a crossing says which series went where.
    """
    from examples.data import ROUNDS, scores_by_round
    from ogviz.theme import SERIES

    scores = scores_by_round()
    strands = [
        Strand(name, values, SERIES[index], spread)
        for index, (name, (values, spread)) in enumerate(scores.items())
    ]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    # Named at their ends rather than in a legend: a legend costs the reader a lookup per line, and
    # the labels are placed by solving all three together, so converging series do not pile up.
    slopegraph(ax, strands, ROUNDS, end_labels=True)
    ax.set_ylabel("Judge score, centred", fontsize=15, fontweight="bold", labelpad=8)
    header_bottom = titled(
        fig,
        "Who improves, and who is found out",
        subtitle="invented scores; the band is the spread across judges",
    )
    # The end labels live outside the panel, so the panel has to stop before the page does.
    fig.tight_layout(rect=(0.0, 0.0, 0.86, header_bottom))
    render(fig, "15_slopegraph")


def controlled_comparison() -> None:
    """A head-to-head where one bar is NOT comparable with the others, and the axis says so.

    Four things a plain grouped bar panel cannot do, all of them about saying what the figure means
    rather than what the numbers are: the reference stands apart on the category axis; a backdrop
    covers only the arms that ARE comparable; the ceiling line is drawn over those arms alone,
    because it says nothing about the reference beside them; and the axis carries three rows —
    which metric, which arm, and what each was trained on.
    """
    from examples.data import ARM_COLORS, ARMS, arm_comparison

    metrics = arm_comparison()
    # The reference sits a gap away from the three arms it is not comparable with.
    positions = [0.0, 1.0, 2.0, 3.45]
    fair = (0, 2)
    ceiling = metrics["Fine score"][0][-1]

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    bar_panel(
        ax,
        [
            Series(
                "Coarse",
                metrics["Coarse score"][0],
                [tint(color, strength=0.45) for color in ARM_COLORS],
            ),
            Series("Fine", metrics["Fine score"][0], list(ARM_COLORS)),
        ],
        list(ARMS),
        positions=positions,
        highlight=fair,
        show_values=False,
    )
    for name, offset in (("Coarse score", -0.155), ("Fine score", 0.155)):
        values, spread = metrics[name]
        error_bars(
            ax,
            [place + offset for place in positions],
            values,
            [low for low, _high in spread],
            [high for _low, high in spread],
        )
    reference_line(
        ax,
        ceiling,
        "reference level",
        span=(positions[fair[0]] - 0.46, positions[fair[1]] + 0.46),
    )
    label_rows(
        ax,
        positions,
        [
            ["coarse / fine"] * len(ARMS),
            list(ARMS),
            ["the three on the left saw the same data"],
        ],
        sizes=[9.5, 12.0, 10.5],
        weights=["normal", "bold", "bold"],
        colors=[None, list(ARM_COLORS), None],
    )
    ax.set_xticks([])
    ax.set_ylabel("Score", fontsize=15, fontweight="bold", labelpad=8)
    ax.set_ylim(0.0, 0.86)
    header_bottom = titled(
        fig,
        "One of these is not like the others",
        subtitle="invented scores; the reference stands apart",
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, header_bottom))
    render(fig, "17_controlled_comparison")


def violin_grid() -> None:
    """One violin panel per region, EACH ON ITS OWN SCALE — and the figure says so.

    This grid used to share one value range across all four panels. Measured with
    `layout.density.panel_emptiness`, that left the two quiet nations 39% and 41% empty at the top:
    the shared limit has to contain every panel's drawn ink, and one panel's three-bracket stack
    reaches far above data the others never approach. Roughly a third of that emptiness was bracket
    headroom only one panel used; the rest was the honest cost of one scale.

    THE COST OF FITTING EACH PANEL IS THAT THE PANELS NO LONGER COMPARE. Four violins the same size
    on the page now span different ranges, and a reader who compares their shapes without reading
    the ticks gets the wrong answer. That is not a thing a figure may leave implicit, so the
    subtitle states it — the same move `controlled_comparison` makes when one of its bars is not
    comparable with the rest.

    `share_value_limits` is the other answer and is the right one whenever the panels are meant to
    be read against each other. It is no longer exercised by any example; its tests still cover it.
    """
    regions = bending_by_nation()
    subject = identity_colors(COHORT)  # one colour per bender, the same in all four panels

    # NOT sharey either: a shared matplotlib axis makes the last panel drawn win, so a panel whose
    # bracket stack is taller gets clipped by a neighbour. Each panel fits itself and keeps it.
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.5))
    for ax, name in zip(axes.flat, NATIONS, strict=True):
        levels, tests = regions[name]
        seat = {level: float(index) for index, level in enumerate(BENDING_LEVELS)}
        group_violins(
            ax,
            [
                (seat[level], levels[level], fill, page_color())
                for level, fill in zip(BENDING_LEVELS, CONDITION_TINTS, strict=True)
            ],
            comparisons=_significant(tests, seat),
            bracket_kwargs={"fontsize": GRID_STAR_SIZE},
            # Every violin holds the same cohort in the same order, so one list of colours serves
            # all three and a reader can follow one bender from novice to master.
            point_colors=[subject] * len(BENDING_LEVELS),
            outline_violins=True,
            violin_kwargs={"alpha": 0.28},
            # The rim is the page rather than a darker tint: at this dot size a dark rim is under a
            # pixel wide, so it antialiases into a grey halo and the cloud reads as smudged instead
            # of as separate dots. A page-coloured rim separates them without darkening them.
            point_kwargs={"edge_width": 0.5, "size": 26.0},
            categories=list(BENDING_LEVELS),
            category_fontsize=15,
            # Turned down from the house default: this row appears in all four panels, and at full
            # weight the twelve numbers become the loudest ink on a figure about twelve shapes.
            mean_fontsize=15,
            mean_weight="normal",
            # PINNED across the panels even though the scales are not. `printed_means` picks its
            # decimals from the largest value in ITS OWN row, and the four rows came out as
            # 0.32 / 0.133 / 0.011 / 0.0120 — four spellings of one quantity, with the Fire Nation's
            # noise-level row reading as the most precise measurement on the figure. The scales may
            # differ; how a number is written is still one decision for the whole grid.
            mean_decimals=2,
        )
        ax.set_title(name, fontsize=17, fontweight="bold", pad=14)
    for ax in axes[:, 0]:
        ax.set_ylabel("Bending strength (arbitrary)", fontsize=16, fontweight="bold", labelpad=8)
    header_bottom = titled(
        fig,
        "Three levels of training, four nations",
        subtitle=(
            "each panel on its own scale \u2014 the panels do not compare; "
            "brackets only where p < 0.05"
        ),
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "03_violin_grid")


def effort_curves() -> None:
    """Measured points over a log cost axis, with the value axis cut and the cut marked."""
    ladders = effort_ladders()
    palette = series_colors(3)
    lines = [
        Line(name, x, y, palette[index], order=index)
        for index, (name, (x, y)) in enumerate(list(ladders.items())[:3])
    ]
    baseline_x, baseline_y = ladders["Baseline"]
    lines.append(Line("Baseline", baseline_x, baseline_y, MUTED_SERIES, muted=True))

    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    line_panel(
        ax,
        lines,
        x_ticks=[0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
        money=True,
        legend_below=True,
    )
    ax.set_ylabel("Pass rate (%)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Cost per task (USD, log scale)", fontsize=16, fontweight="bold")
    broken_zero(ax, floor=value_floor(lines))
    header_bottom = titled(
        fig,
        "Score by effort level",
        subtitle="a made-up benchmark, four made-up systems",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "11_effort_curves")


def comparison_table() -> None:
    """A table drawn as a figure, so it ships into a slide with the same type as the charts.

    Carries a caption, which is off everywhere else: this is the case captions exist for, a figure
    that travels as a file and has to say where its numbers came from. Here that means admitting
    they came from nowhere.

    The shape is the one a comparison table is for. One column dominates, so it is highlighted and
    holds most of the shaded cells — but not all of them, and two of its columns are near-identical,
    which is a finding the table should let the reader see rather than hide.
    """
    rows = [
        Row(
            "Power level",
            (
                Cell("28,000,000", best=True),
                Cell("150,000"),
                Cell("155,000"),
                Cell("152,000"),
                Cell("1,400,000"),
            ),
            sub="scouter, reads high and then explodes",
        ),
        Row(
            "Cool level",
            (Cell("99.4%", best=True), Cell("71.0%"), Cell("73.0%"), Cell("88.0%"), Cell("64.0%")),
            sub="fan poll, entirely invented",
        ),
        Row(
            "Speed",
            (
                Cell("Mach 402", best=True),
                Cell("Mach 61"),
                Cell("Mach 63"),
                Cell("Mach 62"),
                Cell("Mach 140"),
            ),
            sub="Snake Way, one lap",
        ),
        Row(
            "Ki control",
            (
                Cell("96%", sub="effortless", best=True),
                Cell("61%", sub="strained"),
                Cell("63%", sub="strained"),
                Cell("74%", sub="furious"),
                Cell("91%", sub="calm"),
            ),
            sub="share of ki that reaches the target",
            height=1.4,
        ),
        Row(
            "Hair",
            (
                Cell("waist length", best=True),
                Cell("spiked"),
                Cell("spiked"),
                Cell("spiked"),
                Cell("spiked"),
            ),
            sub="a real difference, unlike the two beside it",
        ),
        Row(
            "Stamina drain",
            (
                Cell("14%/min"),
                Cell("3%/min", best=True),
                Cell("9%/min"),
                Cell("11%/min"),
                Cell("5%/min"),
            ),
            sub="lower is better — the one row the fourth form loses",
        ),
        Row(
            "Transformation scream",
            (Cell("42 s", best=True), Cell("18 s"), Cell("19 s"), Cell("31 s"), Cell("11 s")),
            sub="uninterrupted, measured at the cliff edge",
        ),
        Row(
            "Tail",
            (Cell("yes", best=True), Cell(), Cell(), Cell(), Cell()),
            sub="the em dash is 'measured, absent' — not zero",
        ),
    ]
    fig, ax = plt.subplots(figsize=(14.0, 8.4))
    table_panel(
        ax,
        ["Goku SSJ4", "Goku SSJ1", "Goku SSJ2", "Vegeta SSJ2", "Gohan SSJ2"],
        rows,
        highlight=0,
        highlight_color="#E8552D",
    )
    caption(
        fig,
        "Source: nothing. Every number here is invented to show the layout. Notes: the em dash "
        "marks a measurement that was not taken, which is not the same as a zero. Shaded cells are "
        "the strongest value in their row, including the one row where that is not the "
        "highlighted column. Super Saiyan 1 and 2 sit within a few percent of each other on "
        "every row but hair.",
        heading="Super Saiyan forms, compared",
    )
    render(fig, "16_comparison_table")


# Grouped by what the panel IS, and numbered to match. A gallery ordered by the date each example
# was written asks the reader to hold twelve unrelated things in mind; ordered by kind, the violin
# panels are one idea with five variations.
def violin_grid_tall() -> None:
    """THE CONDITION GRID: six measures of one company at the same three stages of the road.

    The other grid holds one quantity and varies the panel's subject; this one varies the QUANTITY
    and holds the comparison fixed, which is the arrangement that lets a reader read straight down
    the column. Every panel asks the same question, so the answer is the only thing that changes.

    NO SHARED SCALE HERE, and that is the difference the two grids exist to show. These measures
    are counts, a weight, hours and a step total in five figures: put them on one axis and five
    panels flatten into a line. So each panel carries its own scale and says its own unit in the y
    label — short, because "hours" is the whole label a panel needs when its title is already
    "Sleep". A long label repeated six times is six copies of the same sentence.
    """
    measures = journey_measures()
    subject = identity_colors(COHORT)  # one colour per walker, the same in all six panels
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 15.5))
    seat = {stage: float(index) for index, stage in enumerate(STAGES_OF_THE_ROAD)}
    for ax, (title, (unit, stages, tests)) in zip(axes.flat, measures.items(), strict=True):
        group_violins(
            ax,
            [
                (seat[stage], stages[stage], fill, page_color())
                for stage, fill in zip(STAGES_OF_THE_ROAD, CONDITION_TINTS, strict=True)
            ],
            comparisons=_significant(tests, seat),
            bracket_kwargs={"fontsize": GRID_STAR_SIZE},
            point_colors=[subject] * len(STAGES_OF_THE_ROAD),
            outline_violins=True,
            violin_kwargs={"alpha": 0.28},
            point_kwargs={"edge_width": 0.5, "size": 26.0},
            categories=list(STAGES_OF_THE_ROAD),
            category_fontsize=14,
            mean_fontsize=15,
            mean_weight="normal",
        )
        # Per-panel scales mean per-panel TICKS, and the step panel is why: matplotlib's default
        # formatter writes 10000, which the gate refuses. Then `ticks_over_data`, because the top
        # of each panel is room held open for the bracket stack rather than values.
        value_ticks(ax, count=4)
        ticks_over_data(ax)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.set_ylabel(unit, fontsize=14, labelpad=8)
    header_bottom = titled(
        fig,
        "One company, six measures, three stages of the road",
        subtitle="invented numbers, own scale per panel; a bracket is drawn only where p < 0.05",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "04_violin_grid_tall")


EXAMPLES = (
    # Violins: one panel, then two, then a grid, then a taller grid, then the two special marks.
    # The order is the order a reader builds the idea up in, not the order these were written.
    two_group_violin,
    display_units,
    violin_grid,
    violin_grid_tall,
    split_violin_pair,
    power_spectrum,
    # Bars: one series, then two, then a decorated one, then the other orientation.
    bars_with_reference,
    grouped_bars,
    headline_bars,
    horizontal,
    # Relationships over a continuous axis: one panel of series, then the composite built from them.
    effort_curves,
    coupling_triangle,
    # What a family of tests leaves once it is treated as a family.
    the_gate,
    # Matrices: many effects at once, then the same shape set as a table.
    effect_matrix,
    # A sequence of stages, where the shape of each series is the comparison.
    rounds_slopegraph,
    comparison_table,
    controlled_comparison,
    # Time against frequency: the one panel whose whole quantity is in the colour.
    stft_spectrogram,
)


def main() -> None:
    use_house_style()
    for example in EXAMPLES:
        example()
        print(f"  {example.__name__}")
    print(f"{len(EXAMPLES)} examples -> {OUT}")


if __name__ == "__main__":
    main()
