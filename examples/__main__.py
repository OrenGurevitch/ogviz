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
# `macosx`, which lays text out differently from `Agg`: `06_stacked_brackets` came out with SEVEN
# y-ticks under `macosx` and FOUR under `Agg`, from identical code and identical data, because the
# text metrics moved the axis limits enough to change what the locator chose.
#
# Every test runs under `Agg`. So without this line the gallery was rendered by something the suite
# never exercises, and could not be reproduced on a headless machine or in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from examples.data import (
    ARM_LABELS,
    BLOCKS,
    CONTROL,
    CONTROL_EDGE,
    DOMAINS,
    FORMATIONS,
    LEAGUE_AVERAGE_XG,
    NATIONS,
    STAGES_OF_THE_ROAD,
    TREATED,
    TREATED_EDGE,
    bending_by_nation,
    burden_by_stage,
    concentration_ppm,
    correlation_with_interval,
    coupled_variables,
    domain_profile,
    effort_ladders,
    expected_goals,
    headline_arms,
    large_magnitudes,
    paired_sensors,
    skewed_groups,
    two_groups,
)
from ogviz import (
    INK,
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
    label_rows,
    legend_pill,
    line_panel,
    multiplicity_ladder,
    reference_line,
    save,
    series_colors,
    share_value_limits,
    slopegraph,
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
from ogviz.panels.lines import MUTED_SERIES

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


def stacked_brackets() -> None:
    """Three comparisons: each star sits against its own line, not midway to the next."""
    rng = np.random.default_rng(9)
    palette = ["#2E7CE0", "#EFA607", "#14A97C"]
    groups = [
        (float(i), rng.normal(i * 0.8, 0.9, 30), colour, "#333333")
        for i, colour in enumerate(palette)
    ]
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    group_violins(
        ax,
        groups,
        comparisons=[(0.0, 1.0, 0.03), (1.0, 2.0, 0.006), (0.0, 2.0, 0.0002)],
    )
    ax.set_ylabel("Measurement (units)", fontsize=17, fontweight="bold", labelpad=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["low", "mid", "high"], fontsize=15)
    ax.tick_params(axis="x", length=0)
    baseline(ax)
    header_bottom = titled(
        fig,
        "Stacked brackets",
        subtitle="each star is anchored by its ink, not its box",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "06_stacked_brackets")


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
    right.axvline(0.0, color="#141413", lw=2.0, zorder=4)
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


def multiplicity_ladder_example() -> None:
    """Fifteen tests, eight of them under 0.05, and what a correction leaves of that.

    The panel a table of stars cannot replace: with fifteen independent tests at 0.05, the chance of
    at least one false positive is about 54%, so eight stars is not eight findings. Drawing the
    Benjamini-Hochberg ramp shows WHY its cutoff falls where it does: the rule takes the largest
    rank whose p clears the ramp, so points above the line at smaller ranks are declared too.
    """
    from examples.data import tournament_p_values

    names, p_values = tournament_p_values()
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    declared = multiplicity_ladder(ax, p_values, labels=names)
    header_bottom = titled(
        fig,
        "Eight stars, three findings",
        subtitle=(
            f"{sum(p < 0.05 for p in p_values)} of {len(p_values)} invented trials fall under "
            f"0.05; Benjamini-Hochberg declares {declared}"
        ),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, header_bottom))
    render(fig, "13_multiplicity_ladder")


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
    """One violin panel per region, on one shared scale.

    The composition case: four panels have to be comparable to each other, which means one value
    range across all of them and the group labels stated once rather than under every panel.
    """
    regions = bending_by_nation()

    # NOT sharey: with a shared axis every panel's own fit writes to the same limits, so the last
    # one drawn wins and a panel whose bracket stack is taller gets clipped by a neighbour. Each
    # panel fits itself, then `share_value_limits` puts them on the union of those answers.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.0))
    # Stacked panels share a boundary, so the bottom tick of one row sits a few pixels from the top
    # tick of the row below and the two read as one number. Pruning the end ticks is the same fix
    # the estimate strips use, and costs nothing here: both extremes fall in the padding.
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    for ax, name in zip(axes.flat, NATIONS, strict=True):
        control, treated, p = regions[name]
        group_violins(
            ax,
            [(0.0, control, CONTROL, CONTROL_EDGE), (1.0, treated, TREATED, TREATED_EDGE)],
            comparisons=[(0.0, 1.0, p)],
            categories=["Novice", "Master"],
            category_fontsize=15,
        )
        ax.set_title(name, fontsize=17, fontweight="bold", pad=14)
    # Each panel measured the headroom its own bracket needs; the shared scale is the union of
    # those answers rather than a number picked in advance.
    share_value_limits(axes.flat)
    for ax in axes[:, 0]:
        ax.set_ylabel("Bending strength (arbitrary)", fontsize=16, fontweight="bold", labelpad=8)
    header_bottom = titled(
        fig,
        "Novice against master benders",
        subtitle="four nations on one scale, so the panels read against each other",
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
    """Six panels, two columns by three rows — the shape a condition grid takes.

    Same rules as the square grid and a different arrangement: every panel fits its own bracket,
    then one shared scale and one line of printed means across all six. Two columns rather than
    three because the panel has to stay wide enough for its two violins and their labels.
    """
    stages = burden_by_stage()
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 15.0))
    for ax, stage in zip(axes.flat, STAGES_OF_THE_ROAD, strict=True):
        companion, bearer, p = stages[stage]
        group_violins(
            ax,
            [(0.0, companion, CONTROL, CONTROL_EDGE), (1.0, bearer, TREATED, TREATED_EDGE)],
            comparisons=[(0.0, 1.0, p)],
            categories=["Companion", "Ring-bearer"],
            category_fontsize=14,
        )
        ax.set_title(stage, fontsize=16, fontweight="bold", pad=12)
    share_value_limits(axes.flat)
    for ax in axes[:, 0]:
        ax.set_ylabel("Burden (arbitrary)", fontsize=15, fontweight="bold", labelpad=8)
    header_bottom = titled(
        fig,
        "The weight of the Ring along the road",
        subtitle="invented numbers; the burden grows toward Mordor and is gone at the Havens",
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
    stacked_brackets,
    # Bars: one series, then two, then a decorated one, then the other orientation.
    bars_with_reference,
    grouped_bars,
    headline_bars,
    horizontal,
    # Relationships over a continuous axis: one panel of series, then the composite built from them.
    effort_curves,
    coupling_triangle,
    # What a family of tests leaves once it is treated as a family.
    multiplicity_ladder_example,
    # Matrices: many effects at once, then the same shape set as a table.
    effect_matrix,
    # A sequence of stages, where the shape of each series is the comparison.
    rounds_slopegraph,
    comparison_table,
    controlled_comparison,
)


def main() -> None:
    use_house_style()
    for example in EXAMPLES:
        example()
        print(f"  {example.__name__}")
    print(f"{len(EXAMPLES)} examples -> {OUT}")


if __name__ == "__main__":
    main()
