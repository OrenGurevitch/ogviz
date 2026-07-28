"""Render every example into `examples/out/`. Run with `uv run python -m examples`.

Each function below is one figure and is meant to be read as much as run: it is the shortest
call that produces the thing, so a reader can see what the library does and what stays theirs.
Colours, statistics and units are always the caller's.
"""

from __future__ import annotations

from pathlib import Path

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
    bar_panel,
    baseline,
    broken_zero,
    caption,
    coupling_panels,
    fit_under_header,
    group_violins,
    hairline_grid,
    legend_pill,
    line_panel,
    save,
    series_colors,
    share_value_limits,
    split_violins,
    table_panel,
    ticks_over_data,
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
    hairline_grid(ax)
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
        # top of this one is room held open for the bracket, not values anything can take.
        ticks_over_data(ax, max(float(v.max()) for v in values.values()))
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
    hairline_grid(ax)
    baseline(ax)
    header_bottom = titled(
        fig,
        "Stacked brackets",
        subtitle="each star is anchored by its ink, not its box",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "03_stacked_brackets")


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
    hairline_grid(ax)
    zero_baseline(ax)
    legend_pill(ax, loc="upper left", ncol=2)
    header_bottom = titled(
        fig,
        "Grouped bars",
        subtitle="value labels clear the whisker cap, on the correct side",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "04_grouped_bars")


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
    hairline_grid(ax)
    header_bottom = titled(
        fig,
        "Expected goals by formation",
        subtitle="invented numbers; the label masks the reference line it crosses",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "05_bars_with_reference")


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
    hairline_grid(left, axis="x")
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
    hairline_grid(right, axis="x")
    right.axvline(0.0, color="#141413", lw=2.0, zorder=4)
    right.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, -0.26), ncol=2)
    titled(fig, "Laid on their side", subtitle="orientation='horizontal' on any mark or panel")
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.09, right=0.985, wspace=0.72)
    render(fig, "06_horizontal")


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
    hairline_grid(ax)
    baseline(ax)
    header_bottom = titled(
        fig,
        "The same quantity, measured two ways",
        subtitle="a shared spine, so a pair that agrees looks like it agrees",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "07_split_violins")


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
        reference_band=(0.70, 0.76, "reported agreement range"),
        value_format="{:.3f}",
    )
    ax.set_ylabel("Score", fontsize=17, fontweight="bold")
    ax.set_ylim(0, 0.86)
    hairline_grid(ax)
    baseline(ax)
    legend_pill(ax, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    header_bottom = titled(
        fig,
        "Headline comparison",
        subtitle="rounded bars, the arm in question shaded, the reference as a band not a line",
    )
    fit_under_header(fig, header_bottom, bottom=0.06)
    render(fig, "08_headline_bars")


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
        right=0.985,
        top=header_bottom - 34.0 / figure_points,
        bottom=62.0 / figure_points,
    )
    render(fig, "09_coupling_panels")


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
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Novice", "Master"], fontsize=15)
        ax.set_title(name, fontsize=17, fontweight="bold", pad=14)
    # Each panel measured the headroom its own bracket needs; the shared scale is the union of
    # those answers rather than a number picked in advance.
    share_value_limits(axes.flat)
    for ax in axes[:, 1]:
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_ylabel("Bending strength (arbitrary)", fontsize=16, fontweight="bold", labelpad=8)
    header_bottom = titled(
        fig,
        "Novice against master benders",
        subtitle="four nations on one scale, so the panels read against each other",
    )
    fit_under_header(fig, header_bottom, bottom=0.0)
    render(fig, "10_violin_grid")


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
    render(fig, "12_comparison_table")


EXAMPLES = (
    two_group_violin,
    display_units,
    stacked_brackets,
    grouped_bars,
    bars_with_reference,
    horizontal,
    split_violin_pair,
    headline_bars,
    coupling_triangle,
    violin_grid,
    effort_curves,
    comparison_table,
)


def main() -> None:
    use_house_style()
    for example in EXAMPLES:
        example()
        print(f"  {example.__name__}")
    print(f"{len(EXAMPLES)} examples -> {OUT}")


if __name__ == "__main__":
    main()
