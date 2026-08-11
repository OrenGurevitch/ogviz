"""The QC gate itself: it must pass every shipped example and fail every planted defect.

Each test plants the defect it guards against and checks the gate reports it.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import INK, bracket_stack, group_violins
from ogviz.marks import iqr_box, mean_line, violin
from ogviz.qc import audit, dots_off_the_marks, significance_gaps, stack_spacing
from ogviz.tags import mark

# Assertions here are about RENDERED TEXT, whose metrics differ by machine — so the
# bundled font is pinned. See `ogviz/conftest.py` for why that is opt-in.
pytestmark = pytest.mark.usefixtures("pinned_font")


def _three_groups(seed: int = 9):
    rng = np.random.default_rng(seed)
    palette = ("#2E7CE0", "#EFA607", "#14A97C")
    # INK for the edge. This mirrored the three-group example that stood at gallery slot 06 until a
    # power spectrum replaced it; the fixture is kept because the SHAPE is what it tests — three
    # groups and a full bracket stack — and that shape now lives in the two condition grids.
    return [(float(i), rng.normal(i * 0.8, 0.9, 30), c, INK) for i, c in enumerate(palette)]


def _gallery_names() -> tuple[str, ...]:
    """Read the gallery off the module that renders it, so a new example is covered by existing.

    A second list here was a copy that could go stale in the direction that matters: an example
    added to the README and not to the list would ship unchecked.
    """
    import examples.__main__ as gallery

    return tuple(example.__name__ for example in gallery.EXAMPLES)


@pytest.mark.parametrize("name", _gallery_names())
def test_every_shipped_example_passes_qc(name: str, tmp_path) -> None:
    """The gate runs on the gallery, so a regression cannot reach the README.

    Rendered into a temporary directory, never into `examples/out`. Each example ends in `save()`,
    and this fixture pins DejaVu — so running the tests wrote thirteen DejaVu-rendered images over
    an Arial-rendered gallery and left them staged. `just` hid it by regenerating afterwards; a
    bare `pytest` did not.
    """
    import examples.__main__ as gallery

    original = gallery.OUT
    gallery.OUT = tmp_path
    try:
        getattr(gallery, name)()  # each calls save(), which asserts the gate
    finally:
        gallery.OUT = original


def test_uneven_stars_are_caught() -> None:
    """The reported defect: one star closer to its bracket than the others.

    Reproduced by measuring the ink with the layout box, which is what `TextPath` effectively
    returns for a label containing a space — a lone "*" anchors correctly and every spaced star
    lands 7.7 pt low.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_ylim(0, 10)
    bracket_stack(ax, [(0.0, 1.0, 0.03)], start=5.0, span=10.0)
    star = ax.texts[-1]
    star.set_position((star.get_position()[0], star.get_position()[1] + 0.6))  # nudge one star
    bracket_stack(ax, [(0.0, 2.0, 0.006)], start=7.0, span=10.0)
    assert any("different distances" in hit for hit in significance_gaps(fig))


def test_a_star_with_no_bracket_is_caught() -> None:
    """What the user actually saw: the line clipped away, the star still drawn."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_ylim(0, 10)
    ax.text(0.5, 5.0, "*", ha="center", fontsize=20, fontweight="bold")
    assert any("no bracket under it" in hit for hit in significance_gaps(fig))


def test_a_crowded_stack_is_caught() -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_ylim(0, 10)
    for start in (5.0, 5.05):
        bracket_stack(ax, [(0.0, 1.0, 0.01)], start=start, span=10.0)
    assert any("apart" in hit or "unevenly" in hit for hit in stack_spacing(fig))


def test_dots_on_the_marks_are_caught() -> None:
    values = np.random.default_rng(0).normal(0.0, 1.0, 40)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_ylim(-4, 4)
    violin(ax, values, 0.0, "#2E7CE0")
    iqr_box(ax, values, 0.0)
    mean_line(ax, values, 0.0)
    scatter = ax.scatter(np.zeros(40), values, s=30)
    mark(scatter, "lane", np.full(40, 0.1))
    mark(scatter, "position", 0.0)
    assert dots_off_the_marks(fig) == ["40 dot(s) sit on the central marks"]


def test_a_correct_panel_says_nothing() -> None:
    """The other half of the contract: the gate must be quiet when the figure is right."""
    fig, ax = plt.subplots(figsize=(8, 8))
    group_violins(ax, _three_groups(), comparisons=[(0.0, 1.0, 0.03), (1.0, 2.0, 0.006)])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["low", "mid", "high"])  # a real panel labels its categories
    assert audit(fig) == []


def test_a_spine_under_its_bars_is_caught() -> None:
    """The reported defect: the bars covered the x axis.

    A bar's base lies exactly on the category spine, and matplotlib draws spines at zorder 2.5 —
    under the bars — so the axis survived only in the gaps and read as a broken line.
    """
    from ogviz import Series, bar_panel
    from ogviz.qc import buried_baselines

    _fig, ax = plt.subplots(figsize=(8, 5))
    bar_panel(ax, [Series("s", np.array([0.3, 0.5, 0.7]), ["#8FA9C9"] * 3)], ["a", "b", "c"])
    assert not buried_baselines(ax.figure), "as shipped, the spine is drawn over its bars"

    ax.spines["bottom"].set_zorder(2.5)
    assert buried_baselines(ax.figure), "a spine under its own bars must be reported"


def test_two_minus_signs_in_one_figure_are_caught() -> None:
    """matplotlib writes U+2212 on a tick; `format` writes a hyphen. A figure must not show both."""
    from ogviz.qc import one_minus_sign

    _fig, ax = plt.subplots()
    ax.set_ylim(-1, 1)
    ax.figure.canvas.draw()
    assert not one_minus_sign(ax.figure)

    ax.text(0.0, -0.5, "-0.42")
    ax.figure.canvas.draw()
    assert one_minus_sign(ax.figure), "a hyphen beside U+2212 ticks must be reported"


def test_a_positive_only_series_prints_no_plus() -> None:
    """A leading plus earns its place only when some other bar carries a minus."""
    from ogviz.panels.bars import default_value_format

    assert default_value_format(np.array([0.3, 0.5])).format(0.5) == "0.500"
    assert default_value_format(np.array([-0.3, 0.5])).format(0.5) == "+0.500"


def test_the_checks_run_against_a_figure_that_knows_nothing_about_ogviz(tmp_path) -> None:
    """The detection-only case: a plain matplotlib script, from a project that never imported us.

    Checks that read an ogviz tag stay quiet; the ones that measure geometry and colour do not.
    """
    from ogviz.qc.__main__ import main

    script = tmp_path / "someone_elses_figure.py"
    script.write_text(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "x = np.linspace(0, 10, 100)\n"
        "ax.plot(x, np.sin(x), color='#2E7CE0', label='alpha')\n"
        "ax.plot(x, np.cos(x), color='#8A63D2', label='beta')\n"
        "ax.text(5.0, 0.0, 'a label right on the curves', ha='center')\n"
        "ax.legend()\n"
    )
    assert main([str(script)]) == 1, "this figure has defects and the exit code must say so"


def test_a_clean_figure_exits_zero(tmp_path) -> None:
    from ogviz.qc.__main__ import main

    script = tmp_path / "fine.py"
    script.write_text(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot([0, 1, 2], [0, 1, 0], color='#2E7CE0')\n"
    )
    assert main([str(script)]) == 0


def test_listing_the_checks_needs_no_target() -> None:
    from ogviz.qc.__main__ import main

    assert main(["--list-checks"]) == 0


def test_a_bar_panel_with_error_bars_is_not_read_as_horizontal() -> None:
    """The QC bug that reported five correct brackets as missing.

    `errorbar` keeps its bars and caps in a LineCollection, so a grouped bar panel's only two-point
    line is the zero baseline — constant y — and the orientation vote came back horizontal. Brackets
    were then measured along x, matched none, and every star was reported bracket-less.
    """
    import numpy as np

    from ogviz import bar_panel, bracket_stack
    from ogviz.panels.bars import Series
    from ogviz.qc import orientation_of, significance_gaps

    values = [0.9, 0.6, 0.45]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bar_panel(ax, [Series("s", values, "#7C9A6E", [0.1] * 3)], ["one", "two", "three"])
    ax.axhline(0.0, color="#333333", lw=1.2)
    fig.canvas.draw()
    votes = [line for line in ax.lines if np.asarray(line.get_xdata()).size == 2]
    assert len(votes) == 1, "the premise: exactly one line votes, and it votes wrongly"
    assert orientation_of(ax) == "vertical"

    reach = max(values)
    for index in range(3):
        bracket_stack(ax, [(index - 0.2, index + 0.2, 0.001)], start=reach + 0.3, span=1.0)
    fig.canvas.draw()
    missing = [c for c in significance_gaps(fig) if "no bracket under it" in c]
    assert not missing, missing


def test_a_marker_only_line_is_not_ink_between_its_markers() -> None:
    """A label in the gaps between error-bar caps touches nothing and must not be flagged.

    `errorbar` draws all the lower caps as ONE marker-only Line2D and all the upper caps as another,
    so the polyline through them runs diagonally across the panel. Testing that polyline reported
    every value label in a bar panel as sitting on the marks.
    """
    import numpy as np

    from ogviz.layout.collision import hits_data, text_box

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(np.arange(6.0), np.arange(1.0, 7.0), yerr=0.15, fmt="none", capsize=4)
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(0.0, 7.0)
    in_the_gap = ax.text(1.5, 2.5, "0.42", ha="center", va="center")
    on_a_cap = ax.text(1.0, 2.0, "0.42", ha="center", va="center")
    fig.canvas.draw()
    assert hits_data(ax, text_box(in_the_gap)) == 0, "nothing is drawn between two caps"
    assert hits_data(ax, text_box(on_a_cap)) > 0, "and a label on a cap is still caught"


def test_a_non_significant_label_is_checked_like_any_other() -> None:
    """`_stars` matched asterisks, so `n.s.` was exempt from the gap-consistency check."""
    import numpy as np

    from ogviz import group_violins
    from ogviz.qc import significance_gaps
    from ogviz.qc.significance import _stars

    rng = np.random.default_rng(0)
    groups = [(float(i), rng.normal(i * 0.2, 1.0, 30), "#E8A838", "#B97C10") for i in range(3)]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    group_violins(ax, groups, comparisons=[(0.0, 1.0, 1e-4), (0.0, 2.0, 0.4)])
    fig.canvas.draw()
    assert "n.s." in [t.get_text().strip() for t in _stars(ax)]
    assert not significance_gaps(fig), "as drawn, the two labels agree"

    for text in ax.texts:
        if text.get_text().strip() == "n.s.":
            x, y = text.get_position()
            text.set_position((x, y - 0.35))
    fig.canvas.draw()
    assert any("different distances" in c for c in significance_gaps(fig))


def test_a_hyphen_inside_a_name_is_not_a_minus_sign() -> None:
    """`COMPASS-31` was reported as mixing two minus signs with matplotlib's own tick labels.

    One complaint on one figure, and the kind that teaches a reader to skim the audit output.
    """
    from ogviz.qc import one_minus_sign

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot([0.0, 1.0], [-0.5, 0.5])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["COMPASS-31", "Diez-Cirarda"])
    fig.canvas.draw()
    assert not one_minus_sign(fig)


def test_a_real_mix_of_minus_glyphs_is_still_caught() -> None:
    """The other direction: narrowing the rule must not blind the check it belongs to."""
    from ogviz.qc import one_minus_sign

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot([0.0, 1.0], [-0.5, 0.5])
    ax.text(0.5, 0.0, f"{-0.42:.2f}", ha="center")  # an ASCII hyphen from format
    fig.canvas.draw()
    assert any("two different minus signs" in c for c in one_minus_sign(fig))


def test_a_knockout_box_excuses_the_ink_it_actually_hides() -> None:
    """The gate cried wolf when a band grew dashed edges under labels that already had boxes.

    Measured on the figure that raised it: label and edge shared 62 px of ink, and NONE of those
    pixels were visible — the knockout covered every one. Paint order decides it, so the same box
    painted UNDER the line excuses nothing.
    """
    from ogviz.qc import colliding_ink

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot([0.0, 1.0], [0.5, 0.5], color="#333333", lw=3)
    label = ax.text(0.5, 0.5, "0.636", ha="center", va="center")
    fig.canvas.draw()
    assert colliding_ink(fig), "a bare label crossed by a rule is a real collision"

    label.set_bbox({"facecolor": "#FCFCFA", "edgecolor": "none", "pad": 2})
    fig.canvas.draw()
    assert not colliding_ink(fig), "and the knockout hides it"

    label.set_zorder(0.5)
    fig.canvas.draw()
    assert colliding_ink(fig), "but a box painted under the line hides nothing"


def test_an_ungrouped_thousand_is_caught_and_a_year_is_not() -> None:
    """The rule holds for numbers a CALLER formatted, which is where it gets forgotten.

    A four-digit number that could be a year is left alone: a year is an identifier rather than a
    quantity, and nothing in a figure carries the fact of which it is.
    """
    from ogviz.qc import ungrouped_thousands

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.set_axis_off()
    for index, printed in enumerate(["1200000", "1,200,000", "999", "2026", "0.5000", "45.6%"]):
        ax.text(0.05, 0.9 - index * 0.15, printed, transform=ax.transAxes)
    fig.canvas.draw()
    complaints = ungrouped_thousands(fig)
    assert len(complaints) == 1, complaints
    assert "1200000" in complaints[0]


def test_the_house_panels_group_their_own_numbers() -> None:
    """Everything the library prints follows the rule without being asked."""

    from ogviz import bar_panel
    from ogviz.panels.bars import Series
    from ogviz.qc import ungrouped_thousands

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    bar_panel(ax, [Series("arm", [1200000.0, 950000.0, 1400000.0], "#7C9A6E")], ["A", "B", "C"])
    fig.canvas.draw()
    printed = [t.get_text() for t in ax.texts]
    assert "1,200,000" in printed, printed
    assert not ungrouped_thousands(fig)


def test_the_gallery_build_pins_its_backend() -> None:
    """The committed figures are a reproducibility claim, and a backend is part of the render.

    On a Mac with a display matplotlib picks `macosx`, which lays text out differently from `Agg`:
    the violin panel that stood at gallery slot 06 came out with seven y-ticks under one and four
    under the other, from identical code and data, because the text metrics moved the axis limits
    enough to change what the locator chose. Every test runs under `Agg`, so an unpinned gallery is
    rendered by something the suite never exercises and cannot be reproduced headlessly.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).parent.parent.parent / "examples" / "__main__.py").read_text()
    pinned = re.search(r'^matplotlib\.use\("Agg"\)$', source, re.M)
    pyplot = re.search(r"^import matplotlib\.pyplot", source, re.M)
    assert pinned, "the gallery build must pin a backend"
    assert pyplot and pinned.start() < pyplot.start(), "and pin it BEFORE pyplot is imported"


def test_audit_renders_a_figure_that_has_no_canvas_yet() -> None:
    """It failed with `'FigureCanvasBase' object has no attribute 'get_renderer'` — an internal.

    A `Figure` built without pyplot cannot produce a renderer, and every check reads rendered
    geometry. Attaching Agg is what the caller wanted when they asked for an audit.
    """
    from matplotlib.figure import Figure

    from ogviz.qc import audit

    bare = Figure(figsize=(4.0, 3.0))
    ax = bare.add_subplot()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    assert audit(bare) == []


def test_a_halo_knocks_a_gridline_out_as_well_as_a_box_does() -> None:
    """`withStroke` in the page colour hugs the glyphs instead of boxing them, which reads better.

    Only the box was recognised, so a figure using the halo was told its labels crossed a gridline
    "with nothing behind it" and had to exclude the complaint by matching on its own text.
    """
    import matplotlib.patheffects as path_effects

    from ogviz.layout.collision import _knocked_out
    from ogviz.theme import page_color

    _fig, ax = plt.subplots()
    plain = ax.text(0.5, 0.5, "plain")
    boxed = ax.text(0.5, 0.4, "boxed", bbox={"facecolor": page_color(), "edgecolor": "none"})
    haloed = ax.text(
        0.5,
        0.3,
        "haloed",
        path_effects=[path_effects.withStroke(linewidth=3, foreground=page_color())],
    )
    coloured = ax.text(
        0.5,
        0.2,
        "red halo",
        path_effects=[path_effects.withStroke(linewidth=3, foreground="#FF0000")],
    )
    assert not _knocked_out(plain)
    assert _knocked_out(boxed)
    assert _knocked_out(haloed)
    assert not _knocked_out(coloured), "a halo in another colour hides nothing"


def test_two_panels_on_one_scale_are_judged_on_what_they_share() -> None:
    """A tick real on the left panel is above every mark on the right, by design.

    And the check must survive: a tick above ALL the shared data is still in the headroom.
    """
    import numpy as np

    from ogviz import group_violins
    from ogviz.qc import ticks_in_the_headroom

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), sharey=True)
    group_violins(
        axes[0],
        [
            (0.0, rng.normal(0.0, 1.0, 40), "#E8A838", "#B97C10"),
            (1.0, rng.normal(3.5, 1.0, 40), "#7C9A6E", "#4A6136"),
        ],
        comparisons=[(0.0, 1.0, 1e-4)],
    )
    group_violins(
        axes[1],
        [
            (0.0, rng.normal(0.0, 0.3, 12), "#E8A838", "#B97C10"),
            (1.0, rng.normal(1.0, 0.3, 12), "#7C9A6E", "#4A6136"),
        ],
        comparisons=[(0.0, 1.0, 0.02)],
    )
    fig.canvas.draw()
    assert not ticks_in_the_headroom(fig)

    # A ladder above everything the pair shares is still reported.
    axes[0].set_yticks([*axes[0].get_yticks(), 12.0, 14.0, 16.0])
    axes[0].set_ylim(axes[0].get_ylim()[0], 17.0)
    fig.canvas.draw()
    assert ticks_in_the_headroom(fig)


def test_two_brackets_that_share_no_x_need_not_clear_each_other() -> None:
    """Two independent comparisons side by side in one panel cannot collide, whatever their heights.

    Reported as "brackets are 0 px apart" by a project drawing one bracket over its first pair of
    groups and another over its last. Levelling by height fixed the equal-height case; this is the
    general rule — a bracket only has to clear one it overlaps along the category axis.
    """
    import numpy as np

    from ogviz import bracket_stack, group_violins
    from ogviz.qc import stack_spacing

    def panel(pairs, starts) -> list[str]:
        rng = np.random.default_rng(0)
        groups = [
            (float(index), rng.normal(index * 0.2, 1.0, 30), "#E8A838", "#B97C10")
            for index in range(5)
        ]
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        group_violins(ax, groups)
        reach = max(float(np.max(values)) for _p, values, _f, _e in groups)
        for pair, start in zip(pairs, starts, strict=True):
            bracket_stack(ax, [(pair[0], pair[1], 1e-3)], start=reach * start, span=1.0)
        fig.canvas.draw()
        return stack_spacing(fig)

    assert not panel([(0.0, 1.0), (3.0, 4.0)], [1.15, 1.22]), "disjoint, different heights"
    assert not panel([(0.0, 1.0), (3.0, 4.0)], [1.15, 1.15]), "disjoint, one height"
    assert panel([(0.0, 2.0), (1.0, 3.0)], [1.15, 1.19]), "overlapping and crowded is still caught"
    assert not panel([(0.0, 2.0), (1.0, 3.0)], [1.15, 1.45]), "overlapping and clear"


def test_a_label_on_its_own_backdrop_is_judged_the_same_way_by_both_checks() -> None:
    """`colliding_ink` believed the backdrop tag and `text_over_data` did not.

    So a project labelling its own shaded window was told the label was fine on the pixels and had
    to move off the marks — one tag, two checks, two answers.
    """
    from ogviz import mark
    from ogviz.qc import audit

    def window(*, tagged: bool) -> list[str]:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.plot([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 1.5, 2.5])
        span = ax.axvspan(0.0, 0.85, color="#EFEDE4", zorder=0)
        if tagged:
            mark(span, "backdrop")
        ax.text(0.42, 2.2, "physiological window", ha="center")
        fig.canvas.draw()
        return audit(fig)

    assert len(window(tagged=False)) == 2, "an untagged shape is data, and a label on it moves"
    assert not window(tagged=True), "a backdrop is not data, and both checks must say so"


def test_a_complaint_names_the_fix_that_matches_what_the_label_sits_on() -> None:
    """A knockout box is a consumer's first move and does nothing for this check.

    When the thing underneath spans the panel it is almost always a shaded region the label NAMES,
    and the fix is to say so rather than to move the label off it.
    """
    from ogviz.qc import audit

    def sits_on(kind: str) -> list[str]:
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.plot([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 1.5, 2.5])
        if kind == "band":
            ax.axvspan(0.0, 0.85, color="#EFEDE4", zorder=0)
            ax.text(0.42, 2.2, "physiological window", ha="center")
        else:
            ax.text(1.5, 1.75, "on the line itself", ha="center")
        fig.canvas.draw()
        return [c for c in audit(fig) if "sits on" in c]

    (band,) = sits_on("band")
    assert "backdrop" in band, band
    (line,) = sits_on("line")
    assert "has to move" in line and "backdrop" not in line, line


def test_a_covered_spine_names_what_covered_it() -> None:
    """A full-width `axhspan` masking a cropped band erases the spine through the headroom."""
    from ogviz.qc import buried_baselines
    from ogviz.theme import page_color

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot([0.0, 1.0, 2.0], [1.0, 2.0, 1.5])
    ax.axhspan(1.8, 2.4, color=page_color(), zorder=4)
    ax.set_ylim(0.8, 2.4)
    fig.canvas.draw()
    (complaint,) = buried_baselines(fig)
    assert "Rectangle" in complaint and "zorder 4" in complaint
    assert "under the frame" in complaint


def test_a_clipped_line_complaint_says_what_actually_fixes_it() -> None:
    """Two natural fixes do nothing, and the second especially looks like it must have worked."""
    from ogviz.layout.overlap import clipped_artists

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot([0.5, 0.5], [1.0, 4.0], color="#333333")
    ax.set_ylim(0.8, 2.4)
    fig.canvas.draw()
    (complaint,) = clipped_artists(fig)
    assert "clipping it does not change this" in complaint
    assert "shorten the data or raise the limit" in complaint
