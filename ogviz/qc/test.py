"""The QC gate itself: it must pass every shipped example and fail every planted defect.

Each test plants the defect it guards against and checks the gate reports it.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import bracket_stack, group_violins, use_house_style
from ogviz.marks import iqr_box, mean_line, violin
from ogviz.qc import audit, dots_off_the_marks, significance_gaps, stack_spacing


@pytest.fixture(autouse=True)
def _style():
    """Pin the bundled font. These assertions are about RENDERED geometry, and Arial on macOS is
    narrower than the DejaVu a Linux runner falls back to — the same layout passes on one and
    collides on the other. Pinning means CI checks the same geometry the author sees, in the
    WIDER of the two, so a layout that passes here passes anywhere."""
    use_house_style()
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    yield
    plt.close("all")


def _three_groups(seed: int = 9):
    rng = np.random.default_rng(seed)
    palette = ("#2E7CE0", "#EFA607", "#14A97C")
    return [(float(i), rng.normal(i * 0.8, 0.9, 30), c, "#333333") for i, c in enumerate(palette)]


def _gallery_names() -> tuple[str, ...]:
    """Read the gallery off the module that renders it, so a new example is covered by existing.

    A second list here was a copy that could go stale in the direction that matters: an example
    added to the README and not to the list would ship unchecked.
    """
    import examples.__main__ as gallery

    return tuple(example.__name__ for example in gallery.EXAMPLES)


@pytest.mark.parametrize("name", _gallery_names())
def test_every_shipped_example_passes_qc(name: str) -> None:
    """The gate runs on the gallery, so a regression cannot reach the README."""
    import examples.__main__ as gallery

    getattr(gallery, name)()  # each calls save(), which asserts the gate


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
    scatter.ogviz_lane = np.full(40, 0.1)
    scatter.ogviz_position = 0.0
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
