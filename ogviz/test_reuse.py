"""Reusability, tested as a contract rather than asserted in a README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz import Series, bar_panel, group_violins, save, use_house_style
from ogviz.marks import VIOLIN_WIDTH


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def _sample(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, 40)


def test_a_panel_does_not_hardcode_its_marks_defaults():
    """The failure this guards: a caller wanting 0.8-wide bodies would silently
    get the house 0.62, so it could use the marks but never the panel above them."""
    wide, ax_wide = plt.subplots()
    group_violins(
        ax_wide,
        [(0.0, _sample(), "#2E7CE0", "#1A4F94")],
        violin_kwargs={"width": 0.8},
        point_kwargs={"width": 0.8},
        show_means=False,
    )
    house, ax_house = plt.subplots()
    group_violins(ax_house, [(0.0, _sample(), "#2E7CE0", "#1A4F94")], show_means=False)

    def body_half_width(ax):
        return float(np.abs(ax.collections[0].get_paths()[0].vertices[:, 0]).max())

    assert body_half_width(ax_wide) > body_half_width(ax_house)
    assert body_half_width(ax_house) == pytest.approx(VIOLIN_WIDTH / 2, abs=1e-9)
    assert wide is not house


def test_spacing_fractions_are_arguments():
    """A caller who needs different headroom must not have to edit the package."""
    _tight, tight = plt.subplots()
    group_violins(tight, [(0.0, _sample(), "#2E7CE0", "#1A4F94")], headroom=0.05)
    _loose, loose = plt.subplots()
    group_violins(loose, [(0.0, _sample(), "#2E7CE0", "#1A4F94")], headroom=1.50)
    assert loose.get_ylim()[1] > tight.get_ylim()[1]


def test_a_panel_refuses_non_finite_values_and_says_how_many():
    """Silently dropping NaN plots an n nobody wrote down — the value error that hides."""
    values = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
    _fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match=r"2 non-finite value\(s\) of 5"):
        group_violins(ax, [(0.0, values, "#2E7CE0", "#1A4F94")])


def test_the_bracket_label_convention_belongs_to_the_project():
    _fig, ax = plt.subplots()
    group_violins(
        ax,
        [(0.0, _sample(1), "#2E7CE0", "#1A4F94"), (1.0, _sample(2) + 3, "#EFA607", "#8A5E00")],
        comparisons=[(0.0, 1.0, 0.004)],
        label_for=lambda p: "p<0.01" if p < 0.01 else "",
        show_means=False,
    )
    assert "p<0.01" in {t.get_text() for t in ax.texts}


def test_save_writes_the_formats_it_is_given(tmp_path):
    _fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = save(plt.gcf(), tmp_path, "figure", formats=("pdf",))
    assert [p.suffix for p in written] == [".pdf"]
    assert written[0].exists()


def test_save_can_leave_the_figure_open_for_more_work(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save(fig, tmp_path, "first", close=False)
    ax.set_title("still editable")
    assert save(fig, tmp_path, "second")[0].exists()


def test_a_bar_panel_carries_its_own_categories_and_series_count():
    """Same panel, one series over three categories and four over two — no reshaping by hand."""
    _fig, one = plt.subplots()
    bar_panel(one, [Series("a", [1.0, 2.0, 3.0], "#2E7CE0")], list("xyz"))
    _fig2, many = plt.subplots(figsize=(10, 4))
    bar_panel(many, [Series(f"s{i}", [1.0, 2.0], f"C{i}") for i in range(4)], list("xy"))
    assert len(one.patches) == 3
    assert len(many.patches) == 8


def test_version_is_importable_so_a_project_can_pin_it():
    import ogviz

    assert ogviz.__version__
