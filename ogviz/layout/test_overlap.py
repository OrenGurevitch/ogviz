import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from ogviz.layout import assert_no_text_overlap, text_overlaps
from ogviz.panels import Series, bar_panel
from ogviz.theme import SERIES, use_house_style


@pytest.fixture(autouse=True)
def _style():
    """Pin the bundled font. These tests assert on RENDERED text geometry, and Arial on macOS is
    narrower than DejaVu on a Linux runner — the same call then overlaps on one and not the
    other. DejaVu ships with matplotlib, so pinning it makes the geometry machine-independent."""
    use_house_style()
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    yield
    plt.close("all")


def test_two_labels_stacked_on_each_other_are_reported():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "a long measurement label", ha="center")
    ax.text(0.5, 0.5, "a long measurement label", ha="center")
    hits = text_overlaps(fig)
    assert len(hits) == 1
    assert "a long measurement label" in hits[0]


def test_separated_labels_are_clean():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.2, 0.2, "left", ha="center")
    ax.text(0.8, 0.8, "right", ha="center")
    assert text_overlaps(fig) == []


def test_touching_labels_pass_but_buried_ones_fail():
    """The threshold is what makes this usable: abutting tick labels are not a defect."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.50, 0.5, "aaaa", ha="center")
    ax.text(0.58, 0.5, "aaaa", ha="center")  # +3.7 px apart in the pinned font: abutting
    assert text_overlaps(fig, min_gap=0.0) == [], "abutting is not a defect"
    assert text_overlaps(fig, min_gap=8.0) != [], "under the floor, it is"


def test_invisible_and_empty_text_is_ignored():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "visible", ha="center")
    ax.text(0.5, 0.5, "hidden", ha="center", visible=False)
    ax.text(0.5, 0.5, "   ", ha="center")
    assert text_overlaps(fig) == []


def _domain_panel(width: float):
    """The real shape this guards: five word-length categories on one tick row."""
    fig, ax = plt.subplots(figsize=(width, 4.5))
    bar_panel(
        ax,
        [Series("z", [0.6, 0.5, 0.6, 0.4, 0.7], SERIES[0])],
        ["fatigue", "sleep", "pain", "cognition", "autonomic"],
        show_values=False,
    )
    return fig


def test_adjacent_labels_that_read_as_one_word_are_reported():
    """Zero overlapping area, still broken: 2.3 px between "cognition" and "autonomic" renders
    joined. The width is measured in the pinned font, so this holds on every machine."""
    fig = _domain_panel(8.0)
    assert text_overlaps(fig, min_gap=0.0) == [], "these do not overlap, they abut"
    assert any("cognition" in hit and "autonomic" in hit for hit in text_overlaps(fig))


def test_labels_that_run_into_each_other_are_reported_too():
    """The hole this closes: a pair overlapping by less than `min_overlap` of area used to skip
    the gap rule as well, so the WORSE condition passed while mere abutting was caught."""
    fig = _domain_panel(7.0)  # the same pair, -11.5 px: actually overlapping
    hits = text_overlaps(fig)
    assert any("cognition" in hit and "autonomic" in hit for hit in hits)
    assert any("runs into" in hit for hit in hits)


def test_a_wide_enough_figure_separates_the_same_labels():
    assert text_overlaps(_domain_panel(13.0)) == []


def test_labels_on_different_rows_are_never_gap_flagged():
    """A y tick and an x tick can be pixels apart horizontally and still be perfectly readable."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.30, 0.90, "upper", ha="center")
    ax.text(0.32, 0.10, "lower", ha="center")
    assert text_overlaps(fig) == []


def test_assert_helper_names_the_colliding_pair():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "Measurement A", ha="center")
    ax.text(0.5, 0.5, "Measurement A", ha="center")
    with pytest.raises(AssertionError, match="Measurement A"):
        assert_no_text_overlap(fig)
