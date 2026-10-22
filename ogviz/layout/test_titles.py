import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import fit_under_header, titled
from ogviz.layout import fit_titles
from ogviz.layout.bounds import figure_text
from ogviz.layout.titles import axes_titles, title_text

pytestmark = pytest.mark.usefixtures("pinned_font")

LONG = "A title much too long for a panel this narrow to carry at its full size"


def _grid(titles: list[str], *, loc: str = "center"):
    fig, axes = plt.subplots(1, len(titles), figsize=(6.0, 3.0))
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=19, loc=loc)
    return fig, list(axes)


def test_titles_shrink_together_to_the_largest_size_that_fits() -> None:
    fig, axes = _grid(["Short", "A moderately long panel title"])
    size = fit_titles(fig, axes, largest=19.0, smallest=8.0)
    assert 8.0 <= size < 19.0
    assert {ax.title.get_fontsize() for ax in axes} == {size}, "one size for the whole set"


def test_titles_that_already_fit_are_left_at_their_size() -> None:
    fig, axes = _grid(["A", "B"])
    assert fit_titles(fig, axes, largest=19.0, smallest=8.0) == 19.0


def test_a_title_no_size_can_fit_raises_and_names_it() -> None:
    fig, axes = _grid(["Short", LONG])
    with pytest.raises(AssertionError, match="Shorten the title") as refused:
        fit_titles(fig, axes, largest=19.0, smallest=12.0)
    assert "A title much too long" in str(refused.value)


def test_a_left_aligned_title_is_found_and_fitted() -> None:
    """Premise: the title is NOT the centre one, which is the only one `get_title()` returns."""
    middling = "A moderately long panel title"
    fig, axes = _grid(["Short", middling], loc="left")
    assert axes[1].get_title() == ""
    assert title_text(axes[1]) == middling
    assert len(axes_titles(axes[1])) == 1
    size = fit_titles(fig, axes, largest=19.0, smallest=6.0)
    assert size < 19.0


def test_the_header_makes_room_for_left_aligned_panel_titles() -> None:
    """Pinning the top under the header reserved nothing for a left title, and the subtitle ran
    through the titles: found on a figure of four left-titled panels that the gate passed."""

    def gap(loc: str) -> float:
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
        for ax in axes.flat:
            ax.set_title("Panel", loc=loc, fontsize=19)
        header = titled(fig, "Header", subtitle="a subtitle that sits under it")
        fit_under_header(fig, header, bottom=0.0)
        fig.canvas.draw()
        subtitle = next(t for t in fig.texts if "subtitle" in t.get_text())
        titles = [t for ax in axes.flat for t in axes_titles(ax)]
        plt.close(fig)
        return subtitle.get_window_extent().y0 - max(t.get_window_extent().y1 for t in titles)

    assert gap("center") > 0, "the premise: a centred title already cleared the subtitle"
    assert gap("left") > 0


def test_every_title_position_is_visible_to_the_text_walkers() -> None:
    fig, ax = plt.subplots()
    ax.set_title("centre")
    ax.set_title("left", loc="left")
    ax.set_title("right", loc="right")
    drawn = {text.get_text() for text, _ in figure_text(fig)}
    assert {"centre", "left", "right"} <= drawn
