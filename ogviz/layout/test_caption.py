"""The caption's one promise: it never comes out wider than the figure."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import use_house_style
from ogviz.layout.caption import caption, longest_unbreakable, overflowing_text

NOTE = (
    "Source: analysis of a random sample of messages, linked to self-reported occupations and "
    "task descriptions. Notes: shares are message-weighted, so higher-volume users contribute "
    "more observations. The chart shows users in the middle 50 percent of the distribution."
)


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def _figure(width: float):
    fig, ax = plt.subplots(figsize=(width, 3.0))
    ax.plot([1, 2, 3], [1, 2, 1.5])
    return fig, ax


def _widest(fig) -> float:
    fig.canvas.draw()
    return max(text.get_window_extent().width for text in fig.texts)


@pytest.mark.parametrize("width", [16.0, 13.0, 9.0, 6.0, 4.0, 3.0, 2.4])
def test_the_caption_fits_at_every_figure_width(width: float) -> None:
    """The guarantee, checked by measuring the render rather than trusting the wrap."""
    fig, _ax = _figure(width)
    caption(fig, NOTE, heading="Figure 6. Smaller workspaces show more boundary crossing")
    assert _widest(fig) <= fig.get_figwidth() * fig.dpi
    assert not overflowing_text(fig)


def test_a_narrow_figure_wraps_to_more_lines_rather_than_overflowing() -> None:
    wide, _ = _figure(13.0)
    caption(wide, NOTE)
    narrow, _ = _figure(5.0)
    caption(narrow, NOTE)
    lines = [len(t.get_text().splitlines()) for t in (wide.texts[0], narrow.texts[0])]
    assert lines[1] > lines[0], "the narrow figure must spend more lines, not more width"


def test_an_unbreakable_word_is_reported_not_silently_shrunk() -> None:
    """Wrapping cannot fix one long token, and a caption set at 4 pt is not a fixed caption."""
    url = "https://example.org/a/path/that/never/breaks/anywhere/at/all/and/keeps/going"
    fig, _ax = _figure(3.0)
    caption(fig, f"Source: {url}")
    assert longest_unbreakable(f"Source: {url}", 9.0) > 3.0 * 72
    complaints = overflowing_text(fig)
    assert complaints and "cannot be wrapped" in complaints[0]
    assert url[:30] in complaints[0], "the message must name the offending word"


def test_neither_block_lands_on_the_axes() -> None:
    fig, ax = _figure(9.0)
    caption(fig, NOTE, heading="A heading that says what the reader should take away")
    fig.canvas.draw()
    panel = ax.get_window_extent()
    for text in fig.texts:
        head = text.get_text()[:30]
        assert not text.get_window_extent().overlaps(panel), f"{head!r} hit the axes"


def test_captions_are_absent_unless_asked_for() -> None:
    fig, _ax = _figure(9.0)
    caption(fig)
    assert not fig.texts
