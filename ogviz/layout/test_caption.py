"""The caption's one promise: it never comes out wider than the figure."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.layout.caption import caption, longest_unbreakable, overflowing_text

# Rendered-text assertions: measured under the font every machine has (see conftest.py).
pytestmark = pytest.mark.usefixtures("pinned_font")

NOTE = (
    "Source: invented readings from a random sample of stations, linked to their declared region "
    "and instrument type. Notes: shares are reading-weighted, so busier stations contribute more "
    "observations. The chart shows stations in the middle 50 percent of the distribution."
)


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
    caption(fig, NOTE, heading="Figure 6. Shorter strands cross the midline more often")
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


def test_a_managed_layout_refuses_the_reservation_and_the_caller_is_told() -> None:
    """`subplots_adjust` is a no-op under a layout engine, and matplotlib only warns about it.

    So the caption reserved nothing and landed on the panels. The premise is the subplot
    parameter itself: it does not move.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.0), layout="constrained")
    ax.plot([0.0, 1.0], [0.0, 1.0])
    before = fig.subplotpars.bottom
    said = caption(fig, "a note long enough to want a couple of lines of room below the panels")
    assert fig.subplotpars.bottom == before, "premise: nothing was reserved"
    assert any("refuses subplots_adjust" in one for one in said)
    plt.close(fig)


def test_a_plain_figure_reserves_its_room_and_reports_nothing() -> None:
    """The same note on a figure with no layout engine: the room appears and nothing is said.

    `NOTE` wraps to several lines at this width, so the room it needs is more than the default
    bottom margin — which is what makes the reservation visible as a rise rather than a change.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    before = fig.subplotpars.bottom
    assert caption(fig, NOTE) == []
    assert fig.subplotpars.bottom > before
    plt.close(fig)


def test_an_unwrapped_label_is_not_reported_as_an_unbreakable_word() -> None:
    """Two different problems were reported as the second one.

    A long title that simply has not been wrapped came back as "'Autonomic' is one word and cannot
    be wrapped", which is false — that title wraps into three lines. A caller acting on it shortens
    a word that was never the problem. Naming the WIDEST word is still right; claiming it is
    unbreakable is only right when it is.
    """
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_title(
        "Autonomic burden across every measured condition and follow-up window", fontsize=16
    )
    fig.canvas.draw()

    said = overflowing_text(fig)
    assert said, "premise: the title really is wider than the canvas"
    assert "has not been wrapped" in said[0]
    assert "cannot be wrapped" not in said[0]
    plt.close(fig)


def test_a_genuinely_unbreakable_word_still_says_so() -> None:
    """The other branch, which is the one the message was written for."""
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_title("Pneumonoultramicroscopicsilicovolcanoconiosisandthensome", fontsize=16)
    fig.canvas.draw()

    said = overflowing_text(fig)
    assert said and "is one word and cannot be wrapped" in said[0]
    assert "shorter wording or smaller type" in said[0]
    plt.close(fig)
