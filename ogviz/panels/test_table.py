"""The table's two axes, its type scale, and the glyphs the display font cannot draw."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import NO, YES, Cell, Row, audit, table_panel, use_house_style
from ogviz.panels.table import VALUE_SIZE
from ogviz.qc import type_too_small
from ogviz.theme import BAD, GLYPH_FAMILY, GOOD, INK, MUTED_INK, family_for


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def _table(**kwargs):
    rows = [
        Row("Video encoder", (Cell(YES, tone="good"), Cell("0.71"))),
        Row("Still encoder", (Cell(NO, tone="bad"), Cell("0.63", best=True))),
        Row("Random init", (Cell(NO, tone="bad"), Cell())),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    table_panel(ax, ["Temporal", "Dice"], rows, **kwargs)
    return fig, ax


def _outlines(ax):
    from matplotlib.patches import FancyBboxPatch

    return [p for p in ax.patches if isinstance(p, FancyBboxPatch)]


def test_a_table_can_be_about_a_row() -> None:
    """`highlight` is a COLUMN index, and a table with entities as rows had no way to say so.

    The failure it replaces is not an exception: a table transposed from metrics-down to arms-across
    kept `highlight=0`, which then outlined a METRIC and read as a claim about it. Nothing errored.
    """
    _fig, ax = _table(highlight_row=0)
    (frame,) = _outlines(ax)
    y = frame.get_y()
    # The second row's band sits below the first: the outline has to be the TOP one.
    assert frame.get_width() > 0.9, "a row outline runs (nearly) the full width"
    assert y + frame.get_height() < 1.0, "and below the header"


def test_a_row_outline_clears_the_label_it_frames() -> None:
    """The frame's stroke ran through the first glyph of the row name, at 23 px of shared ink.

    Caught by `colliding_ink` rather than by eye, which is the check working: the fix was to indent
    the label off the figure's edge, not to excuse the collision — the stroke really did cross it.
    """
    from ogviz.panels.table import LABEL_INDENT

    _fig, ax = _table(highlight_row=1)
    label = next(t for t in ax.texts if t.get_text() == "Still encoder")
    assert label.get_position()[0] >= LABEL_INDENT > 0.0


def test_both_axes_can_be_marked_at_once() -> None:
    _fig, ax = _table(highlight=0, highlight_row=0)
    assert len(_outlines(ax)) == 2


def test_a_row_index_past_the_end_is_refused() -> None:
    with pytest.raises(AssertionError, match="not a row index"):
        _table(highlight_row=9)


def test_a_tone_colours_the_value_and_missing_stays_muted() -> None:
    """Asked for directly: a green tick and a red cross, which `best` could not say."""
    assert Cell(YES, tone="good").ink() == GOOD
    assert Cell(NO, tone="bad").ink() == BAD
    assert Cell("0.71").ink() == INK
    assert Cell().ink() == MUTED_INK, "a missing cell stays muted whatever tone is asked for"


def test_the_tick_and_cross_are_drawn_in_a_font_that_has_them() -> None:
    """Arial has no U+2713 or U+2717, and matplotlib draws a missing glyph as a tofu box.

    Verified against the resolved face rather than assumed, and the fallback is matplotlib's own
    bundled DejaVu, so this holds on a machine with no system fonts at all.
    """
    assert family_for(YES) == GLYPH_FAMILY
    assert family_for(NO) == GLYPH_FAMILY
    assert family_for("0.71 Dice") is None, "ordinary text stays in the display font"

    _fig, ax = _table()
    ticks = [t for t in ax.texts if t.get_text() in (YES, NO)]
    assert ticks, "the table drew them"
    assert all(t.get_fontfamily() == [GLYPH_FAMILY] for t in ticks)
    assert all(t.get_fontfamily() != [GLYPH_FAMILY] for t in ax.texts if t.get_text() == "0.71")


def test_the_glyphs_survive_the_save_gate() -> None:
    """`glyphs_must_render` turns a tofu box into a failure; these must not trip it."""
    from ogviz.theme import glyphs_must_render

    fig, _ax = _table(highlight_row=0)
    with glyphs_must_render():
        fig.canvas.draw()
        fig.savefig("/dev/null", format="png")


def test_font_scale_moves_the_type_and_the_columns_together() -> None:
    """Scaling the type without the measured column widths runs headers into their neighbours."""
    _fig, plain = _table()
    _fig2, bigger = _table(font_scale=1.5)

    def value_size(ax) -> float:
        return max(t.get_fontsize() for t in ax.texts if t.get_text() == "0.71")

    assert value_size(plain) == pytest.approx(VALUE_SIZE)
    assert value_size(bigger) == pytest.approx(VALUE_SIZE * 1.5)

    def label_column(ax) -> float:
        return min(t.get_position()[0] for t in ax.texts if t.get_text() == "0.71")

    assert label_column(bigger) > label_column(plain), "the columns widened with the type"


def test_a_bad_font_scale_is_refused() -> None:
    with pytest.raises(AssertionError, match="font_scale"):
        _table(font_scale=0.0)


def test_a_table_stays_clean_through_the_gate() -> None:
    fig, _ax = _table(highlight_row=0, font_scale=1.2)
    assert audit(fig) == []


def test_cramped_type_is_reported_and_a_comfortable_table_is_not() -> None:
    """The gap `assert_clean` structurally cannot see — cramped type collides with nothing.

    Advisory rather than a gate, and the reason is a measurement: the shape that gets reported as
    unreadable and the densest figure this package ships are about 1.3x apart on this number, which
    is not a margin to fail a build on.
    """
    rows = [Row(f"Metric {i}", (Cell(f"{i / 10:.2f}"), Cell(f"{i / 8:.2f}"))) for i in range(20)]
    fig, ax = plt.subplots(figsize=(15.0, 16.0))
    table_panel(ax, ["A", "B"], rows)
    cramped = type_too_small(fig)
    assert cramped and "short side" in cramped[0]
    assert audit(fig) == [], "and it is NOT a gate failure — it collides with nothing"

    roomy, _ax = _table()
    assert type_too_small(roomy) == []
