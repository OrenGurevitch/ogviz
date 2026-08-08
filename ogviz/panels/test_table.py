"""The table's two axes, its type scale, and the glyphs the display font cannot draw."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import NO, YES, Cell, Row, audit, table_panel
from ogviz.panels.table import VALUE_SIZE
from ogviz.qc import type_too_small
from ogviz.theme import BAD, GLYPH_FAMILY, GOOD, INK, MUTED_INK, family_for


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


def test_a_glyph_the_display_font_lacks_is_drawn_in_one_that_has_it() -> None:
    """The display stack cannot always draw a tick, and matplotlib renders what it cannot as tofu.

    The FONT is pinned rather than taken from the machine, and that is the whole difficulty: which
    answer is correct depends on what is installed. Arial has no U+2713, so on macOS the fallback is
    required; a Linux runner has no Arial at all and falls through to DejaVu Sans, which HAS the
    tick, so there `None` is the right answer and asserting otherwise tests the machine rather than
    the code. DejaVu Serif is bundled with matplotlib and lacks the tick, so pinning it poses the
    question the same way everywhere.
    """
    import matplotlib as mpl

    mpl.rcParams["font.sans-serif"] = ["DejaVu Serif"]
    assert family_for(YES) == GLYPH_FAMILY
    assert family_for(NO) == GLYPH_FAMILY
    assert family_for("0.71 Dice") is None, "ordinary text stays in the display font"


def test_whatever_family_a_cell_ends_up_in_can_render_it() -> None:
    """The contract, stated so it holds on any machine: no cell is drawn as a tofu box.

    Where the glyph is available in the display font this passes by using it, and where it is not
    it passes by falling back — which is why the assertion is about the RESULT rather than about
    which of the two happened.
    """
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.ft2font import FT2Font

    _fig, ax = _table()
    drawn = [text for text in ax.texts if text.get_text() in (YES, NO)]
    assert drawn, "the table drew them"
    for text in drawn:
        face = FT2Font(findfont(FontProperties(family=text.get_fontfamily())))
        assert all(face.get_char_index(ord(ch)) for ch in text.get_text()), (
            f"{text.get_text()!r} would render as tofu in {text.get_fontfamily()}"
        )


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


def test_the_measurement_is_a_pure_function_of_the_inputs() -> None:
    """`_measure` is the half of `table_panel` that draws nothing, and every collision in a table
    is a consequence of these numbers — so they are worth asserting directly rather than only
    through a rendered figure."""
    from ogviz.panels.table import HEADER_HEIGHT, _measure

    rows = [
        Row("first", (Cell("1.00"), Cell("2.00"))),
        Row("second", (Cell("3.00"), Cell("4.00")), height=2.0),
    ]
    grid = _measure(["A", "B"], rows, 1.0)

    assert len(grid.shares) == 2 and len(grid.centres) == 2
    assert len(grid.edges) == 3, "one left edge per column, plus the right edge of the last"
    assert grid.edges == sorted(grid.edges), "columns run left to right"
    assert all(grid.edges[i] < grid.centres[i] < grid.edges[i + 1] for i in range(2))
    assert grid.edges[-1] == pytest.approx(1.0), "the columns fill the width"

    assert len(grid.tops) == len(rows) + 1, "the header band, then one top per row"
    assert grid.tops == sorted(grid.tops, reverse=True), "rows run top to bottom"
    assert grid.tops[0] == pytest.approx(1.0 - HEADER_HEIGHT * grid.unit)
    # The second row asked for twice the height, so its band is twice as deep.
    first, second = grid.tops[0] - grid.tops[1], grid.tops[1] - grid.tops[2]
    assert second == pytest.approx(2.0 * first)


def test_scaling_the_type_widens_every_column() -> None:
    """The measurement takes `font_scale` too — scaling the type without the columns runs a header
    into its neighbour."""
    from ogviz.panels.table import _measure

    rows = [Row("a long row label", (Cell("1.00"),))]
    plain = _measure(["a wide header"], rows, 1.0)
    bigger = _measure(["a wide header"], rows, 1.6)
    assert bigger.edges[0] > plain.edges[0], "the label column widened with the type"
