import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from ogviz.layout import text_overlaps
from ogviz.layout.panels import panel_row, settle_caption, text_width_points, wrap_to_width
from ogviz.theme import use_house_style

CAPTION = (
    "Each dot is one observation, the bar the interquartile range, the line the mean and the "
    "open circle the median. The right-hand group sits lower than the left across every "
    "category, and the composite is the primary comparison."
)


@pytest.fixture(autouse=True)
def _style():
    """Pin the bundled font. These tests assert on RENDERED text geometry, and Arial on macOS is
    narrower than DejaVu on a Linux runner — the same call then overlaps on one and not the
    other. DejaVu ships with matplotlib, so pinning it makes the geometry machine-independent."""
    use_house_style()
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    yield
    plt.close("all")


def test_no_caption_means_no_extra_row():
    """Captions are off by default: a manuscript figure carries none."""
    fig, axes = panel_row(2)
    assert len(axes) == 2
    assert len(fig.axes) == 2, "a caption axes must not be created when no caption is given"


def test_the_caption_gets_its_own_axes():
    fig, axes = panel_row(2, caption=CAPTION)
    assert len(fig.axes) == len(axes) + 1


def test_a_caption_cannot_overlap_an_x_label():
    """The defect the reserved row exists for: a caption at a chosen y collides with x-labels."""
    fig, axes = panel_row(2, caption=CAPTION)
    for ax in axes:
        ax.plot([0, 1], [1.0, 2.0])
        ax.set_xlabel("elapsed time (years)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Treated"])
    assert text_overlaps(fig) == []


def test_two_panels_need_width_for_a_long_x_label():
    """Not a caption problem, and worth separating from one: at the default width two long
    x-labels reach each other and the neighbouring y ticks. The check catches it, so a caller
    finds out at save() rather than in review."""

    def build(width: float):
        fig, axes = panel_row(2, caption=CAPTION, width=width)
        for ax in axes:
            ax.plot([0, 1], [1.0, 2.0])
            ax.set_xlabel("time from first measurement to the imaging visit (years)")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Control", "Treated"])
        return fig

    assert text_overlaps(build(12.8)) != []
    assert text_overlaps(build(18.0)) == []


def test_a_two_line_x_label_no_longer_reaches_the_caption_row():
    """Was an xfail for as long as the caption row existed.

    The row is sized when the panels are created, from the caption's own line count, and at that
    moment the caller has not plotted — the x-label that grows into it does not exist yet. So it is
    settled afterwards instead, from the rendered panels: `settle_caption` takes the lowest ink of
    every panel including its decorations and drops the caption below it. `save` calls it, so a
    figure written the normal way never has to know.
    """
    fig, axes = panel_row(3, caption=CAPTION)
    for ax in axes:
        ax.plot([0, 1, 2], [1.0, 2.0, 1.5])
        ax.set_xlabel("category\n(with a qualifying second line)")

    def caption_collisions(figure):
        return [one for one in text_overlaps(figure) if "Each dot is one" in one]

    fig.canvas.draw()
    assert caption_collisions(fig), "the label does reach the caption before it is settled"

    assert settle_caption(fig) is True
    assert caption_collisions(fig) == []
    # Scoped to the caption on purpose. Three 40-character two-line labels across a 12.8-inch row
    # also collide with EACH OTHER, which is a real defect of this test's figure and not something
    # a caption row can fix — the check reports it, correctly, and it is not what §8 was about.


def test_settle_caption_leaves_a_figure_that_already_clears_alone():
    fig, axes = panel_row(3, caption=CAPTION)
    for ax in axes:
        ax.plot([0, 1, 2], [1.0, 2.0, 1.5])
        ax.set_xlabel("category")
    fig.canvas.draw()
    assert settle_caption(fig) is False
    assert [one for one in text_overlaps(fig) if "Each dot is one" in one] == []


def test_wrapped_lines_never_exceed_the_measured_width():
    """The point of measuring instead of assuming characters-per-inch."""
    limit = 300.0
    lines = wrap_to_width(CAPTION, limit, 8.0)
    assert len(lines) > 1
    assert all(text_width_points(line, 8.0) <= limit for line in lines)


def test_a_wider_caption_row_uses_fewer_lines():
    narrow = wrap_to_width(CAPTION, 200.0, 8.0)
    wide = wrap_to_width(CAPTION, 600.0, 8.0)
    assert len(wide) < len(narrow)


def test_a_single_unbreakable_word_still_yields_a_line():
    assert wrap_to_width("supercalifragilistic", 10.0, 8.0) == ["supercalifragilistic"]


def test_the_caption_row_grows_with_the_line_count():
    short, _ = panel_row(2, caption="One short line.")
    long_caption, _ = panel_row(2, caption=CAPTION * 3)
    assert long_caption.get_figheight() > short.get_figheight()


def test_legend_reservation_narrows_the_panel_row():
    plain, plain_axes = panel_row(2)
    reserved, reserved_axes = panel_row(2, legend=True)
    assert reserved_axes[-1].get_position().x1 < plain_axes[-1].get_position().x1
    assert plain is not reserved
