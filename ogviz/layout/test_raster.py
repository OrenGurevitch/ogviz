"""Reading a rendered figure back as pixels, and the one number that decides what counts as ink.

This module had no test at all until 2026-08-07. `frame_rgb` and `ink_of` were reached only through
`density.ink_mask` and `ink._render`, so the extraction that unified them was covered by nothing of
its own — and, measured, reverting `INK_TOLERANCE` to the value it replaced broke ZERO tests. The
constant ended a real disagreement (two modules answering "is this pixel ink" with 12 and with 10,
so the same faint pixel was ink to one check and page to its neighbour) and nothing protected it.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.layout.density import ink_mask
from ogviz.layout.raster import INK_TOLERANCE, frame_rgb, ink_of


def test_the_ink_tolerance_is_the_strict_one_of_the_two_it_replaced() -> None:
    """Pins the VALUE, because the argument for it is not recoverable from the code.

    Being generous here makes `colliding_ink` miss faint contact, which is a real defect; the only
    cost the other way is a slightly smaller dead-space note, which never fails anything. So of the
    two numbers this replaced, the strict one is correct and the loose one is not — and nothing
    else in the suite would notice it moving back.
    """
    assert INK_TOLERANCE == 10


def test_the_page_colour_is_read_from_the_render_not_from_rcparams() -> None:
    """A figure that sets its own facecolor must be measured against what it actually is.

    Read from the corner pixel, so a figure saved with a colour nobody configured still measures
    correctly. Asserting it here because the docstring is the only other place it is stated.
    """
    fig = plt.figure(figsize=(2.0, 2.0), facecolor="#123456")
    frame = frame_rgb(fig)
    assert tuple(frame[0, 0]) == (0x12, 0x34, 0x56)
    assert not ink_of(frame).any(), "a blank page of any colour carries no ink"
    plt.close(fig)


def test_the_tolerance_decides_from_both_sides() -> None:
    """Built as an array rather than rendered: the question is about the threshold, not a figure.

    `ink_of` measures every pixel against the frame's own corner, so a frame whose corner IS the
    page colour lets both sides of the boundary be stated exactly.
    """
    page = np.array([255, 255, 255], dtype=np.int16)
    frame = np.stack([page, page - (INK_TOLERANCE - 1), page - (INK_TOLERANCE + 1)]).reshape(
        1, 3, 3
    )

    mask = ink_of(frame)
    assert not mask[0, 0], "the page itself is not ink"
    assert not mask[0, 1], "nor is a pixel within the tolerance of it"
    assert mask[0, 2], "a pixel past the tolerance is"


def test_a_drawn_mark_registers_as_ink_through_the_public_helper() -> None:
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_axis_off()
    blank = int(ink_mask(fig).sum())
    ax.plot([0.2, 0.8], [0.2, 0.8], color="#000000", linewidth=6)
    drawn = int(ink_mask(fig).sum())
    assert drawn > blank, "a heavy black line has to add ink"
    plt.close(fig)


def test_a_canvas_that_cannot_be_read_back_says_so() -> None:
    """The failure was an AttributeError from inside a QC helper, naming a matplotlib internal."""
    from matplotlib.backend_bases import FigureCanvasBase

    fig = plt.figure()
    FigureCanvasBase(fig)  # replaces the Agg canvas: no buffer_rgba
    with pytest.raises(AssertionError, match="raster canvas"):
        frame_rgb(fig)
    plt.close(fig)


def test_svg_text_stays_text_for_a_project_that_takes_only_the_ink() -> None:
    """`svg.fonttype` was filed with the TYPE half, and the project needing it calls neither.

    Reported from a consumer: it pins its own family for byte-identical output, so it calls
    `use_house_ink()` alone — and got matplotlib's default, every glyph traced to a `<path>` and not
    one `<text>` element in its whole figure set. That also voids a verification method the
    consumer guidance recommended, diffing the text of two SVGs across a version bump: for that
    project the diff compares nothing against nothing and reports no change, for any bump.
    """
    import matplotlib as mpl

    from ogviz.theme import use_house_ink, use_house_type

    for half in (use_house_ink, use_house_type):
        mpl.rcParams["svg.fonttype"] = mpl.rcParamsDefault["svg.fonttype"]
        half()
        assert mpl.rcParams["svg.fonttype"] == "none", f"{half.__name__} left text as outlines"
