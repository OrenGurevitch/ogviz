"""Writing a figure to disk, with the checks in front of it.

`save` is the only way a figure should leave the process: it runs the gate first and raises instead
of writing, so a broken figure cannot reach a README by being saved from somewhere that forgot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ogviz.layout.axis import settle_axis_labels
from ogviz.layout.header import settle_header
from ogviz.layout.panels import settle_caption
from ogviz.require import require
from ogviz.significance import settle_bracket_labels
from ogviz.theme import glyphs_must_render

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.figure import Figure


def reproducible_metadata(path: Path) -> dict[str, None]:
    """Drop the write date, so a re-render of an unchanged figure produces an unchanged file.

    Two things in a matplotlib SVG change on every run: the `dc:date` stamp, and the random ids
    matplotlib gives its clip paths (`svg.hashsalt` pins those; the house style sets it). Together
    they made `git diff` on the committed gallery useless — thirteen files, every line touched,
    2480 modifications of which none were real, so a diff could not answer "did this change the
    figure". A generated artifact that cannot be diffed cannot be reviewed.

    PNG takes the same treatment through its own key.

    PUBLIC because `save` is not the only way a figure gets written. A caller with a reason to use
    `fig.savefig` directly — a before/after where one half is meant to fail the gate, most obviously
    — still wants the file to be diffable, and the alternative is that they copy the two key names
    and go stale when a third format needs one. This example's own gallery figure was committed with
    a live date stamp for exactly that reason, and churned on every render.
    """
    return {"Date": None} if path.suffix == ".svg" else {"Software": None}


def save(
    fig: Figure,
    directory: Path,
    name: str,
    *,
    dpi: int = 200,
    check_overlap: bool = True,
    formats: Sequence[str] = ("png", "svg"),
    close: bool = True,
    crop: bool = True,
) -> list[Path]:
    """Write `<directory>/<name>.<ext>` per format, checked on the way out.

    Both checks fail the build rather than write a broken figure: a missing glyph renders as a
    tofu box and overlapping labels render as mush, and both otherwise ship unnoticed because a
    figure build scrolls past. Pass `check_overlap=False` for a panel whose text legitimately
    abuts, such as a rendered table, and `close=False` to keep working on the figure.

    `crop=True` (the default) writes `bbox_inches="tight"`: the file is cropped to the artists
    rather than to the canvas, so the declared `figsize` is NOT what lands on disk. It trims dead
    margin, and it keeps a label that reaches past the page instead of cutting it off.

    THE COST, which this docstring claimed the opposite of until 2026-08-04: two figures declaring
    the same canvas do not write the same size. It used to be large — a plain 7x4 in panel at dpi
    100 wrote 602x353 against 829x353 for the same panel with one label reaching past the edge.

    RE-MEASURED 2026-08-12, and it is now small, because the gate closed the case that made it big.
    A label reaching past the canvas is what grew the page, and `text_off_canvas` refuses that
    figure outright — both of the divergent cases above are now rejected before they can be written.
    What remains is the difference in dead margin between two figures that both PASS: a bare panel
    writes 602x353 and one carrying a y-label and a title writes 631x376, which placed at a common
    width in a document is a 1.6% difference in aspect.

    So the argument for inverting this default — that cropping breaks side-by-side use — was
    measured against figures the gate no longer lets through, and 1.6% does not break it. The
    default stays cropped.

    So `crop=False` for a PINNED layout, where the point is that every figure has the same axes
    rectangle: it writes the canvas as declared, and `required_margins` is how the margins get
    chosen. Cropping and pinning are the two coherent choices; picking neither deliberately is how a
    set ends up inconsistent.
    """
    require(
        formats,
        "save needs at least one format",
    )
    # Before the checks, not after: a caption row is reserved when the panels are created and the
    # caller has not plotted yet, so what grows into it can only be measured here.
    settle_header(fig)
    settle_caption(fig)
    # After the caption, and before the checks: a bracket label's gap to its bracket is set in
    # pixels, and anything that rescaled the value axis since then has changed it.
    settle_bracket_labels(fig)
    # Last of the settles, because it reads the ticks where they FINALLY are: anything above that
    # rescaled a value axis moved them. matplotlib centres an axis label on the axes box, and this
    # package pads a panel asymmetrically on purpose — bracket headroom above, a mean lane below —
    # so the box's middle is not where the ticks are.
    settle_axis_labels(fig)
    directory.mkdir(parents=True, exist_ok=True)
    if check_overlap:
        from ogviz.qc import assert_clean

        assert_clean(fig)
    canvas = fig.get_facecolor()
    paths = [directory / f"{name}.{extension}" for extension in formats]
    with glyphs_must_render():
        for path in paths:
            fig.savefig(
                path,
                bbox_inches="tight" if crop else None,
                facecolor=canvas,
                dpi=dpi,
                metadata=reproducible_metadata(path),
            )
    if close:
        plt.close(fig)
    return paths
