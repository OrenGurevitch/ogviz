"""Writing a figure to disk, with the checks in front of it.

`save` is the only way a figure should leave the process: it runs the gate first and raises instead
of writing, so a broken figure cannot reach a README by being saved from somewhere that forgot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ogviz.layout.panels import settle_caption
from ogviz.theme import glyphs_must_render

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.figure import Figure


def _reproducible_metadata(path: Path) -> dict[str, None]:
    """Drop the write date, so a re-render of an unchanged figure produces an unchanged file.

    Two things in a matplotlib SVG change on every run: the `dc:date` stamp, and the random ids
    matplotlib gives its clip paths (`svg.hashsalt` pins those; the house style sets it). Together
    they made `git diff` on the committed gallery useless — thirteen files, every line touched,
    2480 modifications of which none were real, so a diff could not answer "did this change the
    figure". A generated artifact that cannot be diffed cannot be reviewed.

    PNG takes the same treatment through its own key.
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
) -> list[Path]:
    """Write `<directory>/<name>.<ext>` per format, on the figure's own canvas, and check it.

    Both checks fail the build rather than write a broken figure: a missing glyph renders as a
    tofu box and overlapping labels render as mush, and both otherwise ship unnoticed because a
    figure build scrolls past. Pass `check_overlap=False` for a panel whose text legitimately
    abuts, such as a rendered table, and `close=False` to keep working on the figure.
    """
    assert formats, "save needs at least one format"
    # Before the checks, not after: a caption row is reserved when the panels are created and the
    # caller has not plotted yet, so what grows into it can only be measured here.
    settle_caption(fig)
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
                bbox_inches="tight",
                facecolor=canvas,
                dpi=dpi,
                metadata=_reproducible_metadata(path),
            )
    if close:
        plt.close(fig)
    return paths
