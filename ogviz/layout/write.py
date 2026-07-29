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
            fig.savefig(path, bbox_inches="tight", facecolor=canvas, dpi=dpi)
    if close:
        plt.close(fig)
    return paths
