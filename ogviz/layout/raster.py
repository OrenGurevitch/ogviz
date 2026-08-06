"""Reading the rendered figure back as pixels, in one place.

Three functions did this — `density.ink_mask`, `ink._render` and `ink._frame` — each with its own
`getattr(fig.canvas, "buffer_rgba", None)`, its own assertion message, and, between the two modules,
its own idea of how far from the page colour a pixel has to be before it counts as ink. Two
tolerances, 10 and 12, neither chosen against the other: the same faint antialiased pixel was ink to
one check and page to its neighbour, which is the failure the tag vocabulary was written to end, one
layer down.

The tolerance is the strict one of the two. It decides whether a pixel COUNTS, so being generous
about it makes `colliding_ink` miss faint contact — a real defect — while the only thing it costs on
the other side is a slightly smaller dead-space note, which never fails anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

INK_TOLERANCE = 10  # 0-255 per channel; within this of the page colour, a pixel is the page


def frame_rgb(fig: Figure) -> NDArray[np.int16]:
    """The figure as rendered, RGB only, drawn fresh.

    Refuses a canvas it cannot read rather than letting the failure surface as an `AttributeError`
    from inside a QC helper, naming a matplotlib internal and saying nothing about what to do.
    """
    fig.canvas.draw()
    read_back = getattr(fig.canvas, "buffer_rgba", None)
    if read_back is None:
        raise AssertionError(
            "measuring ink needs a raster canvas — run under Agg (matplotlib.use('Agg')), which "
            f"is what the figure builders do; this figure has a {type(fig.canvas).__name__}"
        )
    return np.asarray(read_back(), dtype=np.int16)[:, :, :3]


def ink_of(frame: NDArray[np.int16], *, tolerance: int = INK_TOLERANCE) -> NDArray[np.bool_]:
    """True where the frame differs from its own page colour.

    The page colour comes from the corner pixel of the render rather than from rcParams, so a figure
    that set its own facecolor, or that was saved with one, is measured against what it actually is.
    """
    return np.any(np.abs(frame - frame[0, 0, :]) > tolerance, axis=2)
