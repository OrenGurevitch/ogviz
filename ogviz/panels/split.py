"""Two half-violins back to back, for a paired comparison at each category.

The case this exists for: the same quantity measured two ways on the same subjects — two sensors,
two pipelines, two raters. Side-by-side violins put a gap between things that should be read as
one number measured twice, and a difference of a fraction of a percent disappears into that gap.
Sharing a spine puts the two distributions edge to edge, where a real difference in shape is
visible and an identical pair looks identical.

Each half keeps its own IQR bar, mean line, median dot and printed mean, so the halves are
comparable and neither borrows the other's summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.marks import (
    BOX_COLOR,
    MEAN_HALF_WIDTH,
    VIOLIN_ALPHA,
    Z_IQR,
    Z_MEAN_LINE,
    Z_MEDIAN_DOT,
    Z_VIOLIN,
)
from ogviz.orientation import violin_orientation_kwarg
from ogviz.theme import KNOCKOUT_PAD, VALUE_LABEL_SIZE, page_color

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

Side = int  # -1 for the left half, +1 for the right
SPINE_GAP = 0.004  # keeps the two halves from sharing a pixel column on the centre line
MARK_OFFSET = 0.055  # how far each half's marks sit from the shared spine
PAIR_WIDTH = 0.86  # the width of a PAIR, so each half is half of it and neighbours stay clear


def half_violin(
    ax: Axes,
    values: NDArray[np.float64],
    position: float,
    color: str,
    side: Side,
    *,
    width: float = PAIR_WIDTH,
    alpha: float = VIOLIN_ALPHA,
) -> None:
    """One side of a violin, clipped at the centre line.

    `width` is the width of the whole PAIR, so each half gets half of it. Passing the single-violin
    width here is what made the first version overlap its neighbours: a half as wide as a whole
    violin puts 1.24 units of body into a 1.0-unit slot.

    matplotlib has no half violin, so this draws the whole body and clamps the vertices on one
    side back to the centre. Clipping the vertices rather than the artist keeps the outline exact:
    a rectangular clip path would cut the body off at the axes' pixel grid instead of at the
    position.
    """
    assert side in (-1, 1), f"side must be -1 (left) or +1 (right), got {side}"
    parts = ax.violinplot(
        [values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        **violin_orientation_kwarg("vertical"),  # type: ignore[arg-type]
    )
    edge = position + side * SPINE_GAP
    for body in parts["bodies"]:  # type: ignore[union-attr]
        vertices = body.get_paths()[0].vertices
        if side < 0:
            vertices[:, 0] = np.minimum(vertices[:, 0], edge)
        else:
            vertices[:, 0] = np.maximum(vertices[:, 0], edge)
        body.set_facecolor(color)
        body.set_alpha(alpha)
        body.set_edgecolor("none")
        body.set_zorder(Z_VIOLIN)


def half_marks(
    ax: Axes,
    values: NDArray[np.float64],
    position: float,
    side: Side,
    color: str,
    *,
    offset: float = MARK_OFFSET,
    mean_half_width: float = MEAN_HALF_WIDTH,
) -> None:
    """The IQR bar, mean line and median dot for one half, set off from the shared spine."""
    q1, median, q3 = (float(x) for x in np.percentile(values, [25, 50, 75]))
    at = position + side * offset
    ax.plot([at, at], [q1, q3], color=BOX_COLOR, lw=5.0, zorder=Z_IQR, solid_capstyle="round")
    ax.plot(
        [position, position + side * mean_half_width * 2],
        [float(np.mean(values))] * 2,
        color=color,
        lw=3.0,
        solid_capstyle="round",
        zorder=Z_MEAN_LINE,
    )
    ax.plot(
        [at], [median], "o", mfc=page_color(), mec=BOX_COLOR, mew=1.4, ms=7.0, zorder=Z_MEDIAN_DOT
    )


def split_violins(
    ax: Axes,
    categories: Sequence[str],
    left: Sequence[NDArray[np.float64]],
    right: Sequence[NDArray[np.float64]],
    *,
    left_color: str,
    right_color: str,
    width: float = PAIR_WIDTH,
    show_means: bool = True,
    mean_decimals: int | None = None,
    display_scale: float = 1.0,
    bottom_pad: float = 0.28,
    headroom: float = 0.12,
) -> None:
    """A back-to-back pair of half-violins per category, each half with its own marks.

    `left` and `right` hold one array per category, in the same order as `categories`; they do not
    have to be the same length as each other, since a paired measurement can drop samples on one
    side.
    """
    assert len(left) == len(right) == len(categories), (
        f"{len(categories)} categories, {len(left)} left and {len(right)} right series"
    )
    for name, side in (("left", left), ("right", right)):
        for index, values in enumerate(side):
            missing = int(np.count_nonzero(~np.isfinite(np.asarray(values, dtype=float))))
            assert not missing, (
                f"{name} series for {categories[index]!r} has {missing} non-finite value(s); drop "
                "or impute them in the project, where the choice is visible."
            )

    positions = np.arange(len(categories), dtype=float)
    every = np.concatenate([np.asarray(v, dtype=float) for v in (*left, *right)])
    low, high = float(every.min()), float(every.max())
    span = max(high - low, 1e-9)
    ax.set_ylim(low - bottom_pad * span, high + headroom * span)
    ax.set_xlim(-0.5 - width / 2, positions[-1] + 0.5 + width / 2)

    for at, values in zip(positions, left, strict=True):
        half_violin(ax, np.asarray(values, dtype=float), at, left_color, -1, width=width)
        half_marks(ax, np.asarray(values, dtype=float), at, -1, left_color)
    for at, values in zip(positions, right, strict=True):
        half_violin(ax, np.asarray(values, dtype=float), at, right_color, +1, width=width)
        half_marks(ax, np.asarray(values, dtype=float), at, +1, right_color)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(categories))
    if show_means:
        _printed_pairs(
            ax,
            positions,
            left,
            right,
            left_color=left_color,
            right_color=right_color,
            y=low - bottom_pad * span * 0.62,
            decimals=mean_decimals,
            scale=display_scale,
        )


def _printed_pairs(
    ax: Axes,
    positions: NDArray[np.float64],
    left: Sequence[NDArray[np.float64]],
    right: Sequence[NDArray[np.float64]],
    *,
    left_color: str,
    right_color: str,
    y: float,
    decimals: int | None,
    scale: float,
) -> None:
    """Both means under each category, each in its own half's colour so neither is ambiguous."""
    from ogviz.layout.ticks import auto_decimals, format_value

    means = [float(np.mean(v)) * scale for v in (*left, *right)]
    if decimals is None:
        decimals = auto_decimals(max((abs(m) for m in means if m), default=1.0))
    for at, low_values, high_values in zip(positions, left, right, strict=True):
        for side, values, color in ((-1, low_values, left_color), (1, high_values, right_color)):
            ax.text(
                at + side * 0.22,
                y,
                format_value(float(np.mean(values)), scale=scale, decimals=decimals),
                ha="center",
                va="center",
                fontsize=VALUE_LABEL_SIZE * 0.85,
                fontweight="bold",
                color=color,
                zorder=9,
                bbox={
                    "facecolor": page_color(),
                    "edgecolor": "none",
                    "pad": KNOCKOUT_PAD,
                    "boxstyle": "square",
                },
            )
