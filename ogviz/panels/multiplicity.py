"""How many of a family's findings survive being a family.

Fifteen tests, each with a star beside it, read as fifteen findings. They are not: with fifteen
independent tests at 0.05, the chance of at least one false positive is 1 - 0.95^15, about 54%.
This panel draws the correction rather than asserting it — every p in the family, sorted, against
the two thresholds a reader is likely to have heard of.

  Bonferroni    a flat line at alpha/n. One threshold for the whole family, controlling the chance
                of ANY false positive.
  Benjamini-Hochberg  a ramp, alpha * rank/n. Controls the expected FRACTION of the findings that
                are false, which is the more useful promise when a family is a screen.

The ramp crossing the sorted p-values is the whole point: BH's cutoff is the LARGEST rank whose p
clears alpha * rank/n, and everything at or below that rank is declared, including points sitting
above the line at smaller ranks. Drawing the ramp lets a reader see that; a table of adjusted
p-values hides it.

The panel computes both from the p-values it is given. Nothing here does the inference — the caller
decides which correction its analysis used, and the panel says what each would have declared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout import hairline_grid, legend_pill
from ogviz.orientation import stamp_orientation
from ogviz.require import require
from ogviz.theme import INK, MUTED_INK

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

ALPHA = 0.05
DECLARED_COLOR = "#4A6136"  # survives the correction
REJECTED_COLOR = "#B4B0A4"  # does not
POINT_SIZE = 62.0


def bonferroni_threshold(count: int, *, alpha: float = ALPHA) -> float:
    """The one cutoff every test in a family of `count` must clear."""
    require(
        count > 0,
        "a family needs at least one test",
    )
    return alpha / count


def benjamini_hochberg_rank(sorted_p: NDArray[np.float64], *, alpha: float = ALPHA) -> int:
    """How many of the sorted p-values BH declares: the largest rank with p <= alpha * rank / n.

    Returns 0 when nothing survives. The LARGEST such rank, not the first failure — a p-value above
    the ramp at rank 3 is still declared if rank 7 clears it, which is the step that makes BH more
    powerful than a fixed cutoff and the step a reader cannot see in a table.
    """
    count = len(sorted_p)
    require(
        count > 0,
        "a family needs at least one test",
    )
    ramp = alpha * np.arange(1, count + 1) / count
    clearing = np.nonzero(sorted_p <= ramp)[0]
    return int(clearing[-1]) + 1 if clearing.size else 0


def multiplicity_ladder(
    ax: Axes,
    p_values: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    alpha: float = ALPHA,
    show_bonferroni: bool = True,
    declared_color: str = DECLARED_COLOR,
    rejected_color: str = REJECTED_COLOR,
    legend: bool = True,
    grid: bool = True,
    label_rotation: float = 90.0,
) -> int:
    """Every p in a family, sorted, against the thresholds. Returns how many BH declares.

    Colour marks what survives BH, since that is the correction whose cutoff is not obvious by eye.
    Bonferroni is drawn as a flat line and left uncoloured: where it sits relative to each point is
    plain, and colouring the same points twice would say which correction the caller used, which
    this panel does not know.
    """
    values = np.asarray(p_values, dtype=float)
    require(
        values.ndim == 1 and values.size,
        "multiplicity_ladder needs a family of p-values",
    )
    require(
        np.all((values >= 0.0) & (values <= 1.0)),
        f"p-values must be in [0, 1]; got {values[(values < 0.0) | (values > 1.0)][:3]}",
    )
    require(
        labels is None or len(labels) == len(values),
        f"{len(labels or ())} labels for {len(values)} p-values",
    )

    order = np.argsort(values)
    sorted_p = values[order]
    count = len(sorted_p)
    ranks = np.arange(1, count + 1)
    declared = benjamini_hochberg_rank(sorted_p, alpha=alpha)

    stamp_orientation(ax, "vertical")
    if grid:
        hairline_grid(ax, axis="y")

    ax.plot(
        ranks,
        alpha * ranks / count,
        color=INK,
        linewidth=1.8,
        label=f"Benjamini-Hochberg ({alpha:g} x rank/{count})",
        zorder=3,
    )
    if show_bonferroni:
        ax.axhline(
            bonferroni_threshold(count, alpha=alpha),
            color=MUTED_INK,
            linewidth=1.6,
            linestyle=(0, (5, 3)),
            label=f"Bonferroni ({alpha:g}/{count})",
            zorder=3,
        )
    ax.scatter(
        ranks,
        sorted_p,
        s=POINT_SIZE,
        # A point's colour says whether BH declared it, which is a property of its RANK, so the
        # split is at the rank BH returned rather than at each point's own value.
        c=[declared_color if rank <= declared else rejected_color for rank in ranks],
        edgecolors=INK,
        linewidths=0.8,
        zorder=4,
    )

    if labels is not None:
        ax.set_xticks(ranks)
        # Upright by default. At 45 degrees a family of twelve ordinary names collides — eight
        # complaints from the gate on the first render of this panel — and whether it collides
        # depends on the longest name and the figure width, which the panel does not choose.
        # Vertical labels cannot run into each other however long they are.
        ax.set_xticklabels(
            [labels[int(index)] for index in order],
            rotation=label_rotation,
            ha="center" if label_rotation % 180 == 90 else "right",
        )
    else:
        ax.set_xticks(ranks)
    ax.set_xlim(0.5, count + 0.5)
    ax.set_ylim(0.0, max(float(sorted_p.max()), alpha) * 1.15)
    ax.set_xlabel("Rank")
    ax.set_ylabel("p")
    if legend:
        legend_pill(ax, loc="upper left")
    return declared
