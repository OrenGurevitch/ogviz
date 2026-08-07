"""Which screen axis carries the category, and which carries the value.

Every mark in this package draws along two conceptual axes: the CATEGORY axis, where a group sits
and where jitter spreads, and the VALUE axis, where the datum lives. Vertical and horizontal
figures differ only in which screen axis is which. Naming that once means each mark is written
once, rather than as a pair of near-identical branches that drift.

Everything an orientation decides lives here: how to place a point, which limit setter to call,
which spine is the baseline, and which line-drawing call spans a constant.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt

from ogviz.tags import mark, value_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

Orientation = Literal["vertical", "horizontal"]


ORIENTATIONS = ("vertical", "horizontal")


def _check(orientation: Orientation) -> None:
    """Refuse an orientation this package does not know.

    Raised rather than asserted, as everywhere a CALLER's value is checked here: `python -O`
    deletes an `assert`, and a misspelled orientation would then reach matplotlib as a keyword it
    silently ignores, drawing the panel the other way round.
    """
    if orientation not in ORIENTATIONS:
        raise AssertionError(f"unknown orientation {orientation!r}")


def is_vertical(orientation: Orientation) -> bool:
    _check(orientation)
    return orientation == "vertical"


def stamp_orientation(ax: Axes, orientation: Orientation) -> None:
    """Record on the axes which way this panel runs, for anything that has to ask later.

    Every panel here is TOLD its orientation, and until this existed it threw that away, leaving QC
    to infer it from the marks: a vote on two-point lines, constant-x meaning vertical. A grouped
    bar panel defeats that outright. `errorbar` hides its bars and caps in a LineCollection, so the
    only two-point line in the panel is the zero baseline — constant y — and the panel reads as
    horizontal. QC then measured brackets along x, found none bracket-shaped, and reported all five
    stars as having no bracket under them, on a panel whose brackets were present and correct.

    Inference is a fallback for a figure this package did not draw. Where the answer is known it is
    written down.
    """
    _check(orientation)
    mark(ax, "orientation", orientation)


def read_orientation(ax: Axes) -> Orientation | None:
    """The orientation a panel recorded, or None for axes this package did not draw."""
    return value_of(ax, "orientation")


def place(orientation: Orientation, category: float, value: float) -> tuple[float, float]:
    """(x, y) for a point at `category` on the category axis and `value` on the value axis."""
    return (category, value) if is_vertical(orientation) else (value, category)


def place_many(orientation: Orientation, category, value) -> tuple:
    """`place` for arrays — the SAME operation, typed loosely enough to pass sequences through.

    Nothing is vectorised, because there is nothing to vectorise: this swaps two references and
    matplotlib does the rest. The two names exist only so the typed scalar call stays typed, and
    the body is shared so they cannot come to disagree about which way round the pair goes.
    """
    return place(orientation, category, value)  # type: ignore[arg-type]


def value_limits(ax: Axes, orientation: Orientation) -> Callable[..., object]:
    """The setter for the axis the data lives on."""
    return ax.set_ylim if is_vertical(orientation) else ax.set_xlim


def category_limits(ax: Axes, orientation: Orientation) -> Callable[..., object]:
    return ax.set_xlim if is_vertical(orientation) else ax.set_ylim


def value_span(ax: Axes, orientation: Orientation) -> tuple[float, float]:
    return ax.get_ylim() if is_vertical(orientation) else ax.get_xlim()


def category_ticks(ax: Axes, orientation: Orientation) -> Callable[..., object]:
    return ax.set_xticks if is_vertical(orientation) else ax.set_yticks


def category_tick_labels(ax: Axes, orientation: Orientation) -> Callable[..., object]:
    return ax.set_xticklabels if is_vertical(orientation) else ax.set_yticklabels


def constant_value_line(ax: Axes, orientation: Orientation, value: float, **kwargs) -> Line2D:
    """A rule at a fixed value, spanning the whole category axis."""
    return ax.axhline(value, **kwargs) if is_vertical(orientation) else ax.axvline(value, **kwargs)


def value_transform(ax: Axes, orientation: Orientation):
    """Axes-fraction on the category axis, data on the value axis."""
    return ax.get_yaxis_transform() if is_vertical(orientation) else ax.get_xaxis_transform()


# matplotlib renamed violinplot's `vert: bool` to `orientation: str` in 3.11 and deprecates the
# old spelling; 3.10, the supported floor, accepts only `vert`. Both are versions the dependent
# projects run, so pick by signature once rather than catching a TypeError per call — a swallowed
# TypeError would also hide a genuine argument mistake.
_VIOLINPLOT_TAKES_ORIENTATION = "orientation" in inspect.signature(plt.violinplot).parameters


def violin_orientation_kwarg(orientation: Orientation) -> dict[str, object]:
    """The keyword this matplotlib spells `violinplot`'s orientation with."""
    upright = is_vertical(orientation)  # validate here too, or a bad value reaches matplotlib
    if _VIOLINPLOT_TAKES_ORIENTATION:
        return {"orientation": orientation}
    return {"vert": upright}


def value_scale(ax: Axes, orientation: Orientation) -> str:
    """The scale of the axis the data lives on ("linear", "log", ...)."""
    return ax.get_yscale() if is_vertical(orientation) else ax.get_xscale()


def require_linear_value_axis(ax: Axes, orientation: Orientation, what: str) -> None:
    """Refuse a non-linear value axis, loudly, where the maths assumes a linear one.

    Both the mark clearance and the round-number ticks convert between data and pixels with a
    single ratio, which only exists on a linear scale. On a log axis the ratio came back as 0.0
    and the clearance silently collapsed, and the ticks came back as 200/400/600/800 where a log
    axis wants 1/10/100/1000. Neither raised anything.
    """
    scale = value_scale(ax, orientation)
    if scale != "linear":
        raise AssertionError(
            f"{what} assumes a linear value axis and this one is {scale!r}. The data-to-pixel "
            "ratio it needs does not exist on a non-linear scale, and the result would be "
            "silently wrong."
        )
