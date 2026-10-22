"""Ticks for an axis that ends at a ceiling, which a locator does not know it has.

`MaxNLocator` brackets the DATA: it proposes one tick below the lowest value and one above the
highest. That is right for a measurement with no limit and wrong for two kinds of axis that two
projects each had to correct by hand.

A BOUNDED INSTRUMENT, such as a rating that runs 0 to 9. A tick at 10 is a value the instrument
cannot produce, so it reads as if the scale continued, and clipping the proposal to 8 loses the top
of the scale, which is what a reader needs to see that ratings reached the ceiling. `scale_ticks`
lands on both endpoints.

A BAR PANEL, whose bars grow from zero. A tick above the last value is room that the bars never
use, and a whisker or label floating above the highest tick reads as a mistake in the axis.
`ceiling_ticks` stops at the first tick that covers the data, so the headroom held open for a
bracket stays unlabelled.
"""

from __future__ import annotations

from matplotlib.ticker import MaxNLocator

from ogviz.require import require

SCALE_STEPS = (1, 2, 2.5, 3, 4, 5, 10, 20, 25, 50)  # finest first; the first that fits wins
CEILING_STEPS = (1, 2, 5, 10)
CEILING_BINS = 5


def scale_ticks(ceiling: float, *, max_ticks: int = 6) -> list[float]:
    """Ticks from 0 to `ceiling`, landing on both ends, for a bounded instrument.

    Picks the finest step that divides the range exactly and leaves at most `max_ticks` ticks.
    For 9 that is 3, giving 0, 3, 6, 9: step 1 gives ten ticks, step 1.5 prints a fraction on an
    integer scale, and steps 2 and 4 both miss the 9. When nothing divides the range the two
    endpoints are returned alone, which is honest and never wrong.
    """
    require(ceiling > 0, f"scale_ticks needs a positive ceiling, got {ceiling}")
    for step in SCALE_STEPS:
        count = ceiling / step
        if abs(count - round(count)) < 1e-9 and round(count) + 1 <= max_ticks:
            return [index * step for index in range(round(count) + 1)]
    return [0.0, float(ceiling)]


def ceiling_ticks(high: float, *, bins: int = CEILING_BINS) -> list[float]:
    """Round ticks from 0 up to the first one at or above `high`, and no further.

    Nothing printed floats above the last labelled tick: a bar, a whisker and a value label all end
    at or below it. Never further, so the headroom a bracket stack needs stays unlabelled instead
    of being offered as a scale. `high` is the top of the DATA, whiskers included.
    """
    require(high > 0, f"ceiling_ticks needs a positive top of data, got {high}")
    ticks = [
        float(tick)
        for tick in MaxNLocator(nbins=bins, steps=list(CEILING_STEPS)).tick_values(0, high)
        if tick >= -1e-9
    ]
    first_covering = next(index for index, tick in enumerate(ticks) if tick >= high - 1e-9)
    return ticks[: first_covering + 1]
