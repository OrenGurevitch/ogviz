"""Round-number ticks, and the display scale a stored unit is not always written in.

Two facilities that a figure needs and matplotlib does not give directly.

`round_ticks` picks exactly N round values inside the axis, instead of however many matplotlib's
locator lands on. Four labelled gridlines read; nine do not, and the count changing between panels
of one figure reads as an error.

`display_scale` exists because a value's stored unit and its printed unit differ more often than
not. A trace quantity is stored in ppm and written in ppb; a volume is stored in mm3 and
sometimes written in mL. Without it an axis labelled "ppb" carries -0.002 and a printed mean reads
"-0.00" — the number is right and the figure is wrong, which is worse than a crash. The SCALE is a
display fact, so it lives here; the DATA is never touched.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ogviz.orientation import is_vertical, require_linear_value_axis, value_span
from ogviz.require import require

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ogviz.orientation import Orientation

_NICE_MULTIPLES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
_NICE_STEPS = sorted({m * 10.0**k for k in range(-9, 10) for m in _NICE_MULTIPLES})
INSET_FRACTION = 0.10  # keep ticks off the very ends, where a label collides with the frame


def round_ticks(low: float, high: float, count: int) -> list[float]:
    """Exactly `count` round values inside [low, high], inset from both ends."""
    require(
        count >= 2,
        "round_ticks needs at least two ticks",
    )
    require(
        high > low,
        f"round_ticks got an empty range [{low}, {high}] and would return {count} identical "
        "ticks, which matplotlib draws as one label.",
    )
    margin = (high - low) * INSET_FRACTION
    inner_low, inner_high = low + margin, high - margin
    for step in reversed(_NICE_STEPS):
        first = math.ceil(inner_low / step) * step
        if first + (count - 1) * step <= inner_high:
            return [first + index * step for index in range(count)]
    even = (inner_high - inner_low) / (count - 1)
    return [inner_low + even * index for index in range(count)]


MINUS = "\u2212"  # the typographic minus matplotlib sets its own tick labels with


def typeset(text: str) -> str:
    """Swap the ASCII hyphen for a real minus, matching what matplotlib does to a tick label.

    `"{:.2f}".format(-0.3)` writes a hyphen; matplotlib writes U+2212 on the axis. A panel that
    prints its values therefore lands both glyphs in one figure, at two different widths, for the
    same sign. `axes.unicode_minus` is the rcParam that decides, so this follows it rather than
    forcing the substitution.
    """
    import matplotlib as mpl

    return text.replace("-", MINUS) if mpl.rcParams["axes.unicode_minus"] else text


def format_value(
    value: float,
    *,
    scale: float = 1.0,
    decimals: int | None = None,
    thousands_separator: bool = True,
    strip_trailing_zeros: bool | None = None,
) -> str:
    """A value in its DISPLAY unit. `scale` converts the stored unit; the data is untouched.

    A thousands separator is the HOUSE DEFAULT: from 1000 up, a number is grouped. Below 1000 the
    setting changes nothing, so this is a decision about large numbers only — and there, an
    ungrouped "1200000" has to be counted digit by digit while "1,200,000" is read at a glance.

    Pass `thousands_separator=False` for the cases where grouping is wrong: a year, an identifier, a
    part number. Those are not quantities, and nothing here can tell them apart from one.
    """
    scaled = value * scale
    chosen_here = decimals is None
    if decimals is None:
        decimals = auto_decimals(scaled)
    text = format(scaled, f"{',' if thousands_separator else ''}.{decimals}f")
    # Stripped when the decimals were chosen HERE, kept when the caller stated a count. Stating a
    # count is stating a precision, and a row states it once for the whole row: padding [1.0, 2.5]
    # to "1.00" and "2.50" is what makes them read as one measurement, and stripping would leave
    # "1" beside "2.5". Alone, "0.550" has no such excuse — `auto_decimals` aims at three
    # significant figures, and the third one there is a zero nobody measured.
    if strip_trailing_zeros is None:
        strip_trailing_zeros = chosen_here
    # TWO rules about a zero, and conflating them is what hid the sign of a real measurement.
    #
    # A value that IS zero must not carry a minus. `-0.0` is a float artefact — it turns up at a
    # zero tick and in a difference that cancelled — and a signed "-0.00" on an axis reads as a
    # measurement. The sign goes; the decimals stay, so a row that stated a precision keeps it.
    if scaled == 0.0:
        text = format(0.0, f"{',' if thousands_separator else ''}.{decimals}f")
    if strip_trailing_zeros and "." in text:
        stripped = text.rstrip("0").rstrip(".")
        # A value that merely ROUNDS to zero at the chosen decimals is a different thing, and
        # stripping it flat is how its sign disappeared: -0.0014 at two decimals is "-0.00", which
        # says "small and negative", and "-0" says nothing at all. Measured — before this,
        # `format_value(-0.0014, decimals=2, strip_trailing_zeros=True)` returned "0".
        if scaled == 0.0 or stripped not in ("-0", "0"):
            text = stripped
    return typeset(text)


def auto_decimals(value: float) -> int:
    """Decimal places that keep `value` informative: more for small magnitudes, fewer for large."""
    magnitude = abs(value)
    if not math.isfinite(magnitude) or magnitude == 0:
        return 2
    return int(min(max(2 - math.floor(math.log10(magnitude)), 0), 6))


def value_ticks(
    ax: Axes,
    *,
    count: int = 4,
    scale: float = 1.0,
    decimals: int | None = None,
    thousands_separator: bool = True,
    strip_trailing_zeros: bool | None = None,
    orientation: Orientation = "vertical",
) -> list[float]:
    """Place `count` round ticks on the value axis, labelled in the display unit.

    Call it after the limits are final — it reads them. Returns the tick positions in DATA units,
    so a caller can reuse them; the labels are what carry the scale.

    `strip_trailing_zeros` defaults to `format_value`'s rule rather than to True, which is the fix
    for a contradiction between the two: `format_value` documents that a decimal count the CALLER
    stated is a statement of precision and must be kept, and this function then passed True
    unconditionally and threw it away. Measured — `value_ticks(ax, count=4, decimals=2)` on a 0-3
    axis wrote `0.6 1.2 1.8 2.4` where the caller had asked for `0.60 1.20 1.80 2.40`. The argument
    was supported, documented, and overridden by its own caller.
    """
    require_linear_value_axis(ax, orientation, "value_ticks")
    low, high = value_span(ax, orientation)
    positions = round_ticks(float(low), float(high), count)
    labels = [
        format_value(
            position,
            scale=scale,
            decimals=decimals,
            thousands_separator=thousands_separator,
            strip_trailing_zeros=strip_trailing_zeros,
        )
        for position in positions
    ]
    if is_vertical(orientation):
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    return positions
