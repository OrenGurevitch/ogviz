"""Whether two series that separate on your screen separate for your reader.

The one property none of the geometric checks can see. A figure whose two series are a confident red
and green passes every measurement in this package and is a single grey line to about one man in
twelve, and it is the defect an author is least able to catch by looking.
"""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING

import numpy as np

from ogviz.color import indistinguishable_series

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure


def _needs_arguments(getter: Callable[..., object]) -> bool:
    """Whether this accessor cannot simply be called."""
    try:
        parameters = signature(getter).parameters
    except (TypeError, ValueError):  # a builtin or C function with no introspectable signature
        return False
    return any(
        parameter.default is parameter.empty
        and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters.values()
    )


def _handle_color(handle) -> str | None:
    """The one colour a legend handle stands for, or None if it does not stand for one."""
    from matplotlib.colors import to_hex

    for getter in ("get_color", "get_facecolor", "get_markerfacecolor"):
        found = getattr(handle, getter, None)
        if found is None:
            continue
        # A getter that needs arguments is not the accessor we are after; matplotlib has a few whose
        # names collide. Checked by ASKING the signature rather than by calling and swallowing what
        # comes back: `except (TypeError, ValueError): continue` also hid a getter that raised for a
        # real reason, and the check would then quietly stop covering that series.
        if _needs_arguments(found):
            continue
        value = found()
        flat = np.asarray(value, dtype=object).ravel()
        if flat.size in (3, 4) and all(isinstance(v, (int, float)) for v in flat):
            red, green, blue = (float(flat[index]) for index in range(3))
            return to_hex((red, green, blue))
        if flat.size == 1 and isinstance(flat[0], str):
            return to_hex(str(flat[0]))
    return None


def series_confusable_under_cvd(fig: Figure) -> list[str]:
    """Legend entries that separate for normal vision and merge under colour-vision deficiency.

    Read off the LEGEND, because that is the set the reader is asked to tell apart. Marks with no
    legend entry are not being distinguished by colour in the first place — a violin's fill and its
    edge are one series in two tones, and reporting them would be noise.

    Advisory in spirit but fails the build like the rest: a pair that merges is a figure a reader
    cannot use, and "add a marker or a dash" is a small change to make while the figure is open.

    EVERY LEGEND ON THE FIGURE, not only the ones an axes owns. A legend attached to the FIGURE
    lives in `fig.legends` and `ax.get_legend()` never returns it — so for as long as this walked
    the axes alone, `legend_pill(fig, ...)`, which this package offers and a grid normally wants,
    produced a legend the check could not see. Measured before the fix: a deliberate red/green pair
    drew "0.16 apart under deuteranopia" through an axes legend and NOTHING through a figure one.
    """
    complaints: list[str] = []
    legends = [legend for ax in fig.axes if (legend := ax.get_legend()) is not None]
    legends.extend(fig.legends)
    for legend in legends:
        entries: dict[str, str] = {}
        # NOT `strict=True`. The two are the same length for every legend matplotlib builds, but a
        # checker that raises where it meant to report turns a colour question into a crashed
        # build, and a custom handler is the caller's business rather than a defect in the figure.
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=False):
            color = _handle_color(handle)
            if color is not None:
                entries[text.get_text()] = color
        if len(entries) > 1:
            complaints.extend(indistinguishable_series(entries))
    return complaints
