"""Whether the colours in a figure survive colour-vision deficiency.

Thirteen checks in this package measure where things are and none of them looks at what colour they
are, so a figure whose two series are a confident red and green passes every one of them and is a
single grey line to about one man in twelve. It is the most common accessibility defect in
scientific figures and the one an author is least able to see.

The simulation is Viénot, Brettel and Mollon's, the standard construction: take the colour to LMS,
collapse the missing cone's response onto the plane spanned by the other two, come back. It is a
model, not a photograph of anyone's experience — deuteranomaly is a spectrum and this is the
dichromatic end of it — so it answers "could these two be confused" rather than "this is what they
see". For a check on a palette that is the right question.

Implemented here rather than taken from `colorspacious` or `daltonlens`, both of which do this
properly and more: the matrices are published and twenty lines, and a figure package should not
make a project install a colour-science stack to be told its two series look alike. Use those
libraries when the answer needs to be exact.

`indistinguishable_series` reports pairs, never a verdict on a palette. A figure may legitimately
use two colours that converge if a marker or a dash tells them apart, and this cannot see that.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.colors import to_rgb

Deficiency = Literal["deuteranopia", "protanopia", "tritanopia"]

# sRGB -> LMS (Hunt-Pointer-Estevez, normalised to D65) and back.
_TO_LMS = np.array(
    [
        [0.31399022, 0.63951294, 0.04649755],
        [0.15537241, 0.75789446, 0.08670142],
        [0.01775239, 0.10944209, 0.87256922],
    ]
)
_FROM_LMS = np.linalg.inv(_TO_LMS)
# Each collapses one cone's response onto the plane the other two span.
_COLLAPSE = {
    "protanopia": np.array([[0.0, 1.05118294, -0.05116099], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "deuteranopia": np.array([[1.0, 0.0, 0.0], [0.9513092, 0.0, 0.04866992], [0.0, 0.0, 1.0]]),
    "tritanopia": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.86744736, 1.86727089, 0.0]]),
}
CONFUSABLE_DISTANCE = 0.12  # in linear sRGB; below this two colours are hard to tell apart


def simulate(color: str | tuple[float, float, float], deficiency: Deficiency) -> tuple[float, ...]:
    """`color` as a dichromat with `deficiency` would distinguish it."""
    assert deficiency in _COLLAPSE, f"unknown deficiency {deficiency!r}"
    rgb = np.asarray(to_rgb(color), dtype=float)
    lms = _TO_LMS @ rgb
    seen = _FROM_LMS @ (_COLLAPSE[deficiency] @ lms)
    return tuple(float(np.clip(channel, 0.0, 1.0)) for channel in seen)


def separation(first: str, second: str, deficiency: Deficiency | None = None) -> float:
    """How far apart two colours are, optionally as a dichromat sees them.

    Euclidean in sRGB, which is a crude perceptual metric and an honest one for a threshold: it
    never claims two colours are further apart than they look, only sometimes closer.
    """
    left = np.asarray(simulate(first, deficiency) if deficiency else to_rgb(first), dtype=float)
    right = np.asarray(simulate(second, deficiency) if deficiency else to_rgb(second), dtype=float)
    return float(np.linalg.norm(left - right))


def indistinguishable_series(
    colors: dict[str, str], *, threshold: float = CONFUSABLE_DISTANCE
) -> list[str]:
    """Named colours that separate for normal vision and converge under some deficiency.

    Only reports a pair that is currently FINE and stops being fine — a palette whose colours are
    already close is the caller's own choice and they can see it. What they cannot see is the pair
    that looks well separated on their screen and merges for a reader.
    """
    names = list(colors)
    complaints: list[str] = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if separation(colors[first], colors[second]) < threshold:
                continue  # already close; the author can see that for themselves
            for deficiency in ("deuteranopia", "protanopia", "tritanopia"):
                apart = separation(colors[first], colors[second], deficiency)  # type: ignore[arg-type]
                if apart < threshold:
                    complaints.append(
                        f"{first!r} and {second!r} are distinct now and {apart:.2f} apart under "
                        f"{deficiency} — tell them apart by marker or dash as well as colour"
                    )
                    break
    return complaints
