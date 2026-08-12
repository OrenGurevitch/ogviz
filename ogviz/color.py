"""Whether the colours in a figure survive colour-vision deficiency.

Every other check in this package measures where things ARE, and none of them looks at what colour
they are — so a figure whose two series are a confident red and green passes all of them and is a
single grey line to about one man in twelve. (The count used to be written out here as "thirteen",
and there are twenty-one now: a number that has to be maintained by hand in prose is a number that
goes stale, which is the same reason the README's module tree is generated.) It is the most common
accessibility defect in scientific figures and the one an author is least able to see.

The simulation is Viénot, Brettel and Mollon's, the standard construction: take the colour to LMS,
collapse the missing cone's response onto the plane spanned by the other two, come back. It is a
model, not a photograph of anyone's experience — deuteranomaly is a spectrum and this is the
dichromatic end of it — so it answers "could these two be confused" rather than "this is what they
see". For a check on a palette that is the right question.

Implemented here rather than taken from `colorspacious` or `daltonlens`, both of which do this
properly and more: the matrices are published and twenty lines, and a figure package should not
make a project install a colour-science stack to be told its two series look alike. Use those
libraries when the answer needs to be exact.

WHAT THAT SIMPLIFICATION ACTUALLY COSTS — measured 2026-08-12 against `daltonlens`'s Brettel (1997)
implementation, which is the two-half-plane construction the literature reserves for tritanopia and
which this module does not use. It had been carried as an open worry for a week on the theory that
tritanopia was the weak leg and that a correction might make the shipped palettes report
themselves. Both halves of that were wrong:

  the 114 pairs in SERIES and LINE_SERIES   0 change verdict at the 0.18 threshold
  tritanopia, the deficiency under suspicion  mean |difference| 0.023, and the LEAST affected
  protanopia, which nobody suspected          mean |difference| 0.082, max 0.270
  the tightest pair anywhere, either way      0.216 here, 0.219 under Brettel, both clear of 0.18
  the specific fear, blue against green       moves UP under Brettel (0.232 -> 0.288), not down

On a CONSUMER's own colours, which is the case that matters since a caller passes whatever they
like, 4000 random pairs per deficiency put it at: of the pairs Brettel calls confusable this
catches **88.1%**, missing 0.85% of all pairs and over-reporting 1.32%. So the simplification loses
about one in eight true positives and errs toward complaining in volume. For a screening check
whose every complaint ends in "tell them apart by marker or dash as well", that is the right side
to be wrong on — and it is now a measured limitation rather than an unexamined one.

THE METRIC IS THE SAME STORY. `separation` is Euclidean in sRGB where the accepted choice is a
uniform space; Petroff's accessible-sequence work builds palettes in CAM02-UCS for exactly that
reason. Measured on the same 114 pairs, the two rank pairs 0.896 alike, and a CAM02-UCS threshold
anywhere in (5.2, 8.2) reproduces every verdict the 0.18 sRGB threshold gives — 5.2 being
matplotlib's own red/green, which must be caught, and 8.2 the tightest shipped pair, which must
not be. Switching would change the units, force the threshold to be re-derived, add a dependency,
and return the same answers on everything that ships.

`indistinguishable_series` reports pairs, never a verdict on a palette. A figure may legitimately
use two colours that converge if a marker or a dash tells them apart, and this cannot see that.

HOW MANY SERIES A PALETTE CAN CARRY AT ALL — measured 2026-08-10, while extending the line palette
from five colours to eight. Of 4320 candidates spanning the hue circle at five saturations and six
values, the 1019 that stay clear of all five originals under normal vision AND all three simulated
deficiencies contain **no new hue family**. Twelve hand-picked roses, indigos and moss greens were
tried first and all twelve failed: the roses collapse onto the teal under tritanopia, the greens
onto the orange-red under deuteranopia. What survives is darker, duller versions of hues already in
use — the paired-palette pattern, and the only thing the constraint permits.

So **about five is the ceiling for colour alone**, and eight is the honest maximum with paired
tones. That is a fact about categorical palettes under CVD constraints rather than about this
palette, and it is the reason a ten-condition figure cannot be fixed by choosing better colours:
past that many categories the second channel is not optional. A dash, a marker, a facet, or direct
labelling is what carries the distinction, and colour becomes redundant encoding rather than the
encoding. Every complaint this module raises already ends by saying so for the pair it names; this
is the same advice at the level of the whole figure, where no per-pair check can reach it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.colors import to_rgb

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
# In sRGB as written; below this two colours are hard to tell apart. Chosen from measurement, not
# from taste, once `simulate` was corrected to work in light rather than in encoded values:
#
#   must be caught    matplotlib's own red/green   0.165    the textbook confusable pair
#                     a traffic red/green          0.123
#                     the violet/blue this palette 0.062    already fixed, and it must stay caught
#                     dropped
#   must not be       the tightest pair in SERIES  0.216    #14A97C against #9B3B8F
#                     the tightest in the line wheel 0.216
#
# So anything in (0.165, 0.216] separates the two sets, and 0.18 sits in it with room either side.
# The previous 0.12 was set against the ENCODED distances and let matplotlib's red/green through —
# the open question of 2026-07-31, which measurement in the right space now answers.
CONFUSABLE_DISTANCE = 0.18


def _to_linear(channels: NDArray[np.float64]) -> NDArray[np.float64]:
    """sRGB as written -> sRGB as light, undoing the transfer function.

    The cone matrices below are a statement about LIGHT reaching the eye, and a hex colour is not
    that: sRGB stores a gamma-encoded value, so `#808080` is about 22% of the light of `#FFFFFF`
    rather than 50%. Feeding the encoded number straight into the matrices simulates a colour
    nobody displayed, and the error is largest in the mid-tones, which is where a categorical
    palette lives.

    It went unnoticed for as long as it did because the output looks entirely plausible — the
    simulation still collapses reds and greens onto one another, just by the wrong amount. Measured
    on matplotlib's own red and green under deuteranopia: 0.145 apart encoded, 0.165 apart done
    properly, a 14% error in a number a threshold is compared against.
    """
    return np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)


def _to_srgb(channels: NDArray[np.float64]) -> NDArray[np.float64]:
    """Light -> sRGB as written. The inverse of `_to_linear`, so a simulation comes back comparable
    with the hex colours it started from."""
    clipped = np.clip(channels, 0.0, 1.0)
    return np.where(clipped <= 0.0031308, clipped * 12.92, 1.055 * clipped ** (1 / 2.4) - 0.055)


def simulate(color: str | tuple[float, float, float], deficiency: Deficiency) -> tuple[float, ...]:
    """`color` as a dichromat with `deficiency` would distinguish it.

    Returned in sRGB, the space it was given in, so the result can be handed to matplotlib or
    compared with the original without a further conversion.
    """
    if deficiency not in _COLLAPSE:
        raise AssertionError(f"unknown deficiency {deficiency!r}")
    rgb = _to_linear(np.asarray(to_rgb(color), dtype=float))
    lms = _TO_LMS @ rgb
    seen = _FROM_LMS @ (_COLLAPSE[deficiency] @ lms)
    return tuple(float(channel) for channel in _to_srgb(seen))


def separation(first: str, second: str, deficiency: Deficiency | None = None) -> float:
    """How far apart two colours are, optionally as a dichromat sees them.

    Euclidean in sRGB, which is a crude perceptual metric and an honest one for a threshold: it
    never claims two colours are further apart than they look, only sometimes closer. Both ends are
    measured in the SAME space — the simulation comes back in sRGB — so a distance means the same
    thing whether or not a deficiency was asked for.
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
