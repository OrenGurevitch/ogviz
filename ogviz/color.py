"""Whether the colours in a figure survive colour-vision deficiency.

Every other check in this package measures where things ARE, and none of them looks at what colour
they are — so a figure whose two series are a confident red and green passes all of them and is a
single grey line to about one man in twelve. It is the most common accessibility defect in
scientific figures and the one an author is least able to see. (A count of the other checks used to
sit in this sentence, and had gone stale twice — `--list-checks` is where that number lives.)

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

  the 38 pairs in SERIES and LINE_SERIES,
  under each of the three deficiencies      0 change verdict at the 0.18 threshold
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
reason. Measured on the same 38 pairs x 3 deficiencies, the two rank pairs 0.896 alike, and a
CAM02-UCS threshold
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

from ogviz.require import require

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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
#   must not be       the tightest pair in either  0.216    #14A97C against #9B3B8F, which are in
#                     palette                                both SERIES and LINE_SERIES — one
#                                                            pair, not two measurements
#
# So anything in (0.165, 0.216] separates the two sets, and 0.18 sits in it with room either side.
# The previous 0.12 was set against the ENCODED distances and let matplotlib's red/green through —
# the open question of 2026-07-31, which measurement in the right space now answers.
CONFUSABLE_DISTANCE = 0.18
# How far past `threshold` a `near=` answer has to clear before it counts as a recommendation.
# 1.5x, because the threshold is a screening line rather than a cliff — `separation` catches 88.1%
# of the pairs Brettel calls confusable, so a colour clearing by a thousandth is inside the
# metric's own error and looks identical to the one it replaced. Measured: asked for the nearest
# safe colour to matplotlib's green, the bare threshold returns 0.181 and this returns 0.276.
NEAR_HEADROOM = 1.5


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


def _as_seen(colors: Sequence[str], deficiency: Deficiency | None) -> NDArray[np.float64]:
    """`colors` as an (n, 3) array of sRGB, through `deficiency` when one is given.

    The same answer `simulate` gives one colour at a time, which
    `test_the_vectorised_search_agrees_with_the_one_at_a_time_metric` holds it to.
    """
    if deficiency is None:
        return np.asarray([to_rgb(one) for one in colors], dtype=float).reshape(-1, 3)
    return np.asarray([simulate(one, deficiency) for one in colors], dtype=float).reshape(-1, 3)


def worst_separation(color: str, taken: Iterable[str]) -> float:
    """The closest `color` comes to anything in `taken`, over normal vision AND all three
    deficiencies.

    The number a candidate colour lives or dies by, and a minimum over both axes because a palette
    is only as good as its tightest pair under its worst reader. Returns `inf` for an empty `taken`,
    which is the honest answer — nothing to collide with — and lets a caller score the first colour
    of a palette with the same call as the sixth.
    """
    others = list(taken)
    if not others:
        return float("inf")
    return min(
        separation(color, other, deficiency)
        for other in others
        for deficiency in (None, "deuteranopia", "protanopia", "tritanopia")
    )


# The grid the 2026-08-10 sweep used, kept because its result is what this module's docstring
# reports: 144 hues around the circle, five saturations, six values, 4320 candidates. Spanning
# saturation and value matters more than hue resolution — what survives the constraint is a darker,
# duller version of a hue already in use, never a new hue family.
_HUES = 144
_SATURATIONS = (0.30, 0.45, 0.60, 0.78, 0.95)
_VALUES = (0.35, 0.48, 0.62, 0.75, 0.86, 0.95)


def separated_from(
    taken: Iterable[str], *, threshold: float = CONFUSABLE_DISTANCE, near: str | None = None
) -> str:
    """A colour as far as possible from every one of `taken`, for normal vision and all three
    deficiencies.

    The other half of `indistinguishable_series`, which can say a pair collides and not what to use
    instead — so a project needing the answer writes this search itself, and the search is the easy
    part. WHAT IS HARD IS THE SET, and getting it wrong is the failure this exists to prevent:
    ranked against a palette constant alone, a winner can collide with a neutral the constant does
    not contain; ranked against every colour in a repository, nothing clears the threshold at all,
    because that set is far larger than any one legend. The set that matters is what shares a
    LEGEND, since that is what a reader is asked to tell apart and what the check compares.
    `qc.color.legend_colors` reads it off a rendered figure, and is how a caller should get it.

    `near` asks for the closest acceptable colour to one you already want — for a colour that has
    to keep a meaning, or sit in a house palette. It ranges over everything that clears the
    threshold WITH HEADROOM, not over the winners: without it the answer is the MOST separated
    candidate, with it the one nearest `near` that is still safe. Those are different questions and
    the second is usually the one a caller with an existing palette is asking.

    The headroom is `NEAR_HEADROOM` and it is the difference between a recommendation and a
    restatement. Asked for the nearest safe colour to matplotlib's green, the bare threshold
    returned one scoring 0.181 against a 0.18 line — which simulates to something a reader cannot
    tell from the colour it replaced, so the "fixed" figure looked exactly like the broken one. At
    1.5x it returns 0.276, which is a visible change. The margin is not caution for its own sake:
    this metric catches 88.1% of the pairs Brettel calls confusable, so clearing by a thousandth is
    inside its own measured error.

    RAISES when nothing clears `threshold`, naming the best it found, rather than handing back a
    colour the gate would then refuse.

    **CLEARING THE THRESHOLD IS NOT THE SAME AS BEING USABLE, and this is the thing to know before
    trusting a long palette it builds.** Measured 2026-08-13 by growing the five-colour `SERIES`
    one call at a time: the threshold keeps clearing well past twelve — the worst-case separation
    falls 0.559, 0.499, 0.404, 0.346, 0.313, 0.305, 0.285 for the 6th through 12th — and the
    resulting twelve draw ZERO complaints from `indistinguishable_series`. What runs out long
    before the number does is HUE. Every pick from the sixth on lands within 30 degrees of a hue
    already in the palette, most within 20, and the fourteenth on a hue already there exactly; the
    later colours are light and dark tones of the earlier ones, which is the same result the
    4320-candidate sweep above reports from the other direction.

    So this will cheerfully return a twelfth colour, and a reader asked to tell a pale cyan from a
    bright cyan across a legend cannot use it. The module docstring's "about five is the ceiling
    for colour alone" is a claim about hue families, not about the threshold, and the threshold
    cannot enforce it. Past five series the second channel is not optional whatever this returns.
    """
    from colorsys import hsv_to_rgb

    from matplotlib.colors import to_hex

    others = [to_hex(one) for one in taken]
    # Refused, because the answer for nothing is not a colour: every score stayed at infinity, the
    # threshold passed vacuously, and this handed back the first grid candidate — a dark maroon —
    # or, with `near=`, a near-copy of `near`, both as though they had been computed. Measured.
    require(others, "separated_from needs at least one colour to be separated from")
    candidates = [
        to_hex(hsv_to_rgb(index / _HUES, saturation, value))
        for index in range(_HUES)
        for saturation in _SATURATIONS
        for value in _VALUES
    ]
    # VECTORISED, and it has to be: scored one `separation` call at a time this is
    # len(candidates) x len(taken) x 4 simulations — about 3.7 million for a palette of 200, which
    # took the better part of a minute. Each side is simulated ONCE per deficiency instead, and the
    # distances fall out of one broadcast subtraction. A helper this slow is a helper nobody calls.
    scores = np.full(len(candidates), np.inf)
    if others:
        for deficiency in (None, "deuteranopia", "protanopia", "tritanopia"):
            seen = _as_seen(candidates, deficiency)  # (candidates, 3)
            against = _as_seen(others, deficiency)  # (taken, 3)
            apart = np.linalg.norm(seen[:, None, :] - against[None, :, :], axis=2)
            scores = np.minimum(scores, apart.min(axis=1))
    scored = list(zip(scores.tolist(), candidates, strict=True))
    best = max(score for score, _ in scored)
    require(
        best >= threshold,
        f"no colour clears {threshold} against these {len(others)}; the best is {best:.3f}. "
        "A palette this full cannot be fixed by choosing better colours — about five series is the "
        "ceiling for colour alone. Tell them apart by marker, dash, facet or direct labelling.",
    )
    if near is None:
        # The most separated, and the grid is walked in a fixed order so ties resolve the same way
        # on every run — a palette helper that returned a different colour each call would make a
        # figure irreproducible for no benefit.
        return next(one for score, one in scored if score >= best - 1e-9)
    # `near` chooses among everything COMFORTABLY acceptable, not among the winners. Ordering only
    # the exact ties at the maximum was the first spelling and it does nothing: on a discrete grid
    # the argmax is normally unique, so `near` changed the answer in none of the cases it was
    # written for.
    #
    # And "acceptable" here is the threshold WITH HEADROOM, which the second spelling did not do.
    # Asked for the nearest safe colour to matplotlib's green, it returned one scoring 0.181
    # against a 0.18 threshold — clearing by 0.001, and indistinguishable from the colour it
    # replaced once simulated. That is not a recommendation, it is the same figure. The margin has
    # to exist because the metric is a SCREENING one and says so: measured against Brettel (1997)
    # it catches 88.1% of confusable pairs, so a colour scraping the line is inside its own error.
    safe = threshold * NEAR_HEADROOM
    acceptable = [one for score, one in scored if score >= safe]
    if not acceptable:  # nothing clears the margin; the plain threshold is still a true answer
        acceptable = [one for score, one in scored if score >= threshold]
    return min(acceptable, key=lambda one: separation(one, near))


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
