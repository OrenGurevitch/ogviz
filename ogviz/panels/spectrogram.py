"""Time against frequency, coloured by power — the Short-Time Fourier Transform figure.

The panel for "what frequencies are present, and when". It draws a matrix that a caller has already
computed; the transform itself is not here, for the same reason no p-value is computed anywhere in
this package. Window length, overlap and window shape decide what the picture MEANS, they are the
argument every spectrogram paper spends its methods section on, and a figure library choosing them
silently would be choosing the finding.

Four things separate this from `imshow` of an STFT magnitude, and each is somewhere a hand-rolled
spectrogram goes wrong:

  THE SCALE IS ALWAYS DRAWN. A spectrogram encodes its whole quantity in colour and prints no
  numbers anywhere, so without a key it is a picture rather than a measurement — a reader cannot
  tell 10 dB of dynamic range from 80. This is the difference from `effect_heatmap`, which prints
  the number in every cell and can at least be read without one.

  POWER IS SHOWN IN DECIBELS, because it spans orders of magnitude. On a linear scale a spectrogram
  of anything real is one bright line and a black field: the harmonics that carry the structure are
  three or four decades down and render as the same black as silence.

  ZERO POWER HAS NO LOGARITHM, and a real STFT contains exact zeros — in a padded region, in a
  silent passage, wherever a window sees nothing. `10 * log10(0)` is `-inf`, and matplotlib maps
  a non-finite value to the map's "bad" colour, so untreated those cells come out as holes in a
  colour nobody chose. `to_decibels` floors instead, at a stated number.

  THE FLOOR IS A DECISION AND IS EXPOSED. Everything below it is one colour, so the floor sets how
  much of the quiet part of the signal is visible at all. A default that is silently -120 dB spends
  most of the colour range on noise; one that is silently -20 dB hides the harmonics. It is an
  argument with a documented default rather than a constant.

The frequency axis takes `"log"` for speech and music, where the interesting structure is
multiplicative, and `"linear"` for a signal with harmonics at fixed spacing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.ticker import FuncFormatter

from ogviz.layout.frame import color_scale
from ogviz.layout.ticks import format_value
from ogviz.require import require
from ogviz.theme import MUTED_INK

if TYPE_CHECKING:
    from typing import Literal

    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from numpy.typing import ArrayLike, NDArray

# Taken from matplotlib rather than invented here, and bundled rather than depended on. A
# hand-rolled sequential ramp is the thing this package warns callers against; and a default that
# lives in another distribution would make the rendered gallery depend on whether that distribution
# happens to be installed, which is the one thing a reproducible figure set cannot have.
#
# MEASURED, and the measurements disagree, which is why this comment names both rather than
# declaring a winner:
#
#   adjacent-step separation under simulated CVD   magma 0.159, inferno 0.156, batlow 0.132,
#   (`ogviz.color.separation`, nine stops)         viridis 0.124
#   evenness of CIE L* steps, and lightness range  viridis CV 0.074 / range 75.9, inferno 0.183,
#   (64 stops)                                     magma 0.190 / range 97.7, batlow 0.252 / 75.1
#
# The first metric is sRGB-Euclidean, which this package already records as the wrong space; the
# second reads L* only, where batlow is built for uniformity in full CIELAB. So neither is grounds
# for calling one map better, and the honest summary is that magma, inferno, viridis and batlow are
# all monotonic in lightness and all survive greyscale, while `jet` is not monotonic at all.
# magma is the default for its lightness RANGE — 98 against viridis's 76 — which on a spectrogram
# is dynamic range a reader can actually see.
#
# FOR A FIGURE THAT MUST BE COLOUR-VISION SAFE AND PRINT WELL, Fabio Crameri's scientific colour
# maps are the reference work and are a one-line swap, because `colormap` is just a name:
#
#     pip install cmcrameri
#     spectrogram(ax, ..., colormap="cmc.batlow")
#
# Crameri, F., Shephard, G.E. & Heron, P.J. (2020), "The misuse of colour in science
# communication", *Nature Communications* 11, 5444. See fabiocrameri.ch/colourmaps.
SEQUENTIAL = "magma"

# Decibels below the reference at which everything becomes one colour. 80 dB is the span of a
# 16-bit recording's usable range and the default in every spectrogram tool worth copying; it shows
# harmonics several decades down without spending half the colour range on the noise floor.
FLOOR_DB = -80.0


def to_decibels(
    power: ArrayLike, *, reference: float | None = None, floor_db: float = FLOOR_DB
) -> NDArray[np.float64]:
    """Power to decibels relative to `reference`, with a floor instead of a negative infinity.

    `reference` defaults to the largest value present, so the loudest cell sits at 0 dB and every
    other is read as "this far down from the peak". Pass one to hold a scale across figures — two
    spectrograms with independent references cannot be compared, and they look comparable.

    THE FLOOR IS THE WHOLE POINT. A real STFT contains exact zeros, `log10(0)` is `-inf`, and a
    non-finite value reaches matplotlib as the colormap's "bad" colour — so an untreated
    spectrogram has holes in it, in a colour nobody chose, exactly where the signal is silent.
    Clipping first means silence renders as the bottom of the scale, which is what it is.

    Takes POWER, not amplitude: `10 * log10` rather than `20 * log10`. Handing this the magnitude
    of an STFT instead of its square understates every dynamic range by a factor of two, which is
    the commonest error in a hand-rolled spectrogram and is invisible unless a number is checked.
    Square the magnitude, or pass `numpy.abs(stft) ** 2`.
    """
    values = np.asarray(power, dtype=float)
    require(values.size > 0, "to_decibels needs at least one value")
    require(
        floor_db < 0.0,
        f"the floor is decibels BELOW the reference, so negative; got {floor_db}",
    )

    peak = float(np.nanmax(values)) if reference is None else float(reference)
    require(peak > 0.0, f"the reference power must be positive, got {peak}")
    # Clipped BEFORE the logarithm, so a zero becomes the floor rather than a negative infinity that
    # then has to be repaired. The smallest representable ratio at this floor is what the clip is.
    smallest = peak * 10.0 ** (floor_db / 10.0)
    return 10.0 * np.log10(np.clip(values, smallest, None) / peak)


def spectrogram(
    ax: Axes,
    magnitudes: ArrayLike,
    *,
    times: ArrayLike,
    frequencies: ArrayLike,
    colormap: str = SEQUENTIAL,
    frequency_scale: Literal["linear", "log"] = "linear",
    scale_label: str = "Power (dB)",
    time_label: str = "Time (s)",
    frequency_label: str = "Frequency (Hz)",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
) -> QuadMesh:
    """Draw a precomputed time-frequency matrix, with the colour scale that makes it readable.

    `magnitudes` is `(frequencies, times)` — frequency down the rows, time across the columns,
    which is the shape every STFT implementation returns and therefore the shape not to transpose.
    It is expected in DECIBELS already; run it through `to_decibels` first, which is a separate
    call because the reference and the floor are decisions rather than defaults.

    Drawn with `pcolormesh` rather than `imshow`. The two agree for the uniform grid an STFT
    produces, and only `pcolormesh` is still correct when the axis is logarithmic or the bins are
    not evenly spaced — which is the case the moment a caller switches to a mel or constant-Q
    frequency axis. The cost is a QuadMesh instead of an image, and it is not measurable here.

    Returns the mesh, so a caller can draw over it — a fundamental-frequency track, an onset
    marker, a band of interest.

    KEEP `scale_label` SHORT. It is set along a bar as tall as the panel and rotated, so its length
    competes with the panel height rather than its width — the default was "Power (dB re. peak)"
    and the gate refused the first figure built with it, 3 px over, with the rotated-label hint
    saying that reflowing would make it worse. What the decibels are relative to belongs in the
    caption, where there is room for it.
    """
    values = np.asarray(magnitudes, dtype=float)
    time = np.asarray(times, dtype=float)
    frequency = np.asarray(frequencies, dtype=float)

    require(values.ndim == 2, f"a spectrogram is a 2-D matrix, got shape {values.shape}")
    require(
        values.shape == (frequency.size, time.size),
        f"matrix {values.shape} against {frequency.size} frequencies and {time.size} times — "
        "the matrix is (frequencies, times), which is what every STFT returns; a transpose here "
        "is the usual cause",
    )
    require(time.size > 1 and frequency.size > 1, "a spectrogram needs more than one bin per axis")
    if frequency_scale == "log":
        # A log axis cannot show the DC bin, and an STFT always has one. Left in, matplotlib drops
        # the whole bottom row silently and the figure is missing a band nobody notices is missing.
        require(
            float(frequency[0]) > 0.0,
            "a log frequency axis cannot show the 0 Hz bin — slice it off, and say so in the "
            "caption, rather than letting matplotlib drop it silently",
        )

    mesh = ax.pcolormesh(
        time,
        frequency,
        values,
        cmap=colormap,
        shading="nearest",  # the arrays are bin CENTRES, which is what an STFT reports
        vmin=vmin,
        vmax=vmax,
        rasterized=True,  # see below
    )
    ax.set_yscale(frequency_scale)
    # Held to the bins that exist. `shading="nearest"` widens the mesh by half a bin at each end, so
    # matplotlib's autoscale then offers ticks outside the data — and on a frequency axis the first
    # of those is NEGATIVE, which is not a thing a spectrogram has. Measured on an 8 kHz example:
    # the axis ran to -1,000 Hz before this.
    ax.set_xlim(float(time.min()), float(time.max()))
    ax.set_ylim(float(frequency.min()), float(frequency.max()))
    # Grouped from a thousand, like every other number this package prints. A frequency axis reaches
    # four digits almost immediately — 4000 Hz is an ordinary upper bound for speech — and left to
    # matplotlib it prints "4000", which the house rule calls ungrouped and `ungrouped_thousands`
    # reports. Caught by the gate on the first spectrogram built with this, which is the check doing
    # its job on the package rather than on a caller.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: format_value(value)))
    ax.set_xlabel(time_label)
    ax.set_ylabel(frequency_label)
    ax.tick_params(colors=MUTED_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if colorbar:
        color_scale(ax, mesh, label=scale_label)
    return mesh
