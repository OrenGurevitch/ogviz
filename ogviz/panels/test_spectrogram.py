"""The time-frequency panel: the decibel floor, the shape contract, and the scale.

Every assertion here is about something a hand-rolled spectrogram gets wrong silently — a hole where
a zero was, a transposed matrix that still renders, an axis running to a negative frequency.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.panels.spectrogram import FLOOR_DB, spectrogram, to_decibels
from ogviz.qc import audit


def _grid(rows: int = 16, columns: int = 40):
    """A precomputed time-frequency surface: one band, plus a floor of noise."""
    rng = np.random.default_rng(0)
    frequencies = np.linspace(0.0, 4000.0, rows)
    times = np.linspace(0.0, 1.0, columns)
    power = rng.uniform(1e-6, 1e-5, (rows, columns))
    power[rows // 3, :] = 1.0
    return power, times, frequencies


def test_a_zero_becomes_the_floor_rather_than_a_hole() -> None:
    """`log10(0)` is `-inf`, and matplotlib paints a non-finite value in the colormap's "bad"
    colour — so an untreated spectrogram has holes in it, in a colour nobody chose, exactly where
    the signal is silent. A real STFT contains exact zeros in any padded or silent region."""
    power = np.array([[0.0, 0.5, 1.0]])
    decibels = to_decibels(power, floor_db=-60.0)

    assert np.isfinite(decibels).all(), "no cell may be non-finite"
    assert decibels[0, 0] == pytest.approx(-60.0), "silence lands ON the floor"
    assert decibels[0, 2] == pytest.approx(0.0), "the peak is the reference, so 0 dB"


def test_the_reference_can_be_held_across_figures() -> None:
    """Two spectrograms with independent references cannot be compared, and they look comparable."""
    quiet = np.array([[0.01, 0.02]])
    loud = np.array([[1.0, 2.0]])
    assert to_decibels(quiet).max() == pytest.approx(to_decibels(loud).max()), (
        "each normalised to its own peak, both top out at 0 dB — which is the trap"
    )
    shared = to_decibels(quiet, reference=2.0).max()
    assert shared < to_decibels(loud, reference=2.0).max(), "given one reference, the quiet one is"


def test_power_not_amplitude() -> None:
    """`10 * log10`, not `20`. Handing this a magnitude instead of its square halves every dynamic
    range reported, which is invisible unless a number is checked."""
    assert to_decibels(np.array([1.0, 0.5]))[1] == pytest.approx(-3.0103, abs=1e-3)


def test_a_floor_the_wrong_way_round_is_refused() -> None:
    with pytest.raises(AssertionError, match="decibels BELOW"):
        to_decibels(np.array([1.0, 2.0]), floor_db=60.0)


def test_a_reference_of_zero_is_refused() -> None:
    """Dividing by it is the next line; the message beats a RuntimeWarning and a field of nan."""
    with pytest.raises(AssertionError, match="reference power must be positive"):
        to_decibels(np.array([1.0]), reference=0.0)


def test_a_transposed_matrix_is_refused_by_name() -> None:
    """(frequencies, times) is what every STFT returns, and a transpose still renders — as a
    picture of nothing, with the axes silently meaning each other's quantity."""
    power, times, frequencies = _grid()
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    with pytest.raises(AssertionError, match="transpose"):
        spectrogram(ax, to_decibels(power).T, times=times, frequencies=frequencies)
    plt.close(fig)


def test_the_scale_is_drawn_and_the_figure_passes_the_gate() -> None:
    """A spectrogram prints no numbers anywhere, so without the key it is a picture rather than a
    measurement — a reader cannot tell 10 dB of dynamic range from 80."""
    power, times, frequencies = _grid()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    spectrogram(ax, to_decibels(power), times=times, frequencies=frequencies)
    fig.tight_layout()  # the house type is large; a bare figure puts its x-label off the page
    fig.canvas.draw()

    assert len(fig.axes) == 2, "the panel, and the scale beside it"
    assert audit(fig) == []
    plt.close(fig)


def test_the_frequency_axis_does_not_run_negative() -> None:
    """`shading="nearest"` widens the mesh half a bin at each end, and matplotlib's autoscale then
    offers a tick below zero — which is not a frequency a spectrogram has."""
    power, times, frequencies = _grid()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    spectrogram(ax, to_decibels(power), times=times, frequencies=frequencies)
    fig.canvas.draw()

    assert ax.get_ylim()[0] >= 0.0
    drawn = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert not any(label.startswith("\u2212") for label in drawn), drawn
    plt.close(fig)


def test_the_frequency_ticks_are_grouped_from_a_thousand() -> None:
    """The house rule, and the gate refused the first spectrogram built without it: an axis in Hz
    reaches four digits immediately, and matplotlib prints "4000"."""
    power, times, frequencies = _grid()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    spectrogram(ax, to_decibels(power), times=times, frequencies=frequencies)
    fig.canvas.draw()

    drawn = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert any("," in label for label in drawn), drawn
    plt.close(fig)


def test_a_log_frequency_axis_refuses_the_dc_bin() -> None:
    """A log axis cannot show 0 Hz. Left in, matplotlib drops the bottom row and the figure is
    missing a band nobody notices is missing."""
    power, times, frequencies = _grid()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    with pytest.raises(AssertionError, match="0 Hz bin"):
        spectrogram(
            ax,
            to_decibels(power),
            times=times,
            frequencies=frequencies,
            frequency_scale="log",
        )
    plt.close(fig)


def test_a_log_frequency_axis_works_once_the_dc_bin_is_gone() -> None:
    power, times, frequencies = _grid()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    spectrogram(
        ax,
        to_decibels(power)[1:],
        times=times,
        frequencies=frequencies[1:],
        frequency_scale="log",
    )
    fig.tight_layout()
    fig.canvas.draw()
    assert ax.get_yscale() == "log"
    assert audit(fig) == []
    plt.close(fig)


def test_the_default_floor_is_where_it_says_it_is() -> None:
    """A constant that decides how much of the quiet part is visible at all."""
    assert FLOOR_DB == -80.0
    assert to_decibels(np.array([[0.0, 1.0]])).min() == pytest.approx(FLOOR_DB)


def test_the_corner_states_its_zero_once() -> None:
    """Both axes begin at zero, so the corner would print that origin twice and spell it two ways.

    Measured before the fix, the y axis's "0" and the x axis's "0.00" also overlapped by 7 px, and
    NEITHER GATE SEES IT — `text_overlaps` is a same-row rule and these sit diagonally,
    `colliding_ink` asks about shared pixels and the glyphs miss inside boxes that do overlap. So
    the assertion is here, where the defect lives. The frequency zero is the copy that stays: it is
    a reading, since the transform includes DC, whereas the time origin is the panel's left edge.
    """
    power, times, frequencies = _grid()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    spectrogram(ax, to_decibels(power), times=times, frequencies=frequencies)
    fig.tight_layout()
    fig.canvas.draw()

    # The premise: without it the assertions below hold for a panel that never had two zeros.
    assert ax.get_xticks()[0] == 0.0, "the time axis really does carry a zero tick"
    assert ax.get_yticks()[0] == 0.0, "and so does the frequency axis"

    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    assert x_labels[0] == "", "the time axis drops its copy of the zero"
    assert any(label.strip() for label in x_labels[1:]), "but keeps the rest of its scale"
    assert y_labels[0] == "0", "and the frequency zero, which is a reading, stays"
    plt.close(fig)


def test_no_tick_is_labelled_outside_the_axis() -> None:
    """Fixing the ticks to those inside the limits also drops the phantom ends matplotlib offers —
    an STFT reports window centres including padding, so the locator was labelling a NEGATIVE
    time."""
    power, times, frequencies = _grid()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    spectrogram(ax, to_decibels(power), times=times, frequencies=frequencies)
    fig.canvas.draw()

    low, high = ax.get_xlim()
    assert all(low <= float(tick) <= high for tick in ax.get_xticks())
    plt.close(fig)
