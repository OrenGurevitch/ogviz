"""The coordinate conversions, and the claim that they are written in one place.

The module's whole reason for existing is that the conversion is written ONCE. That is a claim about
the rest of the package as much as about this module, so the last test here checks it by reading the
source — a `dpi / 72.0` written out somewhere else is the module failing at its job, not a style
preference, and it went unnoticed for as long as it did because nothing asked.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import units


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_points_and_pixels_are_inverses() -> None:
    fig = plt.figure(dpi=200.0)
    assert units.px_per_point(fig) == pytest.approx(200.0 / 72.0)
    assert units.to_points(units.to_px(17.0, "pt", fig=fig), fig=fig) == pytest.approx(17.0)
    assert units.to_px(1.0, "in", fig=fig) == pytest.approx(200.0)
    assert units.inches_to_points(1.0) == pytest.approx(72.0)


def test_an_em_with_no_type_size_behind_it_is_refused() -> None:
    """An em is relative to a font. Raised rather than asserted, so `-O` cannot drop it."""
    fig = plt.figure()
    with pytest.raises(AssertionError, match="relative to a type size"):
        units.to_px(1.0, "em", fig=fig)
    assert units.to_px(1.0, "em", fig=fig, em=12.0) == pytest.approx(
        units.to_px(12.0, "pt", fig=fig)
    )


def test_a_midpoint_is_visual_not_arithmetic_on_a_log_axis() -> None:
    """The failure the module was written for, and the reason `align_mean_rows` calls it.

    On an axis running 1 to 1000, the arithmetic midpoint of 1 and 100 is 50.5, which the eye reads
    as most of the way up a gap it sees as centred at 10.
    """
    _fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1.0, 1000.0)
    assert units.midpoint(ax, 1.0, 100.0) == pytest.approx(10.0, rel=1e-6)
    assert units.midpoint(ax, 1.0, 100.0) != pytest.approx(50.5, rel=1e-3)


def test_a_data_value_survives_the_round_trip_through_pixels() -> None:
    _fig, ax = plt.subplots()
    ax.set_ylim(-3.0, 7.0)
    ax.set_xlim(0.0, 4.0)
    for orientation, value in (("vertical", 2.5), ("horizontal", 1.25)):
        pixels = units.value_to_px(ax, value, orientation=orientation)
        back = units.px_to_value(ax, pixels, orientation=orientation)
        assert back == pytest.approx(value)


# The conversions this module owns, as they look when written out by hand.
HAND_ROLLED = ("dpi / 72.0", "dpi/72", "/ 72.0 *", "* 72.0")
# `units.py` is where they belong. The theme's `WORD_LABEL_RATIO`-style arithmetic on type sizes is
# not a coordinate conversion and never involves a dpi, so nothing else is excused here.
ALLOWED = {"units.py"}


def test_no_module_writes_a_points_conversion_out_by_hand() -> None:
    """The one place has to actually be the one place, or it is just a module nobody calls.

    Eleven sites across nine modules had these written out while `ogviz.units` sat with no callers
    at all. Reading the source is the only way to ask this: every one of them was correct
    arithmetic, so no behavioural test could ever have caught the duplication.
    """
    root = Path(__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name in ALLOWED or path.name.startswith("test"):
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#")[0]
            if any(pattern in code for pattern in HAND_ROLLED):
                offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
    assert not offenders, (
        "points conversions written out instead of using ogviz.units:\n" + "\n".join(offenders)
    )
