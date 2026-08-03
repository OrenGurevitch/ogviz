import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.layout.ticks import MINUS, format_value, round_ticks, value_ticks
from ogviz.theme import use_house_style


@pytest.fixture(autouse=True)
def _style():
    use_house_style()
    yield
    plt.close("all")


def test_exactly_the_requested_number_of_ticks():
    """A tick count that changes between panels of one figure reads as an error."""
    for low, high in [(-6.9, 1.2), (0.0, 1.0), (1150.0, 1400.0), (-1e-9, 3e-9)]:
        assert len(round_ticks(low, high, 4)) == 4


def test_ticks_are_round_and_evenly_stepped():
    ticks = round_ticks(-6.9, 1.2, 4)
    steps = {round(b - a, 9) for a, b in itertools.pairwise(ticks)}
    assert len(steps) == 1
    assert all(abs(t / next(iter(steps)) - round(t / next(iter(steps)))) < 1e-9 for t in ticks)


def test_ticks_stay_inside_the_axis():
    low, high = -6.9, 1.2
    assert all(low <= t <= high for t in round_ticks(low, high, 4))


def test_a_display_scale_converts_the_unit_without_touching_the_datum():
    """A quantity stored in ppm and written in ppb. Without this an axis labelled "ppb" carries
    -0.002 and a printed mean reads "-0.00" — right number, wrong figure."""
    assert format_value(-0.00250, scale=1e3, decimals=2) == f"{MINUS}2.50"
    # unscaled, the label carries nothing
    assert format_value(-0.00250, decimals=2) == f"{MINUS}0.00"


def test_thousands_separator_is_available_for_large_units():
    assert format_value(4200.4, decimals=0, thousands_separator=True) == "4,200"


def test_a_zero_tick_never_prints_as_minus_zero():
    """Float sign at an exact zero. A rounded-away small negative KEEPS its sign, though —
    turning -0.00141 into "0" would hide it."""
    assert format_value(-0.0, decimals=0) == "0"
    assert format_value(-0.00250, decimals=2) == f"{MINUS}0.00"


def test_value_ticks_labels_the_axis_in_display_units():
    _fig, ax = plt.subplots()
    ax.set_ylim(-0.0069, 0.0012)
    positions = value_ticks(ax, count=4, scale=1e3)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [f"{MINUS}6", f"{MINUS}4", f"{MINUS}2", "0"]
    assert positions[0] == pytest.approx(-0.006)


def value_ticks_labels(ax):
    return ax.get_xticklabels()


def test_value_ticks_follows_the_orientation():
    _fig, ax = plt.subplots()
    ax.set_xlim(0.0, 10.0)
    value_ticks(ax, count=3, orientation="horizontal")
    labels = [t.get_text() for t in value_ticks_labels(ax)]
    assert labels == ["3", "6", "9"], "the largest round step that fits wins"


def test_a_thousand_and_up_is_grouped_by_default() -> None:
    """The house rule: from 1000 a number is grouped, because an ungrouped 1200000 is counted."""
    from ogviz.layout.ticks import MINUS, format_value

    assert format_value(1000.0) == "1,000"
    assert format_value(1200000.0) == "1,200,000"
    # Grouped AND typeset: the sign is the typographic minus, not a hyphen.
    assert format_value(-45678.0) == f"{MINUS}45,678"


def test_below_a_thousand_the_grouping_rule_changes_nothing() -> None:
    from ogviz.layout.ticks import format_value

    assert format_value(999.0) == "999"
    assert format_value(0.55) == "0.55"


def test_a_trailing_zero_nobody_measured_is_dropped() -> None:
    """`auto_decimals` aims at three significant figures, so 0.55 asks for three decimals.

    The third is a zero the value does not carry, and "0.550" claims a precision to the thousandth
    that was never measured.
    """
    from ogviz.layout.ticks import format_value

    assert format_value(0.55) == "0.55"
    assert format_value(1.0) == "1"
    assert format_value(0.0887) == "0.0887", "a digit that IS measured stays"


def test_a_stated_decimal_count_is_a_stated_precision_and_is_kept() -> None:
    """A row states its count once for the whole row, and the padding is what makes it a row.

    Stripping per value would leave "1" beside "2.5" and the two would read as two different
    measurements — which is the thing a shared format exists to prevent.
    """
    from ogviz.layout.ticks import format_value

    assert [format_value(value, decimals=2) for value in (1.0, 2.5, 0.55)] == [
        "1.00",
        "2.50",
        "0.55",
    ]


def test_grouping_can_be_turned_off_for_a_number_that_is_not_a_quantity() -> None:
    """A year, an identifier, a part number: grouping those is wrong and the caller knows which."""
    from ogviz.layout.ticks import format_value

    assert format_value(2026.0, decimals=0, thousands_separator=False) == "2026"
