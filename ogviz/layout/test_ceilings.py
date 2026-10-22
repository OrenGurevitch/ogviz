import pytest

from ogviz.layout import ceiling_ticks, scale_ticks


def test_a_bounded_scale_lands_on_both_endpoints() -> None:
    """A 0 to 9 rating needs a 9 on the axis and no 10, which a data-bracketing locator gives."""
    assert scale_ticks(9) == [0, 3, 6, 9]
    assert scale_ticks(100) == [0, 20, 40, 60, 80, 100]
    assert scale_ticks(5) == [0, 1, 2, 3, 4, 5]


def test_a_scale_nothing_divides_keeps_its_two_endpoints() -> None:
    """No step lands on seven within six ticks, and two honest endpoints are better than a guess."""
    assert scale_ticks(7) == [0.0, 7.0]


@pytest.mark.parametrize("high", [0.9, 3.2, 11.7, 46.0, 183.0])
def test_ceiling_ticks_stop_at_the_first_tick_that_covers_the_data(high: float) -> None:
    """The last tick covers the data and the one before it does not, so nothing floats above it."""
    ticks = ceiling_ticks(high)
    assert ticks[0] == 0.0
    assert ticks[-1] >= high
    assert len(ticks) == 1 or ticks[-2] < high


@pytest.mark.parametrize("call", [scale_ticks, ceiling_ticks])
def test_a_non_positive_ceiling_is_refused(call) -> None:
    with pytest.raises(AssertionError, match="positive"):
        call(0.0)
