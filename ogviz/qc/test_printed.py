import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import assert_printed
from ogviz.qc import unprinted
from ogviz.qc.printed import MINUS

pytestmark = pytest.mark.usefixtures("pinned_font")


def _figure():
    fig, ax = plt.subplots()
    ax.bar([0, 1], [0.74, 0.5])
    ax.text(0, 0.8, "0.74")
    ax.text(1, 0.6, f"{MINUS}0.12")
    ax.set_xticks([0, 1], ["ours", "theirs"])
    return fig


def test_a_string_on_the_canvas_is_found_in_text_and_in_tick_labels() -> None:
    fig = _figure()
    assert unprinted(fig, ["0.74", "ours", "theirs"]) == []


def test_a_missing_string_is_named_and_a_prefix_does_not_stand_in_for_it() -> None:
    """A figure printing "0.7" has not printed "0.74" (the premise: "0.74" is on it)."""
    fig = _figure()
    assert unprinted(fig, ["0.74"]) == []
    ax = fig.axes[0]
    for text in list(ax.texts):
        if text.get_text() == "0.74":
            text.set_text("0.7")
    assert unprinted(fig, ["0.74", "ours"]) == ["0.74"]


def test_a_hyphen_minus_in_the_expected_string_matches_the_true_minus_drawn() -> None:
    assert unprinted(_figure(), ["-0.12"]) == []


def test_assert_printed_raises_and_lists_every_miss() -> None:
    with pytest.raises(AssertionError, match=r"2 string\(s\)") as refused:
        assert_printed(_figure(), ["0.74", "9.99", "absent"])
    assert "9.99" in str(refused.value)
    assert "absent" in str(refused.value)
