from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.significance import (
    INK_GAP_PX,
    STACK_GAP_PX,
    bracket_stack,
    ink_bounds_points,
    spaced_stars,
    stars,
)


def test_stars_thresholds() -> None:
    assert stars(0.0009) == "***"
    assert stars(0.005) == "**"
    assert stars(0.04) == "*"
    assert stars(0.051) == "n.s."


def test_spaced_stars_only_spaces_asterisks() -> None:
    from ogviz.significance import STAR_SEPARATOR

    assert spaced_stars(0.0009) == STAR_SEPARATOR.join("***")
    assert spaced_stars(0.9) == "n.s.", "n.s. is a word and must not be spaced out"
    assert " " not in spaced_stars(0.9)


def test_asterisk_ink_floats_above_its_baseline() -> None:
    # The premise of the whole module: an asterisk's ink sits well above the baseline, so
    # anchoring the layout box parks the glyph far from the bracket it belongs to.
    low, high = ink_bounds_points("***", 18.0)
    assert low > 2.0, "if the ink started at the baseline, box anchoring would be fine"
    assert high > low


def _star_and_bracket_pixels(ax: plt.Axes) -> tuple[list[float], list[float]]:
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData
    brackets = [
        to_px.transform((0, line.get_ydata()[1]))[1]
        for line in ax.lines
        if len(line.get_ydata()) == 4
    ]
    stars_px = []
    for text in ax.texts:
        baseline = to_px.transform((0, text.get_position()[1]))[1]
        ink_low, ink_high = ink_bounds_points(text.get_text(), text.get_fontsize())
        scale = fig.dpi / 72.0
        stars_px.append((baseline + ink_low * scale, baseline + ink_high * scale))
    return brackets, stars_px


def test_star_ink_sits_just_above_its_own_bracket() -> None:
    _fig, ax = plt.subplots()
    ax.set_ylim(0, 10)
    bracket_stack(ax, [(0.0, 1.0, 0.001)], start=5.0, span=10.0)
    brackets, star_ink = _star_and_bracket_pixels(ax)
    gap = star_ink[0][0] - brackets[0]
    assert gap == pytest.approx(INK_GAP_PX, abs=0.6)
    plt.close("all")


def test_a_star_is_far_closer_to_its_own_bracket_than_to_the_next() -> None:
    # The asymmetry is the point: a star must read as belonging to the line beneath it.
    _fig, ax = plt.subplots()
    ax.set_ylim(0, 40)
    bracket_stack(ax, [(0.0, 1.0, 0.001), (0.0, 2.0, 0.02)], start=5.0, span=40.0)
    brackets, star_ink = _star_and_bracket_pixels(ax)
    below = star_ink[0][0] - brackets[0]  # ink bottom to its own bracket
    above = brackets[1] - star_ink[0][1]  # ink top to the bracket above
    assert below == pytest.approx(INK_GAP_PX, abs=0.6)
    assert above == pytest.approx(STACK_GAP_PX, abs=1.5)
    assert above > 5 * below, "the gaps must be visibly asymmetric, not merely ordered"
    plt.close("all")


def test_the_asymmetry_holds_at_a_different_font_size_and_data_range() -> None:
    # Both gaps are in pixels, so neither the type size nor the axis units may change them.
    ratios = []
    for fontsize, top in ((11.0, 1.0), (26.0, 5000.0)):
        _fig, ax = plt.subplots()
        ax.set_ylim(0, top)
        bracket_stack(
            ax,
            [(0.0, 1.0, 0.001), (0.0, 2.0, 0.001)],
            start=top * 0.2,
            span=top,
            fontsize=fontsize,
        )
        brackets, star_ink = _star_and_bracket_pixels(ax)
        ratios.append((brackets[1] - star_ink[0][1]) / (star_ink[0][0] - brackets[0]))
        plt.close("all")
    assert all(r > 5 for r in ratios)
    assert ratios[0] == pytest.approx(ratios[1], rel=0.25)


def test_empty_comparisons_returns_the_start_and_draws_nothing() -> None:
    _fig, ax = plt.subplots()
    assert bracket_stack(ax, [], start=3.0, span=1.0) == 3.0
    assert not ax.lines and not ax.texts
    plt.close("all")


def test_stack_returns_the_topmost_ink_so_a_caller_can_size_the_axis() -> None:
    _fig, ax = plt.subplots()
    ax.set_ylim(0, 40)
    top = bracket_stack(ax, [(0.0, 1.0, 0.001), (0.0, 2.0, 0.001)], start=5.0, span=40.0)
    assert top > 5.0
    _brackets, star_ink = _star_and_bracket_pixels(ax)
    fig_top_px = max(hi for _lo, hi in star_ink)
    assert ax.transData.transform((0, top))[1] == pytest.approx(fig_top_px, abs=1.0)
    plt.close("all")


def test_label_for_lets_a_project_keep_its_own_star_convention():
    """One project writes "" for non-significant, another "n.s." — placement is what ogviz owns."""
    import matplotlib.pyplot as plt

    from ogviz.theme import use_house_style

    use_house_style()
    fig, ax = plt.subplots()
    ax.set_ylim(0, 10)

    def unspaced_or_blank(p: float) -> str:
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    bracket_stack(
        ax,
        [(0.0, 1.0, 0.004), (0.0, 2.0, 0.40)],
        start=6.0,
        span=10.0,
        label_for=unspaced_or_blank,
    )
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["**"], "a blank label must draw no bracket at all, not an empty one"
    assert len(ax.lines) == 1
    plt.close(fig)
