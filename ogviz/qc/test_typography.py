"""How the figure's numbers are set: one minus sign, grouped thousands, readable type.

These are the checks that read WORDS rather than geometry, and two of the three are in `CHECKS`, so
a gap here is a gate that quietly passes. Writing this module found one: `one_minus_sign` walked
the axes by hand where its two neighbours use the shared `figure_text`, and so could not see a sign
in a figure-level subtitle or a legend entry.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.qc.typography import (
    JOURNAL_MINIMUM_PT,
    one_minus_sign,
    type_too_small,
    ungrouped_thousands,
)

pytestmark = pytest.mark.usefixtures("pinned_font")


def _with_minus_ticks(**kwargs):
    """A panel whose tick labels matplotlib typesets with U+2212."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0), **kwargs)
    ax.plot([0.0, 1.0], [-2.0, 3.0])
    fig.canvas.draw()
    return fig, ax


def test_a_clean_figure_says_nothing() -> None:
    """The premise every test below rests on."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    fig.canvas.draw()
    assert one_minus_sign(fig) == []
    assert ungrouped_thousands(fig) == []
    plt.close(fig)


def test_a_hyphen_beside_matplotlibs_own_minus_is_reported() -> None:
    fig, ax = _with_minus_ticks()
    ax.text(0.5, 0.0, "change of -3.5 units")
    fig.canvas.draw()
    (found,) = one_minus_sign(fig)
    assert "two different minus signs" in found
    plt.close(fig)


def test_the_sign_is_found_in_a_figure_level_label() -> None:
    """A subtitle is not on any axes, and this walked the axes only."""
    fig, _ax = _with_minus_ticks()
    fig.text(0.5, 0.95, "change of -3.5 units", ha="center")
    fig.canvas.draw()
    assert one_minus_sign(fig), "a hyphen in a subtitle is still a hyphen"
    plt.close(fig)


def test_the_sign_is_found_in_a_legend_entry() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot([0.0, 1.0], [-2.0, 3.0], label="drop of -3.5")
    ax.legend()
    fig.canvas.draw()
    assert one_minus_sign(fig)
    plt.close(fig)


def test_both_glyphs_inside_one_label_is_the_case_that_reported_nothing() -> None:
    """Classified as an either/or, a single mixed string landed only in the hyphen set.

    So the figure whose mixture is inside ONE label — the tightest possible case — was the one that
    drew no complaint.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "\u22123.5 to -1.2")  # escaped: ruff flags a bare U+2212
    fig.canvas.draw()
    assert one_minus_sign(fig)
    plt.close(fig)


def test_a_figure_using_only_hyphens_is_left_alone() -> None:
    """The rule is CONSISTENCY, not a preferred glyph — one sign throughout is fine."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "-3.5")
    ax.text(0.5, 0.7, "-1.2")
    fig.canvas.draw()
    assert one_minus_sign(fig) == []
    plt.close(fig)


def test_a_dash_between_words_is_not_a_minus_sign() -> None:
    """`CANDIDATE_MINUS` requires a digit after it and a boundary before it."""
    fig, ax = _with_minus_ticks()
    ax.text(0.5, 0.0, "pre-test 4 of 5")
    fig.canvas.draw()
    assert one_minus_sign(fig) == []
    plt.close(fig)


def test_an_ungrouped_thousand_is_reported_wherever_it_is_printed() -> None:
    for place in ("axes", "figure", "legend"):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if place == "axes":
            ax.text(0.5, 0.5, "12000 samples")
        elif place == "figure":
            fig.text(0.5, 0.95, "12000 samples", ha="center")
        else:
            ax.plot([0.0, 1.0], [0.0, 1.0], label="12000 samples")
            ax.legend()
        fig.canvas.draw()
        assert ungrouped_thousands(fig), f"missed one printed in the {place}"
        plt.close(fig)


def test_a_grouped_thousand_is_left_alone() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.text(0.5, 0.5, "12,000 samples")
    fig.canvas.draw()
    assert ungrouped_thousands(fig) == []
    plt.close(fig)


def test_a_year_is_an_identifier_and_not_a_quantity() -> None:
    """No figure carries the fact of which it is, so a four-digit year is left alone."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.text(0.5, 0.5, "collected in 2019")
    ax.text(0.5, 0.7, "of 2600 items")
    fig.canvas.draw()
    found = ungrouped_thousands(fig)
    assert len(found) == 1 and "2600" in found[0], found
    plt.close(fig)


def test_type_too_small_is_advisory_and_answers_the_journal_question_when_asked() -> None:
    """Without a width it reports a RATIO; told the placed width it reports points.

    The ratio is a property of the figure alone, which is what makes it answerable with no
    argument — and it can only ever stand in for legibility, which is decided by how far the figure
    is scaled when it is placed.
    """
    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    ax.text(0.5, 0.5, "a note set very small indeed", fontsize=3.0)
    fig.canvas.draw()
    placed = type_too_small(fig, column_width=3.0)
    assert placed, "3 pt type scaled down to a 3-inch column is under every journal floor"
    assert f"{JOURNAL_MINIMUM_PT:g}" in placed[0]
    plt.close(fig)


def test_type_too_small_is_silent_on_type_that_clears_the_floor() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.text(0.5, 0.5, "a comfortable note", fontsize=14.0)
    fig.canvas.draw()
    assert type_too_small(fig, column_width=6.0) == []
    plt.close(fig)
