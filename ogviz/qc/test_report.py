"""Reading the subject out of a complaint — every quoting shape `repr` can hand it.

The grouping had no test of its own, and it was wrong for a whole class of label: `repr` switches to
double quotes for a string containing an apostrophe, the pattern matched only single quotes, and so
`"Ohm's law"` grouped under nothing and printed one problem as two lines. That is precisely the
outcome `group_by_subject` exists to prevent, and the same mistake — re-deriving an artist from the
TEXT of a complaint — is recorded in `qc.repair` as having already been made once.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.layout.collision import quoted
from ogviz.qc.report import group_by_subject, subject_of


def _complaint(label: str, tail: str) -> str:
    """A complaint built the way the checks build one: `repr` of `quoted`."""
    return f"{quoted(label)!r} {tail}"


@pytest.mark.parametrize(
    "label",
    [
        pytest.param("plain label", id="plain"),
        pytest.param("Ohm's law", id="apostrophe-so-repr-uses-double-quotes"),
        pytest.param('the "control" arm', id="double-quotes-so-repr-uses-single"),
        pytest.param('it\'s the "control" arm', id="both-so-repr-escapes-one"),
    ],
)
def test_a_label_is_found_whatever_quoting_repr_chose(label: str) -> None:
    assert subject_of(_complaint(label, "sits on 3 mark(s)")) is not None


@pytest.mark.parametrize(
    "label",
    ["plain label", "Ohm's law", 'the "control" arm', 'it\'s the "control" arm'],
)
def test_two_complaints_about_one_label_become_one_line(label: str) -> None:
    """The whole point of the module: one problem is shown once."""
    said = [_complaint(label, "sits on 3 mark(s)"), _complaint(label, "overlaps its neighbour")]
    (line,) = group_by_subject(said)
    assert "sits on 3 mark(s)" in line and "overlaps its neighbour" in line


def test_a_complaint_naming_two_labels_groups_under_the_first() -> None:
    """The pattern is lazy for this reason — greedy, it reads everything between the outer pair."""
    assert subject_of("'a' is too close to 'b'") == "a"


def test_a_complaint_about_the_figure_is_passed_through() -> None:
    """A buried spine has no label as its subject, and inventing one would misfile it."""
    said = ["the bottom spine is covered by 3 mark(s)", "stars sit at uneven distances"]
    assert group_by_subject(said) == said


def test_the_two_minus_signs_complaint_is_not_filed_under_a_number() -> None:
    """It listed its examples as a repr'd LIST, so the first quoted run in it was a value.

    The complaint is about the figure — like a buried spine — but that put it in a group named after
    a number, where any complaint about a label reading the same string joined it.
    """
    from ogviz.qc.typography import one_minus_sign

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_ylim(-2, 2)
    ax.set_yticks([-1.5, 0.0, 1.5])  # matplotlib typesets these with U+2212
    ax.text(0.5, 0.5, "-1.5", transform=ax.transAxes)  # a caller's ASCII hyphen
    fig.canvas.draw()

    (complaint,) = one_minus_sign(fig)
    assert "-1.5" in complaint and "\u22121.5" in complaint, "it still shows both glyphs"
    assert subject_of(complaint) is None, "and names no label, so it groups as a figure-level note"
    plt.close(fig)


def test_the_group_header_is_the_complaint_s_own_quoting() -> None:
    """Re-quoting the subject runs an already-quoted label through `repr` twice."""
    label = 'it\'s the "control" arm'
    said = [_complaint(label, "sits on 3 mark(s)"), _complaint(label, "overlaps its neighbour")]
    (line,) = group_by_subject(said)
    header = line.split(":")[0]
    assert header in said[0], "the header is a slice of the complaint, not a re-rendering of it"
