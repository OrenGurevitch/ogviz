"""Whether `fit_under_header` did what it says it did.

The module had no test file of its own, and the defect below is what that cost: on a figure built
by either of this package's own layout helpers, the layout never ran and the function said it had.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.layout.header import fit_under_header, titled
from ogviz.layout.panels import panel_grid, panel_row
from ogviz.qc.arrangement import layout_not_applied
from ogviz.tags import marked

pytestmark = pytest.mark.usefixtures("pinned_font")


def _headed(build):
    fig, axes = build()
    for ax in axes if isinstance(axes, list) else [axes]:
        ax.plot([0.0, 1.0], [0.0, 1.0])
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        applied = fit_under_header(fig, titled(fig, "A title"))
    return fig, applied, [str(one.message) for one in raised]


@pytest.mark.parametrize(
    ("name", "build"),
    [("panel_row", lambda: panel_row(2)), ("panel_grid", lambda: panel_grid(4, ncol=2))],
)
def test_a_pinned_figure_is_not_reported_as_laid_out(name: str, build) -> None:
    """`tight_layout` skips every axes whose gridspec pins a parameter, and both helpers pin.

    It returned True for as long as this function existed, because the warning matplotlib raises
    for that case is worded differently from the one being matched.
    """
    fig, applied, said = _headed(build)
    assert applied is False, f"{name}: the panels were not laid out, so it must not claim they were"
    assert not any("tight_layout" in one for one in said), (
        f"{name}: the pointless call is skipped, so nothing warns about it: {said}"
    )
    plt.close(fig)


def test_a_pinned_figure_draws_no_complaint_from_the_gate() -> None:
    """Pinned margins are the intended arrangement, not a failure.

    Reporting them would fire on every figure `panel_row` and `panel_grid` build, which is why the
    tag carries a reason rather than a bare flag.
    """
    fig, _applied, _said = _headed(lambda: panel_row(2))
    assert not marked(fig, "layout_refused")
    assert layout_not_applied(fig) == []
    plt.close(fig)


def test_an_ordinary_figure_still_gets_laid_out() -> None:
    """The premise the two above rest on: on a plain gridspec the layout does run and says so."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0))
    for ax in axes:
        ax.plot([0.0, 1.0], [0.0, 1.0])
    before = fig.subplotpars.left
    assert fit_under_header(fig, titled(fig, "A title")) is True
    assert fig.subplotpars.left != before, "and it actually moved the panels"
    plt.close(fig)


def test_a_genuine_refusal_is_still_caught_and_named() -> None:
    """The case that always worked, kept working — and the reason is recorded, not just a flag."""
    from ogviz.tags import value_of

    fig, ax = plt.subplots(figsize=(3.0, 1.4))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_xlabel("a label\nthat is\nseveral\nlines\ntall\nindeed", fontsize=30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        applied = fit_under_header(fig, 0.2)
    if applied:
        pytest.skip("this matplotlib laid the cramped figure out; nothing to assert about refusal")
    assert marked(fig, "layout_refused")
    assert "not applied" in str(value_of(fig, "layout_refused")).lower()
    assert layout_not_applied(fig), "and the gate says it out loud"
    plt.close(fig)


def test_a_partly_pinned_figure_is_not_reported_as_laid_out() -> None:
    """The skip fired only when EVERY axes was pinned; two free subplots beside one hand-added axes
    (which `tight_layout` skips outright) ran the layout, moved the two, and returned True."""
    from ogviz.tags import value_of

    fig, _axes = plt.subplots(1, 2)
    fig.add_axes((0.8, 0.8, 0.1, 0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        applied = fit_under_header(fig, 0.9)
    assert applied is False
    assert "1 of 3 axes pin their own layout" in str(value_of(fig, "layout_pinned"))
    plt.close(fig)
