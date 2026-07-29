"""Repairs, on figures drawn by projects that never imported this package."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ogviz.qc import audit
from ogviz.qc.repair import repair


@pytest.fixture
def foreign_figure():
    """Plain matplotlib: no house style, no tags, a label on the curve and one over a rule."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.linspace(0.0, 10.0, 200)
    ax.fill_between(x, 0.0, np.exp(-x / 3.0), color="#88aacc")
    ax.plot(x, np.exp(-x / 3.0), color="#224466", lw=2.0)
    ax.grid(visible=True, axis="y")
    ax.set_ylim(0.0, 1.2)
    ax.text(1.5, 0.35, "this label is on the curve", fontsize=12)
    ax.text(6.0, 0.6, "this one crosses a gridline", fontsize=12)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def test_repair_clears_a_foreign_figure(foreign_figure) -> None:
    assert audit(foreign_figure), "the fixture is meant to be broken"
    changes = repair(foreign_figure)
    assert any("moved" in change for change in changes)
    assert any("knocked out" in change for change in changes)
    assert not audit(foreign_figure), "and repairable defects should leave nothing behind"


def test_repair_moves_the_label_and_not_the_marks(foreign_figure) -> None:
    """Presentation only. A repair that moved a curve would change what the figure says."""
    ax = foreign_figure.axes[0]
    before = [line.get_ydata().copy() for line in ax.lines]
    limits = (ax.get_xlim(), ax.get_ylim())
    repair(foreign_figure)
    after = [line.get_ydata() for line in ax.lines]
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert (ax.get_xlim(), ax.get_ylim()) == limits


def test_a_colour_pair_is_reported_and_never_silently_changed() -> None:
    """Which of marker, dash or palette to change is a decision, so the repair does not make it."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="#2E7CE0", label="blue")
    ax.plot([0, 1], [1, 0], color="#8A63D2", label="violet")
    ax.legend()
    fig.canvas.draw()

    colours = [line.get_color() for line in ax.lines]
    changes = repair(fig)
    assert not any("colour" in change for change in changes)
    assert [line.get_color() for line in ax.lines] == colours
    assert any("deuteranopia" in complaint for complaint in audit(fig))
    plt.close(fig)


def test_repairing_a_clean_figure_changes_nothing() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0], color="#2E7CE0")
    fig.canvas.draw()
    assert repair(fig) == []
    plt.close(fig)
