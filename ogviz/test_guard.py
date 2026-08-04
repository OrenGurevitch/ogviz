from __future__ import annotations

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz import guard, guarded, is_guarded, unguard
from ogviz.guard import ENV_VAR, FigureRejectedError, guard_from_environment


@pytest.fixture(autouse=True)
def _leave_savefig_as_found():
    """A patched `savefig` must never outlive a test."""
    from matplotlib.figure import Figure

    was = Figure.savefig
    yield
    Figure.savefig = was


def _runs_off_the_page() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.text(1.0, 0.9, "a label that runs clean off the right of the page entirely", ha="left")
    return fig


def _clean() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_ylabel("y")
    return fig


def test_a_plain_savefig_is_audited_once_the_guard_is_on(tmp_path) -> None:
    """The gap this closes: a caller reaches for `fig.savefig`, which knows none of this."""
    target = tmp_path / "rejected.png"
    guard()
    with pytest.raises(FigureRejectedError, match="refused to write"):
        _runs_off_the_page().savefig(target)
    assert not target.exists(), "a rejected figure must not reach disk"


def test_a_clean_figure_passes_silently(tmp_path) -> None:
    target = tmp_path / "clean.png"
    with guarded(mode="raise"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _clean().savefig(target)
    assert target.exists()
    assert not caught


def test_warn_mode_writes_the_figure_and_says_what_is_wrong(tmp_path) -> None:
    """What a project wants on the day it turns this on, before it has fixed anything."""
    target = tmp_path / "warned.png"
    with guarded(mode="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _runs_off_the_page().savefig(target)
    assert target.exists()
    assert any("off the page" in str(w.message) for w in caught)


def test_the_context_manager_restores_whatever_was_there(tmp_path) -> None:
    assert not is_guarded()
    with guarded():
        assert is_guarded()
    assert not is_guarded()


def test_guarding_twice_does_not_stack_wrappers(tmp_path) -> None:
    """Idempotent, so a module that guards on import can be imported twice."""
    guard()
    guard()
    unguard()
    assert not is_guarded(), "one unguard is enough after any number of guards"


def test_ogviz_save_still_works_under_the_guard(tmp_path) -> None:
    """`save` audits and then calls `savefig`; the guard must not audit it a second time."""
    from ogviz import save

    with guarded(mode="raise"):
        paths = save(_clean(), tmp_path, "figure", formats=("png",))
    assert paths[0].exists()


def test_the_environment_variable_picks_the_mode() -> None:
    for setting, expected in (("1", True), ("warn", True), ("repair+advise", True), ("off", False)):
        os.environ[ENV_VAR] = setting
        try:
            unguard()
            assert guard_from_environment() is expected, setting
            assert is_guarded() is expected, setting
        finally:
            del os.environ[ENV_VAR]


def test_a_misspelled_mode_fails_loudly() -> None:
    os.environ[ENV_VAR] = "nonsense"
    try:
        with pytest.raises(ValueError, match="is not a mode"):
            guard_from_environment()
    finally:
        del os.environ[ENV_VAR]


def test_a_bare_import_changes_nothing() -> None:
    """A library that patches matplotlib on import is a bad neighbour; this must stay opt-in."""
    import importlib

    import ogviz

    unguard()
    importlib.reload(ogviz)
    assert not is_guarded()
