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


def test_two_threads_saving_at_once_are_both_audited(tmp_path) -> None:
    """The re-entry flag was a module global, so one thread's audit silenced the other's.

    The flag exists so the guard's own `savefig` at the end does not re-enter the wrapper. Shared
    across threads it meant something else: thread A set it, thread B saw it set, and B skipped its
    audit ENTIRELY and wrote a figure the gate never looked at — the one outcome this module exists
    to prevent. MEASURED with the two saves forced to overlap: 40 of 80 got through, exactly one per
    overlapping pair, which is the mechanism rather than a rare race. A `ContextVar` gives each
    thread its own flag.
    """
    import threading

    with guarded(mode="raise"):
        # The premise, asserted rather than assumed: this figure really is refused on its own. A
        # figure hand-built here was clean under the autouse house style, so the threaded assertion
        # would have passed while measuring nothing.
        alone = _runs_off_the_page()
        with pytest.raises(FigureRejectedError):
            alone.savefig(tmp_path / "probe.png")
        plt.close(alone)

        verdicts: list[str] = []
        barrier = threading.Barrier(2)

        def save(index: int) -> None:
            fig = _runs_off_the_page()
            barrier.wait()  # force the two audits to overlap
            try:
                fig.savefig(tmp_path / f"{index}.png")
                verdicts.append("written past the gate")
            except FigureRejectedError:
                verdicts.append("refused")
            plt.close(fig)

        threads = [threading.Thread(target=save, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert verdicts == ["refused", "refused"], verdicts


def _with_a_missing_glyph() -> plt.Figure:
    """A figure whose text has a character the pinned font has no glyph for.

    DejaVu Sans is pinned by `conftest`, so the character has to be one IT lacks — an Arial-only
    choice would render fine here and the test would measure nothing. U+1F600 is absent from both.
    """
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_title("no glyph for this: \U0001f600")
    return fig


def test_the_missing_glyph_is_a_warning_the_checks_cannot_see() -> None:
    """The premise. `audit` reads finished artists; a missing glyph is emitted while RASTERISING,
    so nothing in `CHECKS` can report it and the guard has to wrap the write itself."""
    from ogviz.qc import audit

    fig = _with_a_missing_glyph()
    with warnings.catch_warnings():  # the render warns by design; that IS the thing being described
        warnings.simplefilter("ignore")
        fig.canvas.draw()
    assert not audit(fig), "if a check can see this, the guard need not wrap the render"
    plt.close(fig)


def test_the_guard_refuses_a_figure_with_no_glyph_for_its_text(tmp_path) -> None:
    """The one protection `save` had that the checks do not, which `guard()` used to drop.

    A project moving from `save` to `guard()` — which the README encourages — silently lost it and
    shipped a tofu box where a tick should be.
    """
    fig = _with_a_missing_glyph()
    with guarded(mode="raise"), pytest.raises(FigureRejectedError, match="no glyph"):
        fig.savefig(tmp_path / "tofu.png")
    assert not (tmp_path / "tofu.png").exists(), "a refused figure was written anyway"
    plt.close(fig)


def test_warn_mode_writes_the_tofu_and_says_so(tmp_path) -> None:
    """`warn` reports and does not block, which is the contract every other check here keeps."""
    fig = _with_a_missing_glyph()
    with guarded(mode="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(tmp_path / "written.png")
    assert (tmp_path / "written.png").exists(), "warn mode must still write"
    assert any("no glyph" in str(w.message) for w in caught), [str(w.message) for w in caught]
    plt.close(fig)


def test_a_figure_whose_glyphs_all_resolve_is_untouched(tmp_path) -> None:
    """The guard must not start refusing ordinary text; every gallery figure goes through here."""
    fig = _clean()
    with guarded(mode="raise"):
        fig.savefig(tmp_path / "fine.png")
    assert (tmp_path / "fine.png").exists()
    plt.close(fig)
