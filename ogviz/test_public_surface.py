"""What `import ogviz` actually hands a caller, checked against what the package claims.

This file exists because the same defect has now happened twice, in two subpackages, and both
times it was invisible: a FUNCTION whose name matches the MODULE holding it loses the name at the
package boundary, so `from ogviz.qc import repair` returns something that is not callable while
every docstring and the README go on describing `repair(fig)`. `layout/panels.py` records the
first instance — `ogviz.panels` shadowing a `panels()` — and the fix there was to rename the
function. The second went unnoticed until a project tried to call it.

Nothing here asserts a particular API. It asserts that the API the package publishes is the one
it is documented to have, which is a question no other test in this suite asks.
"""

from __future__ import annotations

import importlib
import types

import matplotlib

matplotlib.use("Agg")
import pytest

PACKAGES = ("ogviz", "ogviz.layout", "ogviz.qc", "ogviz.panels", "ogviz.marks")


def _all_of(module) -> list[str]:
    """The package's `__all__`, and a FAILURE if it has none.

    `getattr(module, "__all__", ())` made both tests below `[] == []` for `ogviz.marks`, which had
    no `__all__`: two green parametrised cases that checked nothing, and the same silence for any
    package added to `PACKAGES` later.
    """
    assert hasattr(module, "__all__"), f"{module.__name__} has no __all__, so nothing is checked"
    return list(module.__all__)


@pytest.mark.parametrize("package", PACKAGES)
def test_no_exported_name_resolves_to_a_module(package: str) -> None:
    """A name in `__all__` that comes back as a module is a shadowed function.

    The failure it catches is silent in the worst way: the import succeeds, the name exists, and
    the call raises `TypeError: 'module' object is not callable` at the call site, a long way from
    the `__init__` that caused it.
    """
    module = importlib.import_module(package)
    exported = _all_of(module)
    shadowed = [
        name for name in exported if isinstance(getattr(module, name, None), types.ModuleType)
    ]
    assert shadowed == [], (
        f"{package}.__all__ names {shadowed}, which resolve to modules rather than to what they "
        "are documented as. A submodule of the same name is shadowing them."
    )


@pytest.mark.parametrize("package", PACKAGES)
def test_every_exported_name_exists(package: str) -> None:
    """`__all__` is a promise; an entry naming nothing breaks `from ... import *` outright."""
    module = importlib.import_module(package)
    missing = [name for name in _all_of(module) if not hasattr(module, name)]
    assert missing == [], f"{package}.__all__ names {missing}, which do not exist"


def test_the_post_render_passes_are_reachable_from_the_package_root() -> None:
    """`save` runs them; a project writing its own `savefig` has to call them itself.

    That project is the reader `value_label_off_its_marks` addresses by name in its complaint text
    — "`settle_axis_labels(fig)` moves it" — and until 2026-08-14 `import ogviz` could not produce
    it. A complaint that names its own fix is the best kind, and it has to be callable.
    """
    import ogviz

    for name in (
        "settle_axis_labels",
        "settle_bracket_labels",
        "settle_caption",
        "settle_corner_tick",
        "settle_header",
    ):
        assert callable(getattr(ogviz, name, None)), f"ogviz.{name} is not callable"


def test_the_names_the_readme_offers_from_python_all_work() -> None:
    """The README says: "From Python: `audit(fig)`, `repair(fig)`, `assert_clean(fig)`, and
    `type_too_small(fig, column_width=...)`", and says where — `ogviz.qc`. Two of the four are
    also at the top level; the README used to imply all four were."""
    import ogviz
    from ogviz.qc import assert_clean, audit, repair, type_too_small

    assert all(callable(one) for one in (audit, repair, assert_clean, type_too_small))
    assert callable(ogviz.audit) and callable(ogviz.assert_clean)


def test_a_complaint_that_names_a_function_names_one_a_reader_can_import() -> None:
    """Every `name(fig)` a check suggests must resolve somewhere a caller can reach.

    Written against the complaint TEXT rather than a list, so a message reworded to suggest some
    other helper is held to the same standard without anyone remembering to update this.
    """
    import re

    import matplotlib.pyplot as plt

    import ogviz
    from ogviz.qc import CHECKS

    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_ylim(0.0, 4.0)
    ax.set_ylabel("named so the axis check fires")
    ax.set_xlabel("time (s)")
    fig.canvas.draw()

    suggested: set[str] = set()
    for check in CHECKS:
        for complaint in check(fig):
            suggested.update(re.findall(r"`([a-z_]+)\(", complaint))
    assert suggested, "the probe figure drew no complaint that names a function"
    unreachable = [name for name in suggested if not hasattr(ogviz, name)]
    assert unreachable == [], (
        f"complaints tell a reader to call {unreachable}, which `import ogviz` cannot produce"
    )
    plt.close(fig)
