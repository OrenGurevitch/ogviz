"""End-to-end: the public API renders and writes a real two-group figure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from ogviz import baseline, group_violins, hairline_grid, save, titled, use_house_style


def test_a_full_two_group_panel_renders_and_saves(tmp_path: Path) -> None:
    use_house_style()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    group_violins(
        ax,
        [
            (0.0, rng.normal(-0.7, 0.4, 30), "#E8A838", "#B97C10"),
            (1.0, rng.normal(0.6, 0.5, 48), "#7C9A6E", "#4A6136"),
        ],
        comparisons=[(0.0, 1.0, 1e-9)],
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control\nn=30", "Treated\nn=48"])
    ax.set_ylabel("Composite score (z)")
    hairline_grid(ax)
    baseline(ax)
    header = titled(fig, "Composite score", subtitle="invented data, two groups")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, header))
    written = save(fig, tmp_path, "smoke", dpi=80)
    assert all(p.exists() and p.stat().st_size > 5_000 for p in written)


def test_every_exported_name_is_reachable() -> None:
    """`__all__` is a promise. A name in it that does not resolve is a broken import statement.

    `align_brackets` shipped in `__all__` without being imported: `from ogviz import align_brackets`
    raised ImportError while the package looked complete. basedpyright warns about this and the
    warning sat in a build log; a test says it where someone reads it.
    """
    import ogviz

    missing = [name for name in ogviz.__all__ if not hasattr(ogviz, name)]
    # Every facade, not only the top one. `ogviz.panels.__all__` promised `slide_label_clear`
    # while the import had been dropped, and this test could not see it: a subpackage's promise
    # is as breakable as the package's, and is the one a `from ogviz.panels import ...` relies on.
    for module in _facades():
        missing += [
            f"{module.__name__}.{name}"
            for name in getattr(module, "__all__", ())
            if not hasattr(module, name)
        ]
    assert not missing, f"exported but unreachable: {missing}"


def test_the_layout_facade_defines_nothing() -> None:
    """`ogviz.layout` re-exports and nothing else, so a reader can tell what lives where.

    It used to hold ten function bodies alongside the re-exports from six submodules, which left no
    way to tell which names were its own.
    """
    import ogviz.layout

    assert not _definitions(ogviz.layout), f"the facade defines {_definitions(ogviz.layout)}"


def test_the_qc_facade_holds_only_the_registry_and_the_runner() -> None:
    """`ogviz.qc` was 725 lines: every check body plus fourteen private helpers between them.

    What it may still define is the list of checks and the two functions that run it — everything
    else belongs beside the question it answers.
    """
    import ogviz.qc

    assert set(_definitions(ogviz.qc)) <= {"_run", "audit", "assert_clean"}, _definitions(ogviz.qc)


def _definitions(module) -> list[str]:
    """The names a module defines itself, as opposed to the ones it re-exports."""
    import ast
    from pathlib import Path

    source = Path(module.__file__).read_text()
    return [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]


def _facades() -> list:
    """Every subpackage of ogviz that publishes an `__all__`."""
    import importlib
    import pkgutil

    import ogviz

    found = []
    for info in pkgutil.iter_modules(ogviz.__path__, prefix="ogviz."):
        if not info.ispkg:
            continue
        module = importlib.import_module(info.name)
        if hasattr(module, "__all__"):
            found.append(module)
    return found
