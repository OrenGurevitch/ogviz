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
    ax.set_xticklabels(["Control\nn=40", "Treated\nn=55"])
    ax.set_ylabel("Global symptom burden (z)")
    hairline_grid(ax)
    baseline(ax)
    header = titled(fig, "Global symptom burden", subtitle="Five-domain composite index")
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
    assert not missing, f"exported but unreachable: {missing}"


def test_the_layout_facade_defines_nothing() -> None:
    """`ogviz.layout` re-exports and nothing else, so a reader can tell what lives where.

    It used to hold ten function bodies alongside the re-exports from six submodules, which left no
    way to tell which names were its own.
    """
    import ast
    from pathlib import Path

    import ogviz.layout

    source = Path(ogviz.layout.__file__).read_text()
    defined = [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    assert not defined, f"the facade defines {defined}"
