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
