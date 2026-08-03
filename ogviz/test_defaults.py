"""One logical parameter, one default — across the signatures that pass it along.

A composite that takes a parameter and hands it to a part re-declares that part's default in its own
signature. The two agree on the day they are written and nothing keeps them agreeing: change the
part's default and the composite silently keeps the old one, so the same call means two things
depending on which door it came through.

This is the cheap half of the fix. Folding the pass-throughs into one argument would be the other
half and is a breaking change to a package three repositories pin by commit, which is not a call to
make from a smell — see FIXME on `group_violins`' parameter count.
"""

from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")
import pytest

from ogviz.panels.bars import bar_panel, value_labels
from ogviz.panels.violins import group_violins, printed_means

# (composite, part, {name in the composite: name in the part})
PASS_THROUGH = [
    pytest.param(
        group_violins,
        printed_means,
        {
            "mean_fontsize": "fontsize",
            "mean_decimals": "decimals",
            "display_scale": "scale",
            "thousands_separator": "thousands_separator",
        },
        id="group_violins -> printed_means",
    ),
    pytest.param(
        bar_panel,
        value_labels,
        {"value_format": "value_format"},
        id="bar_panel -> value_labels",
    ),
]


@pytest.mark.parametrize(("composite", "part", "mapping"), PASS_THROUGH)
def test_a_pass_through_carries_the_part_s_own_default(composite, part, mapping) -> None:
    outer = inspect.signature(composite).parameters
    inner = inspect.signature(part).parameters
    drifted = {
        f"{composite.__name__}.{out}": (outer[out].default, inner[inn].default)
        for out, inn in mapping.items()
        if outer[out].default != inner[inn].default
    }
    assert not drifted, f"a default was changed in one signature and not the other: {drifted}"


@pytest.mark.parametrize(("composite", "part", "mapping"), PASS_THROUGH)
def test_the_mapping_still_describes_both_signatures(composite, part, mapping) -> None:
    """A renamed parameter must not make this test silently vacuous."""
    outer = set(inspect.signature(composite).parameters)
    inner = set(inspect.signature(part).parameters)
    assert set(mapping) <= outer, f"gone from {composite.__name__}: {set(mapping) - outer}"
    missing = set(mapping.values()) - inner
    assert not missing, f"gone from {part.__name__}: {missing}"
