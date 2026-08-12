"""What every test in this package needs before it draws anything, in one place.

Nineteen modules declared their own autouse `_style` fixture and they had drifted into three
different behaviours: three pinned a font, sixteen did not, and two never closed their figures — so
`ogviz/theme/test.py` and `ogviz/layout/test.py` between them left six open when they finished.
Whether a given test was machine-independent, or started from a clean slate, depended on which copy
of the fixture it happened to inherit. That is the root of two CI failures in one week: a test
asserting rendered text geometry passed locally against Arial and failed on a runner that has none.

So the shared part is shared, and the part that is NOT shared is asked for by name.

`house_style` is autouse: every test starts from the house rcParams and ends with no figures open.

`pinned_font` is opt-in, and a module takes it with

    pytestmark = pytest.mark.usefixtures("pinned_font")

A module needs it when its assertions are about RENDERED TEXT — extents, collisions, wrapping —
because those depend on the font that resolves, and that differs by machine: Arial on macOS, and on
a Linux runner with no Arial the stack falls through to matplotlib's bundled DejaVu, which is wider.
Pinning DejaVu means CI measures what the author measured, in the wider of the two, so a layout that
passes locally passes anywhere. A test about MARKS — dot positions, limits, z-order — does not need
it, and should not take it, because the pin is a claim about text.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from ogviz.theme import use_house_style

# At the REPO ROOT rather than inside the package, deliberately. `ogviz/` is what the wheel ships
# (see `[tool.hatch.build.targets.wheel]`), so a conftest there would put pytest fixtures — and an
# import of pytest — into what consumers install. It also appeared in the generated README module
# tree, which is meant to say what the API is.

PINNED_FAMILY = "DejaVu Sans"  # matplotlib bundles it, so it resolves on any machine


@pytest.fixture(autouse=True)
def house_style():
    """House rcParams before each test, and no figures left open after it."""
    use_house_style()
    yield
    plt.close("all")


@pytest.fixture
def pinned_font(house_style):
    """Pin the bundled font, for tests whose assertions are about rendered text.

    Depends on `house_style` explicitly rather than relying on fixture ordering: `use_house_style`
    sets `font.sans-serif` itself, so this has to run after it or the pin is overwritten.
    """
    del house_style  # ordering only
    mpl.rcParams["font.sans-serif"] = [PINNED_FAMILY]
