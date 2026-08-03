from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ogviz.tags import PREFIX, Tag, mark, marked, value_of

PACKAGE = Path(__file__).parent


def test_a_flag_reads_back_and_an_unset_one_is_false() -> None:
    _fig, ax = plt.subplots()
    assert not marked(ax.patch, "backdrop")
    mark(ax.patch, "backdrop")
    assert marked(ax.patch, "backdrop")


def test_a_tag_can_carry_a_value_and_not_only_a_flag() -> None:
    _fig, ax = plt.subplots()
    line = ax.plot([0, 1], [0, 1])[0]
    mark(line, "position", 3.0)
    assert value_of(line, "position") == 3.0
    assert value_of(line, "lane") is None


def test_no_module_writes_a_tag_by_hand() -> None:
    """A bare `artist.ogviz_thing = True` is how a typo used to disable a rule in silence.

    The vocabulary is a Literal, so `mark(artist, "backdropp")` is a typecheck error at the write
    end and `marked(artist, "bakdrop")` one at the read end. That only holds while every site goes
    through `mark` / `marked` / `value_of`, which is what this asserts.
    """
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name in ("tags.py", "test_tags.py"):  # the vocabulary, and this scanner
            continue
        text = path.read_text()
        for match in re.finditer(rf"\.{PREFIX}\w+\s*=|[\"']{PREFIX}\w+[\"']", text):
            line = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(PACKAGE)}:{line} {match.group().strip()}")
    # The colormap registered by `heatmap` is a matplotlib name, not one of these tags.
    offenders = [entry for entry in offenders if "diverging" not in entry]
    assert not offenders, "tags written or read as bare strings:\n  " + "\n  ".join(offenders)


def test_every_tag_in_the_vocabulary_is_actually_used() -> None:
    """A tag nothing sets or reads is a rule that was removed and left a name behind."""
    sources = "\n".join(
        path.read_text()
        for path in PACKAGE.rglob("*.py")
        if path.name not in ("tags.py", "test_tags.py")
    )
    unused = [tag for tag in get_args(Tag) if f'"{tag}"' not in sources]
    assert not unused, f"declared and never used: {unused}"
