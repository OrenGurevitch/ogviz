"""The `python -m ogviz.qc` entry point: what it accepts, what it writes, and where.

On the untested list until 2026-08-11, which is how three defects lived in it — a live date stamp
on every repaired figure, a path built out of unvalidated caller text, and two ordinary builder
return shapes falling through to "whatever figures happen to be open".
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.qc.__main__ import _figures_from, _filename, main


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("panel one", "panel one"),
        ("../escaped", "_escaped"),
        ("nested/name", "nested_name"),
        ("Fig 1: results", "Fig 1_ results"),
        ("", "figure"),
        ("...", "figure"),
    ],
)
def test_a_label_cannot_carry_the_written_file_out_of_its_directory(label, expected) -> None:
    """`--fix DIR` joins the figure's label onto DIR, and a label is arbitrary caller text.

    The help text promises the originals are not touched; a label of `../escaped` broke that.
    """
    assert _filename(label) == expected


def test_the_written_name_never_contains_a_separator() -> None:
    """The property behind the cases above, so a new character class cannot slip past."""
    import os

    for label in ("a/b", "a\\b", "../..", "a\0b", "  ..  "):
        assert os.sep not in _filename(label)
        assert (os.altsep or "/") not in _filename(label)
        assert _filename(label) not in ("", ".", "..")


def test_a_builder_may_return_a_figure_a_pair_or_a_list() -> None:
    """`(fig, ax)` is the commonest matplotlib idiom and used to fall through silently."""
    first, ax = plt.subplots()
    second = plt.figure()
    assert _figures_from(first, plt) == [first]
    assert _figures_from((first, ax), plt) == [first]
    assert _figures_from([first, second], plt) == [first, second]
    plt.close("all")


def test_a_builder_that_returns_nothing_still_means_the_open_figures() -> None:
    """A builder that draws and leaves them open is the ordinary script shape; still supported."""
    plt.close("all")
    drawn = plt.figure()
    assert _figures_from(None, plt) == [drawn]
    plt.close("all")


def _builder_module(tmp_path, label: str) -> None:
    """A module exposing `build()`, drawing a figure with one defect `repair` can fix."""
    (tmp_path / "builder.py").write_text(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "def build():\n"
        "    fig, ax = plt.subplots()\n"
        "    ax.bar([0, 1], [1.0, 2.0])\n"
        "    ax.text(0, 0.5, 'on the bar', ha='center')\n"
        f"    fig.set_label({label!r})\n"
        "    return fig\n"
    )


def _run(tmp_path, label: str, out):
    import sys

    _builder_module(tmp_path, label)
    sys.path.insert(0, str(tmp_path))
    try:
        return main(["builder:build", "--fix", str(out)])
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("builder", None)
        plt.close("all")


def test_fix_writes_inside_the_directory_it_was_given(tmp_path) -> None:
    """The guard, exercised THROUGH `--fix` rather than by calling `_filename` directly.

    Testing the helper alone passed with the helper unwired: the mutation that dropped it from the
    path was invisible. What matters is where the file lands, so that is what is asserted.
    """
    out = tmp_path / "out"
    _run(tmp_path, "../escaped", out)
    assert not (tmp_path / "escaped.png").exists(), "the label climbed out of the directory"
    assert list(out.glob("*.png")), "and nothing was written where it was asked to be"


def test_fix_output_does_not_carry_the_matplotlib_version(tmp_path) -> None:
    """`--fix` bypasses `save`, so it strips the `Software` chunk itself.

    A bare PNG `savefig` is byte-identical run to run — PNG has no date stamp, unlike SVG — but it
    does record the matplotlib version, so the same repaired figure differs between the two legs
    this package tests on. Asserted on the chunk rather than on a byte comparison, because a byte
    comparison within one leg passes whether or not the stripping happens.
    """
    from PIL import Image

    out = tmp_path / "out"
    _run(tmp_path, "run", out)
    written = next(out.glob("*.png"))
    assert "Software" not in Image.open(written).info, Image.open(written).info


def test_list_checks_prints_and_stops() -> None:
    """`--list-checks` needs no target, and must exit 0 rather than falling into the audit path."""
    assert main(["--list-checks"]) == 0
