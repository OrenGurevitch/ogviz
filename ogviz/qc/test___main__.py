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


def test_list_checks_names_every_tier(capsys) -> None:
    """It printed `CHECKS` alone, so the check `--thorough` adds and the three advisories
    `guard(advise=True)` runs were promised by the help and named nowhere."""
    from ogviz.qc import ADVISORY_CHECKS, CHECKS, THOROUGH_CHECKS

    assert main(["--list-checks"]) == 0
    printed = capsys.readouterr().out
    for check in (*CHECKS, *THOROUGH_CHECKS, *ADVISORY_CHECKS):
        assert check.__name__ in printed, check.__name__
    assert "[advisory]" in printed and "[--thorough]" in printed


def test_fix_still_exits_nonzero_for_a_figure_that_arrived_broken(tmp_path) -> None:
    """The status was the POST-repair count, so a run whose every figure was repairable exited 0
    while the originals on disk were still broken."""
    assert _run(tmp_path, "run", tmp_path / "out") == 1


def test_a_builder_returning_a_tuple_with_no_figure_is_refused() -> None:
    """`(ax1, ax2)` fell through to "every open figure", which the docstring said was no longer
    possible."""
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="no Figure in it"):
        _figures_from((ax, ax), plt)
    plt.close(fig)


def test_two_figures_sharing_a_label_get_two_files(tmp_path) -> None:
    """One `--fix` write silently overwrote the other, and the report described both.

    Three figures labelled the same is not exotic — the shape a builder loop produces — and the
    directory held the last one while the run said three. The FIRST claim keeps the plain name, so
    a project whose labels are already distinct sees the filenames it saw before.
    """
    import sys

    source = tmp_path / "twins.py"
    source.write_text(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "def build():\n"
        "    made = []\n"
        "    for _ in range(3):\n"
        "        fig, ax = plt.subplots(figsize=(4.0, 3.0))\n"
        "        fig.set_label('panel A')\n"
        "        ax.plot([0.0, 1.0], [0.0, 1.0])\n"
        "        made.append(fig)\n"
        "    return made\n"
    )
    out = tmp_path / "out"
    sys.path.insert(0, str(tmp_path))
    try:
        main(["twins:build", "--fix", str(out)])
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("twins", None)
        plt.close("all")

    written = sorted(path.name for path in out.glob("*.png"))
    assert written == ["panel A.png", "panel A_2.png", "panel A_3.png"]


def test_a_target_that_produces_no_figures_says_so_and_exits_nonzero(tmp_path, capsys) -> None:
    """Untested: the branch a mistyped target or a builder that drew nothing lands in.

    It is the one outcome where there is nothing to report per figure, so the exit status has to
    come from somewhere else — and a run that checked zero figures must not read as a clean one.
    """
    import sys

    source = tmp_path / "empty.py"
    source.write_text("x = 1\n")
    sys.path.insert(0, str(tmp_path))
    try:
        assert main([str(source)]) == 1
    finally:
        sys.path.remove(str(tmp_path))
        plt.close("all")
    assert "produced no figures" in capsys.readouterr().out
