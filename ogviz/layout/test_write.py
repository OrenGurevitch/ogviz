"""`save` — the package's front door, and the one function whose promise the README leads on.

It was covered only incidentally, through the gallery build and through whatever other tests
happened to call it. What it PROMISES is a short list, and each item is a separate way to be wrong:
it refuses rather than writes, it writes nothing when it refuses, it closes the figure, it writes
one file per format, and the files it writes can be diffed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from PIL import Image

from ogviz.layout.write import reproducible_metadata, save

pytestmark = pytest.mark.usefixtures("pinned_font")


def _clean(figsize=(6.0, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1])
    return fig


def _broken():
    """A figure `assert_clean` refuses, with its premise asserted by the test that uses it."""
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot([0, 1], [0, 1])
    for _ in range(2):  # two labels at one spot: they overlap by construction
        ax.text(0.5, 0.5, "the same words here", ha="center", fontsize=20)
    return fig


def test_a_refused_figure_leaves_nothing_on_disk(tmp_path) -> None:
    """The promise the README leads on. Refusing while leaving a half-written file would be worse
    than not checking: the next build reads a file that passed nothing."""
    fig = _broken()
    with pytest.raises(AssertionError):
        save(fig, tmp_path, "refused")
    assert not list(tmp_path.glob("refused.*")), "a refused figure wrote a file anyway"
    plt.close(fig)


def test_the_premise_that_figure_is_refused_at_all(tmp_path) -> None:
    """`house_style` is autouse and changes layout, so a figure written to be broken is exactly the
    thing that quietly stops being broken. Without this the test above passes vacuously."""
    from ogviz.qc import audit

    fig = _broken()
    fig.canvas.draw()
    assert audit(fig), "the fixture no longer produces a figure the gate refuses"
    plt.close(fig)


def test_check_overlap_false_writes_the_figure_the_gate_refuses(tmp_path) -> None:
    """The documented escape for a panel whose text legitimately abuts, such as a rendered table."""
    fig = _broken()
    written = save(fig, tmp_path, "allowed", check_overlap=False, formats=("png",))
    assert written == [tmp_path / "allowed.png"]
    assert written[0].exists()


def test_one_file_per_format_and_the_paths_come_back(tmp_path) -> None:
    paths = save(_clean(), tmp_path, "both")
    assert [p.name for p in paths] == ["both.png", "both.svg"]
    assert all(p.exists() for p in paths)


def test_no_format_is_refused(tmp_path) -> None:
    """An empty `formats` would write nothing and report success, which reads as a clean save."""
    fig = _clean()
    with pytest.raises(AssertionError, match="at least one format"):
        save(fig, tmp_path, "none", formats=())
    plt.close(fig)


def test_the_directory_is_created(tmp_path) -> None:
    """A figure build usually runs before its output directory exists."""
    nested = tmp_path / "deep" / "deeper"
    save(_clean(), nested, "made")
    assert (nested / "made.png").exists()


def test_the_figure_is_closed_unless_asked_otherwise(tmp_path) -> None:
    """A build writing thirty figures holds thirty open otherwise; matplotlib warns at twenty."""
    fig = _clean()
    save(fig, tmp_path, "closed", formats=("png",))
    assert not plt.fignum_exists(fig.number)

    kept = _clean()
    save(kept, tmp_path, "kept", formats=("png",), close=False)
    assert plt.fignum_exists(kept.number)
    plt.close(kept)


def test_what_is_written_can_be_diffed(tmp_path) -> None:
    """Both formats carry the stamp that would otherwise change on every render.

    SVG carries a write DATE, which makes a re-render of an unchanged figure a whole-file diff. PNG
    carries no date but does carry the matplotlib VERSION, so an unstripped file differs between the
    two legs this package tests on. Asserted per format, because the two keys are different.
    """
    paths = save(_clean(), tmp_path, "stable")
    png, svg = (p for p in paths if p.suffix == ".png"), (p for p in paths if p.suffix == ".svg")
    assert "Software" not in Image.open(next(png)).info
    assert "dc:date" not in next(svg).read_text()


def test_reproducible_metadata_names_the_key_each_format_stamps() -> None:
    """The two keys are not interchangeable, which is why this is a function and not a constant."""
    from pathlib import Path

    assert reproducible_metadata(Path("x.svg")) == {"Date": None}
    assert reproducible_metadata(Path("x.png")) == {"Software": None}


def test_crop_decides_whether_the_declared_canvas_is_what_lands(tmp_path) -> None:
    """`crop=True` writes the artists' extent, `crop=False` the declared figsize.

    The docstring claimed the opposite until 2026-08-04, and the difference is the whole reason
    `crop=False` exists: a document placing two figures at one width shows them at different scales
    if their cropped sizes differ.
    """
    pinned = save(_clean((7.0, 4.0)), tmp_path, "pinned", formats=("png",), dpi=100, crop=False)
    cropped = save(_clean((7.0, 4.0)), tmp_path, "cropped", formats=("png",), dpi=100, crop=True)
    assert Image.open(pinned[0]).size == (700, 400), "crop=False writes the canvas as declared"
    assert Image.open(cropped[0]).size != (700, 400), "crop=True writes the artists' extent"
