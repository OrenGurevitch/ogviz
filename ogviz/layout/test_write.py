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
    assert reproducible_metadata(Path("x.SVG")) == {"Date": None}, "matplotlib reads it as SVG"
    assert reproducible_metadata(Path("x.jpg")) is None, "Pillow's formats take no metadata"


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


def _refused_figure():
    """A figure the gate refuses on its own — asserted, since `house_style` can clean a fixture."""
    from ogviz.qc import audit

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar([0, 1], [1.0, 2.0])
    ax.text(0, 0.5, "on the bar", ha="center")
    fig.canvas.draw()
    assert audit(fig), "the premise: this figure is refused"
    return fig


def test_a_refusal_writes_nothing_not_even_the_directory(tmp_path) -> None:
    """`mkdir` ran before the gate, so a refused save left an empty tree behind."""
    fig = _refused_figure()
    target = tmp_path / "never" / "made"
    with pytest.raises(AssertionError):
        save(fig, target, "figure")
    assert not target.exists()
    plt.close(fig)


def test_save_without_the_gate_is_not_refused_by_the_guard(tmp_path) -> None:
    """The documented escape hatch and the documented guard could not be used together: `save`
    skipped its gate and the guard then refused the write. Probed 2026-09-01."""
    from ogviz import guarded

    fig = _refused_figure()
    with guarded(mode="raise"):
        written = save(fig, tmp_path, "figure", check_overlap=False, formats=("png",))
    assert written[0].exists()


def test_save_hands_its_settle_adjustments_to_a_caller_who_asks(tmp_path) -> None:
    """Four settle passes each return what they moved, and `save` threw all four away.

    Each of those functions says in its own docstring that it reports because "a silent adjustment
    is unreviewable", and this is the call site that matters. Exercised on the bracket pass, which
    is the one a caller can provoke: rescaling the value axis after the brackets are drawn moves
    them through the data transform while the label's offset, set in points, stays put.
    """
    import numpy as np

    from ogviz import group_violins

    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    group_violins(
        ax,
        [
            (0.0, rng.normal(5.0, 1.0, 40), "#E8A838", "#B97C10"),
            (1.0, rng.normal(7.0, 1.0, 40), "#7C9A6E", "#4A6136"),
        ],
        comparisons=[(0.0, 1.0, 0.001)],
    )
    # What moves the brackets out from under their labels. `ax.margins` cannot do it: this panel
    # set its own limits when it fitted the bracket stack, so autoscaling is off and a margin
    # changes nothing.
    # 5%, chosen so the figure still PASSES the gate: at 36% `unused_value_headroom` refuses it,
    # and the sink would then be exercised only on the way to a raise.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + (high - low) * 0.05)

    lines: list[str] = []
    save(fig, tmp_path, "settled", formats=("png",), settled=lines.append)
    assert lines, "the bracket label had to be re-anchored and nothing said so"


def test_a_name_with_a_separator_in_it_cannot_write_outside_the_directory(tmp_path) -> None:
    """`save`'s `name` is joined onto a directory and had no guard, where `--fix`'s label has one.

    The premise is that the escape really was one: the naive join lands a level up.

    It was flattened to `_escaped.png` until names could reach into a subfolder; now that `/` means
    something, a name that climbs out is refused rather than quietly written somewhere else.
    """
    out = tmp_path / "out"
    assert (out / "../escaped.png").resolve() == tmp_path / "escaped.png", "premise: it escapes"

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(AssertionError, match="must stay inside"):
        save(fig, out, "../escaped", formats=("png",))
    assert not list(tmp_path.rglob("*.png")), "a refused name wrote something"
    plt.close(fig)


@pytest.mark.parametrize(
    "name", ["../escaped", "/absolute/x", "panels/../../x", "a//b", "panels/", "./a", "a/ .. /b"]
)
def test_a_name_that_leaves_the_directory_is_refused_before_anything_happens(
    tmp_path, name
) -> None:
    """Every way out, and the empty and `.` parts, which would land somewhere the caller did not
    write. Refused before the settle passes, so the figure is untouched and still open."""
    fig = _clean()
    with pytest.raises(AssertionError, match="must stay inside"):
        save(fig, tmp_path / "out", name)
    assert not (tmp_path / "out").exists()
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_a_name_may_reach_into_a_subfolder(tmp_path) -> None:
    """`save(fig, out, "panels/a")` writes `out/panels/a.png`, making `panels`. The old answer was
    `out/panels_a.png`, which is why a project wanting the subfolder wrote with `fig.savefig`."""
    written = save(_clean(), tmp_path, "panels/fig: a", formats=("png",))
    assert written == [tmp_path / "panels" / "fig_ a.png"], "each part is still made plain"
    assert written[0].exists()


def test_a_name_without_a_slash_is_written_exactly_as_before(tmp_path) -> None:
    """The widening must not move a single existing file. A backslash is not a separator here on
    any platform, so it is rewritten as it always was."""
    written = save(_clean(), tmp_path, "Fig 1: a\\b", formats=("png",))
    assert written == [tmp_path / "Fig 1_ a_b.png"]


def test_by_format_gives_each_format_its_own_folder(tmp_path) -> None:
    """`out/png/a.png` beside `out/svg/a.svg`, and a subfolder in the name goes under each."""
    written = save(_clean(), tmp_path, "panels/a", by_format=True)
    assert written == [tmp_path / "png" / "panels" / "a.png", tmp_path / "svg" / "panels" / "a.svg"]
    assert all(path.exists() for path in written)


def test_by_format_is_still_one_save(tmp_path, monkeypatch) -> None:
    """The settle passes and the gate run ONCE for every format, as they do without it — two
    calls to `save`, one per folder, is the workaround this replaces, and it gated twice."""
    import ogviz.qc

    gated: list[object] = []
    real = ogviz.qc.assert_clean
    monkeypatch.setattr(ogviz.qc, "assert_clean", lambda fig, **kw: gated.append(real(fig, **kw)))
    save(_clean(), tmp_path, "a", by_format=True, formats=("png", "svg", "pdf"))
    assert len(gated) == 1


def test_metadata_is_merged_over_the_reproducible_defaults(tmp_path) -> None:
    """A caller's `Title` lands, and asking for it does not bring back the stamp the defaults
    remove — which is what passing `metadata=` straight to `fig.savefig` does."""
    png, svg = save(_clean(), tmp_path, "titled", metadata={"Title": "A titled figure"})
    info = Image.open(png).info
    assert info.get("Title") == "A titled figure"
    assert "Software" not in info
    text = svg.read_text()
    assert "A titled figure" in text
    assert "dc:date" not in text


def test_metadata_for_a_format_that_carries_none_is_refused_up_front(tmp_path) -> None:
    """Dropping it silently would write a file without what was asked for, and saying nothing."""
    fig = _clean()
    with pytest.raises(AssertionError, match="cannot attach metadata to 'jpg'"):
        save(fig, tmp_path / "out", "fig", formats=("png", "jpg"), metadata={"Title": "x"})
    assert not (tmp_path / "out").exists()
    plt.close(fig)


def test_a_format_failing_at_write_time_writes_nothing_else(tmp_path) -> None:
    """Some failures exist only once rendering starts — an SVG handed a key its writer does not
    know is one. Every format renders before anything lands, so it cannot strand the PNG."""
    fig = _clean()
    with pytest.raises(ValueError, match="Unknown metadata"):
        save(fig, tmp_path / "out", "fig", metadata={"Software": "x"}, formats=("png", "svg"))
    assert not (tmp_path / "out").exists()
    plt.close(fig)


def test_a_directory_given_as_a_string_is_accepted(tmp_path) -> None:
    """Every path API in the standard library takes a `str`; this raised `AttributeError` from
    `.mkdir` — after the gate had run, with the figure still open."""
    written = save(_clean(), str(tmp_path / "as_text"), "fig", formats=("png",))
    assert written == [tmp_path / "as_text" / "fig.png"]
    assert written[0].exists()


@pytest.mark.parametrize("extension", ["jpg", "jpeg", "tif", "tiff", "webp"])
def test_a_format_that_carries_no_metadata_is_written(tmp_path, extension) -> None:
    """Pillow's formats refuse `metadata=` whatever it holds, and `reproducible_metadata` handed
    every format that is not SVG the PNG key."""
    fig = _clean()
    written = save(fig, tmp_path, "fig", formats=(extension,))
    assert written[0].exists()
    assert not plt.fignum_exists(fig.number)


def test_the_premise_that_pillow_formats_refuse_any_metadata(tmp_path) -> None:
    """A fact about matplotlib — measured, even `{}` is refused. If a future version accepts it,
    this fails and the table in `reproducible_metadata` can widen."""
    fig = _clean()
    with pytest.raises(ValueError, match="metadata not supported"):
        fig.savefig(tmp_path / "x.jpg", metadata={})
    plt.close(fig)


def test_an_unknown_format_is_refused_before_anything_is_written(tmp_path) -> None:
    """A mixed list wrote the formats in front of the bad one and then raised, so a save that
    failed still left half a figure on disk."""
    fig = _clean()
    with pytest.raises(AssertionError, match="nosuch"):
        save(fig, tmp_path / "out", "fig", formats=("png", "svg", "nosuch"))
    assert not (tmp_path / "out").exists(), "a refused save wrote something"
    assert plt.fignum_exists(fig.number), "a refusal leaves the figure open, as the gate's does"
    plt.close(fig)


# The settle passes draw before the gate and warn the same way; that is the defect, not the test.
@pytest.mark.filterwarnings("ignore:.*missing from font")
def test_a_glyph_refusal_writes_nothing(tmp_path) -> None:
    """The glyph gate raises when its block ENDS, and the writes were inside the block: a figure
    with a tofu box was refused and written anyway, in every format.

    The character warns under the pinned DejaVu, and it is one NO other test uses: matplotlib warns
    for a missing glyph once per process per font and character, so sharing `test_guard.py`'s
    Devanagari let whichever ran first consume the warning. `test_guard.py` says why the characters
    are measured rather than picked; Hiragana U+3042 was probed the same way.
    """
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    ax.set_title("no glyph for this: \u3042")
    with pytest.raises(AssertionError, match="no glyph"):
        save(fig, tmp_path / "out", "tofu", check_overlap=False)
    assert not (tmp_path / "out").exists(), "a refused figure was written anyway"
    plt.close(fig)


def test_pdf_and_postscript_lose_the_version_stamp(tmp_path) -> None:
    """Both were handed PNG's key, which neither reads, so each carried the matplotlib version —
    a file that differs between the two matplotlib legs CI runs."""
    version = matplotlib.__version__.encode()
    written = save(_clean(), tmp_path, "fig", formats=("pdf", "eps"))
    for path in written:
        assert version not in path.read_bytes(), path.suffix


def test_an_svg_that_links_its_images_is_written_with_them(tmp_path, monkeypatch) -> None:
    """`svg.image_inline=False` names each image after the SVG and writes it beside it.

    Such an SVG cannot be rendered into a buffer: the writer takes a buffer's missing name as an
    empty one and drops `.image0.png` into the WORKING directory, linked from an SVG that will not
    find it. So it is rendered under its real name elsewhere and carried across with its images.
    The premise is asserted — the buffer route really does strand the image in the working
    directory — so this is not a setting that happens to work either way.
    """
    import io

    import numpy as np

    elsewhere = tmp_path / "working"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    out = tmp_path / "out"
    with matplotlib.rc_context({"svg.image_inline": False}):
        fig, ax = plt.subplots(figsize=(3.0, 2.0))
        ax.imshow(np.arange(16.0).reshape(4, 4))
        fig.savefig(io.BytesIO(), format="svg")
        stranded = list(elsewhere.iterdir())
        assert stranded, "premise: a buffer strands the linked image in the working directory"
        for path in stranded:
            path.unlink()
        paths = save(fig, out, "linked", formats=("svg",), check_overlap=False)
    assert not list(elsewhere.iterdir()), "save stranded a linked image in the working directory"
    tmp_path = out
    assert paths == [tmp_path / "linked.svg"]
    images = sorted(path.name for path in tmp_path.iterdir() if path.suffix == ".png")
    assert images, "the linked image was not written beside the SVG"
    assert all(name in (tmp_path / "linked.svg").read_text() for name in images)
