"""Writing a figure to disk, with the checks in front of it.

`save` is the only way a figure should leave the process: it runs the gate first and raises instead
of writing, so a broken figure cannot reach a README by being saved from somewhere that forgot.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from ogviz.guard import gate_already_run
from ogviz.layout.axis import settle_axis_labels
from ogviz.layout.header import settle_header
from ogviz.layout.panels import settle_caption
from ogviz.require import require
from ogviz.significance import settle_bracket_labels
from ogviz.theme import glyphs_must_render

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.figure import Figure

# The keys holding the write date or the matplotlib version, per format — whichever that format
# stamps. MEASURED on 3.11 by writing each format twice a second apart and diffing the bytes:
#
# - png: `Software` holds the version; there is no date.
# - svg: `dc:date` changes on every run (`Date`). Its `Creator` names the version too and is left in
#   place, because stripping it now would churn every SVG already committed for no reader's benefit.
# - pdf: `CreationDate` changes on every run, and `Creator` and `Producer` both carry the version.
# - ps and eps: `Creator` carries the version. Their date comes from `SOURCE_DATE_EPOCH` and no
#   metadata key removes it, so PostScript is comparable across versions but not across runs.
#
# EVERY OTHER FORMAT TAKES NONE. Pillow's (jpg, tiff, webp, gif, avif) and raw/rgba refuse
# `metadata=` whatever it holds — even `{}` is a `ValueError` — and pgf refuses the keyword itself.
_STAMPS: dict[str, dict[str, None]] = {
    "png": {"Software": None},
    "svg": {"Date": None},
    "svgz": {"Date": None},
    "pdf": {"CreationDate": None, "Creator": None, "Producer": None},
    "ps": {"Creator": None},
    "eps": {"Creator": None},
}


def _format_of(extension: str) -> str:
    """The format matplotlib reads from a file ending `.<extension>`: its suffix, lowercased."""
    return Path(f"figure.{extension}").suffix[1:].lower()


def reproducible_metadata(path: Path) -> dict[str, None] | None:
    """Drop the write date, so a re-render of an unchanged figure produces an unchanged file.

    Two things in a matplotlib SVG change on every run: the `dc:date` stamp, and the random ids
    matplotlib gives its clip paths (`svg.hashsalt` pins those; the house style sets it). Together
    they made `git diff` on the committed gallery useless — thirteen files, every line touched,
    2480 modifications of which none were real, so a diff could not answer "did this change the
    figure". A generated artifact that cannot be diffed cannot be reviewed.

    PNG takes the same treatment through its own key, and PDF and PostScript through theirs; the
    table above says which, and what was measured. A format that carries no metadata — every one
    Pillow writes — gets `None`, the one value `savefig` accepts for it. It was handed PNG's key,
    so `save(formats=("jpg",))` raised.

    PUBLIC because `save` is not the only way a figure gets written. A caller with a reason to use
    `fig.savefig` directly — a before/after where one half is meant to fail the gate, most obviously
    — still wants the file to be diffable, and the alternative is that they copy the two key names
    and go stale when a third format needs one. This example's own gallery figure was committed with
    a live date stamp for exactly that reason, and churned on every render.
    """
    stamps = _STAMPS.get(path.suffix[1:].lower())
    return None if stamps is None else dict(stamps)


def plain_filename(name: str) -> str:
    """Arbitrary text made safe to join onto a directory.

    Everything that is not a plain filename character becomes an underscore, and text that is
    nothing but separators still has to produce a name, so it falls back to `figure`.

    HERE rather than in `ogviz/qc/__main__.py`, where it was written for `--fix`'s figure labels:
    `save`'s own `name` is joined onto a directory in exactly the same way and had no guard at all,
    so `save(fig, out, "../escaped")` wrote a level up from the directory it was given. Two callers
    with the same question, and `qc` already imports from this module, so this is the direction the
    dependency can point.

    A figure LABEL is caller data and a `save` name is usually a developer's literal, which is the
    argument for not guarding the second one. It is not a good argument: the literal is a template
    in a builder loop as often as not.

    `save` now splits its name on `/` first and sends each part through here (`_name_parts`), so a
    subfolder can be asked for; `--fix` still flattens a label whole, since a label is not a path.
    """
    safe = "".join(
        character if character.isalnum() or character in "-_. " else "_" for character in name
    ).strip(" .")
    return safe or "figure"


def _name_parts(name: str) -> list[str]:
    """`save`'s `name` as the folders under `directory` and the file stem, each part made plain.

    A name with no `/` in it is exactly `[plain_filename(name)]`, as before. A `/` reaches into a
    subfolder, and ONLY a subfolder: an absolute name, or a part that is empty, `.` or `..` (or
    nothing but dots and spaces, which `plain_filename` would otherwise quietly turn into
    `figure`), is refused rather than rewritten, because a name meant to climb out and a name that
    lands somewhere else are the same mistake and only one of them says so. `/` is the separator
    on every platform; a backslash is still an ordinary character and still becomes `_`.
    """
    if "/" not in name:
        return [plain_filename(name)]
    parts = name.split("/")
    require(
        not name.startswith("/") and all(part.strip(" .") for part in parts),
        f"save name {name!r} must stay inside the directory it is given: a subfolder like "
        "'panels/a' is allowed, an absolute name or an empty, '.' or '..' part is not",
    )
    return [plain_filename(part) for part in parts]


def save(
    fig: Figure,
    directory: str | os.PathLike[str],
    name: str,
    *,
    dpi: int = 200,
    check_overlap: bool = True,
    formats: Sequence[str] = ("png", "svg"),
    close: bool = True,
    crop: bool = True,
    settled: Callable[[str], None] | None = None,
    by_format: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write `<directory>/<name>.<ext>` per format, checked on the way out.

    Two gates, and both refuse rather than write a broken figure: the glyph gate, because a missing
    glyph renders as a tofu box, and the QC gate — every check in `ogviz.qc.CHECKS` — because
    overlapping labels render as mush, and both otherwise ship unnoticed in a figure build that
    scrolls past. `check_overlap=False` switches off the WHOLE QC gate, not only the overlap check
    (the name predates the gate growing past one check, and is kept so no caller breaks); the
    glyph gate is unconditional. Pass it for a panel whose text legitimately abuts, such as a
    rendered table, and `close=False` to keep working on the figure.

    Under `guard()`, the write inside is not audited a second time: `save` has run the gate, and
    `gate_already_run` says so. Before that a `save(check_overlap=False)` was refused by the guard
    anyway, so the escape hatch and the guard could not both be used.

    `crop=True` (the default) writes `bbox_inches="tight"`: the file is cropped to the artists
    rather than to the canvas, so the declared `figsize` is NOT what lands on disk. It trims dead
    margin, and it keeps a label that reaches past the page instead of cutting it off.

    THE COST, which this docstring claimed the opposite of until 2026-08-04: two figures declaring
    the same canvas do not write the same size. It used to be large — a plain 7x4 in panel at dpi
    100 wrote 602x353 against 829x353 for the same panel with one label reaching past the edge.

    RE-MEASURED 2026-08-12, and it is now small, because the gate closed the case that made it big.
    A label reaching past the canvas is what grew the page, and `text_off_canvas` refuses that
    figure outright — both of the divergent cases above are now rejected before they can be written.
    What remains is the difference in dead margin between two figures that both PASS: a bare panel
    writes 602x353 and one carrying a y-label and a title writes 631x376, which placed at a common
    width in a document is a 1.6% difference in aspect.

    So the argument for inverting this default — that cropping breaks side-by-side use — was
    measured against figures the gate no longer lets through, and 1.6% does not break it. The
    default stays cropped.

    So `crop=False` for a PINNED layout, where the point is that every figure has the same axes
    rectangle: it writes the canvas as declared, and `required_margins` is how the margins get
    chosen. Cropping and pinning are the two coherent choices; picking neither deliberately is how a
    set ends up inconsistent.

    `name` may reach into a subfolder — `save(fig, out, "panels/a")` writes `out/panels/a.png`,
    creating `panels` — and cannot reach out of `directory`: `..`, an empty part or an absolute
    name is refused, where it used to be flattened (`"../escaped"` wrote `_escaped.png`, and before
    that a level up). Each part is still sanitised by `plain_filename`. `_name_parts` has the rule.

    `by_format=True` gives each format its own folder — `out/png/a.png` beside `out/svg/a.svg` —
    for a build that hands the rasters to one reader and the vectors to another. It is still ONE
    save: the settle passes and the gate run once, and every format is written or none is. Wanting
    this was the commonest reason to write with `fig.savefig` and so skip the gate altogether.

    `metadata` is merged over the reproducible defaults for each format (a `Title`, an `Author`; a
    key given `None` removes one), so asking for a key does not bring the date stamp back. A format
    that carries no metadata — every one Pillow writes — cannot take it, and is refused up front
    rather than written without it.

    `directory` may be a `str` or any path-like; it was `.mkdir`-ed as given, so a `str` raised.

    EVERYTHING IS CHECKED BEFORE ANYTHING IS WRITTEN, and every format is rendered before any file
    lands. A format matplotlib cannot write is refused before the settle passes touch the figure,
    and a failure that only exists once rendering starts — the glyph gate, which raises as its block
    ends — leaves no file behind either. A mixed `formats` list used to write the ones in front of
    the bad one, and a figure with a tofu box was refused AND written. A refusal of any kind raises,
    writes nothing and leaves the figure open, as the gate's always has; `close` is about a save
    that succeeded.

    `settled` is handed every adjustment the four settle passes below made, one line at a time,
    before the gate runs. Each of those functions returns what it moved — and each says in its own
    docstring that it does so because "a silent adjustment is unreviewable" — and this, the one
    call site that matters, threw all four answers away. `print` is the obvious sink and a
    `--verbose` flag the obvious caller; a library cannot decide to print, so it takes the sink.
    """
    require(
        formats,
        "save needs at least one format",
    )
    directory = Path(directory)
    supported = fig.canvas.get_supported_filetypes()
    unknown = [extension for extension in formats if _format_of(extension) not in supported]
    require(
        not unknown,
        f"save cannot write {', '.join(map(repr, unknown))}: "
        f"matplotlib writes {', '.join(sorted(supported))}",
    )
    if metadata:
        bare = [extension for extension in formats if _format_of(extension) not in _STAMPS]
        require(
            not bare,
            f"save cannot attach metadata to {', '.join(map(repr, bare))}: matplotlib writes "
            f"metadata only for {', '.join(_STAMPS)}",
        )
    *folders, stem = _name_parts(name)
    # Before the checks, not after: a caption row is reserved when the panels are created and the
    # caller has not plotted yet, so what grows into it can only be measured here.
    moved = list(settle_header(fig))
    if settle_caption(fig):
        moved.append("the caption was pushed below the panels as they finally are")
    # After the caption, and before the checks: a bracket label's gap to its bracket is set in
    # pixels, and anything that rescaled the value axis since then has changed it.
    moved += settle_bracket_labels(fig)
    # Last of the settles, because it reads the ticks where they FINALLY are: anything above that
    # rescaled a value axis moved them. matplotlib centres an axis label on the axes box, and this
    # package pads a panel asymmetrically on purpose — bracket headroom above, a mean lane below —
    # so the box's middle is not where the ticks are.
    moved += settle_axis_labels(fig)
    if settled is not None:
        for line in moved:
            settled(line)
    if check_overlap:
        from ogviz.qc import assert_clean

        assert_clean(fig)
    canvas = fig.get_facecolor()
    paths = [
        directory.joinpath(*([extension] if by_format else []), *folders, f"{stem}.{extension}")
        for extension in formats
    ]
    # INTO MEMORY FIRST, every format, and only then onto disk — measured byte-identical to writing
    # the path directly, for png, svg and pdf. The glyph gate raises as its block exits, so the
    # writes cannot sit inside it, and a later format failing must not strand an earlier one.
    rendered: list[tuple[Path, bytes]] = []
    with glyphs_must_render(), gate_already_run():
        for path in paths:
            stamps = reproducible_metadata(path)
            if stamps is not None and metadata:
                stamps = {**stamps, **metadata}
            # Left out rather than passed as `None` where a format takes none: pgf refuses the
            # keyword itself.
            extra: dict[str, Any] = {} if stamps is None else {"metadata": stamps}
            options: dict[str, Any] = {
                "format": _format_of(path.suffix[1:]),
                "bbox_inches": "tight" if crop else None,
                "facecolor": canvas,
                "dpi": dpi,
                **extra,
            }
            if path.suffix == ".svg" and not plt.rcParams["svg.image_inline"]:
                # An SVG that links its images rather than inlining them names them after its own
                # file and writes them beside it, so it cannot be rendered into a buffer — the
                # writer refuses one. It is rendered into a scratch directory under its real name,
                # and every file that lands there is carried across with it.
                with tempfile.TemporaryDirectory() as scratch:
                    fig.savefig(Path(scratch) / path.name, **options)
                    rendered.extend(
                        (path.parent / written.name, written.read_bytes())
                        for written in sorted(Path(scratch).iterdir())
                    )
                continue
            buffer = io.BytesIO()
            fig.savefig(buffer, **options)
            rendered.append((path, buffer.getvalue()))
    # AFTER the gate: a refused figure used to leave an empty directory tree behind, and the
    # package's claim is that a refusal writes nothing.
    for path, content in rendered:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if close:
        plt.close(fig)
    return paths
