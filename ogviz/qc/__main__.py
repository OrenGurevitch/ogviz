"""`python -m ogviz.qc TARGET` — run the checks over figures another project builds.

Separate from the checks themselves so importing `ogviz.qc` costs nothing but the checks, and so
running the module imports the package exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib.figure import Figure

from ogviz.layout.write import reproducible_metadata
from ogviz.qc import CHECKS, audit
from ogviz.qc.repair import repair
from ogviz.qc.report import group_by_subject
from ogviz.require import require

if TYPE_CHECKING:
    from collections.abc import Sequence


def _figures_from(produced: object, plt) -> list[Figure]:
    """What a builder returned, read as figures — and the three shapes it is allowed to be.

    A `Figure` is the documented one. `(fig, ax)` is the commonest matplotlib idiom there is, and a
    list of figures is what a builder drawing several returns; both used to fall silently through to
    "whatever happens to be open", which is a different set and could quietly be a previous
    import's leftovers. They are read properly now.

    RETURNING NOTHING IS STILL ALLOWED and still means the open figures: a builder that draws and
    leaves them open is the normal script shape, and `plt.close("all")` ran before it, so the open
    set really is this builder's work. What is no longer possible is confusing that case with a
    builder that meant to return something and returned the wrong thing.
    """
    if isinstance(produced, Figure):
        return [produced]
    if isinstance(produced, (tuple, list)):
        found = [item for item in produced if isinstance(item, Figure)]
        if found:
            return found
    return [plt.figure(number) for number in plt.get_fignums()]


def _load_figures(target: str) -> list[Figure]:
    """Every figure `target` produces. Either `module:callable` or the path to a script.

    A callable is preferred: it is explicit about what is being checked and returns the figure
    rather than leaving it lying around. A script path is there because most projects have one that
    draws everything, and rewriting it to expose a callable is not a reasonable price for running a
    check over it.

    Both RUN the code. That is not a sandbox and is not pretending to be one — it is the same thing
    as running the project's own figure build, which is the only way to get artists to inspect.
    """
    import importlib
    import runpy

    import matplotlib.pyplot as plt

    plt.close("all")
    if ":" in target:
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        builder = getattr(module, attribute, None)
        # Written out rather than sent through `require`, here and at the four other places like
        # it, because this one also NARROWS: the type checker learns `builder` is callable from the
        # branch, and cannot learn it from a function call.
        if not callable(builder):
            raise AssertionError(f"{target}: {attribute!r} is not a callable in {module_name}")
        return _figures_from(builder(), plt)
    else:
        path = Path(target)
        require(path.exists(), f"no such file: {target}")
        runpy.run_path(str(path), run_name="__ogviz_qc__")
    return [plt.figure(number) for number in plt.get_fignums()]


def _filename(label: str) -> str:
    """A figure label made safe to join onto a directory.

    A label is arbitrary caller text and this builds a path out of it. One containing a separator
    wrote OUTSIDE the directory the user named — `--fix out/` and a label of `../escaped` puts the
    file a level up — which contradicts the help text's promise that the originals are not touched.
    Everything that is not a plain filename character becomes an underscore, and a label that is
    nothing but separators still has to produce a name, so it falls back to `figure`.
    """
    safe = "".join(
        character if character.isalnum() or character in "-_. " else "_" for character in label
    ).strip(" .")
    return safe or "figure"


def _report_one(figure: Figure, index: int, *, thorough: bool, destination: Path | None) -> int:
    """Audit one figure, say what is wrong with it, and return how much is left outstanding.

    Split out of `main`, which did argument parsing, loading, auditing, repair, writing and
    reporting in one body. This is the half that runs per figure, and having it apart is what lets
    the caller be a `sum(...)` over the figures rather than a loop carrying a running total.

    THE AUDIT RUNS ONCE PER OUTCOME and both uses share it: printed, and counted for the exit
    status. It was called twice, once per use, which on `--thorough` is a second full
    render-per-artist pass over every figure for an answer already in hand.
    """
    # `str(...)`: the stubs type a figure label as `object`, and this one is joined onto a
    # path and printed. Coerced here rather than ignored at each use.
    label = str(figure.get_label() or f"figure_{index + 1}")
    print(f"{label}:")
    found = audit(figure, thorough=thorough)
    for line in group_by_subject(found):
        print(f"  - {line}")

    if destination is None:
        return len(found)

    for change in repair(figure):
        print(f"  fixed: {change}")
    written = destination / f"{_filename(label)}.png"
    # `reproducible_metadata`, because this does not go through `save` — `save` refuses a figure
    # with complaints, and the whole point here is to write one that had them.
    #
    # MEASURED, and it is not the write date: a bare PNG `savefig` is already byte-identical across
    # runs, because PNG carries `Software` where SVG carries `dc:date`. What `Software` holds is the
    # matplotlib VERSION — so the same repaired figure written under 3.10 and under 3.11 differs in
    # its bytes, on a repo whose whole CI shape is two matplotlib legs. Stripping it is what makes
    # `--fix` output comparable between them.
    figure.savefig(written, dpi=200, bbox_inches="tight", metadata=reproducible_metadata(written))
    print(f"  wrote {written}")
    # Re-audited because `repair` has just changed the figure — this is the "what still needs a
    # person" number, and it is a different question from the one printed above.
    remaining = audit(figure, thorough=thorough)
    for line in group_by_subject(remaining):
        print(f"  still needs a person: {line}")
    return len(remaining)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks over figures built by another project. `python -m ogviz.qc TARGET`.

    Exit 0 when every figure is clean, 1 when any complaint is raised, so it drops into CI beside a
    test run. Reports every figure rather than stopping at the first: a build that broke one panel
    has usually broken its neighbours the same way.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ogviz.qc",
        description=(
            "Check matplotlib figures for layout defects. Point it at `module:callable` that "
            "returns a figure, or at a script that draws some. The code is executed."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="`package.module:function`, or the path to a script",
    )
    parser.add_argument(
        "--list-checks", action="store_true", help="print what is checked, and stop"
    )
    parser.add_argument(
        "--thorough",
        action="store_true",
        help=(
            "also run the checks that render once per artist — exact, and seconds rather than "
            "milliseconds on a busy figure"
        ),
    )
    parser.add_argument(
        "--fix",
        metavar="DIR",
        help=(
            "repair what has one obvious fix, write the results to DIR, and report what is left. "
            "The originals are not touched."
        ),
    )
    args = parser.parse_args(argv)

    if args.list_checks:
        for check in CHECKS:
            summary = (check.__doc__ or "").strip().splitlines()[0]
            print(f"  {check.__name__:28s} {summary}")
        return 0

    if args.target is None:
        parser.error("a target is required unless --list-checks is given")
    figures = _load_figures(args.target)
    if not figures:
        print(f"{args.target} produced no figures")
        return 1
    destination = Path(args.fix) if args.fix else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    outstanding = sum(
        _report_one(figure, index, thorough=args.thorough, destination=destination)
        for index, figure in enumerate(figures)
    )
    tail = "outstanding" if destination is not None else ""
    print(f"\n{len(figures)} figure(s), {outstanding} complaint(s) {tail}".rstrip())
    return 1 if outstanding else 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
