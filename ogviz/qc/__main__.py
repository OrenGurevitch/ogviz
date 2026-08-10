"""`python -m ogviz.qc TARGET` — run the checks over figures another project builds.

Separate from the checks themselves so importing `ogviz.qc` costs nothing but the checks, and so
running the module imports the package exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib.figure import Figure

from ogviz.qc import CHECKS, audit
from ogviz.qc.repair import repair
from ogviz.qc.report import group_by_subject
from ogviz.require import require

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        produced = builder()
        if isinstance(produced, Figure):
            return [produced]
    else:
        path = Path(target)
        require(path.exists(), f"no such file: {target}")
        runpy.run_path(str(path), run_name="__ogviz_qc__")
    return [plt.figure(number) for number in plt.get_fignums()]


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

    outstanding = 0
    for index, figure in enumerate(figures):
        label = figure.get_label() or f"figure_{index + 1}"
        print(f"{label}:")
        # Audited ONCE and both used: printed, and counted for the exit status. It was called twice
        # here, once per use, which on `--thorough` is a second full render-per-artist pass over
        # every figure for an answer already in hand.
        found = audit(figure, thorough=args.thorough)
        for line in group_by_subject(found):
            print(f"  - {line}")

        if destination is None:
            outstanding += len(found)
            continue

        for change in repair(figure):
            print(f"  fixed: {change}")
        written = destination / f"{label}.png"
        figure.savefig(written, dpi=200, bbox_inches="tight")
        print(f"  wrote {written}")
        remaining = audit(figure, thorough=args.thorough)
        outstanding += len(remaining)
        for line in group_by_subject(remaining):
            print(f"  still needs a person: {line}")

    tail = "outstanding" if destination is not None else ""
    print(f"\n{len(figures)} figure(s), {outstanding} complaint(s) {tail}".rstrip())
    return 1 if outstanding else 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
