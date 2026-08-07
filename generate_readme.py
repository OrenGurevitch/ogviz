#!/usr/bin/env python
"""Write README.md from README.md.in, substituting the module tree.

Hand-maintaining an API list guarantees it goes stale; `pypatree` reads the package. Run through
`uv run just readme`, and never edit README.md directly — it is generated.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
MARKER = "{{MODULE_TREE}}"
WIDTH = 96  # a signature longer than this is collapsed; the code carries the full one
BRANCH = re.compile(r"[├└]──")  # what pypatree puts in front of a real entry, and only those


def main() -> None:
    template = (ROOT / "README.md.in").read_text()
    assert MARKER in template, f"README.md.in has no {MARKER}"
    raw = subprocess.run(
        ["uv", "run", "pypatree", "ogviz", "--docstrings", "none"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout
    tree = _collapse(raw)
    (ROOT / "README.md").write_text(
        "<!-- generated from README.md.in by generate_readme.py — do not edit -->\n"
        + template.replace(MARKER, tree)
    )
    print(f"README.md written, {len(tree.splitlines())} lines of module tree")


def _collapse(raw: str) -> str:
    """Join pypatree's wrapped signatures and shorten the long ones to `name(...)`.

    pypatree prints a full signature per entry and wraps it over several lines. That is right in a
    terminal and wrong in a README: this package's functions take a lot of keyword arguments, and
    the tree came out at 339 lines, which nobody scans. The tree's job is to say what EXISTS.
    """
    lines: list[str] = []
    for line in raw.splitlines():
        stem = re.match(r"^([\s│├└─]*)", line).group(1)  # type: ignore[union-attr]
        body = line[len(stem) :]
        if lines and not body.strip():
            continue
        # pypatree prefixes every real entry with a branch and never puts one on a
        # continuation, so that is the discriminator. Guessing from the text instead — does it
        # start with a letter, are the parens balanced, did the last line end with an arrow —
        # agreed with it often enough to look right, then failed on a signature wrapping onto
        # `Any]` or `float]`: starts with a letter, closes no paren, follows no arrow. Four
        # signatures in this repo's own tree came out split with the indent guides inside them.
        if lines and not BRANCH.search(stem):
            lines[-1] = lines[-1].rstrip() + " " + body.strip(" │")
            continue
        lines.append(stem + body)
    out = []
    for line in lines:
        # Only the SIGNATURE is normalised, never the stem: the stem's runs of spaces ARE the tree
        # indentation, and squeezing them turns "│   ├──" into "│ ├──" and flattens the whole shape.
        stem = re.match(r"^([\s│├└─]*)", line).group(1)  # type: ignore[union-attr]
        body = line[len(stem) :]
        # pypatree wraps at its own width, so joining leaves "( a, b, )" spacing behind
        body = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", re.sub(r",\s*\)", ")", body)))
        line = stem + re.sub(r"(?<=\S)  +(?=\S)", " ", body)
        if len(line) > WIDTH and "(" in line:
            head, _, _rest = line.partition("(")
            line = f"{head}(...)" + (" -> ..." if "->" in line else "")
        out.append(line.rstrip())
    return "\n".join(out)


if __name__ == "__main__":
    main()
