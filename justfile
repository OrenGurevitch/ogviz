# ogviz — everything runs through `uv run just`

# `figures` is IN the default gate on purpose. Checking that committed images match a
# fresh render needs a tolerance for font differences across machines; regenerating them
# needs nothing, and a stale gallery becomes impossible rather than merely detectable.
default: lint format typecheck test figures readme

lint:
    uv run ruff check --fix ogviz examples

format:
    uv run ruff format ogviz examples

typecheck:
    uv run basedpyright ogviz

test:
    uv run pytest ogviz -q

# deprecations are errors here: matplotlib renames (violinplot vert -> orientation) must
# surface in this package, not in a project's figure build
strict:
    uv run pytest ogviz -q -W error::DeprecationWarning

# rewrite README.md from README.md.in (the module tree comes from pypatree)
readme:
    uv run python generate_readme.py

# regenerate every example image
figures:
    uv run python -m examples
