# ogviz — everything runs through `uv run just`

# Every path the quality tools look at, named once. It was written out per recipe, and the copies
# drifted: CI checked `ogviz` while `just` checked `ogviz examples`, so a lint error in an example
# passed CI, and `generate_readme.py` was in neither.
SOURCES := "ogviz examples generate_readme.py conftest.py"

# `figures` is IN the default gate on purpose. Checking that committed images match a
# fresh render needs a tolerance for font differences across machines; regenerating them
# needs nothing, and a stale gallery becomes impossible rather than merely detectable.
#
# `strict` rather than `test`, so the default gate is at least as strict as CI. It was `test`, and
# the difference is exactly the shape of a session that ends with a red main: everything green
# locally, a deprecation surfacing only on the runner.
default: lint format typecheck strict figures readme

# WHAT CI RUNS, and the only definition of it. CI used to list its own steps, which is how it came
# to check fewer paths than the developer did and to typecheck nothing at all. Read-only: no `--fix`
# and no formatting in place, because a run that repairs the tree cannot fail because of it.
ci: lint-check format-check typecheck strict

lint:
    uv run ruff check --fix {{ SOURCES }}

lint-check:
    uv run ruff check {{ SOURCES }}

format:
    uv run ruff format {{ SOURCES }}

format-check:
    uv run ruff format --check {{ SOURCES }}

# `ogviz` only: the examples and the generator are checked by running them, and the strict settings
# here report stub noise on code that pokes matplotlib internals. See `[tool.basedpyright]`.
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
