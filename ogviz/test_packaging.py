"""What the wheel ships, held to what the source tree calls a test.

The tests live INSIDE the package, so for as long as the wheel was built from `packages = ["ogviz"]`
alone it carried them: 37 of its 90 files, 41% of what a consumer installs, and none of them
importable there — `pytest` is a dev extra, and the autouse `house_style` fixture lives in a root
`conftest.py` that was never in the wheel. `import ogviz.qc.test_qc` in a clean consumer
environment raised `ModuleNotFoundError: No module named 'pytest'`.

The regression this guards is not "someone deletes the exclude". It is the SPLIT: `python_files`
defines what a test module is named, the wheel `exclude` repeats those patterns, and a session that
adds a third naming convention has no reason to think about the second copy. So the test compares
the two rather than checking either against a literal.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def config() -> dict:
    if not PYPROJECT.exists():  # running from an install rather than the source tree
        pytest.skip("no pyproject.toml beside the package")
    return tomllib.loads(PYPROJECT.read_text())


def test_every_test_naming_convention_is_kept_out_of_the_wheel(config: dict) -> None:
    """Whatever pytest collects as a test, the wheel must exclude — matched pattern for pattern."""
    collected = set(config["tool"]["pytest"]["ini_options"]["python_files"])
    excluded = set(config["tool"]["hatch"]["build"]["targets"]["wheel"]["exclude"])

    missing = {name for name in collected if f"**/{name}" not in excluded}
    assert not missing, (
        f"pytest collects {sorted(missing)} as tests and the wheel does not exclude them — "
        f"they would ship to consumers. Add `**/<pattern>` to the wheel exclude."
    )


def test_the_source_tree_really_does_name_its_tests_that_way(config: dict) -> None:
    """The patterns are only worth comparing if they describe the files that exist.

    Both lists could agree with each other and with nothing on disk. This is the third leg: every
    module the tree treats as a test matches one of the declared conventions.
    """
    import fnmatch

    patterns = config["tool"]["pytest"]["ini_options"]["python_files"]
    package = Path(__file__).resolve().parent
    stray = [
        path.relative_to(package.parent)
        for path in package.rglob("test*.py")
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
    ]
    assert not stray, f"named like tests but matching no declared convention: {stray}"


def test_the_metadata_a_person_reads_is_declared(config: dict) -> None:
    """Everything a rendered package page shows, none of which was declared until 2026-08-12.

    Missed because every consumer installs from a git ref and never sees a page: `hatchling` builds
    a perfectly good wheel with no readme, no licence, no author and no links, and nothing in the
    gate looks at what it says. A published name with a blank card is worse than an unpublished one.
    """
    project = config["project"]
    missing = [
        field
        for field in ("readme", "license", "authors", "classifiers", "urls", "description")
        if not project.get(field)
    ]
    assert not missing, f"a package page would be blank in: {missing}"
    assert project["readme"] == "README.md", "the long description is the README, not a stub"
    assert (Path(__file__).resolve().parent.parent / project["readme"]).exists()


def test_the_licence_is_declared_once_and_not_twice(config: dict) -> None:
    """PyPI refuses an upload carrying BOTH an SPDX expression and a `License ::` classifier.

    The classifier is the superseded form and reads as harmless, which is exactly why it gets added
    back — it looks like every other classifier in the list.
    """
    project = config["project"]
    assert isinstance(project.get("license"), str), "declare the licence as an SPDX expression"
    duplicated = [c for c in project.get("classifiers", []) if c.startswith("License ::")]
    assert not duplicated, (
        f"drop these; `license = {project['license']!r}` already says it: {duplicated}"
    )


def test_the_python_floor_is_stated_where_a_reader_will_look(config: dict) -> None:
    """The README told readers the matplotlib floor and not the stricter Python one, so someone on
    3.11 met a resolver error from the page that was supposed to tell them what they needed."""
    floor = config["project"]["requires-python"].lstrip(">=")
    readme = (Path(__file__).resolve().parent.parent / "README.md.in").read_text()
    assert f"Python ≥ {floor}" in readme, f"README.md.in should state Python ≥ {floor}"
