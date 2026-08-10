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
