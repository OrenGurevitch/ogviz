"""`require` exists for one property, and this is the test of it.

Every contract in this package was an `assert` until 2026-08-06, and `assert` is the one statement
Python is allowed to delete. Under `python -O` the gates silently stopped gating: `assert_clean`
passed a figure whose labels shared hundreds of pixels of ink, and `stars(-5.0)` returned the most
significant result there is. The module docstring records that; nothing checked it.

The check has to run in a SUBPROCESS, because `-O` is decided when the interpreter starts and
cannot be turned on for one test.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from ogviz.require import require

PROBE = (
    "from ogviz.require import require\n"
    "try:\n"
    "    require(False, 'the message')\n"
    "except AssertionError as raised:\n"
    "    print('RAISED', raised)\n"
    "else:\n"
    "    print('PASSED')\n"
)


def test_a_false_condition_raises_with_the_message_given() -> None:
    with pytest.raises(AssertionError, match="the reason"):
        require(False, "the reason")


def test_a_truthy_condition_returns_quietly() -> None:
    assert require(True, "never seen") is None
    assert require([0], "a non-empty list is truthy") is None


def test_falsiness_is_the_test_not_identity() -> None:
    """Callers pass `formats`, a list of complaints, a count — not only booleans."""
    for falsy in (0, "", [], {}, None, 0.0):
        with pytest.raises(AssertionError):
            require(falsy, f"{falsy!r} is falsy")


def test_the_refusal_survives_optimised_mode() -> None:
    """THE WHOLE POINT. A bare `assert` disappears here and the contract goes with it."""
    plain = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    )
    optimised = subprocess.run(
        [sys.executable, "-O", "-c", PROBE], capture_output=True, text=True, check=True
    )
    assert plain.stdout.startswith("RAISED"), plain.stdout
    assert optimised.stdout.startswith("RAISED"), (
        "under -O the refusal vanished, which is the failure this module was written to end:\n"
        + optimised.stdout
    )


def test_a_bare_assert_really_does_vanish_when_optimised() -> None:
    """The premise. Without it the test above proves nothing about -O, only that require raises."""
    probe = (
        "try:\n"
        "    assert False, 'gone'\n"
        "except AssertionError:\n"
        "    print('RAISED')\n"
        "else:\n"
        "    print('PASSED')\n"
    )
    optimised = subprocess.run(
        [sys.executable, "-O", "-c", probe], capture_output=True, text=True, check=True
    )
    assert optimised.stdout.startswith("PASSED"), (
        "an assert survived -O, so this interpreter cannot demonstrate the hazard: "
        + optimised.stdout
    )
