"""Group the checks' complaints for a person to read, without merging the checks themselves.

Two checks answering different questions about the same label produce three lines about one
problem: `colliding_ink` reports a pair per artist and `text_over_data` reports the label once. All
three are true and all three are useful when a build fails and someone is debugging a check. None
of that helps the person who ran this over a colleague's figure and wants to know what to fix.

So the grouping happens HERE, at the point of reading, and the checks keep reporting exactly what
they measured. Merging them at source would mean deciding that one check speaks for another, and
the moment their definitions drift the report would be quietly wrong.
"""

from __future__ import annotations

import re
from collections import OrderedDict

QUOTED = re.compile(r"'([^']{1,60})'")


def subject_of(complaint: str) -> str | None:
    """The label a complaint is about, if it names one."""
    found = QUOTED.search(complaint)
    return found.group(1) if found else None


def group_by_subject(complaints: list[str]) -> list[str]:
    """One line per label, with the rest of what was said about it folded in behind it.

    Complaints naming no label are passed through untouched: a buried spine and an uneven bracket
    stack are about the figure, not about a string, and there is nothing to group them under.
    """
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    loose: list[str] = []
    for complaint in complaints:
        subject = subject_of(complaint)
        if subject is None:
            loose.append(complaint)
        else:
            grouped.setdefault(subject, []).append(complaint)

    lines: list[str] = []
    for subject, said in grouped.items():
        if len(said) == 1:
            lines.append(said[0])
            continue
        lines.append(f"{subject!r}: " + "; ".join(_without_subject(one, subject) for one in said))
    return lines + loose


def _without_subject(complaint: str, subject: str) -> str:
    """The complaint with its subject removed, since the group already names it."""
    return complaint.replace(f"{subject!r}", "").replace(f"'{subject}'", "").strip(" -:")
