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

# EITHER quote character, because `repr` picks between them: a label holding an apostrophe comes out
# as "Ohm's law", not 'Ohm\'s law'. Matching only the single quote meant such a label matched
# nothing, fell through to the ungrouped pile, and printed one problem as several lines — which is
# the exact outcome `group_by_subject` exists to prevent. The identical mistake is on the record one
# module along: `qc.repair.knock_out_labels_over_rules` used to split complaints on apostrophes and
# silently repaired nothing whenever a label contained one.
#
# Three things the pattern has to do at once, each of which the obvious version gets wrong:
#   tempered against its OWN delimiter, not both, or the apostrophe case fails again;
#   `\\.` first, so a label carrying both quote characters — where `repr` escapes one — does not
#     end at the escaped quote;
#   LAZY, or a complaint naming two labels reads as everything between the outermost pair.
QUOTED = re.compile(r"""(['"])(?P<subject>(?:\\.|(?!\1).){1,60}?)\1""")


def _quoted_token(complaint: str) -> re.Match[str] | None:
    """The first quoted run in a complaint, quotes included."""
    return QUOTED.search(complaint)


def subject_of(complaint: str) -> str | None:
    """The label a complaint is about, if it names one."""
    found = _quoted_token(complaint)
    return found.group("subject") if found else None


def group_by_subject(complaints: list[str]) -> list[str]:
    """One line per label, with the rest of what was said about it folded in behind it.

    Complaints naming no label are passed through untouched: a buried spine and an uneven bracket
    stack are about the figure, not about a string, and there is nothing to group them under.
    """
    grouped: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
    loose: list[str] = []
    for complaint in complaints:
        found = _quoted_token(complaint)
        if found is None:
            loose.append(complaint)
        else:
            grouped.setdefault(found.group("subject"), []).append((complaint, found.group(0)))

    lines: list[str] = []
    for said in grouped.values():
        if len(said) == 1:
            lines.append(said[0][0])
            continue
        # The header is the TOKEN as the complaint wrote it, not the subject re-quoted. Re-quoting
        # runs a label that already contains a quote character through `repr` a second time, so the
        # group header carries escapes the complaint underneath it does not.
        header = said[0][1]
        lines.append(f"{header}: " + "; ".join(_without_subject(one, token) for one, token in said))
    return lines + loose


def _without_subject(complaint: str, token: str) -> str:
    """The complaint with its subject removed, since the group already names it.

    Given the TOKEN that was matched — quotes and all — rather than the bare subject to be
    re-quoted. Re-quoting guesses which quote character the complaint used and gets it wrong for
    exactly the labels the widened pattern was added to catch, leaving the subject in the middle of
    a line that already names it.
    """
    return complaint.replace(token, "").strip(" -:")
