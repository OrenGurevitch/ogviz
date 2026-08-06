"""Automated checks on a rendered figure, so defects are caught by the build and not by eye.

Every check answers one question about a FINISHED figure, measured from the artists that were
actually drawn. Each exists because the corresponding defect shipped at least once: a star closer to
its bracket than its neighbours, a bracket line clipped out of the axes while its star stayed, two
tick labels touching, a dot on top of the mean line.

`audit` runs them all and returns every complaint. `assert_clean` is the build gate — `save` calls
it, so a project cannot write a figure that fails one of these without being told.

This module is a FACADE and defines nothing but the registry and the runner. The checks live beside
the question they answer: `reading` for what every check has to work out first, then `significance`,
`marks`, `arrangement`, `typography`, `color` and `ink`. It held all of them until 2026-08-01, at
725 lines, with fourteen private helpers interleaved between the checks that used them.
"""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING

from ogviz.layout.bounds import text_off_canvas, text_wider_than_its_panel
from ogviz.layout.caption import overflowing_text
from ogviz.layout.collision import text_over_data
from ogviz.layout.overlap import (
    DEFAULT_MIN_GAP,
    clipped_artists,
    text_hidden_behind_knockouts,
    text_overlaps,
)
from ogviz.layout.panels import grid_warnings
from ogviz.qc.arrangement import (
    layout_not_applied,
    mean_rows_unaligned,
    panels_disagree_about_ticks,
    rows_outside_their_panel,
    ticks_in_the_headroom,
)
from ogviz.qc.color import series_confusable_under_cvd
from ogviz.qc.ink import colliding_ink, drawn_but_invisible
from ogviz.qc.marks import buried_baselines, dots_off_the_marks
from ogviz.qc.reading import (
    GAP_TOLERANCE_PX,
    artist_name,
    bracket_tops_px,
    drawn_artists,
    ensure_rendered,
    is_backdrop,
    is_excused,
    knocked_out_over,
    orientation_of,
)
from ogviz.qc.significance import significance_gaps, stack_spacing
from ogviz.qc.typography import one_minus_sign, type_too_small, ungrouped_thousands

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

__all__ = [
    "CHECKS",
    "GAP_TOLERANCE_PX",
    "THOROUGH_CHECKS",
    "artist_name",
    "assert_clean",
    "audit",
    "bracket_tops_px",
    "buried_baselines",
    "clipped_artists",
    "colliding_ink",
    "dots_off_the_marks",
    "drawn_artists",
    "drawn_but_invisible",
    "ensure_rendered",
    "grid_warnings",
    "is_backdrop",
    "is_excused",
    "knocked_out_over",
    "layout_not_applied",
    "mean_rows_unaligned",
    "one_minus_sign",
    "orientation_of",
    "overflowing_text",
    "panels_disagree_about_ticks",
    "rows_outside_their_panel",
    "series_confusable_under_cvd",
    "significance_gaps",
    "stack_spacing",
    "text_hidden_behind_knockouts",
    "text_off_canvas",
    "text_over_data",
    "text_overlaps",
    "text_wider_than_its_panel",
    "ticks_in_the_headroom",
    "type_too_small",
    "ungrouped_thousands",
]


CHECKS = (
    text_overlaps,
    panels_disagree_about_ticks,
    layout_not_applied,
    series_confusable_under_cvd,
    rows_outside_their_panel,
    mean_rows_unaligned,
    ticks_in_the_headroom,
    colliding_ink,
    text_over_data,
    overflowing_text,
    clipped_artists,
    text_off_canvas,
    text_wider_than_its_panel,
    text_hidden_behind_knockouts,
    significance_gaps,
    stack_spacing,
    dots_off_the_marks,
    buried_baselines,
    one_minus_sign,
    ungrouped_thousands,
    grid_warnings,
)

THOROUGH_CHECKS = (drawn_but_invisible,)


def _run(check: Callable[..., list[str]], fig: Figure, **settings: float) -> list[str]:
    """Call a check, handing it whichever settings its own signature declares.

    `audit` used to name `text_overlaps` in a conditional to give it `min_gap`. That meant the next
    check needing an option had to be added to the same condition, and a check that GAINED an option
    without being added there would be silently called with its defaults — the option would look
    supported and do nothing. Read from the signature, a check opts in by declaring the parameter.
    """
    wanted = signature(check).parameters
    return check(fig, **{name: value for name, value in settings.items() if name in wanted})


def audit(fig: Figure, *, thorough: bool = False, min_gap: float = DEFAULT_MIN_GAP) -> list[str]:
    """Every complaint the checks can make about this figure.

    `thorough` adds the checks that render the figure once per artist. They are exact and slow —
    nine seconds on a six-panel grid against milliseconds for the rest — so they are asked for
    rather than paid for on every save. `python -m ogviz.qc --thorough` is the usual way in.

    `min_gap` is the least space two labels on one row may have and still read as two words. The
    default catches labels that have run together; it does not catch a figure that is merely tight,
    and three "this is too crowded" reports were caught by eye and by nothing else. A project that
    wants breathing room enforced passes its own floor — 32 px is the number one project measured
    its own comfortable figures against. It is a caller's number rather than a default because the
    same figures run 59 complaints at 32 px and none at 5, and one project's comfortable is another
    project's dense.
    """
    ensure_rendered(fig)
    checks = CHECKS + THOROUGH_CHECKS if thorough else CHECKS
    return [complaint for check in checks for complaint in _run(check, fig, min_gap=min_gap)]


def assert_clean(fig: Figure, *, min_gap: float = DEFAULT_MIN_GAP) -> None:
    """The build gate. Reports every complaint at once rather than the first.

    RAISED, not asserted, and the same is true of every other gate in this package. `python -O`
    deletes an `assert` statement outright, so writing the gate that way put the whole contract
    behind an interpreter flag: measured, a figure whose two labels shared 524 px of ink passed this
    function under `-O` without a word. `AssertionError` is kept as the type, since that is what
    callers already catch and what the message reads as.
    """
    complaints = audit(fig, min_gap=min_gap)
    if complaints:
        raise AssertionError("figure QC:\n  - " + "\n  - ".join(complaints))
