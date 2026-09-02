"""The title band, and fitting the panels under it.

`titled` returns where the header ends, so a caller sizes its panels from a measured header rather
than a guessed `rect` top. The title-to-subtitle gap is derived from the type size and the figure
height in points: a fixed figure-fraction gap collides with the title on a short figure and floats
away from it on a tall one.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from ogviz import units
from ogviz.require import require
from ogviz.tags import mark, marked
from ogviz.theme import INK, MUTED_INK, SUBTITLE_SIZE, TITLE_SIZE

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

TITLE_CLEARANCE = 1.35  # an axes title needs its own height plus the pad under it


def panel_left_edge(fig: Figure) -> float:
    """Where the leftmost panel starts, in figure coordinates.

    The anchor a left-aligned header hangs from. A `fig.suptitle` sits in FIGURE coordinates and a
    panel in AXES coordinates, and the two do not line up — a title at x=0.05 beside an axes that
    begins at 0.11 reads as a mistake, which is why a caller doing this by hand ends up reading
    `ax.get_position().x0` after a draw.
    """
    boxes = [ax.get_position().x0 for ax in fig.axes if ax.get_visible()]
    return float(min(boxes)) if boxes else float(fig.subplotpars.left)


def settle_header(fig: Figure) -> list[str]:
    """Re-anchor every left-aligned header line to the panels as they NOW are.

    A left-aligned header is measured against the panels, and `tight_layout` moves the panels after
    the header is written — so the alignment the caller asked for is stale by the time the figure is
    saved. The same failure `settle_caption` and `settle_bracket_labels` exist for, and the reason
    the alignment is stored as intent and resolved once instead of computed at call time.
    """
    left = panel_left_edge(fig)
    moved: list[str] = []
    for text in fig.texts:
        if not marked(text, "header_left"):
            continue
        x, y = text.get_position()
        if abs(x - left) < 1e-9:
            continue
        text.set_position((left, y))
        moved.append(f"re-anchored the header line {text.get_text()[:40]!r} to the panels")
    return moved


def titled(
    fig: Figure,
    title: str,
    *,
    subtitle: str | None = None,
    title_y: float = 0.98,
    size: float = TITLE_SIZE,
    subtitle_size: float = SUBTITLE_SIZE,
    align: Literal["center", "left"] = "center",
) -> float:
    """Bold-sans title with a serif grey subtitle. PASS THE RETURN VALUE TO `fit_under_header`.

    That is the first line because the return value is the point and it is easy to drop: a function
    that draws a title looks like it has finished, and the float it hands back is what makes the
    panels sit under the header instead of through it. A caller who found `titled` and not
    `fit_under_header` hand-rolled the layout three times, and the first attempt put its caption
    straight through the x tick labels.

    `size` is an argument because a title is sized against its figure, not against the house: the
    27 pt default suits a single tall panel and swamps a 12.8-inch report row, where it grows wide
    enough to reach the y tick labels.

    `align="left"` hangs both lines off the leftmost panel's left edge rather than centring them on
    the canvas. It is a different look, not a variant of the same one: the centred header belongs to
    a figure read as a plate, and the left-aligned one to a figure read as the top of a page. Two
    consumer scripts were carrying about eight lines each of hand-placed `fig.text` for it, and
    hand-placing it is what makes it drift when the layout runs afterwards.
    """
    figure_points = units.inches_to_points(fig.get_figheight())
    horizontal = "left" if align == "left" else "center"
    x = panel_left_edge(fig) if align == "left" else 0.5
    heading = fig.suptitle(
        title, fontsize=size, fontweight="bold", color=INK, x=x, y=title_y, ha=horizontal
    )
    header_bottom = title_y - size * 1.25 / figure_points
    lines = [heading]
    if subtitle is not None:
        lines.append(
            fig.text(
                x,
                header_bottom,
                subtitle,
                ha=horizontal,
                va="top",
                fontsize=subtitle_size,
                family="serif",
                color=MUTED_INK,
            )
        )
        header_bottom -= subtitle_size * 1.35 / figure_points
    if align == "left":
        for line in lines:
            mark(line, "header_left")
    return header_bottom


def fit_under_header(
    fig: Figure,
    header_bottom: float,
    *,
    bottom: float = 0.0,
    gap: float = 0.014,
) -> bool:
    """Lay the panels out and PIN their top just under the header `titled` reported.

    `tight_layout(rect=...)` treats the rect as room it may use, not a top edge it must reach: with
    a legend anchored below the axes it leaves a band of empty page between the panels and the
    title — about an eighth of the figure's height, which reads as a mistake rather than as
    breathing room. Doing
    the layout and then pinning the top closes it.

    RETURNS WHETHER THE PANELS WERE ACTUALLY LAID OUT, and there are two ways for that to be False.
    matplotlib refuses, with a warning and no effect, when the axis decorations cannot fit the rect
    — a two-line x tick label is enough. And it silently skips every axes whose gridspec pins any
    parameter of its own, which is what `panel_row` and `panel_grid` both do, so on a figure built
    by either of them the layout never ran at all. That second case returned True for as long as
    this function existed: the warning matplotlib raises for it is worded differently from the one
    being matched, and nothing else was asked. It is detected from the gridspec now, before the
    call, and the pointless call is skipped rather than made — a consumer's build log carried that
    UserWarning on every figure.

    Either way the top pin still applies, and the checks still measure what was actually drawn.

    `gap` is the only free number left, and it is the space BETWEEN the subtitle and the panels
    rather than a guess at where the header ends; `titled` measures that and returns it.

    An axes title grows UPWARD out of its axes, so pinning the subplot top without reserving for it
    drives the titles into the subtitle — which is what pinning a 2x2 grid of named panels did on
    the first attempt. The reservation is measured from the drawn titles rather than assumed from a
    font size, because a title that wrapped to two lines needs twice the room and says so only once
    it has been laid out.
    """
    # ASKED STRUCTURALLY FIRST, because the answer is knowable without running anything and the
    # figures this package builds are the ones it applies to. `tight_layout` silently ignores every
    # axes whose gridspec set any of its own parameters — `locally_modified_subplot_params()` — and
    # BOTH `panel_row` (which pins left/right/top/bottom) and `panel_grid` (which sets hspace and
    # wspace) do exactly that. So on an ogviz-built figure the call moved nothing, warned about it
    # in words this function did not recognise, and returned True.
    pinned = _pinned_axes(fig)
    if pinned and len(pinned) == len(fig.axes):
        # Not calling it at all. It is a no-op here by construction, and the only thing it produced
        # was a UserWarning in every consumer's build log about a figure that is laid out on
        # purpose. The pinned margins stand; the top pin below still applies. Recorded under its
        # OWN tag: this wrote `layout_refused` with an empty reason, so "pinned" and "refused" were
        # one tag with two meanings, told apart by truthiness, and nothing ever read the reason.
        mark(fig, "layout_pinned", f"all {len(fig.axes)} axes pin their own layout")
        _pin_top(fig, header_bottom, gap)
        return False

    # PARTLY pinned is still not laid out: `tight_layout` skips the pinned axes and moves the rest,
    # and this returned True for that — the same wrong answer the branch above exists to stop,
    # surviving for a grid with one hand-added axes beside it.
    applied = not pinned
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always", UserWarning)
        fig.tight_layout(rect=(0.0, bottom, 1.0, header_bottom))
        refused = [one for one in raised if _is_a_refusal(str(one.message))]
        applied = applied and not refused
    # Everything else matplotlib said goes back out. Recording warnings SWALLOWS them, and this
    # consumed the whole batch to read one message — so any other complaint raised during the
    # layout, about a font, a deprecation, an axes it could not place, vanished silently. Only the
    # one being handled here is kept back. `glyphs_must_render` does the same, for the same reason.
    for other in raised:
        if other not in refused:
            warnings.warn_explicit(other.message, other.category, other.filename, other.lineno)
    # Recorded on the figure as well as returned, because the return value went unread for a week
    # and the whole point was that this should not pass unnoticed. A REASON rather than a flag, so
    # the gate can say which of the two happened; `marked()` still reads it as the boolean it was.
    if refused:
        mark(fig, "layout_refused", str(refused[0].message))
    if pinned:
        mark(fig, "layout_pinned", f"{len(pinned)} of {len(fig.axes)} axes pin their own layout")
    _pin_top(fig, header_bottom, gap)
    return applied


# matplotlib's whole refusal vocabulary, from `_tight_layout.py`. It was matched against ONE of
# these, spelled with a capital T, so the lowercase `tight_layout not applied: number of rows...`
# went unrecognised as well as the incompatible-axes one. Matched case-insensitively and by the
# stem, because the rest of each message is the specific complaint.
_REFUSALS = ("tight layout not applied", "tight_layout not applied", "not compatible with")


def _is_a_refusal(message: str) -> bool:
    lowered = message.lower()
    return any(phrase in lowered for phrase in _REFUSALS)


def _pinned_axes(fig: Figure) -> list[Axes]:
    """Axes `tight_layout` will not move, because their gridspec pins parameters of its own.

    Asked of the gridspec rather than inferred from a warning: matplotlib may reword a message at
    any release, and this one it will not, since `locally_modified_subplot_params` is the same
    condition `get_subplotspec_list` tests to decide what to skip.
    """
    pinned: list[Axes] = []
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None:
            pinned.append(ax)  # tight_layout skips these outright
            continue
        grid = spec.get_topmost_subplotspec().get_gridspec()
        # `getattr`, and not because the stubs put the method on `GridSpec` and not on
        # `GridSpecBase`. A NESTED gridspec really is a different class — `GridSpecFromSubplotSpec`
        # — which does not define it at all, so spelling this as a plain attribute is an
        # `AttributeError` on any figure built with `add_gridspec(...)[0].subgridspec(...)`.
        # Absent means "nothing pinned here", which is also what matplotlib assumes of it.
        locally_modified = getattr(grid, "locally_modified_subplot_params", None)
        if callable(locally_modified) and locally_modified():
            pinned.append(ax)
    return pinned


def _pin_top(fig: Figure, header_bottom: float, gap: float) -> None:
    """Put the panels' top just under the header, reserving for any axes titles above them."""
    fig.canvas.draw()
    figure_px = fig.get_figheight() * fig.dpi
    titles_px = max(
        (float(ax.title.get_window_extent().height) for ax in fig.axes if ax.get_title().strip()),
        default=0.0,
    )
    reserved = titles_px / figure_px * TITLE_CLEARANCE
    fig.subplots_adjust(top=max(0.05, header_bottom - gap - reserved))


def room_below(fig: Figure, bottom: float, *, keep_panels: bool = True) -> float:
    """Make room under the axes for `bottom` (a figure fraction) WITHOUT shrinking the panels.

    Text added under an axis — a note row, a group bracket, a second strip — has to come from
    somewhere, and `subplots_adjust(bottom=...)` takes it from the plot. In a figure whose whole
    job is comparing bar heights, that silently shortens every bar.

    So this grows the CANVAS instead: the axes keep the height they have, and the figure gets taller
    by exactly the room the new margin needs. It is `fit_under_header`'s counterpart, and the same
    rule `panel_grid` encodes for a grid — take the room from the page, never from the cell.

    Returns the new figure height in inches. Pass `keep_panels=False` to take the room from the plot
    after all, which is right when the panels have height to spare and the page does not.

    The arithmetic is `height * (top - old_bottom) / (top - bottom)`, which is what a caller
    otherwise writes out by hand and re-derives every time another row appears.
    """
    require(
        0.0 <= bottom < 1.0,
        f"a bottom margin is a figure fraction, got {bottom}",
    )
    pars = fig.subplotpars
    top, old_bottom = float(pars.top), float(pars.bottom)
    require(
        bottom < top,
        f"a bottom margin of {bottom} leaves no room under a top of {top}",
    )
    height = float(fig.get_figheight())
    if keep_panels:
        panel_inches = height * (top - old_bottom)
        height = panel_inches / (top - bottom)
        fig.set_figheight(height)
    fig.subplots_adjust(bottom=bottom)
    return height
