"""Automated checks on a rendered figure, so defects are caught by the build and not by eye.

Every function here answers one question about a FINISHED figure, measured from the artists that
were actually drawn. Each exists because the corresponding defect shipped at least once: a star
that sat closer to its bracket than its neighbours did, a bracket line clipped out of the axes
while its star stayed, two tick labels touching, a dot on top of the mean line.

`audit` runs them all and returns every complaint. `assert_clean` is the build gate — `save` calls
it, so a project cannot write a figure that fails one of these without being told.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ogviz.layout.caption import overflowing_text
from ogviz.layout.collision import quoted, text_over_data
from ogviz.layout.ink import exact_overlaps, hidden_artists
from ogviz.layout.overlap import clipped_artists, text_overlaps
from ogviz.orientation import read_orientation
from ogviz.significance import STACK_GAP_PX, ink_extents_points

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

GAP_TOLERANCE_PX = 1.5  # rendering rounds to the pixel grid; past this it is a real drift
# A dot is flagged only once it is well inside the lane. The lane is PHYSICAL — sized in points of
# ink — so a `tight_layout` after the dots were placed shrinks the axes, the marks then cover more
# data units, and a dot that cleared the lane at draw time is marginally inside it afterwards.
# Real but invisible, and flagging it would fire on every tight_layout. 0.75 still catches a dot
# actually on the marks.
LANE_TOLERANCE = 0.75
MIN_STAR_GAP_PX = 1.0  # below this the glyph is touching its own bracket
BRACKET_POINTS = 4  # a bracket is drawn as four points: down, across, down


def _orientation(ax: Axes) -> str:
    """The panel's orientation: what it recorded, or failing that what its marks suggest.

    The marks are a fallback for a figure this package did not draw. Inferring it where the answer
    was known produced the worst class of QC failure — a confident complaint about a correct
    figure. A grouped bar panel votes exactly once, and wrongly: `errorbar` keeps its bars and caps
    in a LineCollection, so the only two-point line is the zero baseline, whose y is constant, and
    the panel reads as horizontal. Brackets were then measured along x, none matched the bracket
    shape, and all five stars were reported as having no bracket under them.

    The vote itself is unchanged and still right for what it is: an IQR whisker is a two-point line
    on the group's centre, constant x in a vertical panel and constant y in a horizontal one.
    """
    recorded = read_orientation(ax)
    if recorded is not None:
        return recorded
    vertical = horizontal = 0
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if xdata.size != 2:
            continue
        if float(xdata[0]) == float(xdata[1]):
            vertical += 1
        elif float(ydata[0]) == float(ydata[1]):
            horizontal += 1
    return "horizontal" if horizontal > vertical else "vertical"


def _looks_like_stars(text: str) -> bool:
    """A row of asterisks, however `spaced_stars` spaced them. For figures with no tag to read."""
    stripped = text.strip()
    return bool(stripped) and set(stripped.split()) == {"*"}


def _stars(ax: Axes) -> list[Text]:
    """Every label `bracket_stack` placed over a bracket, whatever it says.

    Read from the tag the stack already sets, not from the wording. Matching asterisks instead
    excluded `"n.s."` — so the one label in a stack that is not a row of asterisks was the one
    exempt from the check that they all sit at the same height, and a figure shipped with `n.s.`
    visibly below its neighbours while the audit stayed silent. `label_for` means a project can
    print any wording it likes, so no list of strings could have covered this.

    The wording test remains for a figure this package did not draw, where there is no tag.
    A forest strip marks whole rows and flags those, so they are skipped.
    """
    tagged = [t for t in ax.texts if getattr(t, "ogviz_bracket_star", False)]
    candidates = tagged or [t for t in ax.texts if _looks_like_stars(t.get_text())]
    return [t for t in candidates if not getattr(t, "ogviz_column_star", False)]


def _is_bracket(data: np.ndarray) -> bool:
    """A bracket is [edge, top, top, edge]: two equal middles, two equal lower ends.

    Counting four points alone is not enough — an error bar's cap is also a four-point line, and
    treating those as brackets reported a whole bar panel as an "uneven stack" with brackets 0 px
    apart. The shape is the discriminator.
    """
    if data.size != BRACKET_POINTS:
        return False
    edge_a, top_a, top_b, edge_b = (float(v) for v in data)
    return top_a == top_b and edge_a == edge_b and edge_a < top_a


def _bracket_tops_px(ax: Axes) -> list[float]:
    """Where each bracket's outer edge sits, in display pixels along the axis it grows on."""
    axis = 1 if _orientation(ax) == "vertical" else 0
    tops = []
    for line in ax.lines:
        data = np.asarray(line.get_ydata() if axis == 1 else line.get_xdata(), dtype=float)
        if not _is_bracket(data):
            continue
        point = (0.0, float(np.max(data))) if axis == 1 else (float(np.max(data)), 0.0)
        tops.append(float(ax.transData.transform(point)[axis]))
    return sorted(tops)


def significance_gaps(fig: Figure) -> list[str]:
    """Every star must sit the same distance above its own bracket, and never on it.

    CONSISTENCY is the check, not the absolute number. A `tight_layout` after the marks are drawn
    rescales the axes, and the gap is a pixel quantity converted through the transform, so the
    whole stack drifts a little together — harmless, and even. What is not harmless is one star
    sitting at a different distance from the rest, which is what a reader notices and what a lone
    "*" did for as long as `spaced_stars` existed: `TextPath` counts a space as an empty contour at
    the origin, so the ink bottom of "* * *" measured as zero and every spaced star sat 7.7 pt low
    while a single one was placed correctly.

    Measured from the glyph's INK, never its layout box.
    """
    fig.canvas.draw()
    px_per_point = fig.dpi / 72.0
    complaints: list[str] = []
    for ax in fig.axes:
        tops = _bracket_tops_px(ax)
        gaps: list[tuple[str, float]] = []
        for star in _stars(ax):
            axis = 1 if _orientation(ax) == "vertical" else 0
            size = star.get_fontsize()
            ink_low, _ = ink_extents_points(star.get_text(), float(size), axis=axis)
            baseline_px = float(ax.transData.transform(star.get_position())[axis])
            ink_bottom_px = baseline_px + ink_low * px_per_point
            below = [t for t in tops if t <= ink_bottom_px + GAP_TOLERANCE_PX]
            if not below:
                complaints.append(f"the star {star.get_text()!r} has no bracket under it")
                continue
            gaps.append((star.get_text(), ink_bottom_px - max(below)))
        for label, gap in gaps:
            if gap < MIN_STAR_GAP_PX:
                complaints.append(f"the star {label!r} is {gap:.1f} px from its bracket — touching")
        if len(gaps) > 1:
            spread = max(g for _l, g in gaps) - min(g for _l, g in gaps)
            if spread > GAP_TOLERANCE_PX:
                complaints.append(
                    "stars sit at different distances from their brackets: "
                    + ", ".join(f"{g:.1f}" for _l, g in gaps)
                    + " px"
                )
    return complaints


def _levels(tops: list[float]) -> list[float]:
    """The distinct heights brackets sit at, brackets on one line counted once.

    A panel with one comparison per category puts every bracket at the SAME height — siblings on one
    line, which is what `significance_row` draws. Treating each bracket as a step of its own then
    reported "brackets are 0 px apart", a complaint about the very thing that made the row a row.

    Clustering by height covers all three arrangements with one rule: a stack has one bracket per
    level, a row has one level, and a row of stacks has several brackets on each of several levels.
    """
    levels: list[float] = []
    for top in sorted(tops):
        if not levels or top - levels[-1] > GAP_TOLERANCE_PX:
            levels.append(top)
    return levels


def stack_spacing(fig: Figure) -> list[str]:
    """Stacked brackets must be even, and further apart than a star is from its own line."""
    fig.canvas.draw()
    complaints: list[str] = []
    for ax in fig.axes:
        tops = _levels(_bracket_tops_px(ax))
        if len(tops) < 2:
            continue
        steps = np.diff(tops)
        if float(steps.std()) > GAP_TOLERANCE_PX * 2:
            complaints.append(
                f"brackets are unevenly stacked: steps {np.round(steps, 1).tolist()} px"
            )
        if float(steps.min()) < STACK_GAP_PX * 0.5:
            complaints.append(
                f"brackets are {steps.min():.0f} px apart, closer than a star is to its own line"
            )
    return complaints


def _group_centres(ax: Axes) -> list[float]:
    """The exact x of each group, taken from the vertical marks drawn ON the centre line.

    Estimating it from the dots does not work: the jitter is symmetric about the position but
    avoids a lane at its centre, so both the median and the min-max midpoint land slightly off and
    the lane is then measured from the wrong origin — which flags dots that are perfectly placed.
    The IQR whisker is a two-point line at exactly the position, so read it from there.
    """
    across = 0 if _orientation(ax) == "vertical" else 1
    centres = []
    for line in ax.lines:
        data = np.asarray(line.get_xdata() if across == 0 else line.get_ydata(), dtype=float)
        if data.size == 2 and float(data[0]) == float(data[1]):
            centres.append(float(data[0]))
    return sorted(set(centres))


def _nearest_centre(ax: Axes, offsets: np.ndarray, across: int) -> float | None:
    centres = _group_centres(ax)
    if not centres:
        return None
    middle = float((offsets[:, across].min() + offsets[:, across].max()) / 2)
    return min(centres, key=lambda c: abs(c - middle))


def dots_off_the_marks(fig: Figure) -> list[str]:
    """No jittered point may sit on the central marks it is there to leave readable.

    Compares against the lane `points` RECORDED at draw time, not a recomputed one. The lane is a
    step function of y and its steps move when the axes are resized, so a lane recomputed after a
    `tight_layout` can put a dot in a different band than the one it was placed against — which
    reported perfectly placed dots as violations, twice, while I tuned thresholds at it.
    """
    fig.canvas.draw()
    complaints: list[str] = []
    for ax in fig.axes:
        upright = _orientation(ax) == "vertical"
        across = 0 if upright else 1
        for collection in ax.collections:
            lane = getattr(collection, "ogviz_lane", None)
            position = getattr(collection, "ogviz_position", None)
            if lane is None or position is None:
                continue
            offsets = np.asarray(collection.get_offsets(), dtype=float)
            if offsets.shape[0] != np.asarray(lane).shape[0]:
                continue
            inside = int(np.count_nonzero(np.abs(offsets[:, across] - position) < lane * 0.999))
            if inside:
                complaints.append(f"{inside} dot(s) sit on the central marks")
    return complaints


def _reference_lines(ax: Axes) -> list:
    return [line for line in ax.lines if getattr(line, "ogviz_reference", False)]


def buried_baselines(fig: Figure) -> list[str]:
    """A spine or a threshold must not be covered by the marks it is there to be read against.

    A bar grows from the category axis, so its base lies exactly along that spine. matplotlib
    draws spines at zorder 2.5 and bars above them, so the axis survives only in the gaps between
    bars and reads as a broken line. Caught by comparing z-order against the marks that actually
    overlap the spine, not by assuming which panel type is being drawn.

    Translucency is no defence and is deliberately not exempted: the house bars are drawn at 0.85,
    and 0.85 over a 1.6-point rule is exactly the washed-out segment this check exists to catch.
    Marks a panel puts UNDER the axis on purpose — a highlight column, a reference band — sit below
    the spine's z-order and never reach this test.
    """
    fig.canvas.draw()
    complaints: list[str] = []
    for ax in fig.axes:
        if not ax.axison:
            continue  # `ax.axis("off")` leaves the spine objects visible but draws none of them
        for side, spine in ax.spines.items():
            if not spine.get_visible():
                continue
            spine_box = spine.get_window_extent()
            spine_z = spine.get_zorder()
            buried = 0
            for patch in ax.patches:
                if patch.get_zorder() <= spine_z:
                    continue
                if patch.get_window_extent().overlaps(spine_box):
                    buried += 1
            if buried:
                complaints.append(f"the {side} spine is covered by {buried} mark(s) drawn over it")
        for line in _reference_lines(ax):
            # A threshold is there to be compared against the bars. Behind them it survives only in
            # the gaps between them, and the reader loses the one comparison the line was added to
            # make — the same defect as bars covering the category axis, one artist along.
            line_box = line.get_window_extent()
            over = [
                patch
                for patch in ax.patches
                if patch.get_zorder() > line.get_zorder()
                and patch.get_window_extent().overlaps(line_box)
            ]
            if over:
                value = float(np.asarray(line.get_ydata(), dtype=float)[0])
                complaints.append(
                    f"the reference line at {value:g} is behind {len(over)} mark(s) — "
                    "a threshold has to stay readable across the panel"
                )
    return complaints


def one_minus_sign(fig: Figure) -> list[str]:
    """Every negative number in a figure must use the same glyph for its sign.

    matplotlib typesets its own tick labels with U+2212 while `"{:.2f}".format` writes an ASCII
    hyphen, so a panel that prints its values lands both in one figure — different glyphs, at
    different widths, for the same sign.
    """
    hyphen: set[str] = set()
    minus: set[str] = set()
    for ax in fig.axes:
        for text in [*ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            content = text.get_text().strip()
            if not content or not any(character.isdigit() for character in content):
                continue
            (hyphen if "-" in content else minus if "\u2212" in content else set()).add(content)
    if hyphen and minus:
        return [
            f"two different minus signs in one figure: {sorted(hyphen)[:3]} use a hyphen, "
            f"{sorted(minus)[:3]} use \u2212"
        ]
    return []


def _drawn_artists(ax) -> list:
    """Everything on one axes that a reader can see, in one list."""
    from ogviz.layout.overlap import drawn_tick_labels

    items = [*ax.texts, *ax.lines, *ax.patches, *ax.collections]
    if ax.axison:
        items += drawn_tick_labels(ax)
    return [artist for artist in items if artist.get_visible()]


def mean_rows_unaligned(fig: Figure) -> list[str]:
    """Panels sharing a value scale must print their means on one line, at one size.

    Four rows at four heights read as four different kinds of number. The row's distance from the
    frame is a visual constant a reader uses without noticing, and it only means something if it is
    the same in every panel being compared.

    Only checked where the panels genuinely share a scale — panels on different scales are separate
    figures that happen to share a page, and there is no line for them to share.
    """
    rows = [
        (ax, text) for ax in fig.axes for text in ax.texts if getattr(text, "ogviz_mean_row", False)
    ]
    if len({id(ax) for ax, _text in rows}) < 2:
        return []
    scales = {tuple(round(v, 9) for v in ax.get_ylim()) for ax, _text in rows}
    if len(scales) > 1:
        return []
    heights = {round(float(text.get_position()[1]), 6) for _ax, text in rows}
    sizes = {round(float(text.get_fontsize()), 3) for _ax, text in rows}
    complaints = []
    if len(heights) > 1:
        complaints.append(
            f"printed means sit at {len(heights)} different heights: {sorted(heights)}"
        )
    if len(sizes) > 1:
        complaints.append(f"printed means are set at {len(sizes)} different sizes: {sorted(sizes)}")
    return complaints


def rows_outside_their_panel(fig: Figure) -> list[str]:
    """A printed mean must land between the frame and the marks it belongs to.

    The failure this exists for put the row at -0.25 on an axis running 0.0004 to 0.006 — off the
    panel entirely — because the code measuring "the lowest mark" was reading a scatter's MARKER
    OUTLINE instead of its offsets, and a marker outline is a unit circle about the origin whatever
    the data is. On values of order one that is a small error; on values of order 0.001 the answer
    is not on the page.

    Cheap, and it asks the question directly rather than trusting the measurement that failed.
    """
    from ogviz.layout import drawn_value_extent

    complaints: list[str] = []
    for ax in fig.axes:
        rows = [text for text in ax.texts if getattr(text, "ogviz_mean_row", False)]
        if not rows:
            continue
        extent = drawn_value_extent(ax)
        if extent is None:
            continue
        floor, _top = ax.get_ylim()
        for text in rows:
            where = float(text.get_position()[1])
            if not floor <= where <= extent[0]:
                complaints.append(
                    f"the printed mean {text.get_text()!r} sits at {where:g}, outside the band "
                    f"between the frame ({floor:g}) and the lowest mark ({extent[0]:g})"
                )
    return complaints


def layout_not_applied(fig: Figure) -> list[str]:
    """A figure whose layout engine declined, and which is therefore on default margins.

    `tight_layout` warns and does nothing when the axis decorations will not fit the rect it was
    given, leaving whatever margins were already set. `fit_under_header` catches that and returns
    whether it ran — and nothing read the answer, so a figure could be laid out by nobody and look
    merely a bit loose. Recorded on the figure so the gate can say it out loud.

    Not fatal on its own: default margins are usually survivable and the rest of the checks still
    measure what was actually drawn. It is here because "nobody laid this out" should be a sentence
    someone reads, not a warning swallowed by a build log.
    """
    if getattr(fig, "ogviz_layout_refused", False):
        return [
            "the layout engine declined — the decorations do not fit, so this figure kept default "
            "margins. Give it more height, or shorten what grows out of the axes"
        ]
    return []


def panels_disagree_about_ticks(fig: Figure) -> list[str]:
    """Panels on one value scale must carry the same value ticks.

    The rules are what a reader compares panels with, so different rules in each panel make the
    same height look like different heights. A grid arrived with five in one row and eight in the
    next, because each panel chose its own from its own data before the scale was shared.

    Only checked where the panels genuinely share a scale — panels on different scales are separate
    figures that happen to share a page.
    """
    scales: dict[tuple[float, float], set[tuple[float, ...]]] = {}
    for ax in fig.axes:
        if not ax.axison or not ax.collections:
            continue
        low, high = ax.get_ylim()
        key = (round(low, 9), round(high, 9))
        ticks = tuple(round(float(t), 9) for t in ax.get_yticks() if low <= t <= high)
        scales.setdefault(key, set()).add(ticks)
    complaints = []
    for (low, high), sets in scales.items():
        if len(sets) > 1:
            counts = sorted(len(one) for one in sets)
            complaints.append(
                f"panels sharing the scale {low:g}..{high:g} carry different value ticks "
                f"({counts} of them) — the rules a reader compares the panels with disagree"
            )
    return complaints


def ticks_in_the_headroom(fig: Figure) -> list[str]:
    """A value tick above every mark on the panel, in the space reserved for brackets.

    The space above the data is layout, held open so a bracket stack has somewhere to go. A tick
    and its gridline there say a measurement could sit at that height when none can, and they make
    panels disagree for no reason a reader can see — one whose stack happens to clear a round
    number carries an extra rule, its neighbour does not, and the two are meant to be compared.
    """
    fig.canvas.draw()
    complaints: list[str] = []
    for ax in fig.axes:
        if not _bracket_tops_px(ax):
            # No stack, so nothing is being held open. A scatter with a top margin is ordinary
            # breathing room, not reserved space, and ticks in it are the axis doing its job.
            continue
        reach = _data_reach(ax)
        if reach is None:
            continue
        _low, high = ax.get_ylim()
        stray = [float(tick) for tick in ax.get_yticks() if reach + 1e-9 < float(tick) <= high]
        # One tick above the marks is the axis closing the data in, and is wanted: without it a
        # coarse axis leaves the top of a violin with nothing to read against. Two or more is a
        # ladder climbing through the space held open for brackets.
        if len(stray) > 1:
            complaints.append(
                f"ticks {stray[1:]} climb above the marks into the bracket headroom "
                f"(one bracketing tick at {stray[0]:g} is expected)"
            )
    return complaints


def _data_reach(ax: Axes) -> float | None:
    """The highest value any mark reaches, or None where the panel draws no marks."""
    from ogviz.layout import drawn_value_extent

    extent = drawn_value_extent(ax)
    return extent[1] if extent is not None else None


def series_confusable_under_cvd(fig: Figure) -> list[str]:
    """Legend entries that separate for normal vision and merge under colour-vision deficiency.

    Read off the LEGEND, because that is the set the reader is asked to tell apart. Marks with no
    legend entry are not being distinguished by colour in the first place — a violin's fill and its
    edge are one series in two tones, and reporting them would be noise.

    Advisory in spirit but fails the build like the rest: a pair that merges is a figure a reader
    cannot use, and "add a marker or a dash" is a small change to make while the figure is open.
    """
    from ogviz.color import indistinguishable_series

    complaints: list[str] = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        entries: dict[str, str] = {}
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=False):
            color = _handle_color(handle)
            if color is not None:
                entries[text.get_text()] = color
        if len(entries) > 1:
            complaints.extend(indistinguishable_series(entries))
    return complaints


def _handle_color(handle) -> str | None:
    """The one colour a legend handle stands for, or None if it does not stand for one."""
    from matplotlib.colors import to_hex

    for getter in ("get_color", "get_facecolor", "get_markerfacecolor"):
        found = getattr(handle, getter, None)
        if found is None:
            continue
        try:
            value = found()
        except (TypeError, ValueError):
            continue
        flat = np.asarray(value, dtype=object).ravel()
        if flat.size in (3, 4) and all(isinstance(v, (int, float)) for v in flat):
            red, green, blue = (float(flat[index]) for index in range(3))
            return to_hex((red, green, blue))
        if flat.size == 1 and isinstance(flat[0], str):
            return to_hex(str(flat[0]))
    return None


def drawn_but_invisible(fig: Figure) -> list[str]:
    """Marks that would draw something and are covered by something else.

    The case no geometric check can see: the artist is where it was asked to be, its bounding box is
    fine, and a reader cannot see it. Measured in colour — which pixels change value when it is
    taken away — because in a boolean ink mask a line drawn over a filled area contributes nothing
    whether it is visible or not.

    Backdrops and the marks sitting on them are exempt: a shaded column exists to be covered.
    """
    complaints: list[str] = []
    for ax in fig.axes:
        artists = [
            artist for artist in [*ax.lines, *ax.collections, *ax.patches] if not _backdrop(artist)
        ]
        for index in hidden_artists(fig, artists):
            complaints.append(
                f"a {type(artists[index]).__name__} is drawn and almost entirely covered — "
                "either it is redundant or something is on top of it"
            )
    return complaints


def colliding_ink(fig: Figure) -> list[str]:
    """Artists that genuinely share pixels, decided by the renderer rather than by their boxes.

    The box test cannot settle this. It reports a collision between "Ay" and "1.42" whose boxes
    intersect while no glyph does, and misses one between a label and a curve whose box is mostly
    empty. Boxes are used here only to pick the pairs worth rendering for, which is the split
    Theophil and Schodl's scatter-chart labelling uses and Kakoulis and Tollis survey: a cheap
    test for what MIGHT collide, an exact one for what DOES.

    Deliberately-placed labels are exempt, as everywhere else — a value label sits on its bar and a
    star sits on its bracket by design, and each has a check that measures that relationship
    properly.
    """
    complaints: list[str] = []
    for ax in fig.axes:
        artists = _drawn_artists(ax)
        text_index = {index for index, a in enumerate(artists) if hasattr(a, "get_text")}
        for first, second, shared in exact_overlaps(fig, artists):
            if first not in text_index and second not in text_index:
                continue  # two marks may overlap; that is a chart, not a defect
            if _excused(artists[first], artists[second]) or _excused(
                artists[second], artists[first]
            ):
                continue
            if _backdrop(artists[first]) or _backdrop(artists[second]):
                continue
            complaints.append(
                f"{_name(artists[first])} and {_name(artists[second])} share {shared} px of ink"
            )
    return complaints


def _backdrop(artist) -> bool:
    """A shaded region text is MEANT to sit on — a highlighted column, a reference band.

    Not a mark: it carries no value of its own and is there to tint the marks that do. Text over it
    is the design, and the value labels knock out to its colour rather than to the page precisely so
    they stay readable on it.
    """
    return bool(getattr(artist, "ogviz_backdrop", False))


def _excused(label, other) -> bool:
    """Whether `label` is deliberately placed against `other`, and only against `other`.

    An anchored label is pardoned for touching the thing it labels — a star over its bracket, a
    value printed past its own whisker. It is not pardoned for landing on anything else. Treating
    the flag as a blanket exemption is what let a reference-line label sit on a bar without a word:
    the label is pinned to its line vertically and free to slide along it, so the one axis it could
    actually be wrong on was the one being excused.
    """
    anchor = getattr(label, "ogviz_anchor", None)
    if anchor is not None and anchor is other:
        return True
    # A star in a stack sits between its own bracket and the next one up, and how close it may come
    # to either is measured by `significance_gaps` and `stack_spacing` — in points of ink, against
    # the glyph's own extents, far more precisely than "these two share a pixel". Contact there is
    # their business, not this check's.
    return bool(getattr(label, "ogviz_bracket_star", False)) and bool(
        getattr(other, "ogviz_bracket", False)
    )


def _name(artist) -> str:
    text = getattr(artist, "get_text", None)
    if text is not None:
        return repr(quoted(text()))
    return type(artist).__name__


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
    significance_gaps,
    stack_spacing,
    dots_off_the_marks,
    buried_baselines,
    one_minus_sign,
)


# Two renders per artist, so it costs seconds on a busy panel where the rest of the gate costs
# milliseconds. Out of the default set for that reason alone — it is correct and it is affordable
# occasionally, not on every save.
THOROUGH_CHECKS = (drawn_but_invisible,)


def audit(fig: Figure, *, thorough: bool = False) -> list[str]:
    """Every complaint the checks can make about this figure.

    `thorough` adds the checks that render the figure once per artist. They are exact and slow —
    nine seconds on a six-panel grid against milliseconds for the rest — so they are asked for
    rather than paid for on every save. `python -m ogviz.qc --thorough` is the usual way in.
    """
    checks = CHECKS + THOROUGH_CHECKS if thorough else CHECKS
    return [complaint for check in checks for complaint in check(fig)]


def assert_clean(fig: Figure) -> None:
    """The build gate. Reports every complaint at once rather than the first."""
    complaints = audit(fig)
    assert not complaints, "figure QC:\n  - " + "\n  - ".join(complaints)
