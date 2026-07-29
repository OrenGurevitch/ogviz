# FIXME — open defects and things that need a second look

Audit opened 2026-07-27 after a user-reported bug: `examples/out/03_stacked_brackets.png` showed
`* * *` with no bracket line under it. That turned out to be a class of defect, not one figure.

Each entry states the defect, the failure it causes, and how it was found. Resolved items move to
the bottom with the commit that fixed them.

---

## Open

Nothing. Every item below is closed; the file is kept for what the failures taught, not as a queue.


## Resolved

- **§22 — the bounding-box OVERLAP rule is gone; its spacing rule stays.** They were kept together
  for a while on the grounds that no figure distinguished them, which is an argument for removing
  the redundant one as much as for keeping it. `colliding_ink` answers "do these two collide" on
  rendered pixels and needs no threshold; the box rule answered the same question with a fraction
  of the smaller box's AREA, and could report a pair whose boxes intersect while no glyph does.
  That false positive is what led me to exempt whole classes of artist from the checks earlier,
  and a real defect then hid behind one of those exemptions.
  The SPACING half is untouched and is why the function still exists: two labels 5 px apart with
  glyphs not touching share no pixel and read as one word, which no overlap test of any kind can
  see.
  One claim withdrawn on the way: I had a constructed case where the box rule missed a real
  collision across a 3% overlap. It reproduced in Arial and not in DejaVu, which the suite pins, so
  it was font-dependent and is not part of the argument.


- **§27 — the "drawn but invisible" check works now, and is opt-in.** Two attempts, and the first
  is the interesting half: it measured with a boolean INK mask, and green drawn over blue is ink
  either way — so removing an artist sitting on ANOTHER inked artist changed no mask pixel, and
  every such artist measured as contributing nothing. That is why it once called nearly every label
  buried, and why it disagreed with `artist_ink`: the two were not measuring the same thing.
  Visibility is measured in COLOUR now — which pixels change value when the artist is taken away. A
  line behind a filled area decides 152 px of its own 4,741 footprint; a line on top decides all of
  them.
  It costs two renders per artist, so a bbox-and-z-order gate runs in front — nothing above an
  artist means nothing can hide it — taking a six-panel grid from 12.6 s to 9.1 s. Still far too
  slow for every save, so it is NOT in the default set: `audit(fig, thorough=True)` and
  `python -m ogviz.qc --thorough`. Verified to fire on a buried line and to stay silent across all
  thirteen examples.


### Audit round, 2026-07-29

Found by reading the package through after `02_display_units` shipped visibly broken and none of
the checks noticed. Five fixed, one withdrawn, two carried to Open above.

- **§8 — the caption row could be reached by a two-line x-label**, an xfail for as long as the row
  existed. It is sized when `panel_row` builds it, before the caller has plotted, so build-time care
  could never have got it right. `settle_caption` measures the panels' rendered `get_tightbbox` and
  drops the caption below them; `save` calls it.
- **§17 — two functions decided "scatter or filled shape" with DIFFERENT conditions**, so they could
  disagree about the same collection — and that decision is what broke `02_display_units`.
  `collision.point_offsets` is the single predicate now. My finding said three copies; the third
  reads offsets only for collections it tagged itself and never decides.
- **§18 — `layout/__init__.py` was a facade AND ten function bodies**, with `__all__ ` between two
  import blocks. Split into `header`, `frame`, `axis`, `write`; the facade defines nothing and a
  test holds that.
- **§19 — `align_mean_rows` was violin logic in `layout`**, which panels import, so the dependency
  pointed the wrong way. Moved with `share_value_limits` to `panels/grid.py`.
- **§20 — `qc` and `density` reached into other modules' privates.** A private two modules need is
  not private; both renamed.
- **§21 — four `layout` functions took bare `ax`/`axes` and `orientation: str`**, so no call site of
  theirs was typechecked.
- **§24 — `fit_under_header` returned whether the layout ran and nothing read it.** Now recorded on
  the figure and reported by `layout_not_applied`. `02_display_units` no longer refuses at all —
  the bracket alignment and mean-row placement changed what grows out of the axes.
- **§25 — the `layout` docstring said "No caption helper" while importing one.** The policy changed
  when captions were added and the prose did not.
- **§26 — `align_brackets` was in `__all__` and never imported**, so the export raised while the
  package looked complete. A test now walks `__all__` and asserts every name resolves.
- **§23 — WITHDRAWN.** `pill_frame` and `PANEL_FILL` were reported dead and are not: `legend_pill`
  calls one, which uses the other. The sweep that found them excluded `__init__.py` files while both
  are defined in one. Kept as a note because the same sweep would report them again.


Each has a regression test in `ogviz/test_audit_regressions.py`, named for the failure rather than
the fix, so none of them can come back quietly.

- **§1 — three or more stacked brackets silently lost their lines.** Headroom was a fixed fraction
  of the span whatever the bracket count, so from the third up the stack ran past the axis; and
  because matplotlib clips `Line2D` but not `Text`, the line vanished while its star stayed. Fixed
  by measuring: `bracket_stack` gained `draw=False`, and the panel grows the axis until the stack
  fits. The first attempt at this asserted on every 3-bracket panel instead of fixing it, because
  the stack was anchored to the axis TOP — raising the ceiling lifted the stack by exactly as much
  and stretched the data-per-pixel too, so the loop could never converge. Anchoring the first
  bracket to the DATA makes each pass strictly reduce the overshoot.
- **§2 / §3 — the mark clearance and the round-number ticks were silently wrong on a log axis.**
  The data-to-pixel ratio they need does not exist on a non-linear scale: the y ratio came back
  `0.0` and the ticks came back linear. Both now refuse a non-linear value axis with a sentence
  saying why (`require_linear_value_axis`).
- **§4 — `round_ticks` returned N identical ticks on an empty range.** Asserts `high > low`.
- **§5 — two groups could share a position** and be drawn on top of each other. Asserts distinct.
- **§6 — an impossible p-value printed as three stars.** `stars` asserts `0 <= p <= 1`, which
  covers every path into a label.
- **§9 — `ogviz.panels` resolved to the subpackage, not the function.** It was in `__all__` and the
  README's module table as callable while `ogviz.panels(...)` raised
  `TypeError: 'module' object is not callable`. Renamed to `panel_row`.
- **§10 — a colour list shorter than the values drew anyway.** Asserts one colour, or one per bar.
- **§11 — zero categories raised a raw `IndexError`.** Now says what was wrong.
- **§12 — the text guard could not see clipped content**, so the §1 figure passed it with zero
  problems. `clipped_artists` is a companion check, wired into `save`.
- **§7 — `lc-structural-mri` duplicated `round_ticks` byte for byte.** `_nice_ticks` matched
  `round_ticks` on every range tested and `_format_tick` matched `format_value` on every
  T1 cases including the `-0` guard. Both are one-line delegations now (lc `24c3cdf`), verified by
  rebuilding with the edit stashed and applied on identical data: no figure's text differs.
- **§15 — every spaced star was anchored 7.7 pt too low, since the first commit** (user-reported).
  The module's whole premise is that a star is placed by its INK, not its layout box. But
  `TextPath` emits an empty contour at the origin for a space, and `get_extents` counts it, so the
  ink bottom of `"* * *"` measured as **0.0** instead of 7.734 pt. A lone `"*"` anchored correctly
  and every spaced one sat 10.7 px lower — visible as the bottom star nearly touching its line.
  `spaced_stars` has joined with a space since the beginning, so the flagship feature was wrong
  for every multi-star label for the life of the package. `ink_extents_points` drops whitespace
  before measuring height, and keeps it for width.
- **§16 — there was no automated check that any of this is right.** Every defect so far was found
  by eye, which is why they accumulated. `ogviz/qc.py` measures the finished figure: text
  collisions, clipped artists, star-to-bracket distances (consistency, not just the nominal
  value), even bracket stacking, and dots on the central marks. `save` runs it, and
  `test_qc.py` runs it over all eight shipped examples plus one planted defect per check.
  Three false positives had to be fixed before it was trustworthy — the median x of a jittered
  cloud is not the group position, an error-bar cap is also a four-point line, and a lane
  recomputed after `tight_layout` is not the lane a dot was placed against. A gate that cries wolf
  gets switched off, so `points` now records the lane it used and QC compares against that.
- **§14 — the "synthetic" example data was seeded with a real study's numbers** (user-prompted).
  The samples were generated, but the generators were parameterised with an unpublished study's
  group means and group sizes, so its result was recoverable from the "examples". Every parameter
  is round and arbitrary now, and the README no longer claims what was not true. Drawing a random
  sample is not the same as inventing the distribution.
  The first version of THIS entry quoted those means and cohort sizes while explaining that they
  had been removed, which put them back on a public page. An entry about a leak is not exempt from
  causing one; say what kind of value it was, never the value.
- **§13 — a value label's knockout followed the glyph contours** (user-reported), so a dashed
  reference line still showed through the gaps between and inside the digits. Replaced the stroke
  halo with an opaque page-coloured box: invisible on a plain page, a clean knockout where
  something runs behind it.
