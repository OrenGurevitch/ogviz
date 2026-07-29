# FIXME — open defects and things that need a second look

Audit opened 2026-07-27 after a user-reported bug: `examples/out/03_stacked_brackets.png` showed
`* * *` with no bracket line under it. That turned out to be a class of defect, not one figure.

Each entry states the defect, the failure it causes, and how it was found. Resolved items move to
the bottom with the commit that fixed them.

---

## Open

### Audit 2026-07-29 — read-through of the whole package

Opened after `02_display_units` shipped visibly broken and none of thirteen checks noticed. The
defect is fixed; these are what the read-through found around it. Each says what is wrong and why
it matters, not what to do about it.

- **§16 — FIXED. 🐛 the `layout` docstring contradicted the module.** It says "No caption helper. … A
  caption baked into an image is a second copy that drifts", and the file's own imports bring in
  `caption` from `ogviz.layout.caption`. The policy changed when captions were added and the
  docstring did not. A reader is told the opposite of what the package does.

- **§17 — FIXED (and the count was wrong: two, not three). 🔀 copies of "scatter or shape".** `layout.drawn_value_extent`,
  `collision._collection_paths` and `qc.dots_off_the_marks` each decide separately whether to read
  `get_offsets()` or `get_paths()`. That distinction is exactly what broke `02_display_units`, and
  the fix for it unified two of the three while leaving the third. One of them will drift again.

- **§18 — 🔀 `layout/__init__.py` is a package facade AND twelve function bodies.** It re-exports
  from six submodules and also defines `titled`, `save`, `baseline`, `hairline_grid`,
  `zero_baseline`, `legend_pill`, `pill_frame`, `ticks_over_data`, `share_value_limits`,
  `drawn_value_extent`, `align_mean_rows` and `fit_under_header`. `__all__` sits between two import
  blocks. Nothing tells a reader which names live here and which are passing through.

- **§19 — FIXED. 🔀 `align_mean_rows` was violin logic living in `layout`.** It knows the
  `ogviz_mean_row` tag, which only the violin panels set, and `layout` is imported BY panels — so a
  panel concept sat there with the dependency pointing the wrong way. Moved with
  `share_value_limits`, which calls it, into `ogviz/panels/grid.py`; their tests moved to
  `ogviz/panels/test_grid.py`. `layout` keeps `drawn_value_extent` and `ticks_over_data`, which are
  artist and axis primitives with no panel knowledge.

- **§20 — FIXED. ⚠️ two modules reached into another's privates.** `qc` imports `overlap._drawn_tick_labels`
  and `density` imports `collision._decoration`. Both are needed by more than their own module,
  which means neither should be private — the underscore is now a lie rather than a boundary.

- **§21 — FIXED. 🐛 four functions in `layout` were untyped.**
  `ticks_over_data(ax, ...)`, `share_value_limits(axes, ...)`, `drawn_value_extent(ax)` and
  `align_mean_rows(axes, ...)` take bare `ax`/`axes` and spell orientation `str` rather than the
  `Orientation` literal every other module uses. basedpyright cannot check any of their call sites.

- **§22 — EVALUATED, both kept; half of it is still a duplicate.** Measured on three constructed
  pairs. They are NOT interchangeable: two labels 5 px apart on one row, glyphs not touching, are
  reported by the box test and invisible to the ink test — "cognition" ending 3 px before
  "autonomic" begins renders as one word and shares no pixel. That gap rule is unique and stays.
  Its OTHER half — boxes intersecting by more than a fraction of the smaller area — asks the same
  question as `colliding_ink` with a threshold instead of an answer, and it is the weaker
  instrument of the two. Left in place because removing it would also drop the non-same-row case it
  covers, and no example currently distinguishes them. Revisit the moment it reports a pair that
  `colliding_ink` passes: at that point the threshold is deciding, and it should not be.

- **§23 — WITHDRAWN, the finding was wrong.** I reported `pill_frame` and `PANEL_FILL` as exported
  and called by nothing. They are not: `legend_pill` calls `pill_frame`, which uses `PANEL_FILL`,
  and `pill_frame` is public on purpose so a caller who has already built a legend can restyle it
  without handing its handles to a helper.
  The check that produced the finding excluded `__init__.py` files while both names are DEFINED in
  one, so every use of them was invisible to it. Recorded rather than deleted: a dead-code sweep
  that cannot see intra-module use will report the same two names again.


### §15 — 📚 `fit_under_header` reports a refusal that nothing acts on

`tight_layout` declines, with a warning and no effect, when axis decorations cannot fit the rect;
`02_display_units` hits it through its two-line x tick labels, which is §8 in a second place. The
figure then keeps default margins, which is survivable — the top pin is applied either way and the
checks still run on the result — and `fit_under_header` now returns whether the layout ran rather
than letting the warning vanish into a build log.

Nothing reads the return value. Either a caller should act on it or the gate should, and the real
fix is whatever closes §8: reserve for decorations that grow downward past their allotment. Tried
and rejected: more figure height, a bottom margin, a larger header gap — the constraint is
structural, not room.


### §8 — 🎨 the caption row can still be reached by a two-line x-label

Held as an xfail (`test_a_two_line_x_label_still_reaches_the_caption_row`). The reserved row is
sized from the caption's own line count, so decorations that grow downward past their allotment
still reach it. `constrained` layout reserves correctly but then overrides the row height ratios
and pushes the caption up into the tick labels, so it is not the fix either.

**Fix:** unknown. Measure the axes' rendered bottom after a draw and grow the figure to suit.

---

## Resolved

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

## Ink measurement — a companion check that is NOT shipped (2026-07-28)

`ogviz/layout/ink.py` measures overlap on rendered pixels. The obvious companion — "this artist is
drawn and contributes no pixel, so something covers it" — was written, disagreed with `artist_ink`
called directly on the same artists, and is not shipped.

Part of the cause is now understood and is written into the module: a DIFFERENCE render (with minus
without) gives an artist's marginal contribution, not its footprint, so anything whose pixels are
also painted by something else measures as zero. That is why the first version reported nearly every
label as buried. `artist_ink` renders alone instead, and subtracts a bare render so the axes frame
does not appear in every mask.

What is still unexplained: with the difference implementation, `buried_artists` returned indices for
artists that `artist_ink` reported hundreds of pixels for, in the same process, from the same list.
Renders were measured stable (0 px between consecutive draws), so it was not a cold-font-cache
effect — a theory raised and disproved on the way. Before shipping the check, reproduce that
disagreement and explain it.
