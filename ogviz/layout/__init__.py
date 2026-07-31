"""Everything a figure needs around its marks: the header, the frame, the axis, the page.

A facade and nothing else — every name here is defined in one of the modules below, and this file
exists so a caller writes `from ogviz.layout import save` without needing to know which. It used to
hold ten function bodies as well, so a reader had no way to tell what lived here from what was
passing through.

  header      the title band, and fitting panels under it
  frame       rules, baselines, the legend pill
  axis        which ticks belong to the data, and how far the marks reach
  write       saving, with the checks in front of it
  caption     a heading above and a source note below, neither able to exceed the figure
  collision   whether a label lands on the marks, and where else it could go
  overlap     labels against other labels
  ink         overlap decided on rendered pixels
  density     how much of the page is unused
  ticks       round numbers, and values in their display unit
  panels      a row of panels with a reserved caption row
"""

from ogviz.layout.axis import drawn_value_extent, ticks_over_data
from ogviz.layout.caption import caption, overflowing_text
from ogviz.layout.collision import (
    annotate_clear,
    clear_position,
    hits_data,
    point_offsets,
    text_over_data,
)
from ogviz.layout.density import dead_space, trim_margins
from ogviz.layout.density import measure as measure_density
from ogviz.layout.frame import baseline, hairline_grid, legend_pill, pill_frame, zero_baseline
from ogviz.layout.header import fit_under_header, panel_left_edge, settle_header, titled
from ogviz.layout.overlap import (
    assert_no_text_overlap,
    assert_nothing_clipped,
    clipped_artists,
    text_overlaps,
)
from ogviz.layout.panels import panel_row, text_width_points, wrap_to_width
from ogviz.layout.ticks import auto_decimals, format_value, round_ticks, value_ticks
from ogviz.layout.write import save

__all__ = [
    "annotate_clear",
    "assert_no_text_overlap",
    "assert_nothing_clipped",
    "auto_decimals",
    "baseline",
    "caption",
    "clear_position",
    "clipped_artists",
    "dead_space",
    "drawn_value_extent",
    "fit_under_header",
    "format_value",
    "hairline_grid",
    "hits_data",
    "legend_pill",
    "measure_density",
    "overflowing_text",
    "panel_left_edge",
    "panel_row",
    "pill_frame",
    "point_offsets",
    "round_ticks",
    "save",
    "settle_header",
    "text_over_data",
    "text_overlaps",
    "text_width_points",
    "ticks_over_data",
    "titled",
    "trim_margins",
    "value_ticks",
    "wrap_to_width",
    "zero_baseline",
]
