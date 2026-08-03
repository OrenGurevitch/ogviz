from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ogviz.layout.panels import grid_warnings, panel_grid, rows_that_fit


def test_the_row_budget_matches_the_case_it_was_written_from() -> None:
    """5 in cells two wide, a 6.5 in column, 9 in of page.

    A cell 3.2 in high allows four rows and 3.8 allows three — which is how one grid needing a
    fourth row comes to pin the cell height for every other grid in a document.
    """
    # cell_height = width / ncol / aspect, so a 3.2 in cell at 10 in wide and 2 columns is 5/3.2.
    assert rows_that_fit(2, width=10.0, cell_aspect=5.0 / 3.2) == 4
    assert rows_that_fit(2, width=10.0, cell_aspect=5.0 / 3.8) == 3


def test_a_set_of_grids_is_one_width_whatever_the_panel_count() -> None:
    """Sizing by CELL makes a one-cell grid half as wide as a four-cell one, and a document
    placing both at one column then magnifies the small one."""
    widths = set()
    for count in (1, 2, 4, 7):
        fig, _axes = panel_grid(count)
        widths.add(round(float(fig.get_size_inches()[0]), 6))
        plt.close(fig)
    assert len(widths) == 1, widths


def test_a_cell_keeps_its_aspect_whatever_the_panel_count() -> None:
    for count, ncol in ((1, 2), (4, 2), (6, 3)):
        fig, _axes = panel_grid(count, ncol=ncol)
        columns = min(ncol, count)
        rows = -(-count // columns)
        width, height = fig.get_size_inches()
        assert (width / columns) / (height / rows) == pytest.approx(1.32, abs=1e-6)
        plt.close(fig)


def test_an_empty_slot_is_reported_because_it_is_invisible_until_it_is_drawn() -> None:
    fig, _axes = panel_grid(7, ncol=2)
    complaints = grid_warnings(fig)
    assert any("leaves 1 slot(s) empty" in complaint for complaint in complaints), complaints


def test_a_grid_too_tall_for_the_page_names_the_row_count_that_fits() -> None:
    """And says to take a panel out, because shrinking the cell flattens every other grid."""
    fig, _axes = panel_grid(7, ncol=2)
    complaints = grid_warnings(fig)
    tall = [c for c in complaints if "will not fit" in c]
    assert tall and "3 would" in tall[0], complaints
    assert "shared" in tall[0]


def test_a_grid_that_fits_says_nothing() -> None:
    fig, _axes = panel_grid(4, ncol=2)
    assert not grid_warnings(fig)


def test_a_figure_that_is_not_a_grid_is_not_judged_as_one() -> None:
    """`grid_warnings` is in the gate, so it must be silent about every other figure."""
    fig, _ax = plt.subplots()
    assert not grid_warnings(fig)
