"""Regression tests for repairs of overflowed fort.7 integer fields."""

from __future__ import annotations

from reaxkit.engine.reaxff.generators.fort7_repair import _fix_data_line


def test_fix_data_line_writes_fused_atom_type_and_first_neighbor_split() -> None:
    # Atom type 1 and neighbor id 20594 were fused into ``120594``.  The nine
    # zero neighbor slots complete a frame whose header declares ten bonds.
    line = " ".join(
        ["20596", "120594", *(["0"] * 9), "1", "0.829", *(["0.000"] * 9), "1.665", "1.068", "-1.153"]
    ) + "\n"

    repaired, status = _fix_data_line(line, n_bonds=10)

    assert status == "fixed"
    fields = repaired.split()
    assert fields[:3] == ["20596", "1", "20594"]
    assert fields[3:12] == ["0"] * 9
    assert fields[12] == "1"
    # The parser requires atom number, type, ten neighbors, and molecule id
    # to all be separate integers.
    assert [int(value) for value in fields[:13]] == [20596, 1, 20594, *([0] * 9), 1]


def test_fix_data_line_leaves_already_separated_row_unchanged() -> None:
    line = "1 2 3 8 882 7364 0 1 0.537 0.546 0.546 0.545 0.000 4.013 0.000 1.188\n"

    repaired, status = _fix_data_line(line, n_bonds=5)

    assert status == "unchanged"
    assert repaired == line
