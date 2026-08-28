from __future__ import annotations

from reaxkit.engine.reaxff.io.fort99_handler import Fort99Handler


def test_fort99_handler_accepts_single_and_two_line_records(tmp_path) -> None:
    source = tmp_path / "fort.99"
    source.write_text(
        "FField value QM/Lit value Weight Error Total error\n"
        "water Charge atom: 1  -0.8 -0.7 0.1 1.0 1.0\n"
        "Bond distance: 1 2\n"
        "1.01 1.00 0.02 0.25 1.25\n",
        encoding="utf-8",
    )
    table = Fort99Handler(source).dataframe()

    assert len(table) == 2
    assert table.iloc[0]["section"] == "CHARGE"
    assert table.iloc[1]["section"] == "GEOMETRY"
    assert table.iloc[1]["title"] == "Bond distance: 1 2"
    assert table.iloc[1]["title_lineno"] == 3
    assert table.iloc[1]["lineno"] == 4


def test_fort99_handler_recognizes_cell_parameter_records(tmp_path) -> None:
    source = tmp_path / "fort.99"
    source.write_text(
        "crystal a: 5.10 5.00 1.0 0.01 0.01\n"
        "b:\n"
        "5.20 5.25 1.0 0.01 0.02\n"
        "crystal alpha: 90.1 90.0 1.0 0.01 0.03\n",
        encoding="utf-8",
    )

    table = Fort99Handler(source).dataframe()

    assert table["section"].tolist() == [
        "CELL PARAMETERS",
        "CELL PARAMETERS",
        "CELL PARAMETERS",
    ]
    assert table["title"].tolist() == ["crystal a:", "b", "crystal alpha:"]
    assert table.iloc[1]["title_lineno"] == 2
    assert table.iloc[1]["lineno"] == 3
