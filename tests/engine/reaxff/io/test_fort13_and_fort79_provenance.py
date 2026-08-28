from __future__ import annotations

import math

from reaxkit.engine.reaxff.io.fort13_handler import Fort13Handler
from reaxkit.engine.reaxff.io.fort79_handler import Fort79Handler


def test_fort13_retains_record_order_and_physical_source_line(tmp_path) -> None:
    source = tmp_path / "fort.13"
    source.write_text("10.0 ! baseline\n\ninvalid\n8.0D+00\n", encoding="utf-8")

    table = Fort13Handler(source).dataframe()

    assert table.to_dict("records") == [
        {"row_order": 1, "source_line_number": 1, "epoch": 1, "total_ff_error": 10.0},
        {"row_order": 2, "source_line_number": 4, "epoch": 4, "total_ff_error": 8.0},
    ]


def test_fort79_retains_raw_and_normalized_identifier_and_record_line(tmp_path) -> None:
    source = tmp_path / "fort.79"
    source.write_text(
        "\nValues used for parameter 5102  2\n"
        " 0.100000 0.200000 0.300000\nDifferences found\n"
        " 0.2408814586-316 0.500000 0.600000\n"
        "Parabol: a= 1.000000 b= 2.000000 c= 3.000000\n"
        "Minimum of the parabol 0.250000\n"
        "Difference belonging to minimum of parabol 0.050000\n"
        "New parameter value 0.225000\n"
        "Difference belonging to new parameter value 0.025000\n",
        encoding="utf-8",
    )

    row = Fort79Handler(source).dataframe().iloc[0]

    assert row["row_order"] == 1
    assert row["source_line_number"] == 2
    assert row["raw_identifier"] == "5102  2"
    assert row["identifier"] == "5 102 2"
    assert math.isnan(row["diff1"])
