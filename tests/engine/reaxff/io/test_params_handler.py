from __future__ import annotations

import math

import pytest

from reaxkit.engine.reaxff.io.params_handler import ParamsHandler


def test_params_handler_accepts_optional_bounds(tmp_path) -> None:
    source = tmp_path / "params"
    source.write_text(
        "2 1 1 0.01\n"
        "3 2 4 0.02 -1.5\n"
        "4 3 2 0.03 0.1 2.5 ! bounded\n",
        encoding="utf-8",
    )

    table = ParamsHandler(source).dataframe()

    assert len(table) == 3
    assert math.isnan(table.iloc[0]["min_value"])
    assert math.isnan(table.iloc[0]["max_value"])
    assert table.iloc[1]["min_value"] == pytest.approx(-1.5)
    assert math.isnan(table.iloc[1]["max_value"])
    assert table.iloc[2]["min_value"] == pytest.approx(0.1)
    assert table.iloc[2]["max_value"] == pytest.approx(2.5)
    assert table.iloc[2]["inline_comment"] == "bounded"


def test_params_handler_rejects_invalid_token_count(tmp_path) -> None:
    source = tmp_path / "params"
    source.write_text("3 1 1\n", encoding="utf-8")
    with pytest.raises(Exception, match="Expected 4 to 6 tokens"):
        ParamsHandler(source).dataframe()
