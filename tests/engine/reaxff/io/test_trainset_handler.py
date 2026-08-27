from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reaxkit.core.platform.exceptions import ParseError
from reaxkit.engine.reaxff.io.trainset_handler import TrainsetHandler


def _write_trainset(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "trainset.in"
    path.write_text(text, encoding="utf-8")
    return path


def test_trainset_records_all_section_occurrences_and_source_lines(tmp_path: Path) -> None:
    path = _write_trainset(
        tmp_path,
        """CHARGE
# charge group
charge_a  0.10  1  -0.15  ! inline charge
ENDCHARGE
HEATFO
heat_a  1.0  -19.82
ENDHEATFO
GEOMETRY
geom_rmsg  1.0  0.01
geom_bond  1.0  1 2 1.54
geom_angle  1.0  1 2 3 111.0
geom_torsion  1.0  1 2 3 4 56.0
ENDGEOMETRY
CELL PARAMETERS
cell_a  1.0  a  11.20
ENDCELLPARAMETERS
ENERGY
# first energy block
50.0 + h2o_min/1 - oh1/1 -219.20
ENDENERGY
ENERGY
25.0 + BaZrO3_optx7/1 - BaZrO3_Ba_unr/1 - BaZrO3_Zr_unr/1 -367.1
ENDENERGY
""",
    )

    metadata = TrainsetHandler(path, strict=True).metadata()

    assert metadata["sections"] == [
        "CHARGE",
        "HEATFO",
        "GEOMETRY",
        "CELL_PARAMETERS",
        "ENERGY",
    ]
    assert metadata["section_occurrences"] == [
        {"section": "CHARGE", "occurrence": 1, "section_order": 1, "start_line_number": 1, "end_line_number": 4, "entry_count": 1},
        {"section": "HEATFO", "occurrence": 1, "section_order": 2, "start_line_number": 5, "end_line_number": 7, "entry_count": 1},
        {"section": "GEOMETRY", "occurrence": 1, "section_order": 3, "start_line_number": 8, "end_line_number": 13, "entry_count": 4},
        {"section": "CELL_PARAMETERS", "occurrence": 1, "section_order": 4, "start_line_number": 14, "end_line_number": 16, "entry_count": 1},
        {"section": "ENERGY", "occurrence": 1, "section_order": 5, "start_line_number": 17, "end_line_number": 20, "entry_count": 1},
        {"section": "ENERGY", "occurrence": 2, "section_order": 6, "start_line_number": 21, "end_line_number": 23, "entry_count": 1},
    ]

    energy = metadata["tables"]["ENERGY"]
    assert energy["line_number"].tolist() == [19, 22]
    assert energy.loc[1, ["op1", "id1", "n1", "op2", "id2", "n2", "op3", "id3", "n3"]].tolist() == [
        "+", "BaZrO3_optx7", 1.0, "-", "BaZrO3_Ba_unr", 1.0, "-", "BaZrO3_Zr_unr", 1.0,
    ]


def test_trainset_heatfo_allows_missing_literature_value(tmp_path: Path) -> None:
    path = _write_trainset(
        tmp_path,
        """HEATFO
#Iden Weight Lit
methane 2.00 -17.80 !Heat of formation
chexane 2.00
ENDHEATFO
""",
    )

    table = TrainsetHandler(path, strict=True).heatfo()

    assert table["iden"].tolist() == ["methane", "chexane"]
    assert table.iloc[0]["lit"] == -17.8
    assert pd.isna(table.iloc[1]["lit"])


def test_trainset_preserves_repeated_and_empty_comment_block_locations(
    tmp_path: Path,
) -> None:
    path = _write_trainset(
        tmp_path,
        """ENERGY
# Volume C66_EOS
1.0 + c66_c1_mp_1/1 - c66_0_mp_1/1 0.1
# Volume C66_EOS
1.0 + c66_c1_mp_2/1 - c66_0_mp_2/1 0.2
#
1.0 + product/1 - reactant/1 3.0
ENDENERGY
""",
    )

    table = TrainsetHandler(path, strict=True).energy_terms()

    assert table["group_comment"].tolist() == [
        "Volume C66_EOS",
        "Volume C66_EOS",
        "",
    ]
    assert table["group_comment_line_number"].tolist() == [2, 4, 6]
    assert table["group_comment_occurrence"].tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("50.0 + h2o_min/1 - -219.20", "missing its identifier"),
        ("50.0 * h2o_min/1 - oh1/1 -219.20", "invalid operator '*'"),
        ("50.0 + h2o_min/0 - oh1/1 -219.20", "invalid divisor '/0'"),
    ],
)
def test_strict_trainset_rejects_malformed_energy_with_source_context(
    tmp_path: Path,
    entry: str,
    expected: str,
) -> None:
    path = _write_trainset(tmp_path, f"ENERGY\n{entry}\nENDENERGY\n")

    with pytest.raises(ParseError, match=expected) as exc_info:
        TrainsetHandler(path, strict=True).metadata()

    assert str(path) in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_default_mode_remains_lenient_for_malformed_energy(tmp_path: Path) -> None:
    path = _write_trainset(tmp_path, "ENERGY\n50.0 + h2o_min/1 - -219.20\nENDENERGY\n")

    assert TrainsetHandler(path).metadata()["tables"]["ENERGY"].empty


def test_strict_trainset_accepts_optional_divisors_and_many_energy_operands(
    tmp_path: Path,
) -> None:
    path = _write_trainset(
        tmp_path,
        """ENERGY
12.5 + single -1.2
7.0 + a/1 - b/10.6667 + c/3 - d/4 9.9
ENDENERGY
""",
    )

    energy = TrainsetHandler(path, strict=True).energy_terms()

    assert energy.loc[0, ["op1", "id1", "n1"]].tolist() == ["+", "single", 1.0]
    assert energy.loc[1, ["op1", "id1", "n1", "op2", "id2", "n2", "op3", "id3", "n3", "op4", "id4", "n4"]].tolist() == [
        "+", "a", 1.0, "-", "b", 10.6667, "+", "c", 3.0, "-", "d", 4.0,
    ]


def test_trainset_preserves_an_empty_section_occurrence(tmp_path: Path) -> None:
    path = _write_trainset(tmp_path, "CHARGE\nENDCHARGE\nENERGY\n1.0 + a/1 0.0\nENDENERGY\n")

    metadata = TrainsetHandler(path, strict=True).metadata()

    assert metadata["section_occurrences"][0]["section"] == "CHARGE"
    assert metadata["section_occurrences"][0]["entry_count"] == 0
    assert metadata["tables"]["CHARGE"].empty


def test_trainset_accepts_plural_charges_section_header(tmp_path: Path) -> None:
    path = _write_trainset(
        tmp_path,
        """CHARGES
charge_a  0.10  1  -0.15
charge_b  0.20  2   0.15
ENDCHARGES
""",
    )

    metadata = TrainsetHandler(path, strict=True).metadata()
    charges = metadata["tables"]["CHARGE"]

    assert metadata["sections"] == ["CHARGE"]
    assert charges["line_number"].tolist() == [2, 3]
    assert charges[["iden", "weight", "atom", "lit"]].to_dict(orient="records") == [
        {"iden": "charge_a", "weight": 0.1, "atom": 1, "lit": -0.15},
        {"iden": "charge_b", "weight": 0.2, "atom": 2, "lit": 0.15},
    ]
    assert metadata["section_occurrences"] == [
        {
            "section": "CHARGE",
            "occurrence": 1,
            "section_order": 1,
            "start_line_number": 1,
            "end_line_number": 4,
            "entry_count": 2,
        }
    ]
