"""Tests for multi-structure BOND/ANGLE restraint extraction from geo files."""

from __future__ import annotations

from pathlib import Path

from reaxkit.engine.reaxff.io.geo_restraint_handler import GeoRestraintHandler


def test_geo_restraint_handler_preserves_structure_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "geo"
    path.write_text(
        """BIOGRF 200
DESCRP BAl_1.8
BOND RESTRAINT 1 2 1.8000 5000.0 10.0
END
BIOGRF 200
DESCRP AlNB_100
ANGLE RESTRAINT 2 3 4 100.00 2500.0 5.0
END
BIOGRF 200
DESCRP unrestrained
END
""",
        encoding="utf-8",
    )

    handler = GeoRestraintHandler(path)
    table = handler.dataframe()

    assert table[["descriptor", "restraint_type", "coordinate"]].to_dict(
        orient="records"
    ) == [
        {"descriptor": "BAl_1.8", "restraint_type": "bond", "coordinate": 1.8},
        {"descriptor": "AlNB_100", "restraint_type": "angle", "coordinate": 100.0},
    ]
    assert handler.metadata()["n_structures"] == 2
    assert handler.metadata()["restraint_types"] == ["angle", "bond"]


def test_geo_restraint_handler_extracts_crystx_cells_without_restraints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "geo"
    path.write_text(
        """XTLGRF 200
DESCRP c66_0_mp_2604
CRYSTX 7.08745 7.08745 7.08745 90.0 90.0 90.0
END
XTLGRF 200
DESCRP c66_c1_mp_2604
CRYSTX 7.08760 7.08760 7.08751 90.0 90.0 90.57295
END
""",
        encoding="utf-8",
    )

    handler = GeoRestraintHandler(path)
    cells = handler.cell_dataframe()

    assert cells[["descriptor", "a", "b", "c", "alpha", "beta", "gamma"]].to_dict(
        orient="records"
    ) == [
        {
            "descriptor": "c66_0_mp_2604",
            "a": 7.08745,
            "b": 7.08745,
            "c": 7.08745,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        {
            "descriptor": "c66_c1_mp_2604",
            "a": 7.08760,
            "b": 7.08760,
            "c": 7.08751,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.57295,
        },
    ]
    assert handler.metadata()["n_cell_records"] == 2
    assert handler.metadata()["n_structures"] == 2
