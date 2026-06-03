"""Unit tests for molecule/isomer detection from geometry connectivity."""

from __future__ import annotations

import pandas as pd

from reaxkit.analysis.molecular_analysis.isomer_detection import (
    MoleculeIsomerDetectionRequest,
    MoleculeIsomerDetectionTask,
)
from reaxkit.domain.data_models import GeometryData


def _sample_geometry() -> GeometryData:
    coordinates = pd.DataFrame(
        [
            {"atom_id": 1, "atom_type": "C", "x": 0.0, "y": 0.0, "z": 0.0},
            {"atom_id": 2, "atom_type": "C", "x": 1.0, "y": 0.0, "z": 0.0},
            {"atom_id": 3, "atom_type": "H", "x": 0.0, "y": 1.0, "z": 0.0},
            {"atom_id": 4, "atom_type": "H", "x": 1.0, "y": 1.0, "z": 0.0},
            {"atom_id": 5, "atom_type": "C", "x": 2.0, "y": 0.0, "z": 0.0},
            {"atom_id": 6, "atom_type": "C", "x": 3.0, "y": 0.0, "z": 0.0},
            {"atom_id": 7, "atom_type": "H", "x": 2.0, "y": 1.0, "z": 0.0},
            {"atom_id": 8, "atom_type": "H", "x": 3.0, "y": 1.0, "z": 0.0},
            {"atom_id": 9, "atom_type": "C", "x": 4.0, "y": 0.0, "z": 0.0},
            {"atom_id": 10, "atom_type": "C", "x": 5.0, "y": 0.0, "z": 0.0},
            {"atom_id": 11, "atom_type": "H", "x": 4.0, "y": 1.0, "z": 0.0},
            {"atom_id": 12, "atom_type": "H", "x": 4.0, "y": -1.0, "z": 0.0},
            {"atom_id": 13, "atom_type": "He", "x": 6.0, "y": 0.0, "z": 0.0},
        ]
    )
    connectivity = pd.DataFrame(
        [
            {"source_atom_id": 1, "target_atom_id": 2},
            {"source_atom_id": 2, "target_atom_id": 1},
            {"source_atom_id": 1, "target_atom_id": 3},
            {"source_atom_id": 2, "target_atom_id": 4},
            {"source_atom_id": 5, "target_atom_id": 6},
            {"source_atom_id": 5, "target_atom_id": 7},
            {"source_atom_id": 6, "target_atom_id": 8},
            {"source_atom_id": 9, "target_atom_id": 10},
            {"source_atom_id": 10, "target_atom_id": 9},
            {"source_atom_id": 9, "target_atom_id": 11},
            {"source_atom_id": 9, "target_atom_id": 12},
        ]
    )
    return GeometryData(
        coordinates=coordinates,
        connectivity=connectivity,
        atom_ids=coordinates["atom_id"].astype(int).tolist(),
        elements=coordinates["atom_type"].astype(str).tolist(),
        descriptor="test",
        remark="test",
    )


def _sample_geometry_with_sulfur_groups() -> GeometryData:
    coordinates = pd.DataFrame(
        [
            {"atom_id": 1, "atom_type": "S", "x": 0.0, "y": 0.0, "z": 0.0},
            {"atom_id": 2, "atom_type": "C", "x": 1.0, "y": 0.0, "z": 0.0},
            {"atom_id": 3, "atom_type": "S", "x": 2.0, "y": 0.0, "z": 0.0},
            {"atom_id": 4, "atom_type": "F", "x": 3.0, "y": 0.0, "z": 0.0},
            {"atom_id": 5, "atom_type": "F", "x": 3.5, "y": 0.0, "z": 0.0},
            {"atom_id": 6, "atom_type": "S", "x": 4.0, "y": 0.0, "z": 0.0},
            {"atom_id": 7, "atom_type": "Li", "x": 5.0, "y": 0.0, "z": 0.0},
            {"atom_id": 8, "atom_type": "F", "x": 6.0, "y": 0.0, "z": 0.0},
            {"atom_id": 9, "atom_type": "He", "x": 7.0, "y": 0.0, "z": 0.0},
            {"atom_id": 10, "atom_type": "Ne", "x": 8.0, "y": 0.0, "z": 0.0},
        ]
    )
    connectivity = pd.DataFrame(
        [
            {"source_atom_id": 1, "target_atom_id": 2},
            {"source_atom_id": 3, "target_atom_id": 4},
            {"source_atom_id": 3, "target_atom_id": 5},
            {"source_atom_id": 6, "target_atom_id": 7},
            {"source_atom_id": 6, "target_atom_id": 8},
        ]
    )
    return GeometryData(
        coordinates=coordinates,
        connectivity=connectivity,
        atom_ids=coordinates["atom_id"].astype(int).tolist(),
        elements=coordinates["atom_type"].astype(str).tolist(),
        descriptor="test_s",
        remark="test_s",
    )


def test_molecule_isomer_detection_groups_components_formulas_and_isomers():
    task = MoleculeIsomerDetectionTask()
    result = task.run(_sample_geometry(), MoleculeIsomerDetectionRequest(min_atoms_per_molecule=2))

    assert list(result.formula_table.columns) == ["formula", "molecule_count", "isomer_count"]
    assert list(result.isomer_table.columns) == [
        "formula",
        "isomer_id",
        "molecule_count",
        "atom_count",
        "representative_molecule_id",
    ]
    assert list(result.molecule_table.columns) == [
        "molecule_id",
        "formula",
        "isomer_id",
        "atom_count",
        "atom_ids",
    ]

    assert len(result.formula_table) == 1
    row = result.formula_table.iloc[0]
    assert row["formula"] == "C2H2"
    assert int(row["molecule_count"]) == 3
    assert int(row["isomer_count"]) == 2

    assert len(result.isomer_table) == 2
    counts_by_isomer = dict(
        zip(
            result.isomer_table["isomer_id"].astype(int).tolist(),
            result.isomer_table["molecule_count"].astype(int).tolist(),
        )
    )
    assert counts_by_isomer == {1: 2, 2: 1}

    assert len(result.molecule_table) == 3
    assert set(result.molecule_table["formula"].astype(str).tolist()) == {"C2H2"}
    assert set(result.molecule_table["isomer_id"].astype(int).tolist()) == {1, 2}


def test_molecule_isomer_detection_can_include_single_atom_components():
    task = MoleculeIsomerDetectionTask()
    result = task.run(_sample_geometry(), MoleculeIsomerDetectionRequest(min_atoms_per_molecule=1))

    formulas = result.formula_table.set_index("formula")
    assert int(formulas.loc["C2H2", "molecule_count"]) == 3
    assert int(formulas.loc["He1", "molecule_count"]) == 1
    assert int(formulas.loc["He1", "isomer_count"]) == 1


def test_molecule_isomer_detection_handles_empty_geometry():
    empty = GeometryData(
        coordinates=pd.DataFrame(columns=["atom_id", "atom_type", "x", "y", "z"]),
        connectivity=pd.DataFrame(columns=["source_atom_id", "target_atom_id"]),
    )
    task = MoleculeIsomerDetectionTask()
    result = task.run(empty, MoleculeIsomerDetectionRequest())

    assert result.formula_table.empty
    assert result.isomer_table.empty
    assert result.molecule_table.empty


def test_molecule_isomer_detection_generalized_element_and_motif_filters():
    task = MoleculeIsomerDetectionTask()
    result = task.run(
        _sample_geometry_with_sulfur_groups(),
        MoleculeIsomerDetectionRequest(
            min_atoms_per_molecule=1,
            include_any_elements=("S",),
            exclude_motifs=("SFx",),
        ),
    )

    formulas = set(result.formula_table["formula"].astype(str).tolist())
    assert "C1S1" in formulas
    assert "F1Li1S1" in formulas
    assert "F2S1" not in formulas
    assert "He1" not in formulas
    assert "Ne1" not in formulas


def test_molecule_isomer_detection_include_motif_only():
    task = MoleculeIsomerDetectionTask()
    result = task.run(
        _sample_geometry_with_sulfur_groups(),
        MoleculeIsomerDetectionRequest(
            min_atoms_per_molecule=1,
            include_motifs=("only:F,S;S==1;F>=1",),
        ),
    )

    formulas = set(result.formula_table["formula"].astype(str).tolist())
    assert formulas == {"F2S1"}
