from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from reaxkit.engine.common.generators.ffield_generator import merge_ffields
from reaxkit.engine.common.io.ffield_handler import FFieldHandler


_SRC_FFIELD = (
    Path(__file__).resolve().parents[3]
    / "full_sim_examples"
    / "heatfo_trainset_generation_babo_case"
    / "ffield"
)
_MERGE_SOURCE_FFIELD = (
    Path(__file__).resolve().parents[3]
    / "examples_to_test"
    / "params_interpret_test"
    / "ffield"
)


def test_merge_can_keep_selected_destination_atoms_and_remove_dependent_terms(tmp_path: Path):
    output = tmp_path / "ffield_merge_filtered"

    merge_ffields(
        source=_MERGE_SOURCE_FFIELD,
        destination=_SRC_FFIELD,
        output=output,
        atom_types=("Al",),
        keep_atoms_in_destination=("C", "H"),
    )

    sections = FFieldHandler(output).sections
    assert list(sections["atom"]["symbol"]) == ["C", "H", "Al"]
    assert list(sections["atom"].index) == [1, 2, 3]

    allowed_indices = {1, 2, 3}
    for section, atom_cols in {
        "bond": ("i", "j"),
        "off_diagonal": ("i", "j"),
        "angle": ("i", "j", "k"),
        "torsion": ("i", "j", "k", "l"),
        "hbond": ("i", "j", "k"),
    }.items():
        for col in atom_cols:
            referenced = {int(value) for value in sections[section][col] if int(value) != 0}
            assert referenced <= allowed_indices


def test_merge_keep_destination_atoms_rejects_unknown_symbol(tmp_path: Path):
    with pytest.raises(KeyError, match="Destination ffield missing atom type.*Unobtainium"):
        merge_ffields(
            source=_MERGE_SOURCE_FFIELD,
            destination=_SRC_FFIELD,
            output=tmp_path / "ffield_merge_filtered",
            atom_types=("Al",),
            keep_atoms_in_destination=("C", "Unobtainium"),
        )


def test_build_parser_merge_accepts_keep_atoms_in_dest():
    pytest.importorskip("seaborn")
    from reaxkit.workflows.file_tools import ffield_workflow

    parser = argparse.ArgumentParser()
    ffield_workflow.build_parser(parser, command="merge-ffield")
    args = parser.parse_args(
        [
            "--source",
            str(_MERGE_SOURCE_FFIELD),
            "--destination",
            str(_SRC_FFIELD),
            "--atom-types",
            "Al",
            "--keep-atoms-in-dest",
            "C,H",
        ]
    )

    assert args.keep_atoms_in_dest == "C,H"
