from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.engine.reaxff.generators.ffielld_generator import add_element_to_ffield
from reaxkit.engine.reaxff.io.ffield_handler import FFieldHandler
from reaxkit.workflows.file_tools import ffield_workflow


_SRC_FFIELD = (
    Path(__file__).resolve().parents[1]
    / "full_sim_examples"
    / "heatfo_trainset_generation_babo_case"
    / "ffield"
)


def test_add_element_to_ffield_column_strategy_projects_b_like_terms(tmp_path: Path):
    out_path = tmp_path / "ffield_added"

    summary = add_element_to_ffield(
        destination=_SRC_FFIELD,
        output=out_path,
        element="Al",
        fields=("atom", "bond"),
        similarity_mode="group",
    )

    assert summary.template_atom == "B"
    assert summary.appended["atom"] == 1
    assert summary.appended["bond"] > 0

    sections = FFieldHandler(out_path).sections
    atom_df = sections["atom"]
    sym_to_idx = {str(row["symbol"]): int(idx) for idx, row in atom_df.iterrows()}
    assert "Al" in sym_to_idx

    b_idx = sym_to_idx["B"]
    al_idx = sym_to_idx["Al"]
    o_idx = sym_to_idx["O"]
    bond_df = sections["bond"]

    al_o = (((bond_df["i"] == al_idx) & (bond_df["j"] == o_idx)) | ((bond_df["i"] == o_idx) & (bond_df["j"] == al_idx))).sum()
    al_al = ((bond_df["i"] == al_idx) & (bond_df["j"] == al_idx)).sum()
    b_al = (((bond_df["i"] == b_idx) & (bond_df["j"] == al_idx)) | ((bond_df["i"] == al_idx) & (bond_df["j"] == b_idx))).sum()

    assert int(al_o) >= 1
    assert int(al_al) >= 1
    assert int(b_al) >= 1


def test_build_parser_add_element_command_accepts_new_flags():
    parser = argparse.ArgumentParser()
    ffield_workflow.build_parser(parser, command="add-element-to-ffield")
    args = parser.parse_args(
        [
            "--destination",
            str(_SRC_FFIELD),
            "--element",
            "Al",
            "--fields",
            "atom,bond",
            "--similarity",
            "group",
            "--closest-atom",
            "B",
            "--radius-metrics",
            "all",
        ]
    )
    assert args.destination == str(_SRC_FFIELD)
    assert args.element == "Al"
    assert args.similarity == "group"
    assert args.closest_atom == "B"
    assert args.radius_metrics == "all"
