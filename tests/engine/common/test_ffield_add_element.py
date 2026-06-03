from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.engine.reaxff.generators.ffielld_generator import add_element_to_ffield, merge_ffields
from reaxkit.engine.common.io.ffield_handler import FFieldHandler
from reaxkit.workflows.file_tools import ffield_workflow


_SRC_FFIELD = (
    Path(__file__).resolve().parents[1]
    / "full_sim_examples"
    / "heatfo_trainset_generation_babo_case"
    / "ffield"
)
_MERGE_SOURCE_FFIELD = (
    Path(__file__).resolve().parents[1]
    / "examples_to_test"
    / "params_interpret_test"
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


def test_merge_fill_missing_with_template_adds_missing_destination_terms(tmp_path: Path):
    out_plain = tmp_path / "ffield_merge_plain"
    out_templ = tmp_path / "ffield_merge_templ"

    plain = merge_ffields(
        source=_MERGE_SOURCE_FFIELD,
        destination=_SRC_FFIELD,
        output=out_plain,
        atom_types=("Al",),
        fields=("atom", "bond"),
        fill_missing_with_template=False,
    )
    templ = merge_ffields(
        source=_MERGE_SOURCE_FFIELD,
        destination=_SRC_FFIELD,
        output=out_templ,
        atom_types=("Al",),
        fields=("atom", "bond"),
        fill_missing_with_template=True,
        template_closest_atom="B",
        template_similarity_mode="group",
    )

    assert plain.template_generated["bond"] == 0
    assert templ.template_generated["bond"] >= 1
    assert templ.template_choices.get("Al") == "B"

    plain_sections = FFieldHandler(out_plain).sections
    templ_sections = FFieldHandler(out_templ).sections
    plain_syms = {str(r["symbol"]): int(i) for i, r in plain_sections["atom"].iterrows()}
    templ_syms = {str(r["symbol"]): int(i) for i, r in templ_sections["atom"].iterrows()}
    ba_plain, al_plain = plain_syms["Ba"], plain_syms["Al"]
    ba_templ, al_templ = templ_syms["Ba"], templ_syms["Al"]

    b_plain = plain_sections["bond"]
    b_templ = templ_sections["bond"]
    ba_al_plain = (((b_plain["i"] == ba_plain) & (b_plain["j"] == al_plain)) | ((b_plain["i"] == al_plain) & (b_plain["j"] == ba_plain))).sum()
    ba_al_templ = (((b_templ["i"] == ba_templ) & (b_templ["j"] == al_templ)) | ((b_templ["i"] == al_templ) & (b_templ["j"] == ba_templ))).sum()

    assert int(ba_al_plain) == 0
    assert int(ba_al_templ) >= 1
