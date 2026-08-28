"""Focused tests for classified ENERGY plot collections."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from reaxkit.workflows.force_field_opt.energy_categories import (
    build_energy_category_tables,
    energy_bar_plot_payloads,
    energy_curve_plot_groups,
)


def _energy_row(
    line: int,
    comment_line: int,
    comment: str,
    weight: float,
    lit: float,
    identifiers: list[str],
    operators: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "section": "ENERGY",
        "line_number": line,
        "group_comment_line_number": comment_line,
        "group_comment": comment,
        "weight": weight,
        "lit": lit,
        "inline_comment": "",
    }
    for index, (operator, identifier) in enumerate(
        zip(operators, identifiers), start=1
    ):
        row[f"op{index}"] = operator
        row[f"id{index}"] = identifier
        row[f"n{index}"] = 1.0
    return row


def test_build_energy_category_tables_matches_report_expressions() -> None:
    energy = pd.DataFrame(
        [
            _energy_row(11, 10, "Scan", 1.0, 0.0, ["scan_0", "ref"], ["+", "-"]),
            _energy_row(12, 10, "Scan", 1.0, 2.0, ["scan_2", "ref"], ["+", "-"]),
            _energy_row(21, 20, "Difference", 2.0, 3.0, ["left", "right"], ["+", "-"]),
            _energy_row(
                31,
                30,
                "Reaction",
                3.0,
                -4.0,
                ["product", "reactant_a", "reactant_b"],
                ["+", "-", "-"],
            ),
            _energy_row(
                41,
                40,
                "Restraint N=B_bond_BN",
                25.0,
                -110.0,
                ["BN.opt"],
                ["+"],
            ),
        ]
    )
    training_set = SimpleNamespace(
        energy=energy,
        charge=pd.DataFrame(),
        heatfo=pd.DataFrame(),
        geometry=pd.DataFrame(),
        cell_parameters=pd.DataFrame(),
    )
    data = SimpleNamespace(
        training_set=training_set,
        report=SimpleNamespace(
            linenos=[101, 102, 103, 104, 105],
            sections=["ENERGY"] * 5,
            titles=[
                "Energy +scan_0 -ref",
                "Energy +scan_2 -ref",
                "Energy +left -right",
                "Energy +product -reactant_a -reactant_b",
                "Energy +BN.opt/1.00",
            ],
            ffield_values=[0.1, 2.1, 3.1, -3.9, -109.5],
            qm_values=[0.0, 2.0, 3.0, -4.0, -110.0],
            weights=[1.0, 1.0, 2.0, 3.0, 25.0],
        ),
    )

    tables = build_energy_category_tables(data)

    assert tables["energy_curve"]["report_line_number"].tolist() == [101, 102]
    assert tables["energy_curve"]["curve_coordinate"].tolist() == [0.0, 2.0]
    assert tables["energy_difference"]["report_line_number"].tolist() == [103]
    assert tables["reaction_energy"]["report_line_number"].tolist() == [104]
    assert tables["reaction_energy"]["plot_identifier"].tolist() == ["product"]
    assert tables["single_energy"]["report_line_number"].tolist() == [105]
    assert tables["single_energy"]["plot_identifier"].tolist() == ["BN.opt"]
    assert tables["single_energy"]["group_comment"].tolist() == [
        "Restraint N=B_bond_BN"
    ]


def test_energy_curve_and_bar_payloads_pair_reaxff_with_qm() -> None:
    curve_table = pd.DataFrame(
        {
            "block_id": [10, 10],
            "block_label": ["Scan", "Scan"],
            "curve_coordinate": [0.0, 2.0],
            "curve_coordinate_source": ["identifier suffix"] * 2,
            "ffield_value": [0.1, 2.1],
            "qm_value": [0.0, 2.0],
        }
    )
    bar_table = pd.DataFrame(
        {
            "plot_identifier": ["one", "two", "three"],
            "ffield_value": [1.0, 2.0, 3.0],
            "qm_value": [1.1, 2.1, 3.1],
        }
    )

    groups = energy_curve_plot_groups(curve_table)
    payloads = energy_bar_plot_payloads(
        bar_table,
        entries_per_figure=2,
        filename_prefix="differences",
        title="Differences",
        ylabel="Energy difference (kcal/mol)",
    )

    assert groups[0]["reaxff_y"] == [0.1, 2.1]
    assert groups[0]["qm_y"] == [0.0, 2.0]
    assert [len(payload["labels"]) for payload in payloads] == [2, 1]
    assert [series["color"] for series in payloads[0]["series"]] == [
        "tab:blue",
        "tab:orange",
    ]
    assert all(payload["minimum_category_slots"] == 2 for payload in payloads)
