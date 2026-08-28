"""Focused tests for fort.99-to-trainset annotations."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from reaxkit.workflows.force_field_opt.report_linkage import (
    add_trainset_links,
    build_report_trainset_links,
)


def test_energy_report_rows_link_to_occurrence_aware_trainset_entries() -> None:
    energy = pd.DataFrame(
        {
            "line_number": [10, 20],
            "weight": [1.0, 2.0],
            "op1": ["+", "+"],
            "id1": ["product", "product"],
            "n1": [1.0, 1.0],
            "op2": ["-", "-"],
            "id2": ["reference", "reference"],
            "n2": [1.0, 1.0],
            "lit": [3.0, 4.0],
            "group_comment": ["first group", "second group"],
            "inline_comment": ["first note", "second note"],
        }
    )
    training_set = SimpleNamespace(
        energy=energy,
        charge=pd.DataFrame(),
        heatfo=pd.DataFrame(),
        geometry=pd.DataFrame(),
        cell_parameters=pd.DataFrame(),
    )
    report = SimpleNamespace(
        linenos=[101, 102],
        sections=["ENERGY", "ENERGY"],
        titles=["Energy +product -reference", "Energy +product -reference"],
        ffield_values=[3.1, 4.1],
        qm_values=[3.0, 4.0],
        weights=[1.0, 2.0],
    )

    links = build_report_trainset_links(
        SimpleNamespace(report=report, training_set=training_set)
    )
    annotated = add_trainset_links(
        pd.DataFrame({"lineno": [101, 102], "title": report.titles}),
        links,
        report_line_column="lineno",
    )

    assert annotated["trainset_line_number"].tolist() == [10, 20]
    assert annotated["group_comment"].tolist() == ["first group", "second group"]
    assert annotated["inline_comment"].tolist() == ["first note", "second note"]


def test_add_trainset_links_guarantees_columns_for_unmatched_and_empty_tables() -> None:
    links = pd.DataFrame(
        {
            "report_line_number": [5],
            "trainset_line_number": [50],
            "group_comment": ["group"],
            "inline_comment": ["note"],
        }
    )

    unmatched = add_trainset_links(
        pd.DataFrame({"report_line_number": [6], "value": [1.0]}), links
    )
    empty = add_trainset_links(
        pd.DataFrame(columns=["report_line_number", "value"]), links
    )

    assert pd.isna(unmatched.iloc[0]["trainset_line_number"])
    assert unmatched.iloc[0]["group_comment"] == ""
    assert unmatched.iloc[0]["inline_comment"] == ""
    assert {
        "trainset_line_number",
        "group_comment",
        "inline_comment",
    }.issubset(empty.columns)


def test_truncated_report_expression_links_only_to_unique_prefix_match() -> None:
    energy = pd.DataFrame(
        {
            "line_number": [40],
            "weight": [2.0],
            "op1": ["+"],
            "id1": ["triethborane"],
            "n1": [1.0],
            "op2": ["+"],
            "id2": ["NH3"],
            "n2": [1.0],
            "op3": ["-"],
            "id3": ["dietamineborane"],
            "n3": [1.0],
            "op4": ["-"],
            "id4": ["ethane"],
            "n4": [1.0],
            "lit": [10.0],
            "group_comment": ["reaction group"],
            "inline_comment": ["reaction note"],
        }
    )
    data = SimpleNamespace(
        training_set=SimpleNamespace(
            energy=energy,
            charge=pd.DataFrame(),
            heatfo=pd.DataFrame(),
            geometry=pd.DataFrame(),
            cell_parameters=pd.DataFrame(),
        ),
        report=SimpleNamespace(
            linenos=[400],
            sections=["ENERGY"],
            titles=["Energy +triethborane +NH3 -dietamineborane -e"],
            ffield_values=[11.0],
            qm_values=[10.0],
            weights=[2.0],
        ),
    )

    links = build_report_trainset_links(data)

    assert links["trainset_line_number"].tolist() == [40]
    assert links["group_comment"].tolist() == ["reaction group"]
    assert links["inline_comment"].tolist() == ["reaction note"]
