"""Focused tests for force-field optimization GEOMETRY bar data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reaxkit.domain.data_models import (
    EnergyMinimizationSummaryData,
    ForceFieldOptimizationPlotBundleData,
    ForceFieldOptimizationReportData,
    ForceFieldOptimizationTrainingSetData,
)
from reaxkit.workflows.force_field_opt.geometry_targets import (
    build_geometry_target_table,
    geometry_target_plot_payloads,
)


def test_geometry_table_inherits_identifier_and_matches_atom_tuples() -> None:
    report = ForceFieldOptimizationReportData(
        linenos=np.array([2, 3, 4, 5, 6]),
        sections=np.array(
            ["CHARGE", "GEOMETRY", "GEOMETRY", "GEOMETRY", "GEOMETRY"],
            dtype=object,
        ),
        titles=np.array(
            [
                "chexane Charge atom: 1",
                "Bond distance: 1 2",
                "Valence angle: 1 2 3",
                "Torsion angle: 1 2 3 4",
                "other Bond distance: 2 1",
            ],
            dtype=object,
        ),
        ffield_values=np.array([0.0, 1.50, 110.0, 55.0, 1.40]),
        qm_values=np.array([0.0, 1.54, 111.0, 56.0, 1.42]),
        weights=np.array([0.1, 0.01, 1.0, 1.0, 0.02]),
        errors=np.zeros(5),
        total_ff_error=np.zeros(5),
    )
    geometry = pd.DataFrame(
        {
            "iden": ["chexane", "chexane", "chexane", "chexane", "other"],
            "weight": [0.01, 1.0, 1.0, 1.0, 0.02],
            "at1": [1, 1, 1, pd.NA, 1],
            "at2": [2, 2, 2, pd.NA, 2],
            "at3": [pd.NA, 3, 3, pd.NA, pd.NA],
            "at4": [pd.NA, pd.NA, 4, pd.NA, pd.NA],
            "lit": [1.54, 111.0, 56.0, 0.01, 1.42],
            "group_comment": ["DFT geometry"] * 5,
            "inline_comment": [
                "bond",
                "valence angle",
                "torsion angle",
                "RMSG",
                "reversed bond",
            ],
            "line_number": [10, 11, 12, 13, 14],
        }
    )
    data = ForceFieldOptimizationPlotBundleData(
        report=report,
        geometry_summary=EnergyMinimizationSummaryData(
            identifiers=np.array([], dtype=object)
        ),
        training_set=ForceFieldOptimizationTrainingSetData(
            sections=("GEOMETRY",), geometry=geometry
        ),
    )

    table = build_geometry_target_table(data)

    assert table["label"].tolist() == [
        "chexane [atoms 1 2]",
        "chexane [atoms 1 2 3]",
        "chexane [atoms 1 2 3 4]",
        "other [atoms 1 2]",
    ]
    assert table["geometry_type"].tolist() == [
        "bond",
        "valence_angle",
        "torsion_angle",
        "bond",
    ]
    assert table["trainset_lit"].tolist() == [1.54, 111.0, 56.0, 1.42]
    assert table["inline_comment"].tolist()[-1] == "reversed bond"


def test_geometry_payloads_limit_entries_and_keep_series_colors() -> None:
    table = pd.DataFrame(
        {
            "label": [f"structure_{index} [atoms 1 2]" for index in range(31)],
            "ffield_value": [float(index) for index in range(31)],
            "qm_value": [float(index) + 0.1 for index in range(31)],
        }
    )

    payloads = geometry_target_plot_payloads(table, entries_per_figure=5)

    assert len(payloads) == 7
    assert [len(payload["labels"]) for payload in payloads] == [
        5,
        5,
        5,
        5,
        5,
        5,
        1,
    ]
    assert all(
        [series["color"] for series in payload["series"]]
        == ["tab:blue", "#C0504D"]
        for payload in payloads
    )
    assert all(payload["grid"] is False for payload in payloads)
    assert all(payload["minimum_category_slots"] == 5 for payload in payloads)


def test_geometry_payloads_separate_lengths_from_angles() -> None:
    table = pd.DataFrame(
        {
            "label": [
                "chexane [atoms 1 2]",
                "chexane [atoms 1 2 3]",
                "chexane [atoms 1 2 3 4]",
            ],
            "geometry_type": ["bond", "valence_angle", "torsion_angle"],
            "ffield_value": [1.5, 110.0, 55.0],
            "qm_value": [1.54, 111.0, 56.0],
        }
    )

    payloads = geometry_target_plot_payloads(table, entries_per_figure=6)

    assert len(payloads) == 3
    assert [payload["ylabel"] for payload in payloads] == [
        "Bond length (angstrom)",
        "Angle (degrees)",
        "Angle (degrees)",
    ]
