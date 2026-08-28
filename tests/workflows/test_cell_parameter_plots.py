"""Focused tests for force-field optimization cell-parameter bar data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reaxkit.domain.data_models import (
    EnergyMinimizationSummaryData,
    ForceFieldOptimizationPlotBundleData,
    ForceFieldOptimizationReportData,
    ForceFieldOptimizationTrainingSetData,
)
from reaxkit.workflows.force_field_opt.cell_parameters import (
    build_cell_parameter_table,
    cell_parameter_plot_payloads,
)


def test_cell_table_inherits_identifier_and_matches_trainset_annotations() -> None:
    report = ForceFieldOptimizationReportData(
        linenos=np.array([2, 3, 4]),
        sections=np.array(["CELL PARAMETERS"] * 3, dtype=object),
        titles=np.array(["crystal a:", "b:", "c:"], dtype=object),
        ffield_values=np.array([5.10, 5.20, 7.30]),
        qm_values=np.array([5.00, 5.25, 7.35]),
        weights=np.array([1.0, 1.0, 1.0]),
        errors=np.zeros(3),
        total_ff_error=np.zeros(3),
    )
    cell_parameters = pd.DataFrame(
        {
            "iden": ["crystal"] * 3,
            "type": ["a", "b", "c"],
            "weight": [1.0] * 3,
            "lit": [5.00, 5.25, 7.35],
            "group_comment": ["DFT lattice constants"] * 3,
            "inline_comment": ["axis a", "axis b", "axis c"],
            "line_number": [10, 11, 12],
        }
    )
    data = ForceFieldOptimizationPlotBundleData(
        report=report,
        geometry_summary=EnergyMinimizationSummaryData(
            identifiers=np.array([], dtype=object)
        ),
        training_set=ForceFieldOptimizationTrainingSetData(
            sections=("CELL_PARAMETERS",), cell_parameters=cell_parameters
        ),
    )

    table = build_cell_parameter_table(data)

    assert table["label"].tolist() == [
        "crystal [a]",
        "crystal [b]",
        "crystal [c]",
    ]
    assert table["ffield_value"].tolist() == [5.10, 5.20, 7.30]
    assert table["qm_value"].tolist() == [5.00, 5.25, 7.35]
    assert table["trainset_lit"].tolist() == [5.00, 5.25, 7.35]
    assert table["group_comment"].tolist() == ["DFT lattice constants"] * 3
    assert table["inline_comment"].tolist() == ["axis a", "axis b", "axis c"]


def test_cell_payloads_limit_entries_and_keep_series_colors() -> None:
    table = pd.DataFrame(
        {
            "label": [f"crystal_{index} [a]" for index in range(31)],
            "ffield_value": [float(index) for index in range(31)],
            "qm_value": [float(index) + 0.1 for index in range(31)],
        }
    )

    payloads = cell_parameter_plot_payloads(table, entries_per_figure=5)

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
        == ["tab:blue", "tab:orange"]
        for payload in payloads
    )
    assert all(payload["grid"] is False for payload in payloads)
