"""Focused tests for direct HEATFO optimization targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reaxkit.domain.data_models import (
    EnergyMinimizationSummaryData,
    ForceFieldOptimizationPlotBundleData,
    ForceFieldOptimizationReportData,
    ForceFieldOptimizationTrainingSetData,
)
from reaxkit.workflows.force_field_opt.heatfo import build_heatfo_table


def test_direct_heatfo_rows_inherit_identifiers_and_allow_missing_trainset_lit() -> None:
    report = ForceFieldOptimizationReportData(
        linenos=np.array([2, 3, 4]),
        sections=np.array(["HEATFO", "CHARGE", "HEATFO"], dtype=object),
        titles=np.array(
            [
                "methane Heat of formation:",
                "chexane Charge atom: 1",
                "Heat of formation:",
            ],
            dtype=object,
        ),
        ffield_values=np.array([-17.8, -0.1604, -29.49]),
        qm_values=np.array([-17.8, -0.15, -29.49]),
        weights=np.array([2.0, 0.1, 2.0]),
        errors=np.zeros(3),
        total_ff_error=np.zeros(3),
    )
    heatfo = pd.DataFrame(
        {
            "iden": ["methane", "chexane"],
            "weight": [2.0, 2.0],
            "lit": [-17.8, pd.NA],
            "group_comment": ["Direct heat targets", "Direct heat targets"],
            "inline_comment": ["Heat of formation", ""],
            "line_number": [10, 11],
        }
    )
    data = ForceFieldOptimizationPlotBundleData(
        report=report,
        geometry_summary=EnergyMinimizationSummaryData(
            identifiers=np.array([], dtype=object)
        ),
        training_set=ForceFieldOptimizationTrainingSetData(
            sections=("HEATFO",), heatfo=heatfo
        ),
    )

    table = build_heatfo_table(data)

    assert table["plot_identifier"].tolist() == ["methane", "chexane"]
    assert table["ffield_value"].tolist() == [-17.8, -29.49]
    assert table["qm_value"].tolist() == [-17.8, -29.49]
    assert table.iloc[0]["trainset_lit"] == -17.8
    assert pd.isna(table.iloc[1]["trainset_lit"])
    assert table["report_line_number"].tolist() == [2, 4]
