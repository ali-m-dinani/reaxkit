"""Focused tests for force-field optimization CHARGE bar data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reaxkit.domain.data_models import (
    EnergyMinimizationSummaryData,
    ForceFieldOptimizationPlotBundleData,
    ForceFieldOptimizationReportData,
    ForceFieldOptimizationTrainingSetData,
)
from reaxkit.workflows.force_field_opt.charge import build_charge_table


def test_charge_table_inherits_identifier_and_matches_trainset_annotations() -> None:
    report = ForceFieldOptimizationReportData(
        linenos=np.array([2, 3]),
        sections=np.array(["CHARGE", "CHARGE"], dtype=object),
        titles=np.array(
            ["molecule Charge atom: 1", "Charge atom: 2"], dtype=object
        ),
        ffield_values=np.array([0.25, -0.25]),
        qm_values=np.array([0.3, -0.3]),
        weights=np.array([0.1, 0.1]),
        errors=np.zeros(2),
        total_ff_error=np.zeros(2),
    )
    charge = pd.DataFrame(
        {
            "iden": ["molecule", "molecule"],
            "atom": [1, 2],
            "weight": [0.1, 0.1],
            "lit": [0.3, -0.3],
            "group_comment": ["DFT charges", "DFT charges"],
            "inline_comment": ["cationic atom", "anionic atom"],
            "line_number": [10, 11],
        }
    )
    data = ForceFieldOptimizationPlotBundleData(
        report=report,
        geometry_summary=EnergyMinimizationSummaryData(
            identifiers=np.array([], dtype=object)
        ),
        training_set=ForceFieldOptimizationTrainingSetData(
            sections=("CHARGE",), charge=charge
        ),
    )

    table = build_charge_table(data)

    assert table["label"].tolist() == [
        "molecule [atom 1]",
        "molecule [atom 2]",
    ]
    assert table["ffield_value"].tolist() == [0.25, -0.25]
    assert table["qm_value"].tolist() == [0.3, -0.3]
    assert table["trainset_lit"].tolist() == [0.3, -0.3]
    assert table["group_comment"].tolist() == ["DFT charges", "DFT charges"]
