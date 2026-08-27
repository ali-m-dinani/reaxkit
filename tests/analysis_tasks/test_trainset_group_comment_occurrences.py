"""Focused tests for occurrence-level trainset comment summaries."""

from __future__ import annotations

import pandas as pd

from reaxkit.analysis.force_field.trainset import (
    TrainsetGroupCommentsRequest,
    TrainsetGroupCommentsTask,
)
from reaxkit.domain.data_models import ForceFieldOptimizationTrainingSetData


def test_group_comments_keep_repetitions_empty_blocks_and_data_types() -> None:
    energy = pd.DataFrame(
        {
            "line_number": [11, 12, 21, 22, 31, 32],
            "group_comment": [
                "Volume C66_EOS",
                "Volume C66_EOS",
                "Volume C66_EOS",
                "Volume C66_EOS",
                "",
                "",
            ],
            "group_comment_line_number": [10, 10, 20, 20, 30, 30],
            "group_comment_occurrence": [1, 1, 2, 2, 3, 3],
            "id1": [
                "c66_c1_mp_1",
                "c66_0_mp_1",
                "c66_c1_mp_2",
                "c66_0_mp_2",
                "triethborane",
                "triethborane",
            ],
            "id2": [
                "c66_0_mp_1",
                "c66_0_mp_1",
                "c66_0_mp_2",
                "c66_0_mp_2",
                "NH3",
                "NH3",
            ],
            "id3": [pd.NA, pd.NA, pd.NA, pd.NA, "BNH2_3", "dietamineborane"],
            "id4": [pd.NA, pd.NA, pd.NA, pd.NA, "ethane", "ethane"],
        }
    )
    data = ForceFieldOptimizationTrainingSetData(
        sections=("ENERGY",), energy=energy
    )

    result = TrainsetGroupCommentsTask().run(
        data, TrainsetGroupCommentsRequest(section="all")
    )
    table = result.table

    assert table["group_comment"].tolist() == [
        "Volume C66_EOS",
        "Volume C66_EOS",
        "",
    ]
    assert table["comment_line_number"].tolist() == [10, 20, 30]
    assert table["count"].tolist() == [2, 2, 2]
    assert table["type of data"].tolist() == ["eos", "eos", "reaction_energy"]
    assert "mp_1" in table.iloc[0]["identifiers"]
    assert "mp_2" in table.iloc[1]["identifiers"]


def test_group_comment_types_are_section_and_entry_aware() -> None:
    charge = pd.DataFrame(
        {
            "line_number": [2],
            "group_comment": ["DFT populations"],
            "iden": ["molecule"],
        }
    )
    geometry = pd.DataFrame(
        {
            "line_number": [5, 6],
            "group_comment": ["Optimized geometry", "Optimized geometry"],
            "iden": ["molecule", "molecule"],
            "at1": [1, 1],
            "at2": [2, 2],
            "at3": [pd.NA, 3],
        }
    )
    data = ForceFieldOptimizationTrainingSetData(
        sections=("CHARGE", "GEOMETRY"), charge=charge, geometry=geometry
    )

    table = TrainsetGroupCommentsTask().run(
        data, TrainsetGroupCommentsRequest(section="all")
    ).table

    assert table["type of data"].tolist() == ["charge", "bond, valence_angle"]
