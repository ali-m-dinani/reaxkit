"""Sanity check for ForceFieldOptimizationReportEOSTask via AnalysisExecutor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.force_field.report import (
    FFieldOptimizationReportEOSRequest,
    FFieldOptimizationReportEOSTask,
    _base_other_energy_volume_table,
    _classify_curve_rows,
    _curve_table_from_classified_rows,
    _elastic_strain_metadata,
    _force_field_optimization_curve_tables,
)
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.platform.exceptions import AnalysisError
from reaxkit.domain.data_models import (
    EnergyMinimizationSummaryData,
    ForceFieldOptimizationPlotBundleData,
    ForceFieldOptimizationReportData,
    ForceFieldOptimizationTrainingSetData,
)

RUN_DIR = Path(
    r"C:\Users\alimo\PycharmProjects\pythonProject\reaxkit\examples_to_test"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@pytest.mark.parametrize(
    ("family", "distorted_values", "strain_type"),
    [
        ("c12", {"a": 10.1, "b": 19.8}, "orthorhombic"),
        ("c13", {"a": 10.1, "c": 29.7}, "orthorhombic"),
        ("c23", {"b": 20.2, "c": 29.7}, "orthorhombic"),
        ("c44", {"alpha": 90.572957795}, "shear_angle"),
        ("c55", {"beta": 90.572957795}, "shear_angle"),
        ("c66", {"gamma": 90.572957795}, "shear_angle"),
    ],
)
def test_elastic_strain_uses_family_specific_crystx_components(
    family: str,
    distorted_values: dict[str, float],
    strain_type: str,
) -> None:
    base = {
        "a": 10.0,
        "b": 20.0,
        "c": 30.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }
    distorted = {**base, **distorted_values}
    cells = pd.DataFrame(
        [
            {"descriptor": f"{family}_0", **base},
            {"descriptor": f"{family}_e0001", **distorted},
        ]
    )

    strain = _elastic_strain_metadata(f"{family}_0", f"{family}_e1", cells)

    assert strain["strain_percent"] == pytest.approx(1.0, abs=1.0e-9)
    assert strain["strain_type"] == strain_type


def test_c66_rounded_angle_example_converts_to_one_percent() -> None:
    cells = pd.DataFrame(
        {
            "descriptor": ["c66_0_mp_2604", "c66_c1_mp_2604"],
            "a": [7.08745, 7.08760],
            "b": [7.08745, 7.08760],
            "c": [7.08745, 7.08751],
            "alpha": [90.0, 90.0],
            "beta": [90.0, 90.0],
            "gamma": [90.0, 90.57295],
        }
    )

    strain = _elastic_strain_metadata("c66_0_mp_2604", "c66_c1_mp_2604", cells)

    assert strain["strain_percent"] == pytest.approx(1.0, abs=2.0e-5)


def test_eos_task_propagates_crystx_orthorhombic_strain() -> None:
    report = ForceFieldOptimizationReportData(
        linenos=np.array([1, 2]),
        sections=np.array(["ENERGY", "ENERGY"], dtype=object),
        titles=np.array(
            [
                "Energy +c12_c1_mp_2604/1 -c12_0_mp_2604/1",
                "Energy +c12_e1_mp_2604/1 -c12_0_mp_2604/1",
            ],
            dtype=object,
        ),
        ffield_values=np.array([1.1, 1.2]),
        qm_values=np.array([1.0, 1.0]),
        weights=np.ones(2),
        errors=np.zeros(2),
        total_ff_error=np.zeros(2),
    )
    identifiers = ["c12_0_mp_2604", "c12_c1_mp_2604", "c12_e1_mp_2604"]
    summary = EnergyMinimizationSummaryData(
        identifiers=np.asarray(identifiers, dtype=object),
        minimum_energy=np.array([-10.0, -9.0, -9.0]),
        volume=np.array([1000.0, 999.9, 999.9]),
    )
    energy = pd.DataFrame(
        {
            "line_number": [10, 11],
            "op1": [1, 1],
            "id1": identifiers[1:],
            "n1": [1.0, 1.0],
            "op2": [-1, -1],
            "id2": [identifiers[0], identifiers[0]],
            "n2": [1.0, 1.0],
            "group_comment": ["EOS orthorhombic c12", "EOS orthorhombic c12"],
            "inline_comment": ["compression", "extension"],
        }
    )
    cells = pd.DataFrame(
        {
            "descriptor": identifiers,
            "a": [10.0, 9.9, 10.1],
            "b": [20.0, 20.2, 19.8],
            "c": [30.0, 30.0, 30.0],
            "alpha": [90.0, 90.0, 90.0],
            "beta": [90.0, 90.0, 90.0],
            "gamma": [90.0, 90.0, 90.0],
        }
    )
    data = ForceFieldOptimizationPlotBundleData(
        report=report,
        geometry_summary=summary,
        training_set=ForceFieldOptimizationTrainingSetData(
            sections=("ENERGY",),
            energy=energy,
        ),
        geometry_cells=cells,
    )

    table = FFieldOptimizationReportEOSTask().run(
        data,
        FFieldOptimizationReportEOSRequest(iden="all"),
    ).table

    strains = table.set_index("other_iden")["strain_percent"]
    assert strains["c12_c1_mp_2604"] == pytest.approx(-1.0)
    assert strains["c12_0_mp_2604"] == pytest.approx(0.0)
    assert strains["c12_e1_mp_2604"] == pytest.approx(1.0)
def test_eos_table_preserves_reaxff_and_qm_values() -> None:
    report = ForceFieldOptimizationReportData(
        linenos=np.array([1, 2]),
        sections=np.array(["ENERGY", "ENERGY"], dtype=object),
        titles=np.array(
            [
                "Energy +bulk/1.00 -bulk_0.9/1.00",
                "Energy +bulk/1.00 -bulk_1.1/1.00",
            ],
            dtype=object,
        ),
        ffield_values=np.array([-1.2, -0.8]),
        qm_values=np.array([-1.0, -0.7]),
        weights=np.array([1.0, 1.0]),
        errors=np.array([0.2, 0.1]),
        total_ff_error=np.array([0.2, 0.3]),
    )
    summary = EnergyMinimizationSummaryData(
        identifiers=np.array(["bulk_0.9", "bulk", "bulk_1.1"], dtype=object),
        minimum_energy=np.array([-10.0, -11.0, -9.0]),
        volume=np.array([9.0, 10.0, 11.0]),
    )

    table = _base_other_energy_volume_table(report, summary)

    assert table[["other_iden", "ffield_value", "qm_value"]].to_dict(orient="records") == [
        {"other_iden": "bulk_0.9", "ffield_value": -1.2, "qm_value": -1.0},
        {"other_iden": "bulk", "ffield_value": 0.0, "qm_value": 0.0},
        {"other_iden": "bulk_1.1", "ffield_value": -0.8, "qm_value": -0.7},
    ]


def test_comment_aware_classifier_finds_reference_in_either_operand() -> None:
    report = ForceFieldOptimizationReportData(
        linenos=np.array([10, 11, 12, 13]),
        sections=np.array(["ENERGY"] * 4, dtype=object),
        titles=np.array(
            [
                "Energy +bulk_c1_mp_1008557/1 -bulk_0_mp_1008557/1",
                "Energy +bulk_0_mp_1008557/1 -bulk_0_mp_1008557/1",
                "Energy +H2BCH3/1 -H2BCH3_0_652/1",
                "Energy +H2BCH3/1 -H2BCH3_0_700/1",
            ],
            dtype=object,
        ),
        ffield_values=np.array([0.2, 0.0, -3.0, -2.0]),
        qm_values=np.array([0.1, 0.0, -4.0, -3.0]),
        weights=np.ones(4),
        errors=np.zeros(4),
        total_ff_error=np.zeros(4),
    )
    energy = pd.DataFrame(
        {
            "line_number": [100, 101, 102, 103],
            "op1": [1, 1, 1, 1],
            "id1": [
                "bulk_c1_mp_1008557",
                "bulk_0_mp_1008557",
                "H2BCH3",
                "H2BCH3",
            ],
            "n1": [1.0] * 4,
            "op2": [-1, -1, -1, -1],
            "id2": [
                "bulk_0_mp_1008557",
                "bulk_0_mp_1008557",
                "H2BCH3_0_652",
                "H2BCH3_0_700",
            ],
            "n2": [1.0] * 4,
            "group_comment": [
                "EOS data /// Volume Bulk_EOS",
                "EOS data /// Volume Bulk_EOS",
                "Restraint H2BCH3_bond",
                "Restraint H2BCH3_bond",
            ],
            "inline_comment": ["compressed", "reference", "r1", "r2"],
        }
    )
    summary = EnergyMinimizationSummaryData(
        identifiers=np.array(
            ["bulk_c1_mp_1008557", "bulk_0_mp_1008557"], dtype=object
        ),
        minimum_energy=np.array([-9.0, -10.0]),
        volume=np.array([31.0, 32.0]),
    )
    data = ForceFieldOptimizationPlotBundleData(
        report=report,
        geometry_summary=summary,
        training_set=ForceFieldOptimizationTrainingSetData(
            sections=("ENERGY",), energy=energy
        ),
        geometry_restraints=pd.DataFrame(
            {
                "descriptor": ["H2BCH3_0_652"],
                "restraint_type": ["bond"],
                "coordinate": [0.652],
                "restraint_line_number": [200],
            }
        ),
    )

    tables = _force_field_optimization_curve_tables(data)

    assert set(tables["eos"]["base_iden"]) == {"bulk_0_mp_1008557"}
    assert "bulk_c1_mp_1008557" in set(tables["eos"]["other_iden"])
    assert tables["eos"]["group_comment"].str.contains("Bulk_EOS").all()
    assert not tables["eos"]["base_iden"].str.contains("H2BCH3").any()
    assert set(tables["restraint"]["base_iden"]) == {"H2BCH3"}
    assert {"group_comment", "inline_comment"}.issubset(tables["restraint"].columns)
    assert set(tables["bond"]["base_iden"]) == {"H2BCH3"}
    assert tables["bond"]["scan_coordinate"].dropna().tolist() == [0.652, 0.7]
    assert tables["angle"].empty


def test_volume_comment_eos_accepts_report_terms_without_divisors() -> None:
    report = ForceFieldOptimizationReportData(
        linenos=np.array([122, 123]),
        sections=np.array(["ENERGY", "ENERGY"], dtype=object),
        titles=np.array(
            [
                "Energy -cBN_opt      +cBN_1.20",
                "Energy -cBN_opt      +cBN_1.10",
            ],
            dtype=object,
        ),
        ffield_values=np.array([1.3702, 1.2179]),
        qm_values=np.array([4.79, 1.39]),
        weights=np.array([10.1, 0.1]),
        errors=np.zeros(2),
        total_ff_error=np.zeros(2),
    )
    energy = pd.DataFrame(
        {
            "line_number": [958, 959],
            "op1": ["-", "-"],
            "id1": ["cBN_opt", "cBN_opt"],
            "n1": [64.0, 64.0],
            "op2": ["+", "+"],
            "id2": ["cBN_1.20", "cBN_1.10"],
            "n2": [64.0, 64.0],
            "group_comment": ["Volume cBN_cubic", "Volume cBN_cubic"],
            "inline_comment": ["", ""],
        }
    )
    summary = EnergyMinimizationSummaryData(
        identifiers=np.array(["cBN_opt", "cBN_1.20", "cBN_1.10"], dtype=object),
        minimum_energy=np.array([-10246.4, -10158.7, -10200.0]),
        volume=np.array([364.325, 456.764, 420.0]),
    )
    data = ForceFieldOptimizationPlotBundleData(
        report=report,
        geometry_summary=summary,
        training_set=ForceFieldOptimizationTrainingSetData(
            sections=("ENERGY",), energy=energy
        ),
        geometry_restraints=pd.DataFrame(),
    )

    table = _force_field_optimization_curve_tables(data)["eos"]

    assert set(table["base_iden"]) == {"cBN_opt"}
    assert {"cBN_1.20", "cBN_1.10"}.issubset(set(table["other_iden"]))
    assert table["group_comment"].str.contains("Volume cBN_cubic").all()
    assert table.loc[table["other_iden"].eq("cBN_1.20"), "V_other_iden"].iloc[0] == 456.764


def test_unresolved_repeated_energy_family_is_preserved_as_other_curve() -> None:
    rows = pd.DataFrame(
        {
            "iden1": ["scan_1", "scan_2"],
            "iden2": ["reference", "reference"],
            "ffield_value": [2.0, 1.0],
            "qm_value": [2.5, 1.5],
            "group_comment": ["DFT coordinate scan"] * 2,
            "inline_comment": ["", ""],
            "report_line_number": [10, 11],
            "trainset_line_number": [20, 21],
        }
    )
    classified = _classify_curve_rows(rows, pd.DataFrame())
    table = _curve_table_from_classified_rows(
        classified,
        EnergyMinimizationSummaryData(identifiers=np.array([], dtype=object)),
        pd.DataFrame(),
        curve_type="other_curve",
    )

    assert set(classified["curve_type"]) == {"other_curve"}
    assert {"scan_1", "scan_2"}.issubset(set(table["other_iden"]))
    assert table.loc[table["other_iden"] == "scan_1", "scan_coordinate"].iloc[0] == 1.0


def _run_and_save() -> Path:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    project_root = run_dir / "reaxkit_workspace"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = resolve_engine(str(run_dir), engine=None)

    task = FFieldOptimizationReportEOSTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    request = FFieldOptimizationReportEOSRequest(
        iden="all",
    )
    executor = AnalysisExecutor()

    try:
        result = executor.run(
            task,
            request,
            {
                "run_dir": str(run_dir),
                "project_root": str(project_root),
                "fort99": str(run_dir / "fort.99"),
                "engine": "reaxff",
                "cache": False,
            },
        )
    except AnalysisError as exc:
        msg = str(exc)
        if (
            "expected data type ForceFieldOptimizationPlotBundleData, got NoneType" in msg
            or "Failed to load required data 'ForceFieldOptimizationPlotBundleData'" in msg
        ):
            pytest.skip("ForceFieldOptimizationPlotBundleData is not available for this run_dir.")
        raise

    assert result.request == request
    assert {
        "base_iden",
        "other_iden",
        "V_other_iden",
        "E_other_iden",
        "ffield_value",
        "qm_value",
    }.issubset(set(result.table.columns))

    metadata_path = task_artifacts_dir / "force_field_optimization_report_eos_summary.txt"
    csv_path = task_artifacts_dir / "force_field_optimization_report_eos.csv"
    head_path = task_artifacts_dir / "force_field_optimization_report_eos_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Columns: {list(result.table.columns)}",
                f"Rows: {len(result.table)}",
                f"Request iden: {result.request.iden}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        result.table.to_csv(csv_path, index=False)
        head_path.write_text(result.table.head(12).to_string(index=False) + "\n", encoding="utf-8")
    except PermissionError as exc:
        pytest.skip(f"Artifact file is locked by another process: {exc}")
    return task_artifacts_dir


def test_force_field_optimization_report_eos_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "force_field_optimization_report_eos_summary.txt").exists()
    assert (out_dir / "force_field_optimization_report_eos.csv").exists()
    assert (out_dir / "force_field_optimization_report_eos_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
