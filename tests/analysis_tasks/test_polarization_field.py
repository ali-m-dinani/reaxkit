"""Sanity check for PolarizationFieldTask."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import reaxkit.engine  # noqa: F401 (register engine adapters)
from reaxkit.analysis.electrostatics.electrostatics import (
    PolarizationFieldRequest,
    PolarizationFieldTask,
    _polarization_field_result,
    polarization_field_axis_label,
)
from reaxkit.core.platform.engine_resolver import resolve_engine
from reaxkit.core.platform.exceptions import ParseError
from reaxkit.domain.data_models import (
    ChargeData,
    ConnectivityData,
    ElectricFieldData,
    ElectrostaticsData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.workflows.electrostatics_workflow import (
    _apply_polarization_scale,
    _plot_payload,
    _polarization_summary_path,
    build_parser,
)

RUN_DIR = Path(
    r"C:\Users\alimo\PycharmProjects\pythonProject\reaxkit\examples_to_test"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def test_mean_aggregation_keeps_separate_field_sections_in_frame_order() -> None:
    polarization = pd.DataFrame(
        {
            "frame_index": [0, 1, 2, 3, 4, 5],
            "iter": [0, 10, 20, 30, 40, 50],
            "P_z (uC/cm^2)": [1.0, 3.0, 10.0, 12.0, 100.0, 104.0],
        }
    )
    field = ElectricFieldData(
        applied_field_values=np.asarray([[0.0], [0.0], [1.0], [1.0], [0.0], [0.0]]),
        applied_field_components=["field_z"],
        sampled_field_iterations=np.asarray([0, 10, 20, 30, 40, 50]),
    )

    result = _polarization_field_result(
        polarization,
        field,
        PolarizationFieldRequest(aggregate="mean", field_direction="z"),
    )

    assert result.aggregated_table["frame_index"].tolist() == [0.5, 2.5, 4.5]
    assert result.aggregated_table["P_z (uC/cm^2)"].tolist() == [2.0, 11.0, 102.0]
    assert result.aggregated_table["field_z"].iloc[0] == result.aggregated_table["field_z"].iloc[2]


def test_polarization_field_axis_labels_use_math_subscripts() -> None:
    assert polarization_field_axis_label("field_z") == r"$E_{z}$ (MV/cm)"
    assert polarization_field_axis_label("P_z (uC/cm^2)") == r"$P_{z}$ ($\mu$C/cm$^2$)"


def test_hysteresis_plot_payload_has_markers_and_black_zero_lines() -> None:
    table = pd.DataFrame(
        {"field_z": [-1.0, 0.0, 1.0], "P_z (uC/cm^2)": [-2.0, 1.0, 2.0]}
    )
    payload = _plot_payload(
        "get_polarization_field",
        SimpleNamespace(full_table=table, aggregated_table=table),
        argparse.Namespace(xaxis="field_z", yaxis="pol_z"),
    )

    assert payload is not None
    assert payload["series"][0]["marker"] == "o"
    assert payload["hlines"] == [{"y": 0.0, "color": "black", "linestyle": "--"}]
    assert payload["vlines"] == [{"x": 0.0, "color": "black", "linestyle": "--"}]


@pytest.mark.parametrize("command", ["polarization", "get_polarization_field"])
def test_polarization_commands_accept_scale_by(command: str) -> None:
    parser = build_parser(argparse.ArgumentParser(), command=command)

    args = parser.parse_args(["--scale-by", "10"])

    assert args.scale_by == 10.0


def test_polarization_field_scale_multiplies_tables_and_remnant_values() -> None:
    result = SimpleNamespace(
        full_table=pd.DataFrame(
            {"field_z": [0.0], "P_z (uC/cm^2)": [1.5], "mu_z (debye)": [3.0]}
        ),
        aggregated_table=pd.DataFrame(
            {"field_z": [0.0], "P_z (uC/cm^2)": [1.5], "mu_z (debye)": [3.0]}
        ),
        polarization_zero_crossings=[2.0],
        field_zero_crossings=[1.5],
    )

    _apply_polarization_scale("get_polarization_field", result, 10.0)

    assert result.full_table["P_z (uC/cm^2)"].tolist() == [15.0]
    assert result.aggregated_table["P_z (uC/cm^2)"].tolist() == [15.0]
    assert result.full_table["mu_z (debye)"].tolist() == [3.0]
    assert result.polarization_zero_crossings == [2.0]
    assert result.field_zero_crossings == [15.0]


def test_polarization_scale_multiplies_only_polarization_columns() -> None:
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "P_x (uC/cm^2)": [1.0],
                "P_y (uC/cm^2)": [2.0],
                "P_z (uC/cm^2)": [3.0],
                "mu_z (debye)": [4.0],
                "volume (angstrom^3)": [5.0],
            }
        )
    )

    _apply_polarization_scale("polarization", result, 10.0)

    assert result.table["P_x (uC/cm^2)"].tolist() == [10.0]
    assert result.table["P_y (uC/cm^2)"].tolist() == [20.0]
    assert result.table["P_z (uC/cm^2)"].tolist() == [30.0]
    assert result.table["mu_z (debye)"].tolist() == [4.0]
    assert result.table["volume (angstrom^3)"].tolist() == [5.0]


def test_hysteresis_summary_is_resolved_beside_default_plot(tmp_path: Path) -> None:
    project_root = tmp_path / "reaxkit_workspace"
    args = argparse.Namespace(
        summary="hysteresis_summary.txt",
        save="hysteresis_aggregated.png",
        project_root=str(project_root),
        run_id="run-001",
        analysis_id=None,
    )

    summary_path = _polarization_summary_path("get_polarization_field", args)

    expected_dir = project_root / "analysis" / "get_polarization_field" / "run-001"
    assert summary_path == expected_dir / "hysteresis_summary.txt"
    assert summary_path.parent / "hysteresis_aggregated.png" == expected_dir / "hysteresis_aggregated.png"


@pytest.mark.parametrize(("volume_method", "expected_volume"), [("bbox", 1.0), ("cell", 24.0)])
def test_polarization_field_uses_requested_volume_method(
    volume_method: str,
    expected_volume: float,
) -> None:
    iterations = np.asarray([0])
    elements = ["C", "H", "H", "H"]
    simulation = SimulationData(
        atom_ids=[1, 2, 3, 4],
        iterations=iterations,
        elements=elements,
        cell_lengths=np.asarray([[2.0, 3.0, 4.0]]),
    )
    data = ElectrostaticsData(
        trajectory=TrajectoryData(
            positions=np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
            ),
            elements=elements,
            atom_ids=[1, 2, 3, 4],
            iterations=iterations,
            simulation=simulation,
        ),
        charges=ChargeData(
            charges=np.asarray([[-1.0, 1.0, 0.0, 0.0]]),
            iterations=iterations,
            simulation=simulation,
        ),
        electric_field=ElectricFieldData(
            applied_field_values=np.asarray([[0.0]]),
            applied_field_components=["field_z"],
            sampled_field_iterations=iterations,
        ),
    )

    result = PolarizationFieldTask().run(
        data,
        PolarizationFieldRequest(aggregate="mean", volume_method=volume_method),
    )

    assert result.request.volume_method == volume_method
    assert result.full_table["volume (angstrom^3)"].tolist() == [expected_volume]


def _run_and_save() -> Path:
    run_dir = RUN_DIR
    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {run_dir}")
    project_root = run_dir / "reaxkit_workspace"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = resolve_engine(str(run_dir), engine=None)

    task = PolarizationFieldTask()
    task_name = str(task.__class__.__name__).replace("(", "").replace(")", "")
    task_artifacts_dir = ARTIFACTS_DIR / task_name
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)

    load_args = {
        "run_dir": str(run_dir),
        "project_root": str(project_root),
        "xmolout": str(run_dir / "xmolout"),
        "fort7": str(run_dir / "fort.7"),
        "fort78": str(run_dir / "fort.78"),
        "control": str(run_dir / "control"),
        "cache": False,
    }
    try:
        data = ElectrostaticsData(
            trajectory=adapter.load(TrajectoryData, load_args),
            charges=adapter.load(ChargeData, load_args),
            connectivity=adapter.load(ConnectivityData, load_args),
            electric_field=adapter.load(ElectricFieldData, load_args),
        )
    except ParseError as exc:
        if "fort.78" in str(exc):
            pytest.skip("ElectricFieldData source fort.78 is not available for this run_dir.")
        raise

    request = PolarizationFieldRequest(
        frames=None,
        every=1000,
        aggregate="mean",
        field_direction="z",
        dipole_or_polaization_direction="p_z",
    )
    result = task.run(data, request)

    assert result.request == request
    assert isinstance(result.full_table, pd.DataFrame)
    assert isinstance(result.aggregated_table, pd.DataFrame)
    assert {"iter", "mu_z (debye)", "P_z (uC/cm^2)", "field_z"}.issubset(set(result.full_table.columns))
    assert {"field_z", "P_z (uC/cm^2)"}.issubset(set(result.aggregated_table.columns))

    payload = {
        "full_table": result.full_table.head(20).to_dict(orient="records"),
        "aggregated_table": result.aggregated_table.head(20).to_dict(orient="records"),
        "request": asdict(result.request),
    }
    views = task.recommended_presentations(result, payload)
    assert len(views) >= 3
    assert views[0].view_type == "table"
    assert views[1].view_type == "table"
    assert views[2].view_type == "plot2d"

    metadata_path = task_artifacts_dir / "polarization_field_summary.txt"
    full_csv_path = task_artifacts_dir / "polarization_field_full.csv"
    aggregated_csv_path = task_artifacts_dir / "polarization_field_aggregated.csv"
    head_path = task_artifacts_dir / "polarization_field_head.txt"

    metadata_path.write_text(
        "\n".join(
            [
                f"Detected adapter: {adapter.__class__.__name__}",
                f"Result type: {type(result).__name__}",
                f"Full columns: {list(result.full_table.columns)}",
                f"Full rows: {len(result.full_table)}",
                f"Aggregated columns: {list(result.aggregated_table.columns)}",
                f"Aggregated rows: {len(result.aggregated_table)}",
                f"Request aggregate: {result.request.aggregate}",
                f"Request field_direction: {result.request.field_direction}",
                f"Request dipole_or_polaization_direction: {result.request.dipole_or_polaization_direction}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result.full_table.to_csv(full_csv_path, index=False)
    result.aggregated_table.to_csv(aggregated_csv_path, index=False)
    head_path.write_text(
        result.aggregated_table.head(12).to_string(index=False) + "\n",
        encoding="utf-8",
    )
    return task_artifacts_dir


def test_polarization_field_saves_artifacts() -> None:
    if not RUN_DIR.exists():
        pytest.skip(f"RUN_DIR does not exist: {RUN_DIR}")
    out_dir = _run_and_save()
    assert (out_dir / "polarization_field_summary.txt").exists()
    assert (out_dir / "polarization_field_full.csv").exists()
    assert (out_dir / "polarization_field_aggregated.csv").exists()
    assert (out_dir / "polarization_field_head.txt").exists()


def main() -> None:
    if not RUN_DIR.exists():
        return
    _run_and_save()


if __name__ == "__main__":
    main()
