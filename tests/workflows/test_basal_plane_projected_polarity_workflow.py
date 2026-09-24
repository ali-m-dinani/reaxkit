from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.projected_polarity import (
    BasalPlaneProjectedPolarityRequest,
    calculate_basal_plane_projected_polarity,
)
from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.workflows.ferroelectrics.basal_plane_displacement_for_dipole_moment import (
    projected_polarity_workflow,
)
from reaxkit.domain.data_models import TrajectoryData


def _motif(offset: float, *, inverted: bool) -> list[list[float]]:
    basal_z = 0.2 if inverted else -0.2
    apical_z = -1.8 if inverted else 1.8
    return [
        [offset, 0.0, 0.0],
        [offset + 1.6, 0.0, basal_z],
        [offset - 0.8, 1.386, basal_z],
        [offset - 0.8, -1.386, basal_z],
        [offset, 0.0, apical_z],
    ]


def _trajectory() -> TrajectoryData:
    mixed = _motif(0.0, inverted=False) + _motif(6.0, inverted=True)
    negative = _motif(0.0, inverted=False) + _motif(-6.0, inverted=False)
    return TrajectoryData(
        positions=np.asarray([mixed, negative], dtype=float),
        elements=["Al", "N", "N", "N", "N"] * 2,
        atom_ids=list(range(1, 11)),
        iterations=np.asarray([0, 50]),
    )


def test_command_routes_to_projected_polarity_workflow() -> None:
    route = get_registered_analysis_commands()[projected_polarity_workflow.COMMAND]
    assert route.module_path.endswith("projected_polarity_workflow")


def test_parser_builds_tem_projection_request() -> None:
    parser = projected_polarity_workflow.build_parser(
        argparse.ArgumentParser(), command=projected_polarity_workflow.COMMAND
    )
    args = parser.parse_args([
        "--charge-source", "formal",
        "--formal-charge", "Al=3", "N=-3",
        "--projection-plane", "xz",
        "--projection-bins", "20", "30",
        "--profile-axis", "z",
        "--reference-frame", "0",
        "--component", "z",
        "--plot-2d",
        "--plot-kymograph",
    ])
    request = projected_polarity_workflow.build_request(args)

    assert request.projection_plane == "xz"
    assert request.projection_bins == (20, 30)
    assert request.profile_axis == "z"
    assert request.reference_frame == 0
    assert request.projection_bins[1] == 30
    assert request.component == "z"
    assert not request.include_centers
    assert request.workers == 0
    assert request.chunk_size == 0
    assert args.plot_2d and args.plot_kymograph


def test_projected_parser_accepts_only_native_auto_or_formal_charge_modes() -> None:
    parser = projected_polarity_workflow.build_parser(
        argparse.ArgumentParser(), command=projected_polarity_workflow.COMMAND
    )

    assert parser.parse_args(["--engine", "ams", "--charge-source", "auto"]).charge_source == "auto"
    assert parser.parse_args(["--engine", "reaxff", "--charge-source", "formal"]).charge_source == "formal"
    with pytest.raises(SystemExit):
        parser.parse_args(["--charge-source", "reaxff"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--engine", "lammps"])


def test_heatmap_generators_write_frame_and_evolution_plots(tmp_path) -> None:
    result = calculate_basal_plane_projected_polarity(
        _trajectory(),
        BasalPlaneProjectedPolarityRequest(
            periodic=(False, False, False),
            charge_source="formal",
            formal_charges={"Al": 3.0, "N": -3.0},
            projection_plane="xz",
            projection_bins=(1, 1),
            profile_axis="z",
        ),
    )

    frames = projected_polarity_workflow.generate_projected_polarity_heatmaps(
        result, tmp_path, dpi=72
    )
    kymograph = projected_polarity_workflow.generate_polarity_kymograph(
        result, tmp_path, dpi=72
    )

    assert [path.name for path in frames] == ["frame_000000.png", "frame_000001.png"]
    assert all(path.is_file() for path in frames)
    assert kymograph.name == "projected_polarity_kymograph.png"
    assert kymograph.is_file()


def test_workflow_writes_optional_centers_as_parquet(tmp_path, monkeypatch) -> None:
    result = calculate_basal_plane_projected_polarity(_trajectory(), _request_for_output())
    parser = projected_polarity_workflow.build_parser(
        argparse.ArgumentParser(), command=projected_polarity_workflow.COMMAND
    )
    args = parser.parse_args([
        "--periodic", "none",
        "--charge-source", "formal",
        "--formal-charge", "Al=3", "N=-3",
        "--write-centers",
    ])

    class StubExecutor:
        @staticmethod
        def run(*_args, **_kwargs):
            return result

    monkeypatch.setattr(projected_polarity_workflow, "AnalysisExecutor", StubExecutor)
    monkeypatch.setattr(
        projected_polarity_workflow, "artifact_directory", lambda *_args, **_kwargs: tmp_path
    )
    monkeypatch.setattr(projected_polarity_workflow, "present_result", lambda *_args, **_kwargs: None)

    assert projected_polarity_workflow.run_main(projected_polarity_workflow.COMMAND, args) == 0
    centers = tmp_path / "basal_plane_projected_polarity_centers.parquet"
    assert centers.is_file()
    assert len(pd.read_parquet(centers)) == len(result.centers)


def _request_for_output() -> BasalPlaneProjectedPolarityRequest:
    return BasalPlaneProjectedPolarityRequest(
        periodic=(False, False, False),
        charge_source="formal",
        formal_charges={"Al": 3.0, "N": -3.0},
        projection_plane="xz",
        projection_bins=(1, 1),
        profile_axis="z",
    )


@pytest.mark.parametrize("profile,flag,extension", [("full", False, "parquet"), ("legacy", False, "csv"), ("minimal", True, None)])
def test_output_profile_controls_detail_production(tmp_path, monkeypatch, profile, flag, extension):
    result = calculate_basal_plane_projected_polarity(_trajectory(), _request_for_output())
    parser = projected_polarity_workflow.build_parser(argparse.ArgumentParser(), command=projected_polarity_workflow.COMMAND)
    args = parser.parse_args(["--output-profile", profile] + (["--write-centers"] if flag else []))
    class StubExecutor:
        @staticmethod
        def run(task, request, runtime):
            assert request.include_centers is (extension is not None)
            return result
    monkeypatch.setattr(projected_polarity_workflow, "AnalysisExecutor", StubExecutor)
    monkeypatch.setattr(projected_polarity_workflow, "artifact_directory", lambda *_args: tmp_path)
    monkeypatch.setattr(projected_polarity_workflow, "present_result", lambda *_args: None)
    projected_polarity_workflow.run_main(projected_polarity_workflow.COMMAND, args)
    matches = list(tmp_path.glob("basal_plane_projected_polarity_centers.*"))
    assert len(matches) == (0 if extension is None else 1)
    if matches:
        assert matches[0].suffix == f".{extension}"
