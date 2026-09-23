from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    HBNReferenceLocalPolarizationRequest,
    calculate_hbn_reference_local_polarization,
    write_local_polarization_extxyz,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    REFERENCE_STRUCTURE_PATH,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.projected_polarity import (
    HBNReferenceProjectedPolarityRequest,
    calculate_hbn_reference_projected_polarity,
)
from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.domain.data_models import SimulationData, TrajectoryData
from reaxkit.engine.common.generators.structure_transformers import (
    orthogonalize_hexagonal_cell,
)
from reaxkit.workflows.ferroelectrics.hbn_reference import (
    local_polarization_workflow,
    projected_polarity_workflow,
)


def _trajectory(*, displaced: bool = True) -> TrajectoryData:
    source = read(REFERENCE_STRUCTURE_PATH)
    assert isinstance(source, Atoms)
    reference = orthogonalize_hexagonal_cell(source)
    reference.wrap()
    reference = reference.repeat((2, 1, 1))
    cell = reference.cell.array.copy()
    positions = reference.positions.copy()
    labels = np.asarray(reference.get_chemical_symbols())
    if displaced:
        positions[labels == "Al", 2] += 0.1
    simulation = SimulationData(
        atom_ids=list(range(1, len(reference) + 1)),
        iterations=np.asarray([20]),
        elements=labels.tolist(),
        cell_lengths=np.asarray([np.linalg.norm(cell, axis=1)]),
        cell_angles=np.asarray([[90.0, 90.0, 90.0]]),
    )
    return TrajectoryData(
        positions=positions[None, :, :],
        elements=labels.tolist(),
        atom_ids=list(range(1, len(reference) + 1)),
        iterations=np.asarray([20]),
        simulation=simulation,
    )


def _local_result():
    return calculate_hbn_reference_local_polarization(
        _trajectory(),
        HBNReferenceLocalPolarizationRequest(
            reference_path=REFERENCE_STRUCTURE_PATH,
            replication=(2, 1, 1),
            volume_method="cell",
        ),
    )


def test_commands_route_to_correctly_spelled_hbn_reference_workflows() -> None:
    routes = get_registered_analysis_commands()
    assert routes[local_polarization_workflow.COMMAND].module_path == (
        "reaxkit.workflows.ferroelectrics.hbn_reference.local_polarization_workflow"
    )
    assert routes[projected_polarity_workflow.COMMAND].module_path == (
        "reaxkit.workflows.ferroelectrics.hbn_reference.projected_polarity_workflow"
    )


def test_local_parser_defaults_to_equal_hull_volume_and_accepts_deformation() -> None:
    parser = local_polarization_workflow.build_parser(
        argparse.ArgumentParser(), command=local_polarization_workflow.COMMAND
    )
    default = local_polarization_workflow.build_request(
        parser.parse_args(["--replication", "2", "1", "1"])
    )
    deformation = local_polarization_workflow.build_request(
        parser.parse_args(
            [
                "--replication",
                "2",
                "1",
                "1",
                "--local-volume-method",
                "deformation",
                "--local-grouping",
                "layer",
                "--deformation-neighbors",
                "8",
                "--plot-component",
                "c",
                "--write-extxyz",
            ]
        )
    )

    assert default.local_volume_method == "equal"
    assert default.volume_method == "hull"
    assert deformation.local_volume_method == "deformation"
    assert deformation.local_grouping == "layer"
    assert deformation.deformation_neighbors == 8


def test_projected_parser_reuses_projection_bins_for_profile_axis() -> None:
    parser = projected_polarity_workflow.build_parser(
        argparse.ArgumentParser(), command=projected_polarity_workflow.COMMAND
    )
    args = parser.parse_args(
        [
            "--replication",
            "2",
            "1",
            "1",
            "--component",
            "c",
            "--projection-plane",
            "xz",
            "--projection-bins",
            "1",
            "10",
            "--profile-axis",
            "z",
            "--plot-2d",
            "--plot-evolution",
        ]
    )
    request = projected_polarity_workflow.build_request(args)

    assert request.projection_bins == (1, 10)
    assert request.profile_axis == "z"
    assert request.component == "c"
    assert args.plot_2d and args.plot_kymograph


def test_extxyz_contains_full_atoms_and_local_cell_properties(tmp_path) -> None:
    result = _local_result()
    destination = write_local_polarization_extxyz(result, tmp_path / "local.extxyz")
    lines = destination.read_text(encoding="utf-8").splitlines()

    assert int(lines[0]) == len(result.trajectory.atom_ids)
    assert "local_cell_id:I:1" in lines[1]
    assert "local_layer_id:I:1" in lines[1]
    assert "is_local_cell_center:I:1" in lines[1]
    assert "is_local_layer_center:I:1" in lines[1]
    assert "local_dipole:R:3" in lines[1]
    assert "local_polarization:R:3" in lines[1]
    assert "layer_dipole:R:3" in lines[1]
    assert "layer_polarization:R:3" in lines[1]
    assert 'frame=0' in lines[1]
    assert 'iter=20' in lines[1]


def test_local_and_projected_plot_generators_write_artifacts(tmp_path) -> None:
    local = _local_result()
    two_dimensional = local_polarization_workflow.generate_local_2d_plots(
        local,
        tmp_path,
        plane="xz",
        component="c",
        quantity="polarization",
        bins=(1, 1),
        global_scaling=False,
        dpi=72,
    )
    three_dimensional = local_polarization_workflow.generate_local_3d_plots(
        local,
        tmp_path,
        component="c",
        quantity="dipole",
        global_scaling=False,
        dpi=72,
    )
    projected = calculate_hbn_reference_projected_polarity(
        _trajectory(),
        HBNReferenceProjectedPolarityRequest(
            reference_path=REFERENCE_STRUCTURE_PATH,
            replication=(2, 1, 1),
            component="c",
            projection_plane="xz",
            projection_bins=(1, 1),
            profile_axis="z",
        ),
    )
    frame_plots = projected_polarity_workflow.generate_projected_polarity_heatmaps(
        projected, tmp_path, dpi=72
    )
    kymograph = projected_polarity_workflow.generate_polarity_kymograph(
        projected, tmp_path, dpi=72
    )

    assert len(two_dimensional) == 1 and two_dimensional[0].is_file()
    assert len(three_dimensional) == 1 and three_dimensional[0].is_file()
    assert len(frame_plots) == 1 and frame_plots[0].is_file()
    assert kymograph.is_file()


def test_projected_workflow_writes_whole_slab_polarity_summary(
        tmp_path, monkeypatch
) -> None:
    result = calculate_hbn_reference_projected_polarity(
        _trajectory(),
        HBNReferenceProjectedPolarityRequest(
            reference_path=REFERENCE_STRUCTURE_PATH,
            replication=(2, 1, 1),
            component="c",
            projection_plane="xz",
            projection_bins=(2, 2),
            profile_axis="z",
        ),
    )
    output = tmp_path / "projected_output"
    parser = projected_polarity_workflow.build_parser(
        argparse.ArgumentParser(), command=projected_polarity_workflow.COMMAND
    )
    args = parser.parse_args(["--replication", "2", "1", "1"])

    class StubExecutor:
        @staticmethod
        def run(*_args, **_kwargs):
            return result

    monkeypatch.setattr(projected_polarity_workflow, "AnalysisExecutor", StubExecutor)
    monkeypatch.setattr(
        projected_polarity_workflow,
        "artifact_directory",
        lambda *_args, **_kwargs: output,
    )
    monkeypatch.setattr(
        projected_polarity_workflow,
        "present_result",
        lambda *_args, **_kwargs: None,
    )

    assert projected_polarity_workflow.run_main(
        projected_polarity_workflow.COMMAND, args
    ) == 0
    summary_path = (
            output / "hbn_reference_projected_polarity_whole_slab_summary.csv"
    )
    assert summary_path.is_file()
    summary = pd.read_csv(summary_path)
    assert len(summary) == 1
    assert summary.loc[0, "spatial_scope"] == "whole_slab"
    assert summary.loc[0, "defined_group_count"] == len(result.centers)
    assert summary.loc[0, "defined_group_count"] == result.projected_bins[
        "defined_group_count"
    ].sum()
    assert summary.loc[0, "negative_percentage"] == 100.0
