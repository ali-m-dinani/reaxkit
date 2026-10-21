from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read

from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    HBNReferenceLocalPolarizationRequest,
    calculate_hbn_reference_local_polarization,
    write_local_polarization_extxyz,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    REFERENCE_STRUCTURE_PATH,
    write_oriented_reference_cif,
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


def test_local_parser_defaults_to_equal_bbox_volume_and_accepts_deformation() -> None:
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
    assert default.local_grouping == "al"
    assert default.local_charge_method == "partition"
    neutral = local_polarization_workflow.build_request(parser.parse_args(
        ["--replication", "2", "1", "1", "--local-charge-method", "neutral-center"]
    ))
    assert neutral.local_charge_method == "neutral-center"
    assert default.volume_method == "bbox"
    local_hull = local_polarization_workflow.build_request(parser.parse_args(
        ["--replication", "2", "1", "1", "--volume-method", "local-hull"]
    ))
    assert local_hull.volume_method == "local-hull"
    with pytest.raises(SystemExit):
        parser.parse_args(["--replication", "2", "1", "1", "--local-charge-method", "neutral-al"])
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
    assert not request.include_centers
    assert request.workers == 0
    assert request.chunk_size == 0
    assert args.plot_2d and args.plot_kymograph


def test_hbn_projected_parser_retains_charge_options_only_for_compatibility() -> None:
    parser = projected_polarity_workflow.build_parser(
        argparse.ArgumentParser(), command=projected_polarity_workflow.COMMAND
    )
    required = ["--replication", "2", "1", "1"]

    assert parser.parse_args([*required, "--engine", "ams", "--charge-source", "auto"]).charge_source == "auto"
    assert parser.parse_args([*required, "--engine", "reaxff", "--charge-source", "formal"]).charge_source == "formal"
    assert parser.parse_args([*required, "--charge-source", "reaxff"]).charge_source == "reaxff"
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--engine", "lammps"])


@pytest.mark.parametrize("export_csv", [False, True])
def test_local_csvs_are_opt_in_without_disabling_extxyz(tmp_path, monkeypatch, export_csv):
    from reaxkit.presentation.persist import _write_csvs

    workflow = local_polarization_workflow
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    options = ["--replication", "2", "1", "1", "--write-extxyz",
               "--output-dir", str(tmp_path), "--output-profile", "standard"]
    assert parser.parse_args(options).export_local_csv is False
    if export_csv:
        options.append("--export-local-csv")
    args = parser.parse_args(options)
    result = _local_result()
    monkeypatch.setattr(workflow.AnalysisExecutor, "run", lambda *args, **kwargs: result)
    monkeypatch.setattr(workflow, "present_result", lambda *args, **kwargs: _write_csvs(tmp_path, result))
    assert workflow.run_main(workflow.COMMAND, args) == 0
    for grouping in ("local", "cell", "layer"):
        assert (tmp_path / f"hbn_reference_{grouping}_polarization.csv").exists() == export_csv
        assert (tmp_path / f"hbn_reference_{grouping}_polarization_summary.csv").is_file()
    assert (tmp_path / "hbn_reference_mapping.csv").is_file()
    neighbors = pd.read_csv(tmp_path / "hbn_reference_al_neighbors.csv")
    assert neighbors.groupby("al_atom_id").size().eq(3).all()
    np.testing.assert_allclose(neighbors.groupby("n_atom_id").partition_weight.sum(), 1.0)
    reference = read(tmp_path / "AlN_hbn_replicated_aligned.xyz")
    reference_cif = read(tmp_path / "AlN_hbn_oriented.cif")
    prepared = result.reference_result.reference
    assert reference.info["orthogonalized"] == prepared.orthogonalized
    np.testing.assert_allclose(reference.positions, prepared.aligned_positions, atol=1e-8)
    expected_cell = orthogonalize_hexagonal_cell(read(REFERENCE_STRUCTURE_PATH))
    np.testing.assert_allclose(reference_cif.cell.cellpar(), expected_cell.cell.cellpar(), atol=1e-8)
    assert len(reference_cif) == len(expected_cell)
    assert not (tmp_path / "AlN_hbn_replicated_aligned.cif").exists()
    mapping = result.reference_result.mapping
    np.testing.assert_array_equal(
        reference.arrays["simulation_atom_id"][mapping.reference_atom_index], mapping.atom_id
    )
    np.testing.assert_array_equal(reference.arrays["local_cell_id"], prepared.local_cell_ids)
    assert not (tmp_path / "table.csv").exists()
    assert not (tmp_path / "cell_table.csv").exists()
    assert not (tmp_path / "layer_table.csv").exists()
    frames = read(tmp_path / "hbn_reference_local_polarization.extxyz", index=":")
    assert len(frames) == 1
    assert len(frames[0]) == len(result.trajectory.atom_ids)
    assert frames[0].arrays["is_local_center"].sum() == 8
    assert frames[0].info["local_charge_method"] == "partition"


@pytest.mark.parametrize("orthogonalized", [False, True])
def test_small_reference_cif_follows_resolved_orientation(tmp_path, orthogonalized):
    result = _local_result().reference_result
    result.reference.orthogonalized = orthogonalized
    expected = read(REFERENCE_STRUCTURE_PATH)
    if orthogonalized:
        expected = orthogonalize_hexagonal_cell(expected)
    destination = write_oriented_reference_cif(result, tmp_path / "reference.cif")
    actual = read(destination)
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual.cell.cellpar(), expected.cell.cellpar(), atol=1e-8)


@pytest.mark.parametrize("grouping", ["al", "cell", "layer"])
@pytest.mark.parametrize("charge_method", ["partition", "neutral-center"])
def test_extxyz_contains_only_neighborhood_properties(tmp_path, grouping, charge_method) -> None:
    initial = _local_result()
    result = calculate_hbn_reference_local_polarization(
        initial.trajectory, replace(initial.request, local_grouping=grouping, local_charge_method=charge_method)
    )
    destination = write_local_polarization_extxyz(result, tmp_path / "local.extxyz")
    atoms = read(destination)
    assert set(atoms.arrays) == {
        "numbers", "positions", "atom_number", "local_neighborhood_id", "is_local_center",
        "has_valid_local_volume", "local_volume", "local_dipole", "local_polarization",
        "local_dipole_c", "local_polarization_c", "geometric_displacement",
        "geometric_polarity", "geometric_displacement_c", "geometric_polarity_c",
        "neighborhood_net_charge",
    }
    centers = atoms.arrays["is_local_center"].astype(bool)
    np.testing.assert_array_equal(centers, np.asarray(atoms.get_chemical_symbols()) == "Al")
    assert (atoms.arrays["local_neighborhood_id"][~centers] == -1).all()
    rows = result.layer_table.set_index("representative_atom_id").loc[atoms.arrays["atom_number"][centers]]
    np.testing.assert_array_equal(atoms.arrays["local_neighborhood_id"][centers], rows.local_neighborhood_id)
    assert not {"local_al_id", "local_layer_id", "local_cell_id", "local_group_id"} & set(rows.columns)
    for name, column in {
        "local_volume": "local_volume (angstrom^3)",
        "local_dipole_c": "dipole_c (e*angstrom)",
        "local_polarization_c": "P_c (uC/cm^2)",
        "neighborhood_net_charge": "effective_net_charge (e)",
    }.items():
        np.testing.assert_allclose(atoms.arrays[name][centers], rows[column], rtol=1e-7, atol=1e-7)
        assert np.isnan(atoms.arrays[name][~centers]).all()
    assert atoms.info["local_grouping"] == "neighborhood"
    assert atoms.info["local_charge_method"] == charge_method
    assert atoms.info["frame"] == 0
    assert atoms.info["iter"] == 20


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


@pytest.mark.parametrize("profile", ["standard", "minimal", "full", "legacy"])
@pytest.mark.parametrize("center_flags", [[], ["--no-write-centers"], ["--write-centers"], ["--write-centers", "--no-write-centers"]])
def test_projected_workflow_extxyz_without_centers(tmp_path, monkeypatch, profile, center_flags):
    workflow = projected_polarity_workflow
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args(["--replication", "2", "1", "1", "--write-extxyz", "--extxyz-precision", "10", "--centers-format", "csv", *center_flags])
    args.output_profile = profile
    expected_centers = center_flags == ["--write-centers"] and profile != "minimal"
    assert "--write-extxyz" in parser.format_help()
    assert "--no-write-centers" in parser.format_help()

    class StubExecutor:
        @staticmethod
        def run(task, request, run_args):
            assert request.write_extxyz == (profile != "minimal")
            assert request.include_centers == expected_centers
            assert ("centers" in run_args["_artifact_writer"].specs) == expected_centers
            if request.write_extxyz:
                assert run_args["no_cache"]
            return task.run(_trajectory(), request)

    monkeypatch.setattr(workflow, "AnalysisExecutor", StubExecutor)
    monkeypatch.setattr(workflow, "artifact_directory", lambda *_args: tmp_path)
    monkeypatch.setattr(workflow, "present_result", lambda *_args: None)
    assert workflow.run_main(workflow.COMMAND, args) == 0
    path = tmp_path / "hbn_reference_projected_polarity.extxyz"
    assert path.exists() == (profile != "minimal")
    if path.exists():
        assert len(read(path)) == len(_trajectory().atom_ids)
    assert bool(list(tmp_path.glob("*centers.csv"))) == expected_centers


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
    assert summary.loc[0, "positive_percentage"] == 100.0
