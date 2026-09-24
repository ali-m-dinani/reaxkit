from __future__ import annotations

import argparse

import pytest
from ase import Atoms
from ase.io import read

from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.core.storage.storage_layout import normalize_storage_args
from reaxkit.domain.data_models import ElectrostaticsData, TrajectoryData
from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    HBNReferencePolarizationTask,
)
from reaxkit.workflows.ferroelectrics.hbn_reference import polarization_workflow


def test_command_routes_to_hbn_reference_workflow() -> None:
    route = get_registered_analysis_commands()[polarization_workflow.COMMAND]
    assert route.module_path.endswith("hbn_reference.polarization_workflow")


def test_parser_builds_default_charge_request() -> None:
    parser = polarization_workflow.build_parser(
        argparse.ArgumentParser(), command=polarization_workflow.COMMAND
    )
    args = parser.parse_args(["--replication", "2", "3", "4"])
    request = polarization_workflow.build_request(args)

    assert request.charge_source == "auto"
    assert request.formal_charges == {"Al": 3.0, "N": -3.0}
    assert request.reference_species == {"B": "Al"}
    assert request.orthogonalize == "auto"
    assert request.replication == (2, 3, 4)
    assert request.max_reference_strain == 0.15
    assert request.volume_method == "hull"
    assert request.include_displacements is False


def test_auto_charge_source_uses_fort7_from_pre_snapshot_source(
        tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "xmolout").write_text("trajectory", encoding="utf-8")
    (tmp_path / "fort.7").write_text("charges", encoding="utf-8")
    parser = polarization_workflow.build_parser(
        argparse.ArgumentParser(), command=polarization_workflow.COMMAND
    )
    namespace = parser.parse_args(["--replication", "1", "1", "1"])
    request = polarization_workflow.build_request(namespace)
    runtime = normalize_storage_args(
        {
            **polarization_workflow.runtime_arguments(namespace),
            "project_root": str(tmp_path / "workspace"),
        },
        snapshot=False,
    )

    task = HBNReferencePolarizationTask()
    assert task.required_data_for(request, runtime) is ElectrostaticsData
    assert task.required_data_for(request, None) == (
        TrajectoryData,
        ElectrostaticsData,
    )


def test_parser_accepts_charge_source_and_formal_charge_options() -> None:
    parser = polarization_workflow.build_parser(
        argparse.ArgumentParser(), command=polarization_workflow.COMMAND
    )
    args = parser.parse_args([
        "--replication", "2", "3", "4",
        "--charge-source", "formal",
        "--formal-charge", "B=3",
    ])

    request = polarization_workflow.build_request(args)

    assert request.charge_source == "formal"
    assert request.formal_charges == {"Al": 3.0, "N": -3.0, "B": 3.0}


def test_parser_supports_explicit_reference_orthogonalization() -> None:
    parser = polarization_workflow.build_parser(
        argparse.ArgumentParser(), command=polarization_workflow.COMMAND
    )
    args = parser.parse_args([
        "--replication", "19", "19", "10", "--orthogonalize-reference",
        "--volume-method", "cell",
    ])
    request = polarization_workflow.build_request(args)

    assert request.replication == (19, 19, 10)
    assert request.orthogonalize == "always"
    assert request.volume_method == "cell"


@pytest.mark.parametrize("write_displacements", [False, True])
def test_workflow_writes_requested_reference_outputs(
        tmp_path, write_displacements: bool
) -> None:
    reference_path = polarization_workflow._default_reference_path()
    reference = read(reference_path)
    assert isinstance(reference, Atoms)
    positions = reference.get_positions()
    lengths = reference.cell.lengths()
    angles = reference.cell.angles()
    xmolout = tmp_path / "xmolout"
    lines = [
        str(len(reference)),
        "sim 0 0 "
        + " ".join(str(value) for value in (*lengths, *angles)),
    ]
    lines.extend(
        f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
        for symbol, (x, y, z) in zip(
            reference.get_chemical_symbols(), positions, strict=True
        )
    )
    xmolout.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    parser = polarization_workflow.build_parser(
        argparse.ArgumentParser(), command=polarization_workflow.COMMAND
    )
    command_args = [
        "--engine", "reaxff",
        "--input", str(tmp_path),
        "--xmolout", str(xmolout),
        "--reference", str(reference_path),
        "--replication", "1", "1", "1",
        "--no-orthogonalize-reference",
        "--output-dir", str(output),
        "--project-root", str(tmp_path / "workspace"),
    ]
    if write_displacements:
        command_args.append("--write-displacements")
    args = parser.parse_args(command_args)
    args.output_profile = "standard"

    assert polarization_workflow.run_main(polarization_workflow.COMMAND, args) == 0
    assert (output / "hbn_reference_polarization.csv").is_file()
    assert (output / "hbn_reference_displacements.parquet").is_file() is write_displacements
    assert (output / "hbn_reference_mapping.csv").is_file()
    assert (output / "AlN_hbn_replicated_aligned.xyz").is_file()
