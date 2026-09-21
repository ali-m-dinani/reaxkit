from __future__ import annotations

import argparse
import csv

from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.neighbors import (
    WurtziteNeighborRequest,
)
from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite import (
    neighbors_workflow,
    polarization_workflow,
    polarity_trajectory_workflow,
    polarity_workflow,
)


def _xmolout(tmp_path):
    path = tmp_path / "xmolout"
    path.write_text(
        """5
sim 0 0 20 20 20 90 90 90
Al 0 0 0
N 1.6 0 -0.2
N -0.8 1.386 -0.2
N -0.8 -1.386 -0.2
N 0 0 -1.7
""",
        encoding="utf-8",
    )
    return path


def _base_args(workflow, command, tmp_path):
    parser = workflow.build_parser(argparse.ArgumentParser(), command=command)
    return parser.parse_args([
        "--engine", "reaxff",
        "--input", str(tmp_path),
        "--xmolout", str(_xmolout(tmp_path)),
        "--charge-source", "formal",
        "--formal-charge", "Al=3", "N=-3",
        "--periodic", "none",
        "--project-root", str(tmp_path / "workspace"),
    ])


def _two_frame_reaxff_inputs(tmp_path):
    xmolout = tmp_path / "xmolout"
    xmolout.write_text(
        """5
sim 0 0 20 20 20 90 90 90
Al 0 0 0
N 1.6 0 -0.1
N -0.8 1.386 -0.1
N -0.8 -1.386 -0.1
N 0 0 -1.7
5
sim 10 0 20 20 20 90 90 90
Al 0 0 0
N 1.6 0 -0.2
N -0.8 1.386 -0.2
N -0.8 -1.386 -0.2
N 0 0 -1.7
""",
        encoding="utf-8",
    )
    fort7 = tmp_path / "fort.7"
    fort7.write_text(
        """5 sim Iteration: 0 #Bonds: 0
1 1 1 0.0 0.0 3.0
2 2 1 0.0 0.0 -0.7
3 2 1 0.0 0.0 -0.8
4 2 1 0.0 0.0 -0.9
5 2 1 0.0 0.0 -1.0
0.0 0.0 0.0 0.0
5 sim Iteration: 10 #Bonds: 0
1 1 1 0.0 0.0 2.5
2 2 1 0.0 0.0 -0.6
3 2 1 0.0 0.0 -0.7
4 2 1 0.0 0.0 -0.8
5 2 1 0.0 0.0 -0.9
0.0 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    return xmolout, fort7


def _selected_reaxff_args(workflow, command, tmp_path, output_dir):
    xmolout, fort7 = _two_frame_reaxff_inputs(tmp_path)
    parser = workflow.build_parser(argparse.ArgumentParser(), command=command)
    return parser.parse_args([
        "--engine", "reaxff", "--input", str(tmp_path),
        "--xmolout", str(xmolout), "--fort7", str(fort7),
        "--charge-source", "reaxff", "--periodic", "none",
        "--frames", "1", "--output-dir", str(output_dir),
        "--project-root", str(tmp_path / "workspace"),
    ])


def test_parsers_follow_documentation_and_alias_rules() -> None:
    cases = [
        (neighbors_workflow, "get_three_folded_wurtzite_neighbors"),
        (polarity_workflow, "three-folded-wurtzite-polarity"),
        (polarization_workflow, "three-folded-wurtzite-polarization"),
        (polarity_trajectory_workflow, "three-folded-polarity-extxyz"),
    ]
    for workflow, alias in cases:
        parser = workflow.build_parser(argparse.ArgumentParser(), command=alias)
        assert parser.formatter_class is argparse.RawTextHelpFormatter
        assert parser.get_default("command") == workflow.COMMAND
        assert "\n\nExamples:\n" in parser.description
        assert f"reaxkit {workflow.COMMAND}" in parser.description


def test_commands_route_to_dedicated_three_folded_modules() -> None:
    routes = get_registered_analysis_commands()
    for command, suffix in (
        (neighbors_workflow.COMMAND, "neighbors_workflow"),
        (polarity_workflow.COMMAND, "polarity_workflow"),
        (polarization_workflow.COMMAND, "polarization_workflow"),
        (polarity_trajectory_workflow.COMMAND, "polarity_trajectory_workflow"),
    ):
        assert routes[command].module_path.startswith(
            "reaxkit.workflows.ferroelectrics.three_folded_wurtzite."
        )
        assert routes[command].module_path.endswith(suffix)


def test_neighbor_request_reuses_established_structure_flags() -> None:
    parser = neighbors_workflow.build_parser(
        argparse.ArgumentParser(), command=neighbors_workflow.COMMAND
    )
    args = parser.parse_args([
        "--center", "Zn", "Mg", "--neighbor", "O",
        "--charge-source", "formal", "--formal-charge", "Zn=2", "Mg=2", "O=-2",
        "--periodic", "xy", "--frames", "0:5:2",
    ])
    assert neighbors_workflow.build_request(args) == WurtziteNeighborRequest(
        centers=("Zn", "Mg"), neighbors=("O",), protons=("H",),
        charge_source="formal", formal_charges={"Zn": 2.0, "Mg": 2.0, "O": -2.0},
        neighbor_cutoff=3.0, proton_cutoff=2.0, c_axis=(0.0, 0.0, 1.0),
        periodic=(True, True, False), cell_lengths=None,
        cell_angles=(90.0, 90.0, 90.0), frames=[0, 2, 4], every=1,
    )


def test_neighbor_and_polarity_workflows_write_expected_csvs(tmp_path) -> None:
    neighbor_output = tmp_path / "neighbors"
    neighbor_args = _base_args(
        neighbors_workflow, neighbors_workflow.COMMAND, tmp_path
    )
    neighbor_args.output_dir = neighbor_output
    assert neighbors_workflow.run_main(neighbors_workflow.COMMAND, neighbor_args) == 0
    assert (neighbor_output / "centers.csv").is_file()
    assert (neighbor_output / "neighbors.csv").is_file()

    polarity_output = tmp_path / "polarity"
    polarity_args = _base_args(
        polarity_workflow, polarity_workflow.COMMAND, tmp_path
    )
    polarity_args.output_dir = polarity_output
    assert polarity_workflow.run_main(polarity_workflow.COMMAND, polarity_args) == 0
    assert (polarity_output / "polarity.csv").is_file()
    assert (polarity_output / "polarity_variables.txt").is_file()
    neighbor_csv = polarity_output / "other_helpful_data" / "neighbors.csv"
    assert neighbor_csv.is_file()
    assert "ignored" in neighbor_csv.read_text(encoding="utf-8")


def test_trajectory_workflow_writes_workspace_and_requested_copy(tmp_path) -> None:
    args = _base_args(
        polarity_trajectory_workflow,
        polarity_trajectory_workflow.COMMAND,
        tmp_path,
    )
    requested = tmp_path / "copy" / "surface.extxyz"
    args.output = requested
    assert polarity_trajectory_workflow.run_main(
        polarity_trajectory_workflow.COMMAND, args
    ) == 0
    assert requested.is_file()
    workspace = (
        tmp_path / "workspace" / "analysis"
        / polarity_trajectory_workflow.COMMAND / "analysis"
    )
    assert (workspace / "surface.extxyz").is_file()
    assert (workspace / "polarity_variables.txt").is_file()
    for name in (
        "centers.csv", "neighbors.csv", "polarity.csv",
        "polarity_summary.csv", "proton_proximity_summary.csv",
    ):
        assert (workspace / "other_helpful_data" / name).is_file()


def test_polarization_workflow_writes_bins_summary_and_heatmap(tmp_path) -> None:
    output = tmp_path / "binned_polarization"
    args = _base_args(
        polarization_workflow, polarization_workflow.COMMAND, tmp_path
    )
    args.bins_x = 2
    args.volume_method = "cell"
    args.heatmaps = True
    args.global_scaling = True
    args.output_dir = output

    assert polarization_workflow.run_main(polarization_workflow.COMMAND, args) == 0
    assert (output / "binned_polarization.csv").is_file()
    assert (output / "polarization_summary.csv").is_file()
    assert (output / "heatmaps" / "P_z" / "frame_000000.png").is_file()


def test_frames_select_source_frames_and_use_quick_charge_reader(tmp_path, monkeypatch) -> None:
    def fail_materialized_read(*_args, **_kwargs):
        raise AssertionError("--frames must use the charge-only fort.7 reader.")

    monkeypatch.setattr(Fort7Handler, "_parse", fail_materialized_read)

    neighbor_output = tmp_path / "selected_neighbors"
    neighbor_args = _selected_reaxff_args(
        neighbors_workflow, neighbors_workflow.COMMAND, tmp_path, neighbor_output
    )
    assert neighbors_workflow.run_main(neighbors_workflow.COMMAND, neighbor_args) == 0
    with (neighbor_output / "centers.csv").open(encoding="utf-8", newline="") as handle:
        center_rows = list(csv.DictReader(handle))
    assert {row["frame_index"] for row in center_rows} == {"1"}
    assert {float(row["site_charge (e)"]) for row in center_rows} == {2.5}

    polarity_output = tmp_path / "selected_polarity"
    polarity_args = _selected_reaxff_args(
        polarity_workflow, polarity_workflow.COMMAND, tmp_path, polarity_output
    )
    assert polarity_workflow.run_main(polarity_workflow.COMMAND, polarity_args) == 0
    with (polarity_output / "polarity.csv").open(encoding="utf-8", newline="") as handle:
        polarity_rows = list(csv.DictReader(handle))
    assert {row["frame_index"] for row in polarity_rows} == {"1"}
    assert {row["polarity_label"] for row in polarity_rows} == {"UP"}

    polarization_output = tmp_path / "selected_polarization_bins"
    polarization_args = _selected_reaxff_args(
        polarization_workflow,
        polarization_workflow.COMMAND,
        tmp_path,
        polarization_output,
    )
    polarization_args.volume_method = "cell"
    assert polarization_workflow.run_main(
        polarization_workflow.COMMAND, polarization_args
    ) == 0
    with (polarization_output / "binned_polarization.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        polarization_bin_rows = list(csv.DictReader(handle))
    assert {row["frame_index"] for row in polarization_bin_rows} == {"1"}

    trajectory_copy = tmp_path / "selected_trajectory"
    trajectory_args = _selected_reaxff_args(
        polarity_trajectory_workflow,
        polarity_trajectory_workflow.COMMAND,
        tmp_path,
        trajectory_copy,
    )
    assert polarity_trajectory_workflow.run_main(
        polarity_trajectory_workflow.COMMAND, trajectory_args
    ) == 0
    trajectory_path = trajectory_copy / "three_folded_trajectory_with_polarity.extxyz"
    lines = trajectory_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7
    assert "frame=1 iter=10" in lines[1]
