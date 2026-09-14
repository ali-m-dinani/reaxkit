from __future__ import annotations

import argparse

import pytest

from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler

from reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors import WurtziteNeighborRequest
from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite import neighbors_workflow
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite import polarity_trajectory_workflow
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite import polarity_workflow


@pytest.mark.parametrize(
    ("workflow", "command"),
    [
        (neighbors_workflow, "get-wurtzite-neighbors"),
        (polarity_workflow, "get-wurtzite-polarity"),
        (polarity_trajectory_workflow, "write-trajectory-with-polarity"),
    ],
)
def test_wurtzite_parser_descriptions_explain_usage_and_include_examples(
        workflow, command
) -> None:
    parser = workflow.build_parser(argparse.ArgumentParser(), command=command)

    assert parser.formatter_class is argparse.RawTextHelpFormatter
    assert parser.description.startswith(("Find", "Calculate", "Write"))
    assert "\n\nExamples:\n" in parser.description
    assert f"reaxkit {command}" in parser.description


def test_wurtzite_commands_are_registered_in_dedicated_workflows() -> None:
    routes = get_registered_analysis_commands()
    assert routes["get-wurtzite-neighbors"].module_path.endswith("neighbors_workflow")
    assert routes["get-wurtzite-polarity"].module_path.endswith("polarity_workflow")
    assert routes["write-trajectory-with-polarity"].module_path.endswith(
        "polarity_trajectory_workflow"
    )


def test_neighbor_parser_supports_formal_charges_and_geometry_flags() -> None:
    parser = neighbors_workflow.build_parser(
        argparse.ArgumentParser(), command=neighbors_workflow.COMMAND
    )
    args = parser.parse_args([
        "--center", "Zn", "Mg",
        "--neighbor", "O",
        "--charge-source", "formal",
        "--formal-charge", "Zn=2",
        "--formal-charge", "Mg=2",
        "--formal-charge", "O=-2",
        "--periodic", "xz",
        "--frames", "0:5:2",
    ])
    request = neighbors_workflow.build_request(args)

    assert request == WurtziteNeighborRequest(
        centers=("Zn", "Mg"),
        neighbors=("O",),
        protons=("H",),
        charge_source="formal",
        formal_charges={"Zn": 2.0, "Mg": 2.0, "O": -2.0},
        neighbor_cutoff=3.0,
        proton_cutoff=2.0,
        c_axis=(0.0, 0.0, 1.0),
        periodic=(True, False, True),
        cell_lengths=None,
        cell_angles=(90.0, 90.0, 90.0),
        frames=[0, 2, 4],
        every=1,
    )


def test_polarity_workflow_exposes_plot_modes() -> None:
    parser = polarity_workflow.build_parser(
        argparse.ArgumentParser(), command=polarity_workflow.COMMAND
    )
    args = parser.parse_args([
        "--gen-plots", "--plane", "xz", "--plot-value", "basal-difference",
        "--slice-range", "0", "3.2",
    ])
    assert args.gen_plots is True
    assert args.plane == "xz"
    assert args.plot_value == "basal-difference"


def test_polarity_trajectory_default_name(tmp_path) -> None:
    parser = polarity_trajectory_workflow.build_parser(
        argparse.ArgumentParser(), command=polarity_trajectory_workflow.COMMAND
    )
    args = parser.parse_args([
        "--output-dir", str(tmp_path / "copy"),
        "--project-root", str(tmp_path / "reaxkit_workspace"),
    ])
    assert polarity_trajectory_workflow._output_path(args) == (
            tmp_path
            / "reaxkit_workspace"
            / "analysis"
            / "write-trajectory-with-polarity"
            / "analysis"
            / "trajectory_with_polarity.extxyz"
    ).resolve()
    assert polarity_trajectory_workflow._requested_output_path(args) == (
            tmp_path / "copy" / "trajectory_with_polarity.extxyz"
    ).resolve()


def test_neighbors_workflow_writes_only_compact_named_csvs(tmp_path) -> None:
    xmolout = tmp_path / "xmolout"
    xmolout.write_text(
        """5
sim 0 0 20 20 20 90 90 90
Al 0 0 0
N 0 0 1.8
N 1.7 0 -0.5
N -0.85 1.472 -0.5
N -0.85 -1.472 -0.5
""",
        encoding="utf-8",
    )
    output = tmp_path / "neighbors_output"
    parser = neighbors_workflow.build_parser(
        argparse.ArgumentParser(), command=neighbors_workflow.COMMAND
    )
    args = parser.parse_args(
        [
            "--engine",
            "reaxff",
            "--input",
            str(tmp_path),
            "--xmolout",
            str(xmolout),
            "--charge-source",
            "formal",
            "--formal-charge",
            "Al=3",
            "N=-3",
            "--periodic",
            "none",
            "--output-dir",
            str(output),
            "--project-root",
            str(tmp_path / "reaxkit_workspace"),
        ]
    )

    assert neighbors_workflow.run_main(neighbors_workflow.COMMAND, args) == 0
    assert (output / "centers.csv").is_file()
    assert (output / "neighbors.csv").is_file()
    assert not (output / "centers_and_neighbors.csv").exists()
    assert not (output / "table.csv").exists()

    center_header = (output / "centers.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "proton_cutoff (angstrom)" not in center_header
    assert "charge_source" not in center_header
    assert "neighbor_cutoff (angstrom)" not in center_header


def test_formal_charge_mode_has_no_implicit_anion_flag_or_fallback() -> None:
    parser = polarity_trajectory_workflow.build_parser(
        argparse.ArgumentParser(), command=polarity_trajectory_workflow.COMMAND
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--anion-formal-charge", "-3"])

    args = parser.parse_args(["--charge-source", "formal", "--formal-charge", "N=-3"])
    with pytest.raises(ValueError, match="missing: Al"):
        polarity_trajectory_workflow.build_request(args)


def test_polarity_trajectory_workflow_uses_quick_charges_and_preserves_names(
        tmp_path, monkeypatch
) -> None:
    xmolout = tmp_path / "xmolout"
    xmolout.write_text(
        """6
sim 0 0 20 20 20 90 90 90
Al 0 0 0
N 0 0 1.8
N 1.7 0 -0.5
N -0.85 1.472 -0.5
N -0.85 -1.472 -0.5
H 0 0 1.0
""",
        encoding="utf-8",
    )
    fort7 = tmp_path / "fort.7"
    fort7.write_text(
        """6 sim Iteration: 0 #Bonds: 0
1 1 1 0.0 0.0 3.0
2 2 1 0.0 0.0 -0.7
3 2 1 0.0 0.0 -0.8
4 2 1 0.0 0.0 -0.9
5 2 1 0.0 0.0 -1.0
6 3 1 0.0 0.0 0.4
0.0 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    output = tmp_path / "polarity.extxyz"
    parser = polarity_trajectory_workflow.build_parser(
        argparse.ArgumentParser(), command=polarity_trajectory_workflow.COMMAND
    )
    args = parser.parse_args([
        "--engine", "reaxff",
        "--input", str(tmp_path),
        "--xmolout", str(xmolout),
        "--fort7", str(fort7),
        "--charge-source", "reaxff",
        "--periodic", "none",
        "--output", str(output),
        "--project-root", str(tmp_path / "project"),
    ])

    def fail_materialized_read(*_args, **_kwargs):
        raise AssertionError("The polarity workflow must not materialize full fort.7 tables.")

    monkeypatch.setattr(Fort7Handler, "_parse", fail_materialized_read)
    assert polarity_trajectory_workflow.run_main(polarity_trajectory_workflow.COMMAND, args) == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in lines[2:8]] == ["Al", "N", "N", "N", "N", "H"]
    assert float(lines[3].split()[5]) == -0.7
    assert ":polarity:I:1" in lines[1]
    assert "frame=0 iter=0" in lines[1]
    workspace_dir = (
            tmp_path
            / "project"
            / "analysis"
            / "write-trajectory-with-polarity"
            / "analysis"
    )
    assert (workspace_dir / "polarity.extxyz").is_file()
    assert (workspace_dir / "polarity_variables.txt").is_file()
    helpful = workspace_dir / "other_helpful_data"
    for name in (
            "centers.csv",
            "neighbors.csv",
            "polarity.csv",
            "polarity_summary.csv",
            "proton_proximity_summary.csv",
    ):
        assert (helpful / name).is_file()
    assert not (workspace_dir / "centers.csv").exists()
    assert not (workspace_dir / "centers_and_neighbors.csv").exists()
    assert not (workspace_dir / "polarity.csv").exists()
    assert not (workspace_dir / "polarity_summary.csv").exists()
    assert not (workspace_dir / "proton_proximity_summary.csv").exists()
    assert not (workspace_dir / "apical_neighbors.csv").exists()
    assert not (workspace_dir / "basal_neighbors.csv").exists()
