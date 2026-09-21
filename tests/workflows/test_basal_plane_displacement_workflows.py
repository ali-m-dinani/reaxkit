from __future__ import annotations

import argparse

from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.workflows.ferroelectrics.basal_plane_displacement_for_dipole_moment import (
    dipole_workflow,
    polarization_workflow,
)


def _input(tmp_path):
    (tmp_path / "xmolout").write_text(
        """5
sim 0 0 10 10 10 90 90 90
Al 0 0 0
N 1.6 0 -0.2
N -0.8 1.386 -0.2
N -0.8 -1.386 -0.2
N 0 0 1.8
""",
        encoding="utf-8",
    )


def _args(workflow, tmp_path, output):
    _input(tmp_path)
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    return parser.parse_args([
        "--engine", "reaxff", "--input", str(tmp_path),
        "--xmolout", str(tmp_path / "xmolout"),
        "--charge-source", "formal", "--formal-charge", "Al=3", "N=-3",
        "--periodic", "none", "--frames", "0", "--output-dir", str(output),
        "--project-root", str(tmp_path / "workspace"),
    ])


def test_commands_route_to_basal_plane_workflows() -> None:
    routes = get_registered_analysis_commands()
    assert routes[dipole_workflow.COMMAND].module_path.endswith("dipole_workflow")
    assert routes[polarization_workflow.COMMAND].module_path.endswith("polarization_workflow")


def test_dipole_workflow_writes_site_and_summary_tables(tmp_path) -> None:
    output = tmp_path / "dipole_output"
    args = _args(dipole_workflow, tmp_path, output)
    assert dipole_workflow.run_main(dipole_workflow.COMMAND, args) == 0
    assert (output / "basal_plane_dipoles.csv").is_file()
    assert (output / "basal_plane_ions.csv").is_file()
    assert (output / "basal_plane_dipole_summary.csv").is_file()


def test_polarization_workflow_writes_heatmap_and_supports_frames(tmp_path) -> None:
    output = tmp_path / "polarization_output"
    args = _args(polarization_workflow, tmp_path, output)
    args.volume_method = "cell"
    args.bins_x = 2
    args.heatmaps = True
    args.global_scaling = True
    assert polarization_workflow.run_main(polarization_workflow.COMMAND, args) == 0
    assert (output / "basal_plane_dipoles.csv").is_file()
    assert (output / "basal_plane_ions.csv").is_file()
    assert (output / "basal_plane_polarization.csv").is_file()
    assert (output / "basal_plane_polarization_summary.csv").is_file()
    assert (output / "heatmaps" / "P_z" / "frame_000000.png").is_file()
