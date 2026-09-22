from __future__ import annotations

import argparse

import numpy as np
import pytest

from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.workflows.ferroelectrics.basal_plane_displacement_for_dipole_moment import (
    local_polarization_workflow,
)
from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.local_polarization import (
    BasalPlaneLocalPolarizationRequest,
    calculate_basal_plane_local_polarization,
)
from reaxkit.core.platform.constants import const
from reaxkit.domain.data_models import TrajectoryData


def test_command_routes_to_local_polarization_workflow() -> None:
    route = get_registered_analysis_commands()[local_polarization_workflow.COMMAND]
    assert route.module_path.endswith("local_polarization_workflow")


def test_parser_defaults_to_equal_hull_volume() -> None:
    parser = local_polarization_workflow.build_parser(
        argparse.ArgumentParser(), command=local_polarization_workflow.COMMAND
    )

    request = local_polarization_workflow.build_request(parser.parse_args([]))

    assert request.local_volume_method == "equal"
    assert request.volume_method == "hull"


def test_parser_accepts_coordination_volume() -> None:
    parser = local_polarization_workflow.build_parser(
        argparse.ArgumentParser(), command=local_polarization_workflow.COMMAND
    )
    args = parser.parse_args(
        [
            "--local-volume-method",
            "coordination",
            "--volume-method",
            "cell",
            "--charge-source",
            "formal",
            "--formal-charge",
            "Al=3",
            "N=-3",
            "--write-extxyz",
            "--include-electric-field",
            "--field-direction",
            "x",
        ]
    )

    request = local_polarization_workflow.build_request(args)

    assert request.local_volume_method == "coordination"
    assert request.volume_method == "cell"
    assert request.formal_charges == {"Al": 3.0, "N": -3.0}
    assert request.include_electric_field
    assert request.field_direction == "x"


def _result():
    trajectory = TrajectoryData(
        positions=np.asarray([[
            [0.0, 0.0, 0.0],
            [1.6, 0.0, -0.2],
            [-0.8, 1.386, -0.2],
            [-0.8, -1.386, -0.2],
            [0.0, 0.0, 1.8],
        ]]),
        elements=["Al", "N", "N", "N", "N"],
        atom_ids=[1, 2, 3, 4, 5],
        iterations=np.asarray([10]),
    )
    return calculate_basal_plane_local_polarization(
        trajectory,
        BasalPlaneLocalPolarizationRequest(
            periodic=(False, False, False),
            charge_source="formal",
            formal_charges={"Al": 3.0, "N": -3.0},
            cell_lengths=(10.0, 10.0, 10.0),
            volume_method="cell",
        ),
    )


def test_2d_aggregation_sums_dipole_and_volume_along_omitted_axis() -> None:
    result = _result()

    _, _, polarization = local_polarization_workflow.aggregate_local_values_2d(
        result, 0, plane="xy", component="z", quantity="polarization", bins=(1, 1)
    )
    _, _, dipole = local_polarization_workflow.aggregate_local_values_2d(
        result, 0, plane="xy", component="z", quantity="dipole", bins=(1, 1)
    )

    assert dipole[0, 0] == pytest.approx(-0.6)
    assert polarization[0, 0] == pytest.approx(
        -0.6 / 1000.0 * float(const("ea3_to_uC_cm2"))
    )


def test_plot_generators_write_2d_and_3d_frames(tmp_path) -> None:
    result = _result()

    two_dimensional = local_polarization_workflow.generate_local_2d_plots(
        result, tmp_path, plane="xy", component="z", quantity="polarization",
        bins=(1, 1), global_scaling=False, dpi=72,
    )
    three_dimensional = local_polarization_workflow.generate_local_3d_plots(
        result, tmp_path, component="z", quantity="polarization",
        global_scaling=False, dpi=72,
    )

    assert len(two_dimensional) == 1 and two_dimensional[0].is_file()
    assert len(three_dimensional) == 1 and three_dimensional[0].is_file()


def test_plot_generators_use_restored_source_frame_numbers(tmp_path) -> None:
    result = _result()
    result.trajectory.source_frame_indices = np.asarray([50])
    for table in (result.table, result.summary, result.dipole_result.table):
        table["frame_index"] = 50

    two_dimensional = local_polarization_workflow.generate_local_2d_plots(
        result, tmp_path, plane="xy", component="z", quantity="polarization",
        bins=(1, 1), global_scaling=False, dpi=72,
    )
    three_dimensional = local_polarization_workflow.generate_local_3d_plots(
        result, tmp_path, component="z", quantity="polarization",
        global_scaling=False, dpi=72,
    )

    assert two_dimensional[0].name == "frame_000050.png"
    assert three_dimensional[0].name == "frame_000050.png"
