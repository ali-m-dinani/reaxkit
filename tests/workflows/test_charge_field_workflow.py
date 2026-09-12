from __future__ import annotations

import argparse

import numpy as np

from reaxkit.analysis.ferroelectrics.charge_field import (
    ChargeFieldRequest,
    calculate_charge_field_response,
)
from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.domain.data_models import ChargeData, ElectricFieldData, SimulationData
from reaxkit.workflows.ferroelectrics import charge_field_workflow as workflow


def _result():
    charges = ChargeData(
        charges=np.asarray([[0.1], [0.25]]),
        iterations=np.asarray([0, 10]),
        simulation=SimulationData(atom_ids=[7], elements=["Ti"]),
    )
    field = ElectricFieldData(
        applied_field_values=np.asarray([[0.0], [0.2]]),
        applied_field_components=("field_z",),
        sampled_field_iterations=np.asarray([0, 10]),
    )
    return calculate_charge_field_response(
        charges,
        field,
        ChargeFieldRequest(atom_numbers=[7]),
    )


def test_parser_requires_atoms_and_registers_command() -> None:
    parser = workflow.build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args(
        ["--atom-numbers", "7", "9", "--frames", "0:5:2", "--field-direction", "x"]
    )

    assert workflow.build_request(args) == ChargeFieldRequest(
        atom_numbers=(7, 9),
        frames=[0, 2, 4],
        every=1,
        field_direction="x",
    )
    route = get_registered_analysis_commands()[workflow.COMMAND]
    assert route.module_path == "reaxkit.workflows.ferroelectrics.charge_field_workflow"


def test_dual_axis_plot_and_csv_are_written(tmp_path) -> None:
    result = _result()
    written = workflow.generate_charge_field_plots(
        result,
        tmp_path,
        x_axis="iter",
        dpi=72,
        progress=False,
    )
    result.table.to_csv(tmp_path / "charge_vs_electric_field.csv", index=False)

    assert written == [
        tmp_path / "plots" / "charges" / "atom_7_Ti.png",
        tmp_path / "plots" / "delta_charge" / "atom_7_Ti.png",
    ]
    assert all(path.is_file() for path in written)
    assert (tmp_path / "charge_vs_electric_field.csv").is_file()
    assert np.allclose(result.table["delta_charge"], [0.0, 0.15])
