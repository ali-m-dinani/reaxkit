import json

import pandas as pd
import pytest

from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.cli.main import _ReaxKitArgumentParser
from reaxkit.workflows.ferroelectrics.hbn_reference.domain_wall_velocity_workflow import (
    ALL_COMMANDS, build_parser, run_main,
)


@pytest.mark.parametrize("command", [command for command in ALL_COMMANDS if "fraction" not in command])
def test_registered_csv_workflow(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    centers = []
    for frame in range(5):
        for spatial_bin in range(3):
            positive = max(0, min(100, 50 + 50 * (spatial_bin + 0.5 - frame)))
            rows.append(dict(frame_index=frame, iter=frame, time=frame, u_axis="x", v_axis="z",
                             u_bin=0, v_bin=spatial_bin, positive_count=positive,
                             negative_count=100-positive, zero_count=0, defined_group_count=100,
                             **{"u_center (angstrom)": 0, "v_center (angstrom)": spatial_bin}))
        for group_id in range(8):
            centers.append(dict(frame_index=frame, iter=frame, time=frame, local_group_id=group_id,
                                polarity=-1 if group_id <= frame else 1,
                                **{"reference_center_x (angstrom)": 0, "reference_center_y (angstrom)": 0,
                                   "reference_center_z (angstrom)": group_id}))
    pd.DataFrame(rows).to_csv(source / "hbn_reference_projected_polarity_2d.csv", index=False)
    pd.DataFrame(centers).to_csv(source / "hbn_reference_projected_polarity_centers.csv", index=False)
    parser = build_parser(_ReaxKitArgumentParser(f"reaxkit {command}"), command=command)
    assert "--axis" in parser.format_help()
    assert "--time-per-iteration" in parser.format_help(full=True)
    arguments = ["--input", str(source)]
    if "fraction" in command:
        arguments += ["--length", "10"]
    if "tracking" in command:
        arguments += ["--max-step", "1.1"]
    args = parser.parse_args(arguments)
    assert command in get_registered_analysis_commands()
    assert run_main(command, args) == 0
    destination = next((tmp_path / "reaxkit_workspace" / "other" / command).glob("run_*"))
    for filename in ["samples.csv", "velocities.csv", "diagnostics.csv"]:
        assert (destination / filename).is_file()
        pd.read_csv(destination / filename)
    assert json.loads((destination / "settings.json").read_text())["coordinate_system"] == "reference"
    assert (destination / "velocities.csv").read_text().count("ok") > 0
    interval_path = destination / "velocity_intervals.csv"
    if "tracking" in command:
        assert not interval_path.exists()
    else:
        intervals = pd.read_csv(interval_path)
        assert not intervals.empty
        assert "velocity_m_per_s" in intervals
        assert intervals["status"].eq("ok").any()
