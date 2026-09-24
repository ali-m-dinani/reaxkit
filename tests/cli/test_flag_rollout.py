"""Route repairs, parser compatibility, and live help inventory coverage."""

import argparse
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reaxkit.cli.main import _ReaxKitArgumentParser, build_parser
from reaxkit.cli.inventory import inventory, iter_parsers
from reaxkit.cli.help_metadata import metadata_for_action


REPAIRED = {
    "kinematics": "get_kinematics",
    "largest_molecule_by_mass": "get_largest_molecule_by_mass",
    "largest_molecule_composition": "get_largest_molecule_composition",
    "molecule_lifetime": "get_molecule_lifetime",
    "voronoi": "get_voronoi",
    "connection_list": "get_connection_list",
    "connection_table": "get_connection_table",
    "connection_stats": "get_connection_stats",
    "bond_events": "get_bond_events",
    "coordination": "get_coordination",
    "coordination_relabel": "relabel_traj_using_coordination",
    "hybridization": "get_hybridization",
    "get-control": "get-control_data",
    "gen_control": "gen-control",
    "gen-control": "gen-control",
    "make-control": "gen_template_control",
    "write-control": "gen-control",
    "kinematics_plot3d": "kinematics_plot3d",
    "kinematics_heatmap2d": "kinematics_heatmap2d",
}


def selected(command):
    root = build_parser(command)
    return next(a for a in root._actions if isinstance(a, argparse._SubParsersAction)).choices[command]


@pytest.mark.parametrize("command", REPAIRED)
def test_repaired_route_help_and_namespace(command, capsys):
    parser = selected(command)
    for flag in ("-h", "--help-all", "--all-flags"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([flag])
        assert exc.value.code == 0
        assert "Usage:" in capsys.readouterr().out
    canonical = REPAIRED[command]
    flags = {
        "get_kinematics": ["--key", "velocities"],
        "get-control_data": ["nmdit"],
        "kinematics_plot3d": ["--value", "vz"],
        "kinematics_heatmap2d": ["--value", "vz"],
        "get_coordination": ["--valences", "Al=3,N=3"],
        "relabel_traj_using_coordination": ["--valences", "Al=3,N=3", "--output", "relabeled.xyz"],
        "get_hybridization": ["--hybridizations", "sp=1,sp2=2,sp3=3"],
    }.get(canonical, [])
    actual = vars(parser.parse_args(flags))
    expected = vars(selected(canonical).parse_args(flags))
    actual.pop("_run", None)
    expected.pop("_run", None)
    assert actual == expected


@pytest.mark.parametrize("engine,source", [("reaxff", "custom/coordinates.xyz"), ("ams", "custom/reaxout.kf")])
def test_help_parser_preserves_namespace(engine, source):
    workflow = import_module("reaxkit.workflows.ferroelectrics.hbn_reference.projected_polarity_workflow")
    from reaxkit.core.runtime.cli_policy import add_execution_arguments

    argv = ["--engine", engine, "--input", source, "--replication", "2", "3", "4",
            "--xmolout", "renamed.xyz", "--fort7", "renamed.bo", "--frames", "0:20:2",
            "--execution", "threads", "--workers", "2", "--chunk-size", "3",
            "--project-root", "custom-workspace", "--run-id", "comparison",
            "--analysis-id", "analysis", "--no-input-cache", "--output-profile", "minimal"]
    namespaces = []
    for parser_cls in (argparse.ArgumentParser, _ReaxKitArgumentParser):
        parser = parser_cls()
        workflow.build_parser(parser, command=workflow.COMMAND)
        add_execution_arguments(parser)
        namespaces.append(vars(parser.parse_args(argv)))
    assert namespaces[0] == namespaces[1]


def test_live_inventory_matches_snapshot_and_help_covers_every_action(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    expected = json.loads((root / "cli-help-inventory.json").read_text(encoding="utf-8"))
    assert inventory() == expected
    assert json.loads((root / "cli-help-inventory.errors.json").read_text()) == {}

    # Inspect rendered logical rows, independently of terminal wrapping.
    captured = []
    def capture(headers, rows, width, wrap_cols):
        if headers[0] == "Flag":
            captured.extend(row[0] for row in rows)
        return "table"
    monkeypatch.setattr(_ReaxKitArgumentParser, "_render_table", staticmethod(capture))
    for path, parser in iter_parsers():
        actions = [a for a in parser._actions if not isinstance(a, argparse._SubParsersAction)
                   and a.help != argparse.SUPPRESS]
        for full in (False, True):
            captured.clear()
            parser.format_help(full=full)
            expected_flags = [parser._normalize(parser._format_flags(a)) for a in actions
                              if metadata_for_action(a, parser)[1] != "internal"
                              and (full or metadata_for_action(a, parser)[1] == "short")]
            assert sorted(captured) == sorted(expected_flags), (path, full)


@pytest.mark.parametrize("command", ["kinematics_plot3d", "kinematics_heatmap2d"])
def test_restored_spatial_parser_reaches_plot_runner(command, monkeypatch, tmp_path):
    workflow = import_module("reaxkit.workflows.kinematics_workflow")
    parser = selected(command)
    args = parser.parse_args(["--value", "vz", "--atoms", "1", "2", "--project-root", str(tmp_path)])
    requests = []
    class Executor:
        def run(self, task, request, settings):
            requests.append(request)
            table = pd.DataFrame({"atom_index": [1, 2], "x": [0., 1.], "y": [0., 1.],
                                  "z": [0., 1.]}) if request.key == "coordinates" else pd.DataFrame(
                                      {"atom_index": [1, 2], "vz": [2., 3.]})
            return SimpleNamespace(table=table)
    plots = []
    monkeypatch.setattr(workflow, "AnalysisExecutor", Executor)
    monkeypatch.setattr(workflow, "scatter3d_points", lambda *a, **kw: plots.append((a, kw)))
    monkeypatch.setattr(workflow, "heatmap2d_from_3d", lambda *a, **kw: plots.append((a, kw)))
    assert args._run(args) == 0
    assert [r.key for r in requests] == ["coordinates", "velocities"]
    assert plots[0][0][1].tolist() == [2., 3.]
    if command.endswith("heatmap2d"):
        assert plots[0][1]["plane"] == "xy"
        assert plots[0][1]["bins"] == 100


@pytest.mark.parametrize("command,runner", [("get-control", "_run_get"), ("make-control", "_run_make"),
                                          ("write-control", "_run_write"), ("gen_control", "_run_write")])
def test_control_alias_dispatches_to_intended_operation(command, runner, monkeypatch):
    workflow = import_module("reaxkit.workflows.file_tools.control_workflow")
    called = []
    monkeypatch.setattr(workflow, runner, lambda args: called.append(args) or 0)
    args = selected(command).parse_args(["nmdit"] if command == "get-control" else [])
    assert args._run(args) == 0
    assert called == [args]


def test_control_aliases_generate_modify_and_read_real_files(tmp_path, capsys):
    common = ["--project-root", str(tmp_path / "workspace")]
    args = selected("make-control").parse_args([
        *common, "--output", "source.control", "--parameter", "nmdit", "--value", "123"])
    assert args._run(args) == 0
    source = next(tmp_path.rglob("source.control"))
    args = selected("write-control").parse_args([
        *common, "--input", str(source.parent), "--control", str(source),
        "--output", "updated.control", "--parameter", "nmdit", "--value", "321"])
    assert args._run(args) == 0
    updated = next(tmp_path.rglob("updated.control"))
    capsys.readouterr()
    args = selected("get-control").parse_args([
        *common, "nmdit", "--input", str(updated.parent), "--control", str(updated), "--engine", "reaxff"])
    assert args._run(args) == 0
    assert "nmdit = 321" in capsys.readouterr().out
