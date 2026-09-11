"""Tests for the composite force-field optimization report workflow."""

from __future__ import annotations

import argparse
import json

from reaxkit.cli.main import _canonicalize_direct_command
from reaxkit.core.registry.analysis_cli_routing_registry import (
    get_registered_analysis_commands,
)
from reaxkit.workflows.force_field_opt import get_ffield_opt_report as workflow


def test_report_declares_its_command_and_complete_pipeline() -> None:
    assert workflow.ALL_COMMANDS == ("get-ffield-opt-report",)
    assert workflow.REPORT_COMMANDS == (
        "get_trainset_data",
        "get_trainset_group_comments",
        "get_ffield_opt_results",
        "get_ffield_opt_plots",
        "get_ffield_opt_bulk_modulus",
    )
    assert "get-ffield-opt-report" in get_registered_analysis_commands()
    assert _canonicalize_direct_command(
        ["reaxkit", "get_ffield_opt_report"]
    )[1] == "get-ffield-opt-report"


def test_report_parser_is_ready_for_the_current_directory() -> None:
    parser = workflow.build_parser(
        argparse.ArgumentParser(), command="get-ffield-opt-report"
    )

    args = parser.parse_args([])

    assert args.run_dir == "."
    assert args.fort99 == "fort.99"
    assert args.fort74 == "fort.74"
    assert args.trainset == "trainset.in"
    assert args.geo == "geo"
    assert "reaxkit get-ffield-opt-report" in parser.description


def test_report_runs_each_existing_workflow_and_writes_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    report_root = tmp_path / "report"
    calls: list[tuple[str, argparse.Namespace]] = []

    def record(command: str, args: argparse.Namespace) -> int:
        calls.append((command, args))
        return 0

    monkeypatch.setattr(workflow.trainset_workflow, "run_main", record)
    monkeypatch.setattr(workflow.ffield_workflow, "run_main", record)
    monkeypatch.setattr(workflow.plots_workflow, "run_main", record)

    def normalize(values: dict[str, object]) -> dict[str, object]:
        normalized = dict(values)
        normalized.update(
            {
                "run_id": "run-1",
                "analysis_id": "analysis-1",
                "project_root": str(tmp_path / "workspace"),
                "_snapshot_source_dir": str(tmp_path),
            }
        )
        return normalized

    monkeypatch.setattr(workflow, "normalize_storage_args", normalize)
    parser = workflow.build_parser(
        argparse.ArgumentParser(), command="get-ffield-opt-report"
    )
    args = parser.parse_args(["--output", str(report_root)])

    assert workflow.run_main("get-ffield-opt-report", args) == 0
    assert [command for command, _ in calls] == list(workflow.REPORT_COMMANDS)
    assert calls[0][1].export == str(report_root / "trainset_data")
    assert calls[1][1].export == str(report_root / "trainset_group_comments.csv")
    assert calls[2][1].export == str(report_root / "ffield_opt_results.csv")
    assert calls[3][1].output == str(report_root / "plots")
    assert calls[3][1].geo == str(tmp_path / "geo")
    assert calls[4][1].export == str(report_root / "bulk_modulus.csv")

    manifest = json.loads(
        (report_root / "report_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["delegated_commands"] == list(workflow.REPORT_COMMANDS)
    assert all(status["exit_code"] == 0 for status in manifest["statuses"])
