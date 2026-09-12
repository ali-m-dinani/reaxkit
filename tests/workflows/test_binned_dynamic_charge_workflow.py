from __future__ import annotations

import argparse

import pandas as pd

from reaxkit.workflows.ferroelectrics.binned_dynamic_charge_workflow import (
    build_parser,
    build_request,
)
from reaxkit.workflows.ferroelectrics import binned_dynamic_charge_workflow as workflow


def test_parser_builds_three_dimensional_bin_request() -> None:
    parser = build_parser(
        argparse.ArgumentParser(), command="get_binned_dynamic_charges"
    )
    args = parser.parse_args(
        [
            "--bins-x",
            "2",
            "--plane",
            "xz",
            "--bins-z",
            "4",
            "--frames",
            "0:6:2",
            "--every",
            "2",
            "--average",
        ]
    )

    request = build_request(args)

    assert request.plane == "xz"
    assert (request.bins_x, request.bins_y, request.bins_z) == (2, None, 4)
    assert request.selected_frames == [0, 2, 4]
    assert request.every == 2
    assert request.average is True


def test_run_main_enables_quick_charge_streaming_and_progress(monkeypatch, tmp_path) -> None:
    parser = build_parser(argparse.ArgumentParser(), command=workflow.COMMAND)
    args = parser.parse_args([
        "--plane", "xz", "--bins-x", "2", "--bins-z", "3",
        "--skip-plots", "--output-dir", str(tmp_path),
    ])
    captured = {}

    def fake_run(_self, _task, request, runtime_args):
        captured["request"] = request
        captured["runtime_args"] = runtime_args
        return type("Result", (), {"table": pd.DataFrame()})()

    monkeypatch.setattr(workflow.AnalysisExecutor, "run", fake_run)
    monkeypatch.setattr(workflow, "present_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "_quick_n_frames_from_control", lambda _path: 17)

    assert workflow.run_main(workflow.COMMAND, args) == 0
    assert captured["runtime_args"]["_quick_charge_only"] is True
    assert captured["runtime_args"]["scope"] == "total"
    assert captured["runtime_args"]["frames"] is None
    assert captured["runtime_args"]["progress"] is True
    assert captured["request"]._expected_frames == 17
