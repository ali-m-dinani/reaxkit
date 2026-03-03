from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd

from reaxkit.presentation import dispatcher
from reaxkit.workflows import trajectory_workflow


def _base_args(**overrides):
    values = {
        "engine": None,
        "run_dir": ".",
        "xmolout": "xmolout",
        "log": None,
        "plot": None,
        "show": False,
        "save": None,
        "export": None,
        "grid": None,
        "frames": None,
        "every": 1,
        "atom_ids": None,
        "atom_types": None,
        "dims": ("x", "y", "z"),
        "origin": "first",
        "atom_ids_a": None,
        "atom_ids_b": None,
        "atom_types_a": None,
        "atom_types_b": None,
        "bins": 200,
        "r_max": None,
        "backend": "freud",
        "property": None,
        "prop": None,
        "xaxis": "frame",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_main_resolves_alias_before_executor(monkeypatch):
    captured = {}

    def fake_run(self, task, request, args):
        captured["task_name"] = task.__class__.__name__
        captured["request_name"] = request.__class__.__name__
        return SimpleNamespace(table=pd.DataFrame({"iter": [0], "msd": [0.0]}))

    monkeypatch.setattr(trajectory_workflow.AnalysisExecutor, "run", fake_run)
    monkeypatch.setattr(trajectory_workflow, "present_result", lambda command, result, args, plot_payload_builder=None: None)

    rc = trajectory_workflow.run_main("mean-square-displacement", _base_args())

    assert rc == 0
    assert captured["task_name"] == "MSDTask"
    assert captured["request_name"] == "MSDRequest"


def test_run_main_rdf_property_uses_property_task(monkeypatch):
    captured = {}

    def fake_run(self, task, request, args):
        captured["task_name"] = task.__class__.__name__
        captured["request_name"] = request.__class__.__name__
        return SimpleNamespace(table=pd.DataFrame({"frame_index": [0], "area": [1.0]}))

    monkeypatch.setattr(trajectory_workflow.AnalysisExecutor, "run", fake_run)
    monkeypatch.setattr(trajectory_workflow, "present_result", lambda command, result, args, plot_payload_builder=None: None)

    rc = trajectory_workflow.run_main("rdf_property", _base_args(prop="area", atom_types_a=["Al"], atom_types_b=["N"]))

    assert rc == 0
    assert captured["task_name"] == "RDFPropertyTask"
    assert captured["request_name"] == "RDFPropertyRequest"


def test_present_result_exports_csv_and_renders_plot(monkeypatch, tmp_path):
    calls = {}

    def fake_plot(payload):
        calls["payload"] = payload
        return None

    monkeypatch.setattr(dispatcher, "render_plot", fake_plot)
    result = SimpleNamespace(table=pd.DataFrame({"r": [0.5, 1.0], "g": [0.9, 1.2]}))
    args = _base_args(
        plot="single",
        save=str(tmp_path / "rdf.png"),
        export=str(tmp_path / "rdf.csv"),
    )

    dispatcher.present_result("rdf", result, args, plot_payload_builder=trajectory_workflow._plot_payload)

    assert (tmp_path / "rdf.csv").exists()
    assert calls["payload"]["save"] == str(tmp_path / "rdf.png")
    assert calls["payload"]["plot_type"] == "single_plot"


def test_msd_subplot_payload_carries_grid():
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 1, 0, 1],
                "iter": [0, 1, 0, 1],
                "atom_id": [1, 1, 2, 2],
                "msd": [0.0, 1.0, 0.0, 2.0],
            }
        )
    )

    payload = trajectory_workflow._plot_payload("msd", result, _base_args(plot="subplot", grid="2x1"))

    assert payload["plot_type"] == "multi_subplots"
    assert payload["grid"] == "2x1"


def test_rdf_property_payload_uses_selected_property_and_xaxis():
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 1],
                "iter": [10, 20],
                "area": [1.5, 1.7],
            }
        )
    )

    payload = trajectory_workflow._plot_payload("rdf_property", result, _base_args(property="area", xaxis="iter"))

    assert payload["plot_type"] == "single_plot"
    assert payload["x"] == [10, 20]
    assert payload["y"] == [1.5, 1.7]
    assert payload["title"] == "RDF Area"


def test_rdf_single_plot_payload_builds_one_series_per_frame():
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 0, 1, 1],
                "iter": [10, 10, 20, 20],
                "r": [0.5, 1.0, 0.5, 1.0],
                "g": [0.8, 1.2, 0.9, 1.1],
            }
        )
    )

    payload = trajectory_workflow._plot_payload("rdf", result, _base_args(plot="single"))

    assert payload["plot_type"] == "single_plot"
    assert len(payload["series"]) == 2
    assert payload["series"][0]["label"] == "frame 0 (iter 10)"
    assert payload["series"][1]["label"] == "frame 1 (iter 20)"


def test_rdf_subplot_payload_builds_one_subplot_per_frame():
    result = SimpleNamespace(
        table=pd.DataFrame(
            {
                "frame_index": [0, 0, 1, 1],
                "iter": [10, 10, 20, 20],
                "r": [0.5, 1.0, 0.5, 1.0],
                "g": [0.8, 1.2, 0.9, 1.1],
            }
        )
    )

    payload = trajectory_workflow._plot_payload("rdf", result, _base_args(plot="subplot", grid="2x1"))

    assert payload["plot_type"] == "multi_subplots"
    assert payload["grid"] == "2x1"
    assert len(payload["subplots"]) == 2
