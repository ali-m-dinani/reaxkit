from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

from reaxkit.workflows import active_site_workflow


def _base_args(**overrides):
    values = {
        "engine": None,
        "input": ".",
        "run_dir": ".",
        "fort7": "fort.7",
        "xmolout": "xmolout",
        "summary": None,
        "log": None,
        "plot": None,
        "show": False,
        "save": None,
        "export": None,
        "grid": None,
        "report": False,
        "report_format": "both",
        "frame": 0,
        "bo_threshold": 0.3,
        "bond_mode": "bo",
        "bond_scale": 1.2,
        "alpha_radius": 0.0,
        "gap_deg": 220.0,
        "carbon_element": "C",
        "include_noncarbon": True,
        "strict_tract": False,
        "soap": False,
        "soap_ref_path": None,
        "soap_r_cut": 5.0,
        "soap_n_max": 9,
        "soap_l_max": 9,
        "soap_zeta": 2,
        "frames": None,
        "every": 1,
        "mode": "auto",
        "r_co": 1.65,
        "r_csi": 2.10,
        "persist": 50,
        "oxygen_element": "O",
        "silicon_element": "Si",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_active_site_structural_run_main_bundles_tables_and_passes_strict(monkeypatch):
    captured = {}

    def fake_run(self, task, request, args):
        captured["task_name"] = task.__class__.__name__
        captured["request"] = request
        captured["args"] = dict(args)
        return SimpleNamespace(
            table=pd.DataFrame({"atom_id": [1], "d_pyr": [0.1], "label": ["basal"]}),
            tract_table=pd.DataFrame({"atom_id": [1]}),
            summary={},
            request=request,
            soap_descriptors=np.asarray([[0.1, 0.2]], dtype=float),
        )

    def fake_present(command, result, args, plot_payload_builder=None, report_payload_builder=None):
        captured["command"] = command
        captured["present_result"] = result
        captured["present_args"] = args
        captured["plot_payload_builder"] = plot_payload_builder

    monkeypatch.setattr(active_site_workflow.AnalysisExecutor, "run", fake_run)
    monkeypatch.setattr(active_site_workflow, "present_result", fake_present)

    rc = active_site_workflow.run_main(
        "active_site_structural",
        _base_args(strict_tract=True, frame=3, bond_mode="distance", bond_scale=1.2),
    )

    assert rc == 0
    assert captured["command"] == "active_site_structural"
    assert captured["task_name"] == "ActiveSiteStructuralTask"
    assert captured["request"].strict_tract is True
    assert captured["request"].frame == 3
    assert captured["request"].bond_mode == "distance"
    assert captured["request"].bond_scale == 1.2
    assert isinstance(captured["present_result"].table, pd.DataFrame)
    assert isinstance(captured["present_result"].tract_table, pd.DataFrame)
    assert isinstance(captured["present_result"].soap_descriptors, np.ndarray)


def test_active_site_events_run_main_bundles_tables_and_passes_strict(monkeypatch):
    captured = {}

    def fake_run(self, task, request, args):
        captured["task_name"] = task.__class__.__name__
        captured["request"] = request
        return SimpleNamespace(
            table=pd.DataFrame({"atom_id": [1], "n_events_O": [1], "n_events_Si": [0]}),
            tract_table=pd.DataFrame({"atom_id": [1]}),
            summary={},
            request=request,
        )

    def fake_present(command, result, args, plot_payload_builder=None, report_payload_builder=None):
        captured["command"] = command
        captured["present_result"] = result

    monkeypatch.setattr(active_site_workflow.AnalysisExecutor, "run", fake_run)
    monkeypatch.setattr(active_site_workflow, "present_result", fake_present)

    rc = active_site_workflow.run_main(
        "active_site_events",
        _base_args(strict_tract=True, frames=["0:20:5"], every=2, mode="bo"),
    )

    assert rc == 0
    assert captured["command"] == "active_site_events"
    assert captured["task_name"] == "ActiveSiteEventsTask"
    assert captured["request"].strict_tract is True
    assert captured["request"].every == 2
    assert captured["request"].mode == "bo"
    assert isinstance(captured["present_result"].table, pd.DataFrame)
    assert isinstance(captured["present_result"].tract_table, pd.DataFrame)
