from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows import active_site_workflow
from reaxkit.workflows.result_bundle import bundle_canonical_and_tract_tables


def test_bundle_canonical_and_tract_tables_persists_both_tables(tmp_path):
    raw = SimpleNamespace(
        canonical=pd.DataFrame({"atom_id": [1, 2], "value": [10, 20]}),
        tract=pd.DataFrame({"atom_id": [1, 2], "label": ["a", "b"]}),
    )
    bundled = bundle_canonical_and_tract_tables(raw, canonical_attr="canonical", tract_attr="tract")
    args = argparse.Namespace(
        plot=None,
        show=False,
        save=None,
        export=None,
        run_id="run_bundle",
        project_root=str(tmp_path),
        analysis_id=None,
    )

    present_result("active_site_structural", bundled, args)

    out_dir = tmp_path / "analysis" / "active_site_structural" / "run_bundle"
    assert (out_dir / "table.csv").exists()
    assert (out_dir / "tract_table.csv").exists()


def test_bundle_canonical_and_tract_tables_persists_soap_descriptors_npy(tmp_path):
    soap = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    raw = SimpleNamespace(
        canonical=pd.DataFrame({"atom_id": [1, 2], "value": [10, 20]}),
        tract=pd.DataFrame({"atom_id": [1, 2], "label": ["a", "b"]}),
        soap_descriptors=soap,
    )
    bundled = bundle_canonical_and_tract_tables(raw, canonical_attr="canonical", tract_attr="tract")
    args = argparse.Namespace(
        plot=None,
        show=False,
        save=None,
        export=None,
        run_id="run_bundle",
        project_root=str(tmp_path),
        analysis_id=None,
    )

    present_result("active_site_structural", bundled, args)

    out_dir = tmp_path / "analysis" / "active_site_structural" / "run_bundle"
    npy_path = out_dir / "soap_descriptors.npy"
    assert npy_path.exists()
    loaded = np.load(npy_path)
    assert np.array_equal(loaded, soap)


def test_active_site_structural_persists_tract_style_figures(tmp_path):
    table = pd.DataFrame(
        {
            "atom_id": [1, 2, 3, 4],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "d_pyr": [0.1, 0.2, 0.05, 0.0],
            "is_undercoord": [False, True, False, False],
            "label": ["basal", "edge_armchair", "defect", "edge_zigzag"],
            "grain_id": [0, 0, 1, -1],
        }
    )
    raw = SimpleNamespace(
        canonical=table,
        tract=table.copy(),
    )
    bundled = bundle_canonical_and_tract_tables(raw, canonical_attr="canonical", tract_attr="tract")
    args = argparse.Namespace(
        plot=None,
        show=False,
        save=None,
        export=None,
        run_id="run_bundle",
        project_root=str(tmp_path),
        analysis_id=None,
    )

    present_result("active_site_structural", bundled, args)

    out_dir = tmp_path / "analysis" / "active_site_structural" / "run_bundle"
    assert (out_dir / "run_bundle_dpyr_map.png").exists()
    assert (out_dir / "run_bundle_label_map.png").exists()
    assert (out_dir / "run_bundle_grain_map.png").exists()
    assert (out_dir / "run_bundle_dpyr_hist.png").exists()


def test_active_site_structural_writes_pdf_report(tmp_path):
    table = pd.DataFrame(
        {
            "atom_id": [1, 2, 3, 4],
            "element": ["C", "C", "C", "C"],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "n_bonds": [3, 2, 3, 2],
            "is_undercoord": [False, True, False, True],
            "d_pyr": [0.1, 0.0, 0.25, 0.0],
            "defect_type": ["none", "none", "haeckelite_like", "non6_cluster"],
            "label": ["basal", "edge_armchair", "defect", "edge_zigzag"],
            "grain_id": [0, 0, 1, -1],
        }
    )
    raw = SimpleNamespace(
        canonical=table,
        tract=table.copy(),
        summary={
            "is_periodic": True,
            "ring_histogram": {"6": 10, "5": 2, "7": 1},
            "defect_cluster_counts": {"haeckelite_like": 1, "non6_cluster": 1},
            "n_grains": 2,
            "n_bonds_total": 5,
            "dpyr_stats": {"mean_abs": 0.175, "median_abs": 0.175, "frac_above_tau": 0.5},
        },
        request=SimpleNamespace(carbon_element="C"),
    )
    bundled = bundle_canonical_and_tract_tables(raw, canonical_attr="canonical", tract_attr="tract")
    args = argparse.Namespace(
        plot=None,
        show=False,
        save=None,
        export=None,
        report=True,
        report_format="pdf",
        frame=0,
        xmolout="xmolout",
        input=".",
        run_id="run_bundle",
        project_root=str(tmp_path),
        analysis_id=None,
    )

    present_result(
        "active_site_structural",
        bundled,
        args,
    )

    analysis_dir = tmp_path / "analysis" / "active_site_structural" / "run_bundle"
    report_dir = tmp_path / "reports" / "active_site_structural" / "run_bundle"
    assert (report_dir / "run_bundle_report.pdf").exists()
    assert (report_dir / "run_bundle_report_data.json").exists()

    settings = json.loads((analysis_dir / "settings.json").read_text(encoding="utf-8"))
    reports = (settings.get("artifacts") or {}).get("reports") or []
    assert "run_bundle_report.pdf" in reports


def test_active_site_events_writes_pdf_report(tmp_path):
    table = pd.DataFrame(
        {
            "atom_id": [10, 11, 12],
            "element": ["C", "C", "C"],
            "n_events_O": [2, 0, 1],
            "n_events_Si": [0, 1, 0],
            "is_reactive_O": [True, False, True],
            "is_reactive_Si": [False, True, False],
            "is_reactive_any": [True, True, True],
            "total_bound_frames_O": [4, 0, 2],
            "total_bound_frames_Si": [0, 3, 0],
            "mean_contact_O_when_bound": [1.55, np.nan, 1.60],
            "mean_contact_Si_when_bound": [np.nan, 2.05, np.nan],
        }
    )
    raw = SimpleNamespace(
        canonical=table,
        tract=table.copy(),
        summary={
            "mode": "dist",
            "frames_analyzed": 20,
            "frame_first": 0,
            "frame_last": 95,
            "every": 5,
            "persist": 3,
            "n_carbon": 3,
            "n_reactive_O": 2,
            "n_reactive_Si": 1,
            "n_reactive_any": 3,
            "r_CO": 1.65,
            "r_CSi": 2.10,
        },
    )
    bundled = bundle_canonical_and_tract_tables(raw, canonical_attr="canonical", tract_attr="tract")
    args = argparse.Namespace(
        plot=None,
        show=False,
        save=None,
        export=None,
        report=True,
        report_format="pdf",
        xmolout="xmolout",
        input=".",
        run_id="run_events_bundle",
        project_root=str(tmp_path),
        analysis_id=None,
    )

    present_result("active_site_events", bundled, args)

    analysis_dir = tmp_path / "analysis" / "active_site_events" / "run_events_bundle"
    report_dir = tmp_path / "reports" / "active_site_events" / "run_events_bundle"
    assert (report_dir / "run_events_bundle_report.pdf").exists()
    report_data = report_dir / "run_events_bundle_report_data.json"
    assert report_data.exists()
    payload = json.loads(report_data.read_text(encoding="utf-8"))
    detailed = next((s for s in payload.get("sections", []) if s.get("title") == "Detailed Summary"), None)
    assert isinstance(detailed, dict)
    assert isinstance(detailed.get("table"), dict)
    headers = detailed["table"].get("headers") or []
    assert headers == ["Variable", "Value", "Description"]

    settings = json.loads((analysis_dir / "settings.json").read_text(encoding="utf-8"))
    reports = (settings.get("artifacts") or {}).get("reports") or []
    assert "run_events_bundle_report.pdf" in reports


def test_bundle_canonical_and_tract_tables_requires_dataframes():
    raw = SimpleNamespace(canonical=[1, 2, 3], tract=pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="Expected `canonical` to be a pandas DataFrame"):
        bundle_canonical_and_tract_tables(raw, canonical_attr="canonical", tract_attr="tract")
