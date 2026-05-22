from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reaxkit.domain.data_models import GeometryData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def _examples_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "examples_to_test"


def test_reaxff_adapter_maps_geo_to_geometry_data_with_initial_role():
    examples = _examples_dir()
    geo_path = examples / "geo"
    if not geo_path.exists():
        pytest.skip(f"Sample geo file not found: {geo_path}")

    adapter = ReaxFFAdapter()
    out = adapter.load_geometry({"geo": str(geo_path)})

    assert isinstance(out, GeometryData)
    assert isinstance(out.coordinates, pd.DataFrame)
    assert isinstance(out.connectivity, pd.DataFrame)
    assert out.metadata is not None
    assert out.metadata.get("geometry_role") == "initial"
    assert str(out.metadata.get("source_file", "")).lower() == "geo"


def test_reaxff_adapter_maps_fort90_to_geometry_data_with_final_role():
    examples = _examples_dir()
    fort90_path = examples / "fort.90"
    if not fort90_path.exists():
        pytest.skip(f"Sample fort.90 file not found: {fort90_path}")

    adapter = ReaxFFAdapter()
    out = adapter.load_final_geometry({"final_geometry": str(fort90_path)})

    assert isinstance(out, GeometryData)
    assert isinstance(out.coordinates, pd.DataFrame)
    assert isinstance(out.connectivity, pd.DataFrame)
    assert not out.connectivity.empty
    assert list(out.connectivity.columns) == ["source_atom_id", "target_atom_id"]
    assert out.metadata is not None
    assert out.metadata.get("geometry_role") == "final"
    assert str(out.metadata.get("source_file", "")).lower() == "fort.90"


def test_reaxff_adapter_load_geometry_respects_final_geometry_role_flag():
    examples = _examples_dir()
    fort90_path = examples / "fort.90"
    if not fort90_path.exists():
        pytest.skip(f"Sample fort.90 file not found: {fort90_path}")

    adapter = ReaxFFAdapter()
    out = adapter.load_geometry(
        {
            "geometry_role": "final",
            "final_geometry": str(fort90_path),
        }
    )

    assert isinstance(out, GeometryData)
    assert out.metadata is not None
    assert out.metadata.get("geometry_role") == "final"
    assert str(out.metadata.get("source_file", "")).lower() == "fort.90"
