from __future__ import annotations

import json
from pathlib import Path

import pytest

from reaxkit.engine.reaxff.io.geo_handler import GeoHandler


REPO_ROOT = Path(__file__).resolve().parents[4]
FORT90_PATH = REPO_ROOT / "examples_to_test" / "fort.90"
ARTIFACT_DIR = REPO_ROOT / "tests" / "artifacts" / "reaxf_io" / "fort90_geo_handler"


def test_geo_handler_parses_fort90_and_saves_artifacts() -> None:
    if not FORT90_PATH.exists():
        pytest.skip(f"Sample fort.90 file not found: {FORT90_PATH}")

    handler = GeoHandler(FORT90_PATH)
    atoms_df = handler.dataframe()
    connectivity_df = handler.connectivity()
    metadata = handler.metadata()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    atoms_path = ARTIFACT_DIR / "atoms.csv"
    connectivity_path = ARTIFACT_DIR / "connectivity.csv"
    metadata_path = ARTIFACT_DIR / "metadata.json"

    atoms_df.to_csv(atoms_path, index=False)
    connectivity_df.to_csv(connectivity_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    assert list(atoms_df.columns) == ["atom_id", "atom_type", "x", "y", "z"]
    assert list(connectivity_df.columns) == ["source_atom_id", "target_atom_id"]
    assert len(atoms_df) > 0
    assert len(connectivity_df) > 0
    assert metadata["n_atoms"] == len(atoms_df)
    assert metadata["n_connectivity_edges"] == len(connectivity_df)

    assert atoms_path.exists()
    assert connectivity_path.exists()
    assert metadata_path.exists()
