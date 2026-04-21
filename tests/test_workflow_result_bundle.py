from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from reaxkit.presentation.dispatcher import present_result
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


def test_bundle_canonical_and_tract_tables_requires_dataframes():
    raw = SimpleNamespace(canonical=[1, 2, 3], tract=pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError, match="Expected `canonical` to be a pandas DataFrame"):
        bundle_canonical_and_tract_tables(raw, canonical_attr="canonical", tract_attr="tract")
