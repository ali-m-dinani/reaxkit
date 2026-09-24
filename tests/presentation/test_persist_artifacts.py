from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import pandas as pd

from reaxkit.presentation.persist import persist_analysis_result


@dataclass
class _Result:
    table: pd.DataFrame
    details: pd.DataFrame

    @property
    def csv_tables(self):
        return {"summary": self.table, "details": self.details}

    artifact_tiers = {"summary": "core", "details": "detail"}
    artifact_defaults = {"summary": True, "details": False}


def _args(tmp_path, profile: str):
    return SimpleNamespace(
        project_root=str(tmp_path),
        run_id=f"run-{profile}",
        analysis_id=f"run-{profile}",
        output_profile=profile,
    )


def test_standard_profile_uses_atomic_writer_and_omits_opt_in_detail(tmp_path):
    result = _Result(
        table=pd.DataFrame({"frame_index": [0], "value": [1.0]}),
        details=pd.DataFrame({"frame_index": [0], "atom_id": [1]}),
    )

    output = persist_analysis_result("demo", result, _args(tmp_path, "standard"))

    assert (output / "summary.csv").is_file()
    assert not (output / "details.csv").exists()
    manifest = json.loads((output / "artifacts.json").read_text(encoding="utf-8"))
    statuses = {item["name"]: item["status"] for item in manifest["artifacts"]}
    assert statuses == {"summary": "written", "details": "omitted"}
    assert manifest["run_metadata"]["command"] == "demo"
    assert not list(output.glob(".rk-*.tmp"))


def test_full_profile_retains_declared_detail_artifacts(tmp_path):
    result = _Result(
        table=pd.DataFrame({"frame_index": [0], "value": [1.0]}),
        details=pd.DataFrame({"frame_index": [0], "atom_id": [1]}),
    )

    output = persist_analysis_result("demo", result, _args(tmp_path, "full"))

    assert (output / "summary.csv").is_file()
    assert (output / "details.parquet").is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(output / "details.parquet"), result.details)
