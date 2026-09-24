from __future__ import annotations

import json

import pandas as pd
import pytest

from reaxkit.core.runtime.artifacts import ArtifactSpec, ArtifactWriter


def test_csv_batches_are_published_atomically_with_manifest(tmp_path):
    spec = ArtifactSpec("series", "series.csv", "core", True, "csv", True)

    with ArtifactWriter(tmp_path, [spec]) as writer:
        sink = writer.sink("series")
        assert sink is not None
        sink.append(pd.DataFrame({"frame_index": [3], "value": [1.5]}))
        sink.append(pd.DataFrame({"frame_index": [7], "value": [2.5]}))
        assert not (tmp_path / "series.csv").exists()

    table = pd.read_csv(tmp_path / "series.csv")
    pd.testing.assert_frame_equal(
        table, pd.DataFrame({"frame_index": [3, 7], "value": [1.5, 2.5]})
    )
    manifest = json.loads((tmp_path / "reaxkit_artifacts.json").read_text())
    entry = manifest["artifacts"][0]
    assert entry["status"] == "written"
    assert entry["row_count"] == 2
    assert entry["source_frame_min"] == 3
    assert entry["source_frame_max"] == 7
    assert not list(tmp_path.glob(".rk-*.tmp"))


def test_failure_removes_temporary_artifacts(tmp_path):
    spec = ArtifactSpec("series", "series.csv", "core", True, "csv", True)

    with pytest.raises(RuntimeError, match="worker failed"):
        with ArtifactWriter(tmp_path, [spec]) as writer:
            writer.append("series", [{"frame_index": 0, "value": 1.0}])
            raise RuntimeError("worker failed")

    assert not (tmp_path / "series.csv").exists()
    assert not (tmp_path / "reaxkit_artifacts.json").exists()
    assert not list(tmp_path.glob(".rk-*.tmp"))


def test_standard_profile_omits_disabled_detail_table(tmp_path):
    specs = (
        ArtifactSpec("core", "core.csv", "core", True, "csv", True),
        ArtifactSpec("details", "details.parquet", "detail", False, "parquet", True),
    )
    with ArtifactWriter(tmp_path, specs, profile="standard") as writer:
        writer.append("core", [{"frame_index": 0}])

    assert (tmp_path / "core.csv").is_file()
    assert not (tmp_path / "details.parquet").exists()
    manifest = json.loads((tmp_path / "reaxkit_artifacts.json").read_text())
    details = next(item for item in manifest["artifacts"] if item["name"] == "details")
    assert details["status"] == "omitted"
    assert details["reason"] == "disabled_by_output_profile"


def test_parquet_sink_writes_multiple_batches(tmp_path):
    pytest.importorskip("pyarrow")
    spec = ArtifactSpec("details", "details.parquet", "detail", True, "parquet", True)

    with ArtifactWriter(tmp_path, [spec]) as writer:
        writer.append("details", [{"frame_index": 1, "value": 2.0}])
        writer.append("details", [{"frame_index": 2, "value": 3.0}])

    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "details.parquet"),
        pd.DataFrame({"frame_index": [1, 2], "value": [2.0, 3.0]}),
    )
