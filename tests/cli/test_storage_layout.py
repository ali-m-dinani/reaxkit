from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from reaxkit.cli.path import resolve_output_path
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, normalize_storage_args
from reaxkit.presentation.dispatcher import present_result


def test_normalize_storage_args_maps_defaults_to_run_raw(tmp_path: Path):
    args = {
        "run_id": "run_91ac0e",
        "project_root": str(tmp_path),
        "input": ".",
        "run_dir": ".",
        "xmolout": "xmolout",
        "fort7": "fort.7",
    }

    out = normalize_storage_args(args)

    expected_raw = tmp_path / "data" / "raw" / "run_91ac0e"
    assert out["input"] == str(expected_raw)
    assert out["run_dir"] == str(expected_raw)
    assert out["xmolout"] == str(expected_raw / "xmolout")
    assert out["fort7"] == str(expected_raw / "fort.7")
    assert (tmp_path / "inputs" / "run_91ac0e").exists()
    assert expected_raw.exists()


def test_register_parsed_dataset_reuses_same_id(tmp_path: Path):
    layout = ReaxkitStorageLayout(project_root=tmp_path)
    layout.ensure_run_layout("run_A")
    (layout.raw_run_dir("run_A") / "xmolout").write_text("same content", encoding="utf-8")

    first = layout.register_parsed_dataset(run_id="run_A", parser_version="TrajectoryTask:TrajectoryData", engine="reaxff")
    second = layout.register_parsed_dataset(run_id="run_A", parser_version="TrajectoryTask:TrajectoryData", engine="reaxff")

    assert first == second
    assert (layout.parsed_dir(first) / "meta.json").exists()
    assert (layout.run_index_path("run_A")).exists()


def test_resolve_output_path_run_scoped(tmp_path: Path):
    out = resolve_output_path(
        "msd.csv",
        "msd",
        run_id="run_X",
        project_root=tmp_path,
        analysis_id="analysis_123",
    )
    assert out == tmp_path / "analysis" / "msd" / "analysis_123" / "msd.csv"
    assert out.parent.exists()


def test_present_result_exports_to_run_scoped_path(tmp_path: Path):
    result = SimpleNamespace(table=pd.DataFrame({"a": [1], "b": [2]}))
    args = SimpleNamespace(
        export="result.csv",
        save=None,
        plot=None,
        show=False,
        run_id="run_Z",
        project_root=str(tmp_path),
        analysis_id=None,
    )
    present_result("msd", result, args)

    expected = tmp_path / "analysis" / "msd" / "run_Z" / "result.csv"
    assert expected.exists()


def test_present_result_auto_generates_run_id(tmp_path: Path):
    result = SimpleNamespace(table=pd.DataFrame({"a": [1]}))
    args = SimpleNamespace(
        export="result.csv",
        save=None,
        plot=None,
        show=False,
        run_id=None,
        project_root=str(tmp_path),
        analysis_id=None,
    )
    present_result("msd", result, args)

    assert getattr(args, "run_id", None) is not None
    expected = tmp_path / "analysis" / "msd" / str(args.run_id) / "result.csv"
    assert expected.exists()
