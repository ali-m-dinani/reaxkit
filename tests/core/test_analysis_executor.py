from __future__ import annotations

from reaxkit.core.runtime.analysis_executor import AnalysisExecutor


def test_engine_detection_path_prefers_explicit_input_over_snapshot_source():
    args = {
        "input": "30_1073_ams.rkf",
        "_snapshot_source_dir": "mixed_directory_with_reaxff_markers",
        "xmolout": "xmolout",
        "fort7": "fort.7",
    }

    assert AnalysisExecutor._engine_detection_path(args) == "30_1073_ams.rkf"


def test_engine_detection_path_uses_snapshot_source_when_input_is_default():
    args = {
        "input": ".",
        "_snapshot_source_dir": "run_directory",
        "xmolout": "xmolout",
        "fort7": "fort.7",
    }

    assert AnalysisExecutor._engine_detection_path(args) == "run_directory"
