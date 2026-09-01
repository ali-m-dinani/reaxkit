from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.domain.data_models import TrajectoryData


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


def test_engine_detection_path_ignores_storage_synthesized_input():
    args = {
        "input": "reaxkit_workspace/data/raw/run_123",
        "_input_was_explicit": False,
        "_snapshot_source_dir": "run_directory",
    }

    assert AnalysisExecutor._engine_detection_path(args) == "run_directory"


@dataclass
class _FrameRequest:
    frames: list[int]
    every: int = 1


@dataclass
class _FrameResult:
    table: pd.DataFrame
    request: _FrameRequest


class _FrameTask:
    required_data = TrajectoryData

    def run(self, _data, request):
        assert request.frames == [0, 1]
        return _FrameResult(
            table=pd.DataFrame({"frame_index": request.frames, "value": [1.0, 2.0]}),
            request=request,
        )


def test_partial_frame_execution_uses_compact_indices_and_restores_source_indices():
    request = _FrameRequest(frames=[0, 50])
    data = TrajectoryData(
        positions=np.zeros((2, 1, 3), dtype=float),
        elements=["Al"],
        atom_ids=[1],
        iterations=np.asarray([0, 500], dtype=int),
        source_frame_indices=np.asarray([0, 50], dtype=int),
    )

    result = AnalysisExecutor._run_task(_FrameTask(), data, request, reporter=None)

    assert result.table["frame_index"].tolist() == [0, 50]
    assert result.request is request


def test_executor_builds_selective_load_plan_only_for_explicit_frames():
    assert AnalysisExecutor._requested_frame_indices(_FrameRequest([0, 50, 100]), TrajectoryData) == [0, 50, 100]
    assert AnalysisExecutor._requested_frame_indices(_FrameRequest([]), TrajectoryData) is None


def test_selective_sources_are_not_copied_into_the_run_snapshot(tmp_path):
    xmolout = tmp_path / "xmolout"
    xmolout.write_text("trajectory", encoding="utf-8")
    args = {"_snapshot_source_dir": str(tmp_path)}

    remaining, borrowed = AnalysisExecutor._use_selective_sources(
        args,
        ("xmolout", "summary.txt"),
    )

    assert remaining == ("summary.txt",)
    assert borrowed == {"xmolout": str(xmolout.resolve())}
    assert args["xmolout"] == str(xmolout.resolve())


def test_executor_passes_requested_data_fields_to_adapter(monkeypatch):
    captured = {}

    class DummyAdapter:
        def supports_streaming(self, data_type, args):
            return False

        def required_input_files(self, data_type, args):
            captured["snapshot_fields"] = args.get("_required_data_fields")
            return ()

        def load(self, data_type, args, reporter=None):
            captured["load_fields"] = args.get("_required_data_fields")
            return "payload"

    class FieldAwareTask:
        required_data = object

        @staticmethod
        def required_data_fields_for(request, args):
            return (request,)

        def run(self, data, request):
            return data

    monkeypatch.setattr(
        "reaxkit.core.runtime.analysis_executor.resolve_engine",
        lambda path, engine=None: DummyAdapter(),
    )

    result = AnalysisExecutor().run(
        FieldAwareTask(),
        request="potential_energy",
        args={"engine": "reaxff", "no_cache": True},
    )

    assert result == "payload"
    assert captured["snapshot_fields"] == ("potential_energy",)
    assert captured["load_fields"] == ("potential_energy",)


def test_selective_streaming_requires_task_opt_in():
    class DummyAdapter:
        @staticmethod
        def supports_streaming(data_type, args):
            return True

    class StreamTask:
        @staticmethod
        def run_stream(frames, request):
            return None

    adapter = DummyAdapter()
    selected = [0, 1, 2]

    assert not AnalysisExecutor._streaming_enabled(StreamTask(), adapter, object, {}, selected)

    StreamTask.supports_selective_streaming = True
    assert AnalysisExecutor._streaming_enabled(StreamTask(), adapter, object, {}, selected)


def test_ams_and_lammps_selective_sources_are_not_copied(tmp_path):
    rkf = tmp_path / "reaxout.rkf"
    dump = tmp_path / "dump.lammpstrj"
    rkf.write_text("rkf", encoding="utf-8")
    dump.write_text("dump", encoding="utf-8")
    args = {"_snapshot_source_dir": str(tmp_path)}

    remaining, borrowed = AnalysisExecutor._use_selective_sources(
        args,
        ("reaxout.kf", "reaxout.rkf", "dump.lammpstrj", "log.lammps"),
    )

    assert remaining == ("reaxout.kf", "log.lammps")
    assert borrowed == {
        "reaxout.rkf": str(rkf.resolve()),
        "dump.lammpstrj": str(dump.resolve()),
    }
    assert args["rkf"] == str(rkf.resolve())
    assert args["dump"] == str(dump.resolve())
