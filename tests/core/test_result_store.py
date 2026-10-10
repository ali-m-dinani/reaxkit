import json

import pandas as pd
import pytest

from reaxkit.core.runtime.result_store import ResultStore


def frame(source):
    return {"main": pd.DataFrame({"frame_index": [source], "value": [source / 3]}),
            "detail": pd.DataFrame({"frame_index": [source, source], "atom": [1, 2]})}


@pytest.mark.parametrize("handled", [False, True])
def test_export_failure_preserves_analysis_for_retry(tmp_path, handled):
    from argparse import Namespace
    from reaxkit.presentation.workflow_artifacts import workflow_artifact_policy, register_checkpoint

    args = Namespace()
    def command():
        with workflow_artifact_policy(args):
            store = ResultStore(tmp_path, {}, batch_frames=1)
            register_checkpoint(store)
            store.append(0, 5, frame(5))
            store.transition("analysis_complete")
            if handled:
                args._checkpoint_exit_code = 1
            else:
                raise OSError("export failed")
    if handled:
        command()
    else:
        with pytest.raises(OSError, match="export failed"):
            command()
    with ResultStore(tmp_path, {}, resume=True) as store:
        assert store.manifest["state"] == "analysis_complete"
        assert store.table("main").materialize().frame_index.tolist() == [5]


@pytest.mark.parametrize("format_name", ["csv", "parquet"])
def test_resume_transactions_and_streaming_export(tmp_path, format_name):
    with ResultStore(tmp_path, {"version": 1}, batch_frames=2, format_name=format_name) as store:
        store.append(0, 9, frame(9))
        store.append(1, 2, frame(2))
        store.append(2, 15, frame(15))
        assert store.committed_frames == 2
    with ResultStore(tmp_path, {"version": 1}, resume=True) as store:
        assert store.committed_frames == 2
        store.append(2, 15, frame(15))
        store.transition("analysis_complete")
        table = store.table("main")
        assert len(table) == 3
        assert table.materialize().frame_index.tolist() == [9, 2, 15]
        table.to_csv(tmp_path / "export.csv")
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "export.csv"), table.materialize())
        store.transition("complete")


@pytest.mark.parametrize("stage", ["writing", "staged", "published", "manifest"])
def test_failure_never_commits_half_batch(tmp_path, stage):
    def fail(point):
        if point == stage:
            raise OSError("injected disk failure")
    with pytest.raises(OSError):
        with ResultStore(tmp_path, {}, batch_frames=1) as store:
            store.append(0, 7, frame(7))
            store.fault = fail
            store.append(1, 8, frame(8))
    with ResultStore(tmp_path, {}, resume=True, batch_frames=1) as store:
        assert store.committed_frames == 1
        store.transition("running")
        store.append(1, 8, frame(8))
        store.transition("analysis_complete")
        assert store.table("detail").materialize().frame_index.tolist() == [7, 7, 8, 8]


def test_lock_identity_corruption_and_empty_selection(tmp_path):
    with ResultStore(tmp_path, {"scientific": 1}, batch_frames=1) as store:
        with pytest.raises(RuntimeError, match="writer lock"):
            ResultStore(tmp_path, {"scientific": 1}, resume=True)
        with pytest.raises(ValueError, match="No selected"):
            store.transition("analysis_complete")
        store.append(0, 0, frame(0))
        with pytest.raises(ValueError, match="duplicates"):
            store.append(0, 0, frame(0))
        with pytest.raises(ValueError, match="Every enabled"):
            store.append(1, 1, {"main": frame(1)["main"]})
    with pytest.raises(ValueError, match="incompatible"):
        ResultStore(tmp_path, {"scientific": 2}, resume=True)
    batch = tmp_path / "batches" / "batch-00000000"
    metadata = json.loads((batch / "batch.json").read_text())
    (batch / metadata["tables"]["main"]["file"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Damaged"):
        ResultStore(tmp_path, {"scientific": 1}, resume=True)


def test_hbn_resume_skips_committed_kernels(tmp_path, monkeypatch):
    from tests.analysis.test_hbn_reference_polarization import _trajectory_from_reference
    from reaxkit.analysis.ferroelectrics.hbn_reference import polarization as module
    from reaxkit.core.runtime.frame_tables import source_frame_index
    from reaxkit.analysis.ferroelectrics.polarization_stream import stream_polarization
    from reaxkit.core.runtime.checkpoint_results import CheckpointAccumulator

    request = module.HBNReferencePolarizationRequest(reference_path=module.REFERENCE_STRUCTURE_PATH,
                                                    replication=(2, 1, 1), charge_source="formal", volume_method="cell")
    data = [_trajectory_from_reference(module.REFERENCE_STRUCTURE_PATH) for _ in range(3)]
    for index, trajectory in enumerate(data):
        trajectory.source_frame_indices = [index]
    task = module.HBNReferencePolarizationTask()
    expected = stream_polarization(task, iter(data), request, "hbn")
    original = module.calculate_hbn_reference_polarization
    calls = []
    def calculate(trajectory, *args, **kwargs):
        source = source_frame_index(trajectory, 0)
        calls.append(source)
        if source == 2:
            raise RuntimeError("injected kernel failure")
        return original(trajectory, *args, **kwargs)
    monkeypatch.setattr(module, "calculate_hbn_reference_polarization", calculate)
    with pytest.raises(Exception, match="injected kernel failure"):
        with ResultStore(tmp_path, {}, batch_frames=1) as store:
            request._result_store = store
            stream_polarization(task, iter(data), request, "hbn")
    calls.clear()
    def resumed(trajectory, *args, **kwargs):
        calls.append(source_frame_index(trajectory, 0))
        return original(trajectory, *args, **kwargs)
    monkeypatch.setattr(module, "calculate_hbn_reference_polarization", resumed)
    with ResultStore(tmp_path, {}, resume=True) as store:
        store.transition("running")
        request._result_store = store
        actual = stream_polarization(task, iter(data), request, "hbn")
        assert calls == [2]
        pd.testing.assert_frame_equal(actual.table.materialize(), expected.table)
        store.transition("complete")
    with ResultStore(tmp_path, {}, resume=True) as store:
        calls.clear()
        request._result_store = store
        actual = CheckpointAccumulator(store).finish(request)
        assert calls == []
        pd.testing.assert_frame_equal(actual.table.materialize(), expected.table)


@pytest.mark.parametrize("stage", ["writing", "staged", "published", "manifest"])
def test_process_death_preserves_only_committed_batches(tmp_path, stage):
    import subprocess
    import sys
    script = '''
import os
import sys
import pandas as pd
from reaxkit.core.runtime.result_store import ResultStore
store = ResultStore(sys.argv[1], {}, batch_frames=1)
store.append(0, 5, {"table": pd.DataFrame({"value": [5]})})
def crash(stage):
    if stage == sys.argv[2]:
        os._exit(77)
store.fault = crash
store.append(1, 9, {"table": pd.DataFrame({"value": [9]})})
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path), stage], timeout=30)
    assert process.returncode == 77
    with pytest.raises(RuntimeError, match="writer lock"):
        ResultStore(tmp_path, {}, resume=True)
    (tmp_path / "writer.lock").unlink()
    with ResultStore(tmp_path, {}, resume=True, batch_frames=1) as store:
        assert store.committed_frames == 1
        store.append(1, 9, {"table": pd.DataFrame({"value": [9]})})
        store.transition("analysis_complete")
        assert store.table("table").materialize().value.tolist() == [5, 9]
        store.transition("complete")
