import json

import pytest
import yaml

from reaxkit.optimization.adapters.flocky import convert
from reaxkit.optimization.config import load_config
from reaxkit.optimization.conversion import collect_native
from reaxkit.optimization.orchestrator import prepare
from reaxkit.optimization.registry import specification
from reaxkit.optimization.schedulers.slurm import JOB_TEMPLATE


def test_native_cycle_checkpoints(tmp_path):
    (tmp_path / "opti_log.out.10").write_text("0 5.0e+00\n1 4.0\n")
    (tmp_path / "opti_log.out.2").write_text("0 10.0\n1 9.0\n")
    (tmp_path / "ffield.gbest.2.1.0").write_text("synthetic force field")
    (tmp_path / "results.out.2").write_text("ambiguous prediction report")
    result = convert(tmp_path)
    assert [row["cycle"] for row in result.progress] == [2, 2, 10, 10]
    assert [row["best_score"] for row in result.progress] == [10, 9, 5, 4]
    assert all(row["candidate_evaluations"] is None for row in result.progress)
    assert len({row["event_id"] for row in result.progress}) == 4
    assert result.artifacts[0]["restart"] == "parameter_only"
    assert not result.predictions


def test_partial_duplicate_and_nonfinite(tmp_path):
    (tmp_path / "opti_log.out.0").write_text("0 inf\n1 3\n1 2\nbad row\n2 1")
    result = convert(tmp_path)
    assert len(result.progress) == 2
    assert result.progress[0]["status"] == "nonfinite"
    assert result.progress[0]["score"] is None
    assert any("partial" in warning for warning in result.warnings)
    assert any("duplicate" in warning for warning in result.warnings)


def test_empty_output(tmp_path):
    assert not convert(tmp_path).progress
    assert specification("flocky")["cluster_validated"] is False


def test_nonfinite_checkpoints_are_not_scientific_results(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "opti_log.out.0").write_text("0 inf\n1 inf\n")
    report = collect_native("flocky", native, tmp_path / "normalized")
    assert report["progress_count"] == 2
    assert report["finite_progress_count"] == 0
    assert any("nonfinite fitness" in warning for warning in report["warnings"])
    (native / "opti_log.out.0").write_text("0 inf\n1 3.5\n")
    report = collect_native("flocky", native, tmp_path / "normalized")
    assert report["finite_progress_count"] == 1


@pytest.mark.parametrize("swarm_size", [2, 4])
def test_external_contract(tmp_path, swarm_size):
    inputs = {}
    for name in ("ffield", "params", "geo", "trainset", "control"):
        (tmp_path / name).write_text("fixture\n")
        inputs[name] = name
    inputs["params_bounds_order"] = "min_max"
    (tmp_path / "job.sh").write_text(JOB_TEMPLATE)
    config = dict(schema_version=1, optimizers=["flocky"], replicas=1, seed=None,
                  inputs=inputs, output="campaign", job="job.sh",
                  optimizer_settings=dict(flocky=dict(
                      runtime=dict(mode="external_runner", command=["python", "runner.py", "{run_config}"]),
                      settings=dict(swarm_size=swarm_size))))
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config))
    root = prepare(load_config(path))
    request = json.loads((root / "runs/flocky/replica-0001/run-config.json").read_text())
    assert request["contract"] == "reaxkit-external-runner-v1"
    assert request["settings"]["swarm_size"] == swarm_size
    assert request["settings"]["iterations"] == 2
