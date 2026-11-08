import hashlib
import json

from reaxkit.optimization.adapters.glompo import convert
from reaxkit.optimization.registry import specification


def write_export(root, rows=None):
    root.mkdir(parents=True, exist_ok=True)
    native = b"synthetic HDF5 fixture; not parsed by ReaxKit"
    (root / "glompo_log.h5").write_bytes(native)
    if rows is None:
        rows = [dict(schema_version=1, optimizer_id="optimizer_1", call_id=index,
                     score=score, parameters=[0.5], source="glompo_log.h5",
                     source_dataset="/optimizer_1/iter_hist", source_row=index - 1)
                for index, score in enumerate([3.0, 1.5], 1)]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (root / "glompo_history.jsonl").write_bytes(payload)
    (root / "glompo_history.meta.json").write_text(json.dumps(dict(
        schema_version=1, source="glompo_log.h5", count=len(rows),
        source_sha256=hashlib.sha256(native).hexdigest(),
        history_sha256=hashlib.sha256(payload).hexdigest())))
    return rows


def test_history_and_profile(tmp_path):
    root = tmp_path / "manager"
    write_export(root)
    (root / "glompo-profile.json").write_text(json.dumps(dict(
        profile="ams_params", engine="AMS", objective="native SSE",
        dataset_scope="converted", parameter_space="scaled_0_1")))
    result = convert(tmp_path)
    assert [row["score"] for row in result.progress] == [3, 1.5]
    assert result.progress[-1]["best_score"] == 1.5
    assert result.progress[-1]["candidate_id"] == "optimizer_1:call:2"
    assert result.progress[-1]["parameter_space"] == "scaled_0_1"
    assert result.progress[-1]["engine_identity"] == "AMS"
    assert result.predictions == []
    assert result.artifacts[0]["restart"] == "neither"
    assert "manager/glompo_log.h5" in result.sources
    assert specification("glompo")["seed"] == "unsupported"


def test_stale_history_is_not_accepted(tmp_path):
    write_export(tmp_path)
    (tmp_path / "glompo_log.h5").write_bytes(b"changed")
    result = convert(tmp_path)
    assert not result.progress
    assert any("re-export" in warning for warning in result.warnings)


def test_duplicate_ids_rejected(tmp_path):
    rows = write_export(tmp_path)
    write_export(tmp_path, [rows[0], rows[0]])
    assert not convert(tmp_path).progress


def test_benchmark_and_nonfinite(tmp_path):
    rows = write_export(tmp_path)
    rows[1].update(score=None, raw_score="nan")
    write_export(tmp_path, rows)
    result = convert(tmp_path)
    assert result.progress[1]["status"] == "nonfinite"
    assert result.progress[1]["best_score"] == 3
    assert any("Benchmark" in warning for warning in result.warnings)


def test_missing_export_keeps_native_artifact(tmp_path):
    (tmp_path / "glompo_log.h5").write_bytes(b"fixture")
    result = convert(tmp_path)
    assert not result.progress
    assert len(result.artifacts) == 1
    assert any("--export-native" in warning for warning in result.warnings)


def test_initial_candidate_requires_matching_profile_vector(tmp_path):
    write_export(tmp_path)
    profile = dict(profile="ams_params", parameter_space="scaled_0_1",
                   initial_candidate_policy="exact_first_evaluation",
                   initial_parameters_scaled=[0.5])
    path = tmp_path / "glompo-profile.json"
    path.write_text(json.dumps(profile))
    result = convert(tmp_path)
    assert result.progress[0]["is_initial_candidate"] is True
    assert result.progress[0]["record_kind"] == "initial_parameter_evaluation"
    assert result.progress[1]["is_initial_candidate"] is False
    profile["initial_parameters_scaled"] = [1.0]
    path.write_text(json.dumps(profile))
    result = convert(tmp_path)
    assert not any(record["is_initial_candidate"] for record in result.progress)
    assert any("original-field baseline" in warning for warning in result.warnings)


def test_old_profile_does_not_claim_a_baseline(tmp_path):
    write_export(tmp_path)
    (tmp_path / "glompo-profile.json").write_text(json.dumps(dict(profile="ams_params")))
    result = convert(tmp_path)
    assert len(result.progress) == 2
    assert not any(record["is_initial_candidate"] for record in result.progress)
    assert any("original-field baseline" in warning for warning in result.warnings)


def test_external_runner_campaign(tmp_path):
    import yaml
    from reaxkit.optimization.config import load_config
    from reaxkit.optimization.orchestrator import prepare
    from reaxkit.optimization.schedulers.slurm import JOB_TEMPLATE

    inputs = {}
    for name in ("ffield", "params", "geo", "trainset", "control"):
        (tmp_path / name).write_text("fixture\n")
        inputs[name] = name
    inputs["params_bounds_order"] = "min_max"
    (tmp_path / "job.sh").write_text(JOB_TEMPLATE)
    settings = dict(schema_version=1, optimizers=["glompo"], replicas=1, seed=None,
                    inputs=inputs, output="campaign", job="job.sh",
                    optimizer_settings=dict(glompo=dict(runtime=dict(
                        mode="external_runner", command=["/path/to/python", "runner.py", "{run_config}"]))))
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    root = prepare(load_config(path))
    request = json.loads((root / "runs/glompo/replica-0001/run-config.json").read_text())
    assert request["contract"] == "reaxkit-external-runner-v1"
    assert request["seed"] is None
    assert request["settings"]["max_function_calls"] == 12
    assert request["settings"]["population_size"] == 4
