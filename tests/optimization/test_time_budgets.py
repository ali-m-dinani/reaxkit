import json
import sys

import pytest

from reaxkit.optimization.config import load_config
from reaxkit.optimization.conversion import collect_native
from reaxkit.optimization.models import OptimizationError
from reaxkit.optimization.orchestrator import prepare
from reaxkit.optimization.storage import read_json
from reaxkit.optimization.worker import run_worker
from .test_configuration import settings_file, edit_config


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_invalid_budget(settings_file, value):
    edit_config(settings_file, lambda data: data.update(optimization_time_limit_seconds=value))
    with pytest.raises(OptimizationError):
        load_config(settings_file)


def test_budget_inheritance(settings_file):
    edit_config(settings_file, lambda data: data.update(optimization_time_limit_seconds=1800))
    runtime = load_config(settings_file)["optimizer_settings"]["jax_soppi"]["runtime"]
    assert runtime["optimization_time_limit_seconds"] == 1800
    assert runtime["shutdown_grace_seconds"] == 600
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(
        optimization_time_limit_seconds=60))
    assert load_config(settings_file)["optimizer_settings"]["jax_soppi"]["runtime"]["optimization_time_limit_seconds"] == 60


def test_unverified_commercial_stop_rejected(settings_file):
    def update(data):
        data["optimizers"] = ["ams_cmaes"]
        data["optimizer_settings"]["ams_cmaes"] = data["optimizer_settings"].pop("jax_soppi")
        data["optimizer_settings"]["ams_cmaes"]["settings"] = {}
        data["optimization_time_limit_seconds"] = 60
    edit_config(settings_file, update)
    with pytest.raises(OptimizationError, match="not verified"):
        load_config(settings_file)


def test_live_history_partial_line_and_immutable_best(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "optimization-events.jsonl").write_text(
        json.dumps(dict(completed=1, elapsed_seconds=2, score=3)) + "\n" + '{"completed":',
        encoding="utf-8")
    (native / "ffield_best").write_text("first candidate")
    report = collect_native("ezff", native, tmp_path / "normalized")
    assert report["progress_count"] == 1
    artifacts = read_json(tmp_path / "normalized" / report["snapshot"] / "artifacts.json")
    preserved = tmp_path / "normalized" / artifacts[0]["preserved_path"]
    (native / "ffield_best").write_text("second candidate")
    collect_native("ezff", native, tmp_path / "normalized")
    assert preserved.read_text() == "first candidate"


def test_worker_requests_stop_from_timing_handshake(settings_file, monkeypatch):
    code = """
import json, os, pathlib, time
timing = pathlib.Path(os.environ['REAXKIT_TIMING_FILE'])
stop = pathlib.Path(os.environ['REAXKIT_STOP_FILE'])
record = dict(initialization_seconds=2, optimization_elapsed_seconds=0,
              updated_at_epoch=time.time(), state='running', graceful=False)
timing.write_text(json.dumps(record))
while not stop.exists():
    time.sleep(.01)
record.update(state='completed', graceful=True, termination_reason='time_limit',
              optimization_elapsed_seconds=.1)
timing.write_text(json.dumps(record))
"""
    script = settings_file.parent / "runner.py"
    script.write_text(code)
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(
        command=[sys.executable, str(script), "{run_config}"],
        optimization_time_limit_seconds=.05, shutdown_grace_seconds=5, timeout_seconds=10))
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert run_worker(directory) == 0
    state = read_json(directory / "worker.json")
    assert state["termination_mode"] == "graceful"
    assert state["termination_reason"] == "time_limit"
    assert state["initialization_seconds"] == 2


def test_unresponsive_runner_is_interrupted_not_graceful(settings_file, monkeypatch):
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(
        command=[sys.executable, "-c", "import time; time.sleep(60)", "{run_config}"],
        optimization_time_limit_seconds=.01, shutdown_grace_seconds=.01, timeout_seconds=10))
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert run_worker(directory) == 124
    state = read_json(directory / "worker.json")
    assert state["termination_mode"] == "interrupted"
    assert state["initialization_seconds"] is None
