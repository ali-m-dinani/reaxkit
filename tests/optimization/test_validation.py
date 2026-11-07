import sys

from reaxkit.optimization.config import load_config
from reaxkit.optimization.orchestrator import prepare
from reaxkit.optimization.storage import read_json, write_json
from reaxkit.optimization.validation import smoke_report
from reaxkit.optimization.worker import run_worker
from .test_configuration import edit_config, settings_file
from .test_lifecycle import FakeSlurm


def test_empty_smoke_report_is_not_cluster_validation(settings_file):
    root = prepare(load_config(settings_file))
    report = smoke_report(root, FakeSlurm())
    assert not report["automated_checks_pass"]
    assert not report["cluster_validated"]
    assert not report["checks"]["two_methods_two_replicas"]
    assert report["review_status"] == "manual_review_required"
    assert len(report["manual_acceptance_gates"]) >= 4


def test_smoke_report_cli_returns_failure_for_missing_evidence(settings_file, capsys):
    from reaxkit.cli.main import build_parser
    from reaxkit.optimization.cli import run_main
    root = prepare(load_config(settings_file))
    args = build_parser("optimize-ffield").parse_args(["optimize-ffield", "--smoke-report", str(root)])
    assert run_main(args) == 1
    assert '"cluster_validated": false' in capsys.readouterr().out


def test_smoke_report_checks_job_identity_and_artifact_changes(settings_file, monkeypatch):
    code = "import pathlib; pathlib.Path('history.jsonl').write_text('{\"step_index\":0,\"current_error\":3}\\n'); pathlib.Path('ffield_best').write_text('field')"
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(command=[sys.executable, "-c", code, "{run_config}"]))
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert run_worker(directory) == 0
    write_json(directory / "job/submission.json", {"state": "submitted", "job_id": "112"})
    report = smoke_report(root, FakeSlurm())
    assert not report["runs"][0]["checks"]["worker_job_matches_submission"]
    assert report["runs"][0]["checks"]["artifacts_preserved"]
    (directory / "native/ffield_best").write_text("changed")
    assert not smoke_report(root, FakeSlurm())["runs"][0]["checks"]["artifacts_preserved"]


def test_runtime_watchdog_keeps_partial_results(settings_file, monkeypatch):
    code = "import pathlib,time; pathlib.Path('history.jsonl').write_text('{\"step_index\":0,\"current_error\":4}\\n'); time.sleep(60)"
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(command=[sys.executable, "-c", code, "{run_config}"], timeout_seconds=1))
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    assert run_worker(directory) == 124
    worker = read_json(directory / "worker.json")
    assert worker["state"] == "timed_out"
    assert worker["last_conversion"]["progress_count"] == 1


def test_missing_executable_is_recorded_as_failed_worker(settings_file, monkeypatch):
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(command=["missing-reaxkit-test-executable", "{run_config}"]))
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    assert run_worker(directory) == 1
    worker = read_json(directory / "worker.json")
    assert worker["state"] == "failed"
    assert worker["scientific_results"] == "unavailable"
