import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reaxkit.optimization.config import load_config
from reaxkit.optimization.models import OptimizationError
from reaxkit.optimization.orchestrator import collect, prepare, status, submit
from reaxkit.optimization.schedulers.slurm import Slurm, SubmissionUnknown, validate_job
from reaxkit.optimization.storage import read_json, write_json
from reaxkit.optimization.worker import run_worker
from .test_configuration import settings_file, edit_config


class FakeSlurm:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def submit(self, *arguments):
        self.calls.append(arguments)
        if self.failure:
            raise self.failure
        return {"job_id": str(100 + len(self.calls)), "cluster": None}

    def statuses(self, identifiers):
        return {identifier: {"state": "RUNNING", "source": "fake"} for identifier in identifiers}

    def find_submission(self, name, date):
        return ["987"]


def linux_campaign(settings_file, monkeypatch):
    monkeypatch.setattr("reaxkit.optimization.orchestrator.platform.system", lambda: "Linux")
    root = prepare(load_config(settings_file))
    monkeypatch.setattr("reaxkit.optimization.orchestrator.os", SimpleNamespace(name="posix"))
    return root


def test_submit_once_and_status_collect_do_not_resubmit(settings_file, monkeypatch):
    root = linux_campaign(settings_file, monkeypatch)
    scheduler = FakeSlurm()
    first = submit(root, scheduler)
    assert len(scheduler.calls) == 2
    assert all(row["state"] == "submitted" for row in first)
    submit(root, scheduler)
    status(root, scheduler)
    collect(root)
    assert len(scheduler.calls) == 2


@pytest.mark.parametrize("error,state", [
    (SubmissionUnknown("timeout"), "unknown"),
    (OptimizationError("rejected"), "submission_failed"),
    (FileNotFoundError("sbatch absent"), "submission_failed"),
])
def test_partial_submission_is_durable_and_not_retried(settings_file, monkeypatch, error, state):
    root = linux_campaign(settings_file, monkeypatch)
    scheduler = FakeSlurm(error)
    result = submit(root, scheduler)
    assert result[0]["state"] == state
    assert len(scheduler.calls) == 1
    runs = read_json(root / "campaign.json")["runs"]
    assert read_json(root / runs[1] / "job" / "submission.json")["state"] == "staged"
    scheduler.failure = None
    submit(root, scheduler)
    assert len(scheduler.calls) == 2
    assert read_json(root / runs[0] / "job" / "submission.json")["state"] == state
    if state == "unknown":
        rows = status(root, scheduler)
        assert rows[0]["reconciliation_candidates"] == ["987"]


def test_slurm_argument_list_and_ambiguous_success(settings_file, monkeypatch):
    config = load_config(settings_file)
    backend = Slurm()
    calls = []

    def command(arguments):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "123;cluster\n", "")
    monkeypatch.setattr(backend, "_run", command)
    result = backend.submit(Path(config["job"]), Path("/worker with space.sh"), Path("/run with space"), "unique-name", Path("/logs"))
    assert result["job_id"] == "123"
    assert str(Path("/worker with space.sh")) == calls[0][-1]
    assert "--chdir=" + str(Path("/run with space")) in calls[0]
    monkeypatch.setattr(backend, "_run", lambda arguments: subprocess.CompletedProcess(arguments, 0, "unrecognized", ""))
    with pytest.raises(SubmissionUnknown):
        backend.submit(Path(config["job"]), Path("/worker"), Path("/run"), "name", Path("/logs"))


def test_scheduler_accounting_absence_not_success(monkeypatch):
    backend = Slurm()
    monkeypatch.setattr(backend, "_run", lambda arguments: subprocess.CompletedProcess(arguments, 0, "", ""))
    assert backend.statuses(["42"])["42"]["state"] == "UNKNOWN"
    monkeypatch.setattr(backend, "_run", lambda arguments: subprocess.CompletedProcess(
        arguments, 0, "" if arguments[0] == "squeue" else "42|TIMEOUT|0:15\n42.batch|FAILED|1:0\n", ""
    ))
    assert backend.statuses(["42"])["42"]["state"] == "TIMEOUT"


def test_job_contract_preserves_user_modules_and_rejects_arrays(settings_file):
    path = settings_file.parent / "job.sh"
    text = path.read_text().replace('exec bash "$1"', 'module load site-specific-version\nexport NSCM="$SLURM_NTASKS"\nexec bash "$1"')
    path.write_text(text)
    assert "module load site-specific-version" in validate_job(path)
    path.write_text(text.replace("#SBATCH --nodes=1", "#SBATCH --array=0-3"))
    with pytest.raises(OptimizationError, match="arrays"):
        validate_job(path)


def test_external_runner_executes_with_isolated_directory_and_final_collection(settings_file, monkeypatch):
    code = "import json,pathlib,sys; config=json.load(open(sys.argv[1])); pathlib.Path('history.jsonl').write_text(json.dumps(dict(step_index=0,current_error=3,best_error=3))+'\\n')"
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(command=[sys.executable, "-c", code, "{run_config}"]))
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert run_worker(directory) == 0
    state = read_json(directory / "worker.json")
    assert state["state"] == "completed"
    assert state["scientific_results"] == "available"
    assert state["last_conversion"]["progress_count"] == 1
    with pytest.raises(OptimizationError, match="already started"):
        run_worker(directory)


def test_worker_failure_preserves_reports(settings_file, monkeypatch):
    edit_config(settings_file, lambda data: data["optimizer_settings"]["jax_soppi"]["runtime"].update(command=[sys.executable, "-c", "raise SystemExit(7)", "{run_config}"]))
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    assert run_worker(directory) == 7
    state = read_json(directory / "worker.json")
    assert state["state"] == "failed"
    assert (directory / "native" / "stderr.log").exists()


def test_mutated_inputs_and_staged_config_rejected(settings_file, monkeypatch):
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    (directory / "input" / "ffield").write_text("changed")
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    with pytest.raises(OptimizationError, match="Staged input"):
        run_worker(directory)


def test_periodic_collection_then_interruption(settings_file, monkeypatch):
    import signal
    from reaxkit.optimization.worker import InterruptedRun
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    monkeypatch.setattr("reaxkit.optimization.worker.require_allocation", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "125")
    calls = {"waits": 0, "stopped": False}

    class Process:
        def wait(self, timeout=None):
            calls["waits"] += 1
            if calls["waits"] == 1:
                (directory / "native" / "history.jsonl").write_text(
                    '{"step_index":0,"current_error":5,"best_error":5}\n'
                )
                raise subprocess.TimeoutExpired("runner", timeout)
            raise InterruptedRun("simulated allocation termination")

    monkeypatch.setattr("reaxkit.optimization.worker.subprocess.Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr("reaxkit.optimization.worker.stop_process", lambda process: calls.update(stopped=True))
    before = signal.getsignal(signal.SIGTERM)
    assert run_worker(directory) == 143
    state = read_json(directory / "worker.json")
    assert state["state"] == "interrupted" and calls["stopped"]
    assert state["last_conversion"]["progress_count"] == 1
    assert signal.getsignal(signal.SIGTERM) == before


def test_success_before_failed_submission_retains_both_ledgers(settings_file, monkeypatch):
    root = linux_campaign(settings_file, monkeypatch)

    class PartialSlurm(FakeSlurm):
        def submit(self, *arguments):
            if self.calls:
                self.calls.append(arguments)
                raise OptimizationError("allocation rejected")
            return super().submit(*arguments)

    scheduler = PartialSlurm()
    result = submit(root, scheduler)
    assert [row["state"] for row in result] == ["submitted", "submission_failed"]
    submit(root, scheduler)
    assert len(scheduler.calls) == 2


def test_timeout_and_environment_override_handling(settings_file, monkeypatch):
    config = load_config(settings_file)
    backend = Slurm()

    def timeout(arguments):
        raise subprocess.TimeoutExpired(arguments, 30)

    monkeypatch.setattr(backend, "_run", timeout)
    with pytest.raises(SubmissionUnknown):
        backend.submit(Path(config["job"]), Path("/worker"), Path("/native"), "name", Path("/logs"))
    backend = Slurm()
    monkeypatch.setenv("SBATCH_ARRAY_INX", "0-99")
    captured = {}

    def capture(arguments, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(arguments, 0, "123", "")

    monkeypatch.setattr("reaxkit.optimization.schedulers.slurm.subprocess.run", capture)
    backend._run(["sbatch", "--parsable"])
    assert "SBATCH_ARRAY_INX" not in captured["env"]
    assert not captured.get("shell", False)


def test_worker_launcher_modification_blocks_submission(settings_file, monkeypatch):
    root = linux_campaign(settings_file, monkeypatch)
    directory = root / read_json(root / "campaign.json")["runs"][0]
    (directory / "job" / "worker.sh").write_text("changed launcher")
    scheduler = FakeSlurm()
    with pytest.raises(OptimizationError, match="worker changed"):
        submit(root, scheduler)
    assert not scheduler.calls
