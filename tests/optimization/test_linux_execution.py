"""Actual Linux shell/process tests using synthetic runners, never engines/Slurm."""

import os
import platform
import signal
import subprocess
import sys
import time

import pytest

from reaxkit.optimization.config import load_config
from reaxkit.optimization.orchestrator import collect, prepare
from reaxkit.optimization.storage import read_json
from .test_configuration import edit_config, settings_file


pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="Requires real Linux process/signal semantics")


RUNNER = """import json, pathlib, signal, subprocess, sys, time
root = pathlib.Path.cwd()
mode = sys.argv[1]
if mode == 'missing':
    raise SystemExit(0)
root.joinpath('history.jsonl').write_text(json.dumps({'step_index': 0, 'current_error': 7.25, 'best_error': 7.25}) + '\\n')
root.joinpath('ffield_best').write_text('synthetic best-field artifact\\n')
if mode == 'success':
    raise SystemExit(0)
child_code = "import pathlib,signal,time; signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path('child.stopped').write_text('stopped'), exit(0))); pathlib.Path('child.ready').touch(); time.sleep(120)"
child = subprocess.Popen([sys.executable, '-c', child_code])
deadline = time.monotonic() + 10
while not root.joinpath('child.ready').exists():
    if time.monotonic() > deadline:
        raise SystemExit(4)
    time.sleep(.02)
root.joinpath('ready').touch()
time.sleep(120)
"""


@pytest.mark.parametrize("mode,code,state", [
    ("success", 0, "completed"), ("missing", 0, "completed"),
    ("timeout", 124, "timed_out"), ("signal", 143, "interrupted"),
])
def test_linux_shell_worker_and_real_process_group(settings_file, mode, code, state):
    runner = settings_file.parent / "synthetic runner.py"
    runner.write_text(RUNNER)

    def update(data):
        data["collection_interval_seconds"] = 1
        data["optimizer_settings"]["jax_soppi"]["runtime"].update(
            command=[sys.executable, str(runner), mode, "{run_config}"],
            timeout_seconds=5 if mode == "timeout" else 30,
        )
    edit_config(settings_file, update)
    root = prepare(load_config(settings_file))
    directory = root / read_json(root / "campaign.json")["runs"][0]
    for script in (directory / "job/submit.sh", directory / "job/worker.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True, timeout=10)
    environment = dict(os.environ, SLURM_JOB_ID="987654321", SLURM_NTASKS="1", SLURM_JOB_NUM_NODES="1")
    process = subprocess.Popen(["bash", str(directory / "job/worker.sh")], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        if mode == "signal":
            deadline = time.monotonic() + 20
            while not (directory / "native/ready").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("Synthetic child failed to start")
                time.sleep(0.05)
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=40)
        assert process.returncode == code, (stdout, stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    worker = read_json(directory / "worker.json")
    assert worker["state"] == state
    if mode == "missing":
        assert worker["scientific_results"] == "unavailable"
    else:
        assert worker["last_conversion"]["progress_count"] == 1
        assert (directory / "native/ffield_best").read_text() == "synthetic best-field artifact\n"
    if mode in ("timeout", "signal"):
        assert (directory / "native/child.stopped").is_file()
    collected = collect(root)[0]
    assert collected["snapshot"] == worker["last_conversion"]["snapshot"]
