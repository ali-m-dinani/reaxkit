"""Slurm submission/status with durable callers and no automatic retries."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

from ..models import OptimizationError

CONTRACT = "# reaxkit-worker-contract: 1"
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=reaxkit-fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=PARTITION

# reaxkit-worker-contract: 1
set -e
# Replace ACCOUNT and PARTITION; add memory and module/environment setup.
# The submitted worker runs once; it launches your optimizer inside this allocation.
# Keep the worker invocation last, replacing the old hard-coded engine command.
exec bash "$1"
"""


def validate_job(path):
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("#!/") or CONTRACT not in text.splitlines():
        raise OptimizationError(f"{path}: add {CONTRACT!r} and end with exec bash \"$1\"; use --generate-job-template")
    commands = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not commands or commands[-1] != 'exec bash "$1"':
        raise OptimizationError(f"{path}: final command must be exec bash \"$1\"")
    for line in text.splitlines():
        if line.lstrip().startswith("#SBATCH"):
            tokens = shlex.split(line.lstrip()[7:])
            for token in tokens:
                if token.startswith(("--array", "--wrap")) or token == "-a" or re.match(r"-a\S", token):
                    raise OptimizationError(f"{path}: Slurm arrays/wrapped commands are not supported")
    return text


def generate_job_template(path):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(JOB_TEMPLATE)
    return Path(path)


class SubmissionUnknown(OptimizationError):
    """Submission may have succeeded; reconcile before any retry."""


class Slurm:
    def _run(self, arguments):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
        return subprocess.run(arguments, capture_output=True, text=True, timeout=30, env=environment, check=False)

    def submit(self, job, worker, native, name, log_directory):
        validate_job(job)
        arguments = [
            "sbatch", "--parsable", f"--job-name={name}", f"--comment={name}",
            f"--chdir={native}", f"--output={log_directory / 'slurm-%j.out'}",
            f"--error={log_directory / 'slurm-%j.err'}", str(job), str(worker),
        ]
        try:
            result = self._run(arguments)
        except subprocess.TimeoutExpired as error:
            raise SubmissionUnknown("sbatch timed out; submission outcome is unknown") from error
        if result.returncode:
            raise OptimizationError(f"sbatch rejected the job: {result.stderr.strip()}")
        response = result.stdout.strip()
        if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", response):
            raise SubmissionUnknown(f"Unrecognized sbatch response: {response!r}")
        return {"job_id": response.split(";")[0], "cluster": response.split(";")[1] if ";" in response else None}

    def statuses(self, job_ids):
        job_ids = list(dict.fromkeys(job_ids))
        if not job_ids:
            return {}
        if any(not re.fullmatch(r"\d+", job_id) for job_id in job_ids):
            raise OptimizationError("Invalid Slurm job ID")
        states = {job_id: {"state": "UNKNOWN", "source": "unavailable"} for job_id in job_ids}
        queries = (
            (["squeue", "--noheader", "--jobs=" + ",".join(job_ids), "--format=%i|%T"], "squeue"),
            (["sacct", "--noheader", "--parsable2", "--jobs=" + ",".join(job_ids), "--format=JobIDRaw,State,ExitCode"], "sacct"),
        )
        for arguments, source in queries:
            try:
                result = self._run(arguments)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode:
                continue
            for line in result.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) < 2 or parts[0] not in states:
                    continue
                if states[parts[0]]["source"] == "squeue":
                    continue
                states[parts[0]] = {"state": parts[1], "source": source, "exit_code": parts[2] if len(parts) > 2 else None}
        return states

    def find_submission(self, name, date):
        found = set()
        queries = (
            ["squeue", "--noheader", "--name=" + name, "--format=%i|%j"],
            ["sacct", "--noheader", "--parsable2", "--starttime=" + date,
             "--name=" + name, "--format=JobIDRaw,JobName%100"],
        )
        for arguments in queries:
            try:
                result = self._run(arguments)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode:
                continue
            for line in result.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 2 and parts[1] == name and re.fullmatch(r"\d+", parts[0]):
                    found.add(parts[0])
        return sorted(found)
