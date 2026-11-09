"""One external optimizer runner inside a Slurm allocation."""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import time
import re
from pathlib import Path

from .models import OptimizationError
from .launching import adapter, expand_command, prepare_launch, verify_staged
from .orchestrator import collect_run
from .storage import atomic_text, digest, lock, now, read_json, write_json


class InterruptedRun(Exception):
    pass


class RunTimeout(Exception):
    pass


def mcff_stop_text(content):
    layouts = (
        (r"(?m)^(\s*)([012])(?=[ \t]+!Key:[ \t]*1[ \t]*-[ \t]*tell program to use the values below,[ \t]*2[ \t]*-[ \t]*stop[ \t]*$)", "2"),
        (r"(?m)^(\s*)([01])(?=[ \t]+StopKey\b)", "1"),
        (r"(?m)^([ \t]*StopKey[ \t]+)([01])\b", "1"),
    )
    matches = [(match, value) for pattern, value in layouts for match in re.finditer(pattern, content)]
    if len(matches) != 1:
        raise OptimizationError("Unrecognized or ambiguous AMS istop layout; refusing to guess stop key")
    match, value = matches[0]
    return content[:match.start(2)] + value + content[match.end(2):]


def require_allocation():
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID"):
        raise OptimizationError("Optimization workers require a Linux Slurm allocation")


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def run_worker(directory):
    require_allocation()
    directory = Path(directory).resolve()
    with lock(directory, ".worker.lock"):
        state_path = directory / "worker.json"
        if state_path.exists():
            raise OptimizationError("Worker already started; explicit native recovery belongs in a new run")
        run = read_json(directory / "run.json")
        if digest(directory / "run-config.json") != run["config_sha256"]:
            raise OptimizationError("Staged run configuration changed")
        request = read_json(directory / "run-config.json")
        for key, checksum in request["input_hashes"].items():
            if digest(Path(request["inputs"][key])) != checksum:
                raise OptimizationError(f"Staged input changed: {key}")
        verify_staged(directory, request.get("staged_hashes", {}), request.get("external_file_hashes", {}))
        environment = os.environ.copy()
        environment.update(request["runtime"].get("environment", {}))
        request["runtime"].setdefault("mode", "external_runner")
        plan = request.get("launch") or prepare_launch(request)
        environment.update(plan["environment"])
        for key in ("REAXKIT_TIMING_FILE", "REAXKIT_STOP_FILE", "REAXKIT_OPTIMIZATION_TIME_LIMIT_SECONDS"):
            environment.pop(key, None)
        state = {"state": "running", "started_at": now(), "job_id": os.environ["SLURM_JOB_ID"],
                 "exit_code": None, "collection_errors": [], "commands": [],
                 "allocation": {key: value for key, value in os.environ.items() if key in (
                     "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
                     "SLURM_JOB_GPUS", "SLURM_CLUSTER_NAME")},
                 "seed_forwarded": None}
        write_json(state_path, state)
        process = None
        handlers = {}
        started = time.monotonic()
        timeout_seconds = request["runtime"].get("timeout_seconds")
        optimization_limit = request["runtime"].get("optimization_time_limit_seconds")
        grace = request["runtime"].get("shutdown_grace_seconds", 600)
        timing_path = directory / "native" / "optimization-timing.json"
        stop_path = directory / "native" / "optimization.stop"
        instrumented = request["optimizer"] not in ("ams_mcff", "ams_cmaes", "standalone_soppi", "ogolem")
        environment["REAXKIT_STOP_FILE"] = str(stop_path)
        if instrumented:
            environment["REAXKIT_TIMING_FILE"] = str(timing_path)
            environment["REAXKIT_STOP_FILE"] = str(stop_path)
            if optimization_limit is not None:
                environment["REAXKIT_OPTIMIZATION_TIME_LIMIT_SECONDS"] = str(optimization_limit)
        optimizer_started = None
        stop_requested = None
        next_collection = started
        state.update(initialization_seconds=None, optimization_elapsed_seconds=None,
                     termination_reason=None, termination_mode="running")

        def monitor_budget():
            nonlocal stop_requested
            elapsed = None
            if timing_path.is_file():
                try:
                    evidence = read_json(timing_path)
                    state["optimizer_timing"] = evidence
                    state["initialization_seconds"] = evidence.get("initialization_seconds")
                    state["optimization_elapsed_seconds"] = evidence.get("optimization_elapsed_seconds")
                    elapsed = evidence.get("optimization_elapsed_seconds")
                    if elapsed is not None and evidence.get("state") != "completed":
                        elapsed += max(0, time.time() - evidence.get("updated_at_epoch", time.time()))
                except (OSError, ValueError):
                    pass
            if optimizer_started is None or optimization_limit is None:
                return
            if not instrumented or not timing_path.is_file():
                elapsed = time.monotonic() - optimizer_started
                state["budget_clock_scope"] = "optimizer_process_including_initialization"
            if elapsed is not None and elapsed >= optimization_limit and stop_requested is None:
                atomic_text(stop_path, "time_limit\n")
                if request["optimizer"] == "ogolem":
                    atomic_text(Path(plan["results_directory"]) / "STOP", "time_limit\n")
                if request["optimizer"] == "ams_mcff":
                    control = directory / "native" / "istop"
                    if not control.is_file():
                        raise OptimizationError("AMS istop is missing; cannot request a graceful stop")
                    if control.is_file():
                        content = control.read_text()
                        atomic_text(control, mcff_stop_text(content))
                stop_requested = time.monotonic()
                state["stop_requested_at"] = now()
            if stop_requested is not None and time.monotonic() - stop_requested >= grace:
                raise RunTimeout("Cooperative shutdown grace expired")

        def interrupted(signum, frame):
            raise InterruptedRun(f"Received signal {signum}")

        def collect_available():
            try:
                report = collect_run(directory)
                state["last_conversion"] = report
            except (OSError, ValueError) as error:
                state["collection_errors"].append(str(error))

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                handlers[signum] = signal.signal(signum, interrupted)
            native = directory / "native"
            with (native / "stdout.log").open("ab") as stdout, (native / "stderr.log").open("ab") as stderr:
                for step in plan["commands"]:
                    if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                        raise RunTimeout(f"Worker runtime limit exceeded ({timeout_seconds} seconds)")
                    command = expand_command(step["argv"], environment)
                    state["commands"].append({"argv": command, "cwd": step["cwd"], "phase": step["phase"]})
                    write_json(state_path, state)
                    process = subprocess.Popen(command, cwd=step["cwd"], env=environment, stdout=stdout, stderr=stderr,
                                               start_new_session=(os.name == "posix"))
                    if step["phase"] == "optimizer":
                        optimizer_started = time.monotonic()
                        state["seed_forwarded"] = plan["seed_forwarded"]
                        write_json(state_path, state)
                    while True:
                        monitor_budget()
                        interval = run["collection_interval_seconds"]
                        if optimization_limit is not None:
                            interval = min(interval, 1.0)
                        if timeout_seconds is not None:
                            remaining = timeout_seconds - (time.monotonic() - started)
                            if remaining <= 0:
                                raise RunTimeout(f"Worker runtime limit exceeded ({timeout_seconds} seconds)")
                            interval = min(interval, remaining)
                        try:
                            state["exit_code"] = process.wait(timeout=interval)
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() >= next_collection:
                                collect_available()
                                next_collection = time.monotonic() + run["collection_interval_seconds"]
                            write_json(state_path, state)
                    if state["exit_code"] != 0:
                        break
                    if step["phase"] == "prepare":
                        adapter(request["optimizer"]).after_prepare(request)
            state["state"] = "completed" if state["exit_code"] == 0 else "failed"
            state["termination_mode"] = "graceful" if state["exit_code"] == 0 else "failed"
            state["termination_reason"] = "time_limit_requested" if stop_requested else "optimizer_exit"
        except RunTimeout as error:
            state.update(state="timed_out", error=str(error), exit_code=124)
            state.update(termination_mode="interrupted", termination_reason="hard_timeout")
            for signum in handlers:
                signal.signal(signum, signal.SIG_IGN)
            stop_process(process)
        except InterruptedRun as error:
            state.update(state="interrupted", error=str(error), exit_code=143)
            state.update(termination_mode="interrupted", termination_reason="signal")
            for signum in handlers:
                signal.signal(signum, signal.SIG_IGN)
            stop_process(process)
        except Exception as error:
            state.update(state="failed", error=str(error), exit_code=1)
            state.update(termination_mode="failed", termination_reason="worker_error")
            stop_process(process)
        finally:
            if timing_path.is_file():
                try:
                    evidence = read_json(timing_path)
                    state["optimizer_timing"] = evidence
                    state["initialization_seconds"] = evidence.get("initialization_seconds")
                    state["optimization_elapsed_seconds"] = evidence.get("optimization_elapsed_seconds")
                    if state["state"] == "completed" and evidence.get("graceful"):
                        state["termination_reason"] = evidence.get("termination_reason")
                except (OSError, ValueError):
                    pass
            collect_available()
            state["finished_at"] = now()
            state["elapsed_seconds"] = time.monotonic() - started
            state["scientific_results"] = "available" if state.get("last_conversion", {}).get("finite_progress_count", 0) else "unavailable"
            write_json(state_path, state)
            for signum, previous in handlers.items():
                signal.signal(signum, previous)
        return state["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    return run_worker(args.run_directory)


if __name__ == "__main__":
    raise SystemExit(main())
