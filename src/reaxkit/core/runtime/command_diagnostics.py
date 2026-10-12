"""Per-command reports with a native stack watchdog and independent observer."""

from __future__ import annotations

from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from uuid import uuid4


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class _ConsoleTee:
    def __init__(self, original, destination, lock):
        self.original = original
        self.destination = destination
        self.lock = lock

    def write(self, text):
        with self.lock:
            result = self.original.write(text)
            self.destination.write(text)
            self.destination.flush()
        return result

    def flush(self):
        with self.lock:
            self.original.flush()
            self.destination.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class CommandDiagnostics:
    """Capture one parsed CLI invocation without replacing signal handlers."""

    def __init__(self, project_root, command, argv, run_id, *, interval=60.0, enabled=True, trace=None,
                 profile=False, profile_clock="wall"):
        self.profile = None
        if profile:
            if not enabled:
                raise ValueError("--profile requires diagnostics; remove --no-diagnostics.")
            from reaxkit.core.runtime.command_profile import CommandProfile
            self.profile = CommandProfile(profile_clock)
        self.enabled = enabled
        self.interval = interval
        self.trace = trace
        self.started = time.monotonic()
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(command))[:80]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.directory = Path(project_root).resolve() / "logs" / "diagnostics" / f"{name}_{stamp}_{uuid4().hex[:8]}"
        self.report = dict(schema_version=1, command=command, argv=list(argv), run_id=run_id,
                           started_at=_timestamp(), status="running", pid=os.getpid(),
                           main_thread_id=threading.get_ident(),
                           hostname=socket.gethostname(), cwd=str(Path.cwd()), python=sys.executable,
                           python_version=sys.version, stack_interval_seconds=interval,
                           scheduler={key: os.environ[key] for key in (
                               "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK",
                               "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST") if key in os.environ})
        self.exit_code = None
        self.observer = None
        self.raw = None
        self.console = None
        self.report["profiling"] = dict(enabled=profile, clock=profile_clock if profile else None)

    def __enter__(self):
        if not self.enabled:
            return self
        self.directory.mkdir(parents=True, exist_ok=False)
        import reaxkit
        self.report["reaxkit_module"] = reaxkit.__file__
        _write_json(self.directory / "report.json", self.report)
        self.original_stdout, self.original_stderr = sys.stdout, sys.stderr
        try:
            self.console = (self.directory / "console.log").open("w", encoding="utf-8", buffering=1)
            lock = threading.RLock()
            sys.stdout = _ConsoleTee(sys.stdout, self.console, lock)
            sys.stderr = _ConsoleTee(sys.stderr, self.console, lock)
            print(f"[ReaxKit] Diagnostic report: {self.directory}", file=sys.stderr, flush=True)
            self.raw = (self.directory / "stacks.raw.log").open("wb", buffering=0)
            faulthandler.dump_traceback_later(self.interval, repeat=True, file=self.raw)
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with (self.directory / "observer.log").open("wb") as errors:
                self.observer = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), str(self.directory),
                     str(os.getpid()), str(self.started)], stdin=subprocess.DEVNULL,
                    stdout=errors, stderr=errors, creationflags=creationflags,
                )
            if self.profile is not None:
                self.profile.start()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def record_exception(self, error):
        if self.enabled:
            with (self.directory / "exception.log").open("a", encoding="utf-8") as stream:
                traceback.print_exception(type(error), error, error.__traceback__, file=stream)

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        if self.profile is not None:
            try:
                self.profile.finish(self.directory)
            except Exception as error:
                self.report["profiling"]["error"] = str(error)
                print(f"[ReaxKit] Could not finalize profile: {error}", file=sys.stderr)
        if self.raw is not None:
            faulthandler.cancel_dump_traceback_later()
            self.raw.close()
        sys.stdout, sys.stderr = self.original_stdout, self.original_stderr
        if self.console is not None:
            self.console.close()
        if exc is not None:
            self.record_exception(exc)
        (self.directory / "observer.stop").touch()
        if self.observer is not None:
            try:
                self.observer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.observer.terminate()
                self.observer.wait(timeout=5)
                self.report["observer_error"] = "Observer did not stop within five seconds."
            if self.observer.returncode:
                self.report["observer_exit_code"] = self.observer.returncode
        self.report.update(finished_at=_timestamp(), elapsed_seconds=time.monotonic()-self.started,
                           exit_code=self.exit_code,
                           status="completed" if exc is None and self.exit_code == 0 else "failed")
        if self.trace is not None:
            self.report["execution"] = self.trace.to_record()
            (self.directory / "steps.log").write_text(self.trace.render(), encoding="utf-8")
        _write_json(self.directory / "report.json", self.report)
        return False


def _observe(directory, pid, started):
    """Annotate native dumps and sample resources even when the parent holds the GIL."""
    import psutil
    from reaxkit.core.runtime.stack_summary import StackSummary

    process = psutil.Process(pid)
    metadata = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    summary = StackSummary(metadata["stack_interval_seconds"], metadata["main_thread_id"])
    summary.write(directory)
    pending = b""
    samples = 0
    last_resource = 0.0
    with (directory / "stacks.raw.log").open("rb") as source, \
            (directory / "stacks.log").open("w", encoding="utf-8", buffering=1) as stacks, \
            (directory / "resources.jsonl").open("w", encoding="utf-8", buffering=1) as resources:
        stacks.write("Elapsed times are observer timestamps (approximately 0.25-second resolution).\n")
        while True:
            elapsed = time.monotonic() - started
            stopping = (directory / "observer.stop").exists() or not process.is_running()
            received = source.read()
            pending += received
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line.startswith(b"Timeout ("):
                    samples += 1
                    stacks.write(f"\n=== Stack sample {samples} | elapsed {elapsed:.3f} s | {_timestamp()} ===\n")
                stacks.write(line.decode("utf-8", errors="replace") + "\n")
                summary.feed(line.decode("utf-8", errors="replace"))
            if received or stopping:
                summary.write(directory)
            if elapsed - last_resource >= 5.0 or last_resource == 0.0 or stopping:
                try:
                    cpu = process.cpu_times()
                    memory = process.memory_info()
                    resources.write(json.dumps(dict(timestamp=_timestamp(), elapsed_seconds=elapsed,
                        cpu_user_seconds=cpu.user, cpu_system_seconds=cpu.system,
                        rss_bytes=memory.rss, threads=process.num_threads())) + "\n")
                except psutil.Error:
                    pass
                last_resource = elapsed
            if stopping:
                stacks.write(pending.decode("utf-8", errors="replace"))
                break
            time.sleep(0.25)


if __name__ == "__main__":
    _observe(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
