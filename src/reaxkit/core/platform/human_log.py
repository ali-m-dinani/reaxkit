"""Append-only, human-readable execution traces for ReaxKit commands."""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from reaxkit.core.platform.paths import io_path
from reaxkit.core.storage.storage_layout import generate_run_id


_WRITE_LOCK = Lock()
_ACTIVE_TRACE: ContextVar["HumanReadableRunLog | None"] = ContextVar(
    "reaxkit_human_readable_trace",
    default=None,
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _clean(value: Any) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "not measured"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {remainder:.3f} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes} min {remainder:.3f} s"


@dataclass
class _StepRecord:
    name: str
    started_at: datetime = field(default_factory=_now)
    started_counter: float = field(default_factory=perf_counter)
    finished_at: datetime | None = None
    seconds: float | None = None
    status: str = "RUNNING"
    details: list[tuple[str, str]] = field(default_factory=list)
    results: list[tuple[str, str]] = field(default_factory=list)
    children: list["_StepRecord"] = field(default_factory=list)


class HumanLogStep(AbstractContextManager["HumanLogStep"]):
    """A timed node in a :class:`HumanReadableRunLog` execution tree."""

    def __init__(self, trace: "HumanReadableRunLog", record: _StepRecord):
        self._trace = trace
        self._record = record

    def detail(self, name: str, value: Any) -> None:
        self._record.details.append((_clean(name), _clean(value)))

    def result(self, name: str, path: str | Path) -> None:
        self._record.results.append((_clean(name), str(Path(path).resolve())))

    def __enter__(self) -> "HumanLogStep":
        self._trace._enter_step(self._record)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._record.finished_at = _now()
        self._record.seconds = perf_counter() - self._record.started_counter
        self._record.status = "FAILED" if exc is not None else "SUCCESS"
        if exc is not None:
            self.detail("error", f"{type(exc).__name__}: {exc}")
        self._trace._exit_step(self._record)
        return False


class HumanReadableRunLog(AbstractContextManager["HumanReadableRunLog"]):
    """Collect one request as a hierarchy and append it as a complete block.

    The same block is written to ``logs/human_readable.log`` and to a
    run-specific ``logs/run_<run_id>.human.log`` file.  Existing diagnostic and
    machine-readable timing logs are intentionally left unchanged.
    """

    def __init__(
        self,
        logs_dir: str | Path,
        *,
        command: str,
        run_id: str | None = None,
        request_name: str | None = None,
    ) -> None:
        self.logs_dir = Path(logs_dir).resolve()
        self.command = _clean(command)
        self.run_id = _clean(run_id or generate_run_id())
        self.request_name = _clean(request_name or "Command execution")
        self.started_at = _now()
        self._started_counter = perf_counter()
        self.finished_at: datetime | None = None
        self.seconds: float | None = None
        self.status = "RUNNING"
        self.error: str | None = None
        self.details: list[tuple[str, str]] = []
        self.results: list[tuple[str, str]] = []
        self.steps: list[_StepRecord] = []
        self._stack: list[_StepRecord] = []
        self._written = False
        self._context_token: Token[HumanReadableRunLog | None] | None = None

    def detail(self, name: str, value: Any) -> None:
        self.details.append((_clean(name), _clean(value)))

    def result(self, name: str, path: str | Path) -> None:
        self.results.append((_clean(name), str(Path(path).resolve())))

    def fail(self, error: BaseException | str) -> None:
        """Mark a handled command failure before leaving the context."""
        self.status = "FAILED"
        if isinstance(error, BaseException):
            self.error = f"{type(error).__name__}: {error}"
        else:
            self.error = _clean(error)

    def completed_step(
        self,
        name: str,
        *,
        seconds: float | None,
        details: dict[str, Any] | None = None,
        results: dict[str, str | Path] | None = None,
        parent: str | None = None,
    ) -> None:
        """Add a step whose elapsed time was measured by another runtime layer."""
        container = self._stack[-1].children if self._stack else self.steps
        if parent:
            parent_name = _clean(parent)
            parent_record = next((item for item in container if item.name == parent_name), None)
            if parent_record is None:
                parent_record = _StepRecord(name=parent_name, status="SUCCESS")
                parent_record.started_counter = 0.0
                container.append(parent_record)
            container = parent_record.children

        clean_name = _clean(name)
        existing = next((item for item in container if item.name == clean_name), None)
        record = existing or _StepRecord(name=clean_name)
        if existing is None:
            container.append(record)
        record.finished_at = _now()
        record.seconds = float(seconds) if seconds is not None else None
        record.status = "SUCCESS"
        if details:
            record.details.extend(
                (_clean(key), _clean(value))
                for key, value in details.items()
                if value is not None and value != ""
            )
        if results:
            record.results.extend(
                (_clean(key), str(Path(value).resolve()))
                for key, value in results.items()
                if value is not None and value != ""
            )

    def step(self, name: str, **details: Any) -> HumanLogStep:
        record = _StepRecord(name=_clean(name))
        record.details.extend((_clean(key), _clean(value)) for key, value in details.items())
        if self._stack:
            self._stack[-1].children.append(record)
        else:
            self.steps.append(record)
        return HumanLogStep(self, record)

    def _enter_step(self, record: _StepRecord) -> None:
        self._stack.append(record)

    def _exit_step(self, record: _StepRecord) -> None:
        if self._stack and self._stack[-1] is record:
            self._stack.pop()

    def __enter__(self) -> "HumanReadableRunLog":
        self._context_token = _ACTIVE_TRACE.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            self.status = "FAILED"
            self.error = f"{type(exc).__name__}: {exc}"
        elif self.status == "RUNNING":
            self.status = "SUCCESS"
        try:
            self.finish()
        finally:
            if self._context_token is not None:
                _ACTIVE_TRACE.reset(self._context_token)
                self._context_token = None
        return False

    def finish(self) -> None:
        """Finalize and append the trace. Calling this method twice is safe."""
        if self._written:
            return
        self.finished_at = _now()
        self.seconds = perf_counter() - self._started_counter
        if self.status == "RUNNING":
            self.status = "SUCCESS"
        block = self.render()
        machine_line = json.dumps(self.to_record(), sort_keys=True) + "\n"
        io_path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        run_log_name = (
            f"{self.run_id}.human.log"
            if self.run_id.startswith("run_")
            else f"run_{self.run_id}.human.log"
        )
        human_paths = (
            self.logs_dir / "human_readable.log",
            self.logs_dir / run_log_name,
        )
        machine_run_name = run_log_name.removesuffix(".human.log") + ".machine.jsonl"
        machine_paths = (
            self.logs_dir / "machine_readable.jsonl",
            self.logs_dir / machine_run_name,
        )
        with _WRITE_LOCK:
            for path in human_paths:
                with io_path(path).open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(block)
            for path in machine_paths:
                with io_path(path).open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(machine_line)
        self._written = True

    @staticmethod
    def _pairs_to_records(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"name": name, "value": value} for name, value in pairs]

    @classmethod
    def _step_to_record(cls, step: _StepRecord) -> dict[str, Any]:
        return {
            "name": step.name,
            "status": step.status.lower(),
            "started_at": step.started_at.isoformat(),
            "finished_at": step.finished_at.isoformat() if step.finished_at else None,
            "duration_seconds": step.seconds,
            "details": cls._pairs_to_records(step.details),
            "results": cls._pairs_to_records(step.results),
            "steps": [cls._step_to_record(child) for child in step.children],
        }

    def to_record(self) -> dict[str, Any]:
        """Return the complete request tree as a JSON-serializable record."""
        finished_at = self.finished_at or _now()
        return {
            "schema": "reaxkit.execution_trace",
            "schema_version": 1,
            "request": {
                "name": self.request_name,
                "command": self.command,
                "requested_at": self.started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": self.seconds,
                "status": self.status.lower(),
                "error": self.error,
            },
            "run": {
                "run_id": self.run_id,
                "logs_directory": str(self.logs_dir),
                "details": self._pairs_to_records(self.details),
            },
            "steps": [self._step_to_record(step) for step in self.steps],
            "results": self._pairs_to_records(self.results),
        }

    @staticmethod
    def _render_pairs(lines: list[str], pairs: list[tuple[str, str]], indent: int) -> None:
        prefix = "  " * indent
        for name, value in pairs:
            lines.append(f"{prefix}{name}: {value}")

    @classmethod
    def _render_step(cls, lines: list[str], step: _StepRecord, indent: int) -> None:
        prefix = "  " * indent
        lines.append(f"{prefix}- {step.name}")
        lines.append(f"{prefix}  status: {step.status}")
        lines.append(f"{prefix}  duration: {_duration(step.seconds)}")
        cls._render_pairs(lines, step.details, indent + 1)
        if step.results:
            lines.append(f"{prefix}  results:")
            cls._render_pairs(lines, step.results, indent + 2)
        if step.children:
            lines.append(f"{prefix}  substeps:")
            for child in step.children:
                cls._render_step(lines, child, indent + 2)

    def render(self) -> str:
        """Render the completed request using spaces only (no terminal glyphs)."""
        finished_at = self.finished_at or _now()
        lines = [
            "=" * 88,
            f"REQUEST: {self.request_name}",
            f"  requested_at: {self.started_at.isoformat(timespec='seconds')}",
            f"  command: {self.command}",
            f"  status: {self.status}",
            f"  finished_at: {finished_at.isoformat(timespec='seconds')}",
            f"  total_duration: {_duration(self.seconds)}",
            "  run:",
            f"    run_id: {self.run_id}",
            f"    logs_directory: {self.logs_dir}",
        ]
        self._render_pairs(lines, self.details, 2)
        if self.steps:
            lines.append("    steps:")
            for step in self.steps:
                self._render_step(lines, step, 3)
        if self.results:
            lines.append("    results:")
            self._render_pairs(lines, self.results, 3)
        if self.error:
            lines.append(f"    error: {_clean(self.error)}")
        lines.extend(["=" * 88, ""])
        return "\n".join(lines)


def current_human_log() -> HumanReadableRunLog | None:
    """Return the trace activated by the current CLI or standalone request."""
    return _ACTIVE_TRACE.get()
