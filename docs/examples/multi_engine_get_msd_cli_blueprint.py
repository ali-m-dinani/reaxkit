"""
Blueprint example: engine-agnostic CLI flow for MSD.

This is intentionally minimal and educational (not connected to reaxkit internals yet).
It shows:
1) Engine selection with `_pick_engine(...)`
2) A lazy `TrajectoryData` container that loads through the selected adapter
3) An engine-agnostic `MSDTask(request)` that only sees normalized trajectory data

Run:
    python docs/examples/multi_engine_get_msd_cli_blueprint.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


# ---------------------------------------------------------------------------
# Common domain data used by analyses (engine-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryPayload:
    """Normalized trajectory content consumed by engine-agnostic analyses."""

    positions: list[list[tuple[float, float, float]]]  # frames -> atoms -> (x,y,z)
    atom_ids: list[int]
    times: list[float]


class TrajectoryData:
    """Lazy trajectory holder.

    The first call to `.get()` triggers loading via a provider callback.
    This lets tasks depend only on `TrajectoryData`, while adapters remain outside tasks.
    """

    def __init__(self, loader: Callable[[], TrajectoryPayload]) -> None:
        self._loader = loader
        self._payload: TrajectoryPayload | None = None

    def is_loaded(self) -> bool:
        return self._payload is not None

    def get(self) -> TrajectoryPayload:
        if self._payload is None:
            self._payload = self._loader()
        return self._payload


@dataclass
class MSDRequest:
    atom_ids: list[int] | None = None
    dt: float = 1.0


@dataclass
class MSDResult:
    lag: list[int]
    msd: list[float]


# ---------------------------------------------------------------------------
# Analysis task (single implementation for every engine)
# ---------------------------------------------------------------------------


class MSDTask:
    """Engine-agnostic MSD implementation over `TrajectoryData`."""

    def __init__(self, request: MSDRequest) -> None:
        self.request = request

    def run(self, data: TrajectoryData) -> MSDResult:
        payload = data.get()  # lazy load happens here (if not loaded yet)
        if len(payload.positions) < 2:
            return MSDResult(lag=[], msd=[])

        selected = self.request.atom_ids or payload.atom_ids
        indices = [payload.atom_ids.index(aid) for aid in selected]

        first = payload.positions[0]
        values: list[float] = []
        lag: list[int] = []

        for t in range(1, len(payload.positions)):
            accum = 0.0
            for i in indices:
                x0, y0, z0 = first[i]
                xt, yt, zt = payload.positions[t][i]
                accum += (xt - x0) ** 2 + (yt - y0) ** 2 + (zt - z0) ** 2
            values.append(accum / len(indices))
            lag.append(t)

        return MSDResult(lag=lag, msd=values)


# ---------------------------------------------------------------------------
# Engine adapters (engine-specific readers + validation)
# ---------------------------------------------------------------------------


class EngineAdapter(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def detect(run_dir: Path) -> int:
        """Return confidence 0..100."""

    @abstractmethod
    def validate(self, *, input_file: Path | None, xmolout: Path | None, summary: Path | None) -> None:
        """Raise ValueError if required inputs are missing for this engine."""

    @abstractmethod
    def load_trajectory(self, *, input_file: Path | None, xmolout: Path | None, summary: Path | None) -> TrajectoryPayload:
        """Read engine-specific files and normalize into TrajectoryPayload."""


class ReaxFFAdapter(EngineAdapter):
    name = "reaxff"

    @staticmethod
    def detect(run_dir: Path) -> int:
        has_xmol = (run_dir / "xmolout").exists()
        has_summary = (run_dir / "summary.txt").exists()
        return 90 if has_xmol else (60 if has_summary else 0)

    def validate(self, *, input_file: Path | None, xmolout: Path | None, summary: Path | None) -> None:
        _ = summary
        if input_file is not None:
            raise ValueError("For ReaxFF, use --xmolout (not --input).")
        if xmolout is None:
            raise ValueError("ReaxFF MSD workflows require --xmolout.")

    def load_trajectory(self, *, input_file: Path | None, xmolout: Path | None, summary: Path | None) -> TrajectoryPayload:
        _ = (xmolout, summary)
        return TrajectoryPayload(
            positions=[
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                [(0.1, 0.0, 0.0), (1.1, 0.0, 0.0)],
                [(0.2, 0.0, 0.0), (1.2, 0.0, 0.0)],
            ],
            atom_ids=[1, 2],
            times=[0.0, 1.0, 2.0],
        )


class AMSAdapter(EngineAdapter):
    name = "ams"

    @staticmethod
    def detect(run_dir: Path) -> int:
        return 90 if any(run_dir.glob("*.rkf")) else 0

    def validate(self, *, input_file: Path | None, xmolout: Path | None, summary: Path | None) -> None:
        if xmolout is not None or summary is not None:
            raise ValueError("For AMS, use --input ams.rkf (not --xmolout/--summary).")
        if input_file is None:
            raise ValueError("AMS workflows require --input ams.rkf (or auto-detected .rkf file).")

    def load_trajectory(self, *, input_file: Path | None, xmolout: Path | None, summary: Path | None) -> TrajectoryPayload:
        _ = summary
        _ = input_file
        return TrajectoryPayload(
            positions=[
                [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
                [(0.0, 0.2, 0.0), (2.0, 0.2, 0.0)],
                [(0.0, 0.4, 0.0), (2.0, 0.4, 0.0)],
            ],
            atom_ids=[1, 2],
            times=[0.0, 0.5, 1.0],
        )


ADAPTERS: list[type[EngineAdapter]] = [ReaxFFAdapter, AMSAdapter]


def _pick_engine(run_dir: Path, forced: str | None) -> EngineAdapter:
    if forced is not None:
        for cls in ADAPTERS:
            if cls.name == forced:
                return cls()
        raise ValueError(f"Unknown engine '{forced}'. Available: {[c.name for c in ADAPTERS]}")

    scored: list[tuple[int, type[EngineAdapter]]] = [(cls.detect(run_dir), cls) for cls in ADAPTERS]
    score, winner = max(scored, key=lambda t: t[0])
    if score == 0:
        raise ValueError("Could not detect engine. Use --engine reaxff|ams.")
    return winner()


def run_msd_cli(
    *,
    run_dir: str,
    engine: str | None = None,
    input_file: str | None = None,
    xmolout: str | None = None,
    summary: str | None = None,
    atom_ids: Iterable[int] | None = None,
) -> tuple[str, MSDResult]:
    """Single task-driven command path, e.g. `reaxkit get_msd [--engine ...]`."""
    run_path = Path(run_dir)

    # 1) Pick engine (explicit override or auto-detect)
    adapter = _pick_engine(run_path, forced=engine)

    input_path = Path(input_file) if input_file else None
    xmol_path = Path(xmolout) if xmolout else None
    summary_path = Path(summary) if summary else None

    adapter.validate(input_file=input_path, xmolout=xmol_path, summary=summary_path)

    # 2) Create task and run it on a lazy trajectory container
    trajectory = TrajectoryData(
        loader=lambda: adapter.load_trajectory(
            input_file=input_path,
            xmolout=xmol_path,
            summary=summary_path,
        )
    )

    task = MSDTask(request=MSDRequest(atom_ids=list(atom_ids) if atom_ids else None))
    result = task.run(trajectory)
    return adapter.name, result


if __name__ == "__main__":
    selected_engine, result = run_msd_cli(
        run_dir=".",
        engine="reaxff",
        xmolout="xmolout",
        atom_ids=[1],
    )
    print(f"ReaxFF demo via engine={selected_engine}:", result)

    selected_engine, result = run_msd_cli(
        run_dir=".",
        engine="ams",
        input_file="ams.rkf",
        atom_ids=[1],
    )
    print(f"AMS demo via engine={selected_engine}:", result)
