"""LAMMPS ``log.lammps`` file handler."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LAMMPSLogHandler:
    """Reader for ``log.lammps`` files using ``lammps.formats.LogFile``."""

    def __init__(self, file_path: str | Path):
        self.path = Path(file_path)

    @staticmethod
    def _load_logfile(path: Path):
        try:
            from lammps.formats import LogFile
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LAMMPS log loading requires 'lammps.formats.LogFile'. "
                "Install the LAMMPS Python package and retry."
            ) from exc
        return LogFile(str(path))

    @staticmethod
    def _normalize_runs(runs: list[dict]) -> list[dict[str, np.ndarray]]:
        normalized: list[dict[str, np.ndarray]] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            normalized.append({str(key): np.asarray(values) for key, values in run.items()})
        return normalized

    @staticmethod
    def _select_thermo_run(runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        for run in runs:
            if "Density" in run:
                return run
        return runs[0] if runs else {}

    def read(self) -> dict[str, object]:
        """Parse ``log.lammps`` and return runs, errors, and selected thermo table."""
        if not self.path.exists():
            raise FileNotFoundError(f"LAMMPS log file not found: {self.path}")

        parsed = self._load_logfile(self.path)
        runs = self._normalize_runs(getattr(parsed, "runs", None) or [])
        errors = list(getattr(parsed, "errors", None) or [])
        thermo = self._select_thermo_run(runs)
        return {"runs": runs, "errors": errors, "thermo": thermo}
