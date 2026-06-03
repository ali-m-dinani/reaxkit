"""Parse LAMMPS dump trajectories into per-frame coordinate payloads.

This module supports both xyz-like dumps and native ``ITEM:`` formatted dumps,
normalizing frame data into table/array forms used by the LAMMPS adapter. It
focuses on trajectory parsing, frame iteration, and optional progress updates.

**Usage context**

- Trajectory ingestion: Produces frame coordinates/labels for ``TrajectoryData``.
- Box metadata: Captures box bounds when present in ``ITEM:`` dumps.
- Streaming workflows: Exposes random-access and iterative frame APIs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from reaxkit.engine.reaxff.io.base import BaseHandler
from .lammps_log_handler import LAMMPSLogHandler


class LAMMPSDumpHandler(BaseHandler):
    """Parser for LAMMPS dump trajectories.

    Supported formats:
    1) XYZ-like blocks:
       ``n_atoms`` -> comment with timestep -> ``atom_type x y z`` rows
    2) Native ``ITEM:`` dump blocks.
    """

    _TIMESTEP_PATTERN = re.compile(r"timestep\s*:\s*([+-]?\d+)", flags=re.IGNORECASE)

    def __init__(self, file_path: str | Path = "dump.xyz", *, reporter=None, progress_every: int = 5000):
        """Initialize dump parser state and optional progress reporter."""
        super().__init__(file_path)
        self._frames: list[pd.DataFrame] = []
        self._box_bounds: list[list[tuple[float, ...]] | None] = []
        self._format: str = "unknown"
        self._n_atoms: int | None = None
        self._reporter = reporter
        self._progress_every = max(1, int(progress_every))
        self._last_report_line = -1

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Parse dump content and return simulation index table plus metadata."""
        with open(self.path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        total_lines = len(non_empty)
        self._last_report_line = -1
        self._report_progress(0, total_lines, "Parsing LAMMPS dump", force=True)
        if not non_empty:
            self._report_progress(total_lines, total_lines, "Finished parsing LAMMPS dump", force=True)
            return pd.DataFrame(columns=["num_of_atoms", "iter"]), {
                "format": "empty",
                "n_atoms": 0,
                "n_frames": 0,
                "has_box_bounds": False,
            }

        if non_empty[0].startswith("ITEM:"):
            df = self._parse_item_dump(non_empty)
            fmt = "item_dump"
        else:
            df = self._parse_xyz_like(non_empty)
            fmt = "xyz_like"

        self._report_progress(total_lines, total_lines, "Finished parsing LAMMPS dump", force=True)
        self._format = fmt
        meta: dict[str, Any] = {
            "format": fmt,
            "n_atoms": self._n_atoms,
            "n_frames": len(self._frames),
            "has_box_bounds": any(b is not None for b in self._box_bounds),
        }
        return df, meta

    def _report_progress(self, lines_read: int, total_lines: int, stage: str, *, force: bool = False) -> None:
        """Emit throttled progress events to the optional reporter callback."""
        if not callable(self._reporter):
            return
        if not force and lines_read < total_lines:
            if self._last_report_line >= 0 and (lines_read - self._last_report_line) < self._progress_every:
                return
        self._last_report_line = int(lines_read)
        remaining = max(0, int(total_lines) - int(lines_read))
        msg = f"{stage} ({int(lines_read)}/{int(total_lines)} lines, {remaining} remaining)"
        self._reporter("load", int(lines_read), int(total_lines), msg)

    def _parse_xyz_like(self, lines: list[str]) -> pd.DataFrame:
        """Parse xyz-like dump blocks into frame tables and iteration index rows."""
        rows: list[dict[str, Any]] = []
        i = 0
        n_lines = len(lines)

        while i < n_lines:
            self._report_progress(i, n_lines, "Parsing LAMMPS xyz-like dump")
            n_atoms = int(lines[i].split()[0])
            if self._n_atoms is None:
                self._n_atoms = n_atoms
            i += 1
            if i >= n_lines:
                break

            header = lines[i]
            i += 1
            iter_num = self._extract_timestep(header, fallback=len(rows))

            atom_rows: list[list[Any]] = []
            for _ in range(n_atoms):
                if i >= n_lines:
                    break
                parts = lines[i].split()
                i += 1
                if len(parts) < 4:
                    continue
                atom_rows.append([parts[0], float(parts[1]), float(parts[2]), float(parts[3])])

            frame_df = pd.DataFrame(atom_rows, columns=["atom_type", "x", "y", "z"])
            self._frames.append(frame_df)
            self._box_bounds.append(None)
            rows.append({"num_of_atoms": n_atoms, "iter": int(iter_num)})
            self._report_progress(i, n_lines, "Parsing LAMMPS xyz-like dump")

        return pd.DataFrame(rows, columns=["num_of_atoms", "iter"])

    def _parse_item_dump(self, lines: list[str]) -> pd.DataFrame:
        """Parse native ``ITEM:`` dump blocks into frame tables and metadata rows."""
        rows: list[dict[str, Any]] = []
        i = 0
        n_lines = len(lines)

        while i < n_lines:
            self._report_progress(i, n_lines, "Parsing LAMMPS ITEM dump")
            if not lines[i].startswith("ITEM: TIMESTEP"):
                i += 1
                continue
            i += 1
            if i >= n_lines:
                break
            iter_num = int(float(lines[i].strip()))
            i += 1

            if i >= n_lines or not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
                break
            i += 1
            if i >= n_lines:
                break
            n_atoms = int(float(lines[i].strip()))
            if self._n_atoms is None:
                self._n_atoms = n_atoms
            i += 1

            if i >= n_lines or not lines[i].startswith("ITEM: BOX BOUNDS"):
                break
            i += 1
            bounds: list[tuple[float, ...]] = []
            for _ in range(3):
                if i >= n_lines:
                    break
                vals = tuple(float(v) for v in lines[i].split())
                bounds.append(vals)
                i += 1

            if i >= n_lines or not lines[i].startswith("ITEM: ATOMS"):
                break
            atom_cols = lines[i].split()[2:]
            i += 1

            atom_rows_raw: list[list[str]] = []
            for _ in range(n_atoms):
                if i >= n_lines:
                    break
                atom_rows_raw.append(lines[i].split())
                i += 1

            frame_df = self._frame_from_item_rows(atom_cols, atom_rows_raw)
            self._frames.append(frame_df)
            self._box_bounds.append(bounds if bounds else None)
            rows.append(
                {
                    "num_of_atoms": n_atoms,
                    "iter": int(iter_num),
                    "xlo": bounds[0][0] if len(bounds) > 0 else math.nan,
                    "xhi": bounds[0][1] if len(bounds) > 0 and len(bounds[0]) > 1 else math.nan,
                    "ylo": bounds[1][0] if len(bounds) > 1 else math.nan,
                    "yhi": bounds[1][1] if len(bounds) > 1 and len(bounds[1]) > 1 else math.nan,
                    "zlo": bounds[2][0] if len(bounds) > 2 else math.nan,
                    "zhi": bounds[2][1] if len(bounds) > 2 and len(bounds[2]) > 1 else math.nan,
                }
            )
            self._report_progress(i, n_lines, "Parsing LAMMPS ITEM dump")

        return pd.DataFrame(rows)

    def _frame_from_item_rows(self, atom_cols: list[str], atom_rows_raw: list[list[str]]) -> pd.DataFrame:
        """Build normalized ``atom_type/x/y/z`` frame table from raw ITEM rows."""
        raw = pd.DataFrame(atom_rows_raw, columns=atom_cols)
        x_col = self._pick_coord_col(raw.columns, axis="x")
        y_col = self._pick_coord_col(raw.columns, axis="y")
        z_col = self._pick_coord_col(raw.columns, axis="z")

        if "element" in raw.columns:
            atom_type = raw["element"].astype(str)
        elif "type" in raw.columns:
            atom_type = raw["type"].astype(str)
        else:
            atom_type = raw.iloc[:, 0].astype(str)

        frame = pd.DataFrame(
            {
                "atom_type": atom_type,
                "x": pd.to_numeric(raw[x_col], errors="coerce"),
                "y": pd.to_numeric(raw[y_col], errors="coerce"),
                "z": pd.to_numeric(raw[z_col], errors="coerce"),
            }
        )
        return frame

    @staticmethod
    def _pick_coord_col(columns: Any, axis: str) -> str:
        """Select preferred coordinate column name for one axis."""
        preferred = [axis, f"{axis}u", f"{axis}s", f"{axis}su"]
        for name in preferred:
            if name in columns:
                return name
        raise ValueError(f"Cannot find {axis}-coordinate column in ITEM: ATOMS header: {list(columns)}")

    @classmethod
    def _extract_timestep(cls, header: str, fallback: int) -> int:
        """Extract timestep from header text or return fallback integer."""
        m = cls._TIMESTEP_PATTERN.search(header)
        if m:
            return int(m.group(1))
        return int(fallback)

    def n_frames(self) -> int:
        """Return the number of parsed frames."""
        return int(self.metadata().get("n_frames", 0))

    def n_atoms(self) -> int | None:
        """Return detected atom count per frame, parsing on first access if needed."""
        if self._n_atoms is None:
            _ = self.dataframe()
        return self._n_atoms

    def frame(self, i: int) -> dict[str, Any]:
        """Return one parsed frame payload by index.

        Parameters
        ----------
        i : int
            Zero-based frame index.

        Returns
        -------
        dict[str, Any]
            Frame payload containing index, iteration, coordinates, atom types,
            and optional ``box_bounds``.

        Examples
        --------
        ```python
        fr0 = handler.frame(0)
        coords = fr0["coords"]
        ```
        """
        df = self.dataframe()
        if i < 0 or i >= len(self._frames):
            raise IndexError(f"frame index {i} out of range [0, {len(self._frames) - 1}]")
        frame_df = self._frames[i]
        row = df.iloc[i]
        payload: dict[str, Any] = {
            "index": i,
            "iter": int(row["iter"]) if "iter" in df.columns else i,
            "coords": frame_df[["x", "y", "z"]].to_numpy(dtype=float),
            "atom_types": frame_df["atom_type"].astype(str).tolist(),
        }
        if i < len(self._box_bounds) and self._box_bounds[i] is not None:
            payload["box_bounds"] = self._box_bounds[i]
        return payload

    def iter_frames(self, step: int = 1) -> Iterator[dict[str, Any]]:
        """Iterate parsed frames with optional stride.

        Parameters
        ----------
        step : int, optional
            Positive stride for frame sampling.

        Returns
        -------
        Iterator[dict[str, Any]]
            Iterator of frame payload dictionaries from ``frame(i)``.

        Examples
        --------
        ```python
        for fr in handler.iter_frames(step=10):
            ...
        ```
        """
        for i in range(0, self.n_frames(), max(1, int(step))):
            yield self.frame(i)


__all__ = ["LAMMPSDumpHandler", "LAMMPSLogHandler"]
