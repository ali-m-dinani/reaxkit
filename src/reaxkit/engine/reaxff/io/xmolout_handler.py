"""
ReaxFF trajectory output (xmolout) handler.

This module provides a handler for parsing ReaxFF ``xmolout`` files,
which store atomic trajectories from MD runs or MM minimizations.

``xmolout`` files contain repeated coordinate frames with associated
cell parameters and energies and are commonly used for visualization
and structural analysis.

**Usage context**

- ReaxFF parsing: Read ReaxFF text outputs into normalized tabular structures.
- Workflow ingestion: Provide canonical handler interfaces used by adapters/workflows.
- Diagnostics/export: Preserve parsed metadata for reporting and downstream conversion.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import pickle
import shutil
from time import perf_counter
from typing import List, Optional, Iterator, Dict, Any
import numpy as np
import pandas as pd
from reaxkit.core.platform.exceptions import ParseError
from reaxkit.core.storage.frame_store import FrameOffset, FrameStore, IndexCoverage
from reaxkit.engine.reaxff.io.base import BaseHandler
from reaxkit.engine.reaxff.io.frame_header_validation import validate_geometry_name


_FRAME_CACHE_BATCH_SIZE = 16
_OFFSET_CACHE_BATCH_SIZE = 128


def _parse_xmolout_header(
        raw: str,
        *,
        path: str | Path,
        frame_index: int,
        line_number: int | None,
) -> tuple[str, int, list[float]]:
    """Parse one xmolout frame header and provide a format-specific error."""
    values = raw.split()
    location = f"frame {frame_index}"
    if line_number is not None:
        location += f", line {line_number}"
    if len(values) != 9 or not values[1].lstrip("-").isdigit():
        raise ParseError(
            f"Malformed xmolout frame header in '{Path(path)}' ({location}). "
            "Expected: geometry_name iteration energy a b c alpha beta gamma. "
            f"Header: {raw.strip()!r}"
        )
    name = validate_geometry_name(
        values[0],
        file_kind="xmolout",
        path=path,
        frame_index=frame_index,
        line_number=line_number,
        header=raw,
    )
    try:
        numeric_values = [float(value) for value in values[2:]]
    except ValueError as exc:
        raise ParseError(
            f"Malformed xmolout frame header in '{Path(path)}' ({location}): "
            f"energy or cell fields are not numeric. Header: {raw.strip()!r}"
        ) from exc
    return name, int(values[1]), numeric_values


class XmoloutHandler(BaseHandler):
    """
    Parser for ReaxFF trajectory output files (``xmolout``).

    This class parses ``xmolout`` files and exposes both a per-frame
    summary table and per-frame atomic coordinate tables.

    Parsed Data
    -----------
    Summary table
        One row per frame, returned by ``dataframe()``, with columns:
        ["num_of_atoms", "iter", "E_pot",
         "a", "b", "c", "alpha", "beta", "gamma"]

        Duplicate iteration indices are removed by keeping the last
        occurrence.

    Per-frame atom tables
        Stored in ``self._frames``, one table per frame, where each table
        has at least the columns:
        ["atom_type", "x", "y", "z"]

        Any additional per-atom columns present in the file are preserved
        per frame. If their names are not provided explicitly, they are
        auto-named as ``unknown_1``, ``unknown_2``, …

    Metadata
        Returned by ``metadata()``, containing:
        ["simulation_name", "n_atoms", "n_frames", "has_time"]

    Notes
    -----
    - Frames are inferred from the repeating ``#atoms → header → atoms`` pattern.
    - The number of atoms is assumed constant across all frames.
    - This handler supports lightweight frame access via ``frame(i)``
      and streaming access via ``iter_frames(step=...)``.
    """

    def __init__(
            self,
            file_path: str | Path = "xmolout",
            *,
            extra_atom_cols: Optional[list[str]] = None,
            frame_indices: Optional[list[int]] = None,
            reporter=None,
            input_cache: bool = True,
            frame_cache_root: str | Path | None = None,
    ):
        """
        Initialize the instance.

        Parameters
        ----------
        file_path : str | Path
            Parameter description.
        extra_atom_cols : Optional[list[str]]
            Parameter description.

        """
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []  # list of per-frame atom tables
        self._n_atoms: Optional[int] = None
        self.simulation_name: str = ""
        self._extra_atom_cols = list(extra_atom_cols) if extra_atom_cols else None
        self._frame_indices = (
            tuple(dict.fromkeys(int(i) for i in frame_indices if int(i) >= 0))
            if frame_indices is not None
            else None
        )
        self._reporter = reporter
        self._input_cache = bool(input_cache)
        self._frame_cache_root = Path(frame_cache_root) if frame_cache_root else None
        self._frame_cache_stats: dict[str, Any] = {}

    # ---- FileHandler requirement
    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
         parse.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, Any]]
            Return value description.

        """
        if self._frame_indices is not None:
            return self._parse_selected_frames()

        sim_rows: List[list] = []
        frames: List[pd.DataFrame] = []

        sim_cols = ["num_of_atoms", "iter", "E_pot", "a", "b", "c", "alpha", "beta", "gamma"]
        base_atom_cols = ["atom_type", "x", "y", "z"]
        total_lines = self._count_lines()

        with open(self.path, "r") as fh:
            atom_buf: List[list] = []
            atom_count = 0
            current_atom_cols: Optional[List[str]] = None
            n_atoms: Optional[int] = None
            lines_read = 0

            frame_index = -1
            expecting_header = False
            for line in fh:
                lines_read += 1
                if self._reporter and (lines_read % 5000 == 0 or lines_read == total_lines):
                    self._reporter("load", lines_read, total_lines, "Parsing xmolout")
                vals = line.strip().split()
                if not vals:
                    continue

                # #atoms line
                if len(vals) == 1 and vals[0].isdigit():
                    frame_index += 1
                    n_atoms = int(vals[0])
                    self._n_atoms = n_atoms
                    sim_rows.append([n_atoms])  # placeholder row; will complete after header line
                    atom_buf, atom_count = [], 0
                    current_atom_cols = None
                    expecting_header = True
                    continue

                # header line (name iter E a b c alpha beta gamma)
                if expecting_header:
                    name, iteration, numeric_values = _parse_xmolout_header(
                        line,
                        path=self.path,
                        frame_index=frame_index,
                        line_number=lines_read,
                    )
                    if not self.simulation_name:
                        self.simulation_name = name
                    row = [self._n_atoms, iteration, *numeric_values]
                    sim_rows[-1] = row
                    expecting_header = False
                    continue

                # atom coordinates (optionally with extra columns)
                if self._n_atoms and len(vals) >= 4:
                    # lazily determine expected columns for this frame
                    if current_atom_cols is None:
                        n_extras = max(0, len(vals) - 4)
                        if self._extra_atom_cols:
                            names = list(self._extra_atom_cols)[:n_extras]
                            if len(names) < n_extras:
                                names += [f"unknown_{i + 1}" for i in range(n_extras - len(names))]
                        else:
                            names = [f"unknown_{i + 1}" for i in range(n_extras)]
                        current_atom_cols = base_atom_cols + names

                    base = [vals[0]] + list(map(float, vals[1:4]))
                    expected_extras = len(current_atom_cols) - 4
                    extras_vals = [float(x) for x in vals[4:4 + expected_extras]]
                    # pad if fewer provided
                    while len(extras_vals) < expected_extras:
                        extras_vals.append(float('nan'))
                    atom_buf.append(base + extras_vals)
                    atom_count += 1

                    if atom_count == self._n_atoms:
                        frames.append(pd.DataFrame(atom_buf, columns=current_atom_cols))
                        atom_buf, atom_count = [], 0
                        current_atom_cols = None

        # Build per-frame summary table
        df = pd.DataFrame(sim_rows, columns=sim_cols)

        # Deduplicate by iter (keep last)
        if not df.empty and "iter" in df.columns:
            keep_idx = df.drop_duplicates("iter", keep="last").index
            frames = [frames[i] for i in keep_idx if i < len(frames)]
            df = df.loc[keep_idx].reset_index(drop=True)

        # Save frames
        self._frames = frames

        meta: Dict[str, Any] = {
            "simulation_name": self.simulation_name,
            "n_atoms": self._n_atoms,
            "n_frames": len(self._frames),
            "has_time": False,
        }
        if self._reporter:
            self._reporter("load", total_lines, total_lines, "Finished parsing xmolout")
        return df, meta

    def _frame_store(
            self,
            *,
            streaming: bool = False,
            coordinates_only: bool = False,
    ) -> FrameStore | None:
        """Return the shared source-frame store, or ``None`` when disabled."""
        if not self._input_cache:
            return None
        root = self._frame_cache_root
        if root is None:
            configured = os.environ.get("REAXKIT_FRAME_CACHE_DIR", "").strip()
            root = Path(configured) if configured else self._cache_root().parent
        try:
            representation = (
                "xmolout-coordinate-stream-v1"
                if streaming and coordinates_only
                else "xmolout-stream-v1"
                if streaming
                else "xmolout-full-frame-v1"
            )
            capabilities = (
                ("coordinates", "cell", "energy")
                if streaming and coordinates_only
                else ("coordinates", "atom-extras", "cell", "energy")
            )
            return FrameStore.for_source(
                root,
                self.path,
                engine="reaxff",
                source_kind="xmolout",
                parser=f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                parser_version="4",
                representation=representation,
                capabilities=capabilities,
                options={"extra_atom_cols": self._extra_atom_cols or []},
            )
        except (OSError, RuntimeError, ValueError):
            return None

    @staticmethod
    def _next_nonempty_binary(handle) -> tuple[int, bytes, int] | None:
        while True:
            start = handle.tell()
            raw = handle.readline()
            if not raw:
                return None
            if raw.strip():
                return start, raw, handle.tell()

    def _scan_offsets(
            self,
            store: FrameStore,
            *,
            through_index: int,
            progress_stage: str = "load",
            progress_total: int = 0,
            progress_message: str = "Indexing xmolout frames",
    ) -> dict[str, Any]:
        """Extend the byte index through ``through_index`` without building frames."""
        coverage = store.get_coverage()
        started_at = perf_counter()
        bytes_read = 0
        indexed = 0
        if coverage.complete or coverage.next_frame_index > through_index:
            return {"indexed_frames": 0, "index_bytes": 0, "index_seconds": 0.0}

        pending_offsets: list[FrameOffset] = []

        def flush_offsets(next_frame_index: int, next_byte_offset: int) -> None:
            if pending_offsets:
                store.put_offsets(pending_offsets)
                pending_offsets.clear()
            store.set_coverage(IndexCoverage(next_frame_index, next_byte_offset, False))

        with self.path.open("rb") as handle:
            handle.seek(coverage.next_byte_offset)
            frame_index = coverage.next_frame_index
            while frame_index <= through_index:
                item = self._next_nonempty_binary(handle)
                if item is None:
                    flush_offsets(frame_index, handle.tell())
                    store.set_coverage(IndexCoverage(frame_index, handle.tell(), True))
                    break
                byte_start, raw_count, _ = item
                values = raw_count.decode("utf-8").strip().split()
                if len(values) != 1 or not values[0].isdigit():
                    bytes_read += handle.tell() - byte_start
                    continue
                atom_count = int(values[0])
                header_item = self._next_nonempty_binary(handle)
                if header_item is None:
                    flush_offsets(frame_index, byte_start)
                    store.set_coverage(IndexCoverage(frame_index, byte_start, True))
                    break
                _, raw_header, _ = header_item
                name, iteration, _ = _parse_xmolout_header(
                    raw_header.decode("utf-8"),
                    path=self.path,
                    frame_index=frame_index,
                    line_number=None,
                )
                if not self.simulation_name:
                    self.simulation_name = name
                complete = True
                for _atom_index in range(atom_count):
                    if self._next_nonempty_binary(handle) is None:
                        complete = False
                        break
                if not complete:
                    flush_offsets(frame_index, byte_start)
                    store.set_coverage(IndexCoverage(frame_index, byte_start, True))
                    break
                byte_end = handle.tell()
                pending_offsets.append(
                    FrameOffset(frame_index, byte_start, byte_end, iteration, atom_count)
                )
                frame_index += 1
                indexed += 1
                bytes_read += byte_end - byte_start
                if len(pending_offsets) >= _OFFSET_CACHE_BATCH_SIZE:
                    flush_offsets(frame_index, byte_end)
                if self._reporter and progress_total > 0:
                    self._reporter(
                        progress_stage,
                        indexed,
                        progress_total,
                        progress_message,
                    )
            else:
                # Coverage is a safe resume point even though EOF is not yet known.
                flush_offsets(frame_index, handle.tell())
        return {
            "indexed_frames": indexed,
            "index_bytes": bytes_read,
            "index_seconds": perf_counter() - started_at,
        }

    def _stream_selected_frames_one_pass(
            self,
            store: FrameStore,
            requested_order: list[int],
            *,
            coordinates_only: bool,
    ) -> Iterator[Dict[str, Any]]:
        """Populate a cold stream cache without indexing and rereading the source."""
        pending: dict[int, dict[str, Any]] = {}
        parsed_frames = 0
        if callable(self._reporter):
            self._reporter("stream", 0, len(requested_order), "Reading xmolout frames")
        sequential = XmoloutHandler(
            self.path,
            extra_atom_cols=self._extra_atom_cols,
            frame_indices=requested_order,
            reporter=self._reporter,
            input_cache=False,
        )
        for record in sequential.stream_file_frames(coordinates_only=coordinates_only):
            source_index = int(record["source_index"])
            cached_record = dict(record)
            cached_record["simulation_name"] = sequential.simulation_name
            pending[source_index] = cached_record
            parsed_frames += 1
            if len(pending) >= _FRAME_CACHE_BATCH_SIZE:
                store.put_frames(pending)
                pending.clear()
            yield record
        if pending:
            store.put_frames(pending)
        if callable(self._reporter):
            self._reporter(
                "stream",
                parsed_frames,
                len(requested_order),
                "Read xmolout frames",
            )
        self.simulation_name = sequential.simulation_name
        self._frame_cache_stats = {
            "requested": len(requested_order),
            "hits": 0,
            "misses": len(requested_order),
            "parsed_frames": parsed_frames,
            "source_bytes": int(self.path.stat().st_size),
            "indexed_frames": 0,
            "index_bytes": 0,
            "index_seconds": 0.0,
            "one_pass": True,
        }

    def _parse_indexed_frame(self, handle, offset: FrameOffset) -> dict[str, Any]:
        """Parse one canonical full frame from a verified byte range."""
        handle.seek(offset.byte_start)
        count_item = self._next_nonempty_binary(handle)
        if count_item is None:
            raise ParseError(f"Missing xmolout frame {offset.frame_index} in '{self.path}'.")
        values = count_item[1].decode("utf-8").strip().split()
        if len(values) != 1 or not values[0].isdigit():
            raise ParseError(f"Malformed xmolout atom count in '{self.path}' (frame {offset.frame_index}).")
        n_atoms = int(values[0])
        header_item = self._next_nonempty_binary(handle)
        if header_item is None:
            raise ParseError(f"Missing xmolout frame header in '{self.path}' (frame {offset.frame_index}).")
        name, iteration, numeric_values = _parse_xmolout_header(
            header_item[1].decode("utf-8"),
            path=self.path,
            frame_index=offset.frame_index,
            line_number=None,
        )
        atom_rows: list[list[Any]] = []
        atom_columns: list[str] | None = None
        for _atom_index in range(n_atoms):
            atom_item = self._next_nonempty_binary(handle)
            if atom_item is None:
                raise ParseError(f"Truncated xmolout atom block in '{self.path}' (frame {offset.frame_index}).")
            atom_values = atom_item[1].decode("utf-8").strip().split()
            if len(atom_values) < 4:
                raise ParseError(f"Malformed xmolout atom row in '{self.path}' (frame {offset.frame_index}).")
            if atom_columns is None:
                n_extras = max(0, len(atom_values) - 4)
                if self._extra_atom_cols:
                    extra_names = list(self._extra_atom_cols)[:n_extras]
                    extra_names.extend(
                        f"unknown_{i + 1}" for i in range(len(extra_names), n_extras)
                    )
                else:
                    extra_names = [f"unknown_{i + 1}" for i in range(n_extras)]
                atom_columns = ["atom_type", "x", "y", "z", *extra_names]
            extras = [float(value) for value in atom_values[4:len(atom_columns)]]
            extras.extend([float("nan")] * (len(atom_columns) - 4 - len(extras)))
            atom_rows.append(
                [atom_values[0], *[float(value) for value in atom_values[1:4]], *extras]
            )
        return {
            "summary": [n_atoms, iteration, *numeric_values],
            "frame": pd.DataFrame(
                atom_rows,
                columns=atom_columns or ["atom_type", "x", "y", "z"],
            ),
            "simulation_name": name,
            "source_bytes": int(offset.byte_end - offset.byte_start),
        }

    def _load_indexed_records(
            self,
            store: FrameStore,
            requested: list[int],
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        cache_started = perf_counter()
        records = store.get_frames(requested)
        cache_seconds = perf_counter() - cache_started
        missing = [index for index in requested if index not in records]
        through_index = max(missing, default=-1)
        coverage = store.get_coverage()
        index_work = (
            max(0, through_index - coverage.next_frame_index + 1)
            if missing and not coverage.complete
            else 0
        )
        total_work = index_work + len(requested)
        if self._reporter:
            self._reporter("load", 0, total_work, "Reading xmolout frames")
        scan_stats = self._scan_offsets(
            store,
            through_index=through_index,
            progress_total=total_work,
            progress_message="Reading xmolout frames",
        ) if missing else {
            "indexed_frames": 0,
            "index_bytes": 0,
            "index_seconds": 0.0,
        }
        offsets = store.get_offsets(missing)
        completed = len(records)
        if self._reporter:
            self._reporter(
                "load",
                index_work + completed,
                total_work,
                "Reading xmolout frames",
            )
        source_started = perf_counter()
        parsed: dict[int, dict[str, Any]] = {}
        source_bytes = 0
        if offsets:
            with self.path.open("rb") as handle:
                for frame_index in missing:
                    offset = offsets.get(frame_index)
                    if offset is None:
                        continue
                    record = self._parse_indexed_frame(handle, offset)
                    parsed[frame_index] = record
                    source_bytes += int(record.get("source_bytes", 0))
                    completed += 1
                    if self._reporter:
                        self._reporter(
                            "load",
                            index_work + completed,
                            total_work,
                            "Reading xmolout frames",
                        )
        source_seconds = perf_counter() - source_started
        write_started = perf_counter()
        store.put_frames(parsed)
        write_seconds = perf_counter() - write_started
        records.update(parsed)
        stats = {
            "requested": len(requested),
            "hits": len(requested) - len(missing),
            "misses": len(missing),
            "parsed_frames": len(parsed),
            "source_bytes": source_bytes,
            "cache_read_seconds": cache_seconds,
            "source_read_seconds": source_seconds,
            "cache_write_seconds": write_seconds,
            **scan_stats,
        }
        return records, stats

    def _parse_selected_frames(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load explicit frames from the shared frame store with a safe fallback."""
        requested = list(self._frame_indices or ())
        store = self._frame_store()
        if store is None:
            return self._parse_selected_frames_sequential()
        records, stats = self._load_indexed_records(store, requested)
        self._frame_cache_stats = stats
        source_indices = [index for index in requested if index in records]
        sim_rows = [records[index]["summary"] for index in source_indices]
        frames = [records[index]["frame"] for index in source_indices]
        if source_indices:
            first = records[source_indices[0]]
            self.simulation_name = str(first.get("simulation_name") or "")
            self._n_atoms = int(first["summary"][0])
        sim_cols = ["num_of_atoms", "iter", "E_pot", "a", "b", "c", "alpha", "beta", "gamma"]
        df = pd.DataFrame(sim_rows, columns=sim_cols)
        if not df.empty:
            keep_idx = df.drop_duplicates("iter", keep="last").index.tolist()
            frames = [frames[index] for index in keep_idx]
            source_indices = [source_indices[index] for index in keep_idx]
            df = df.iloc[keep_idx].reset_index(drop=True)
        self._frames = frames
        meta: Dict[str, Any] = {
            "simulation_name": self.simulation_name,
            "n_atoms": self._n_atoms,
            "n_frames": len(frames),
            "has_time": False,
            "source_frame_indices": source_indices,
            "partial": True,
            "frame_cache": stats,
        }
        return df, meta

    def _parse_selected_frames_sequential(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Parse only explicitly requested frames and stop after the last one."""
        sim_cols = ["num_of_atoms", "iter", "E_pot", "a", "b", "c", "alpha", "beta", "gamma"]
        base_atom_cols = ["atom_type", "x", "y", "z"]
        requested = list(self._frame_indices or ())
        requested_set = set(requested)
        max_requested = max(requested, default=-1)
        records: dict[int, tuple[list[Any], pd.DataFrame]] = {}

        current_index = -1
        current_selected = False
        current_summary: list[Any] | None = None
        atom_buf: list[list[Any]] = []
        atom_count = 0
        current_atom_cols: list[str] | None = None

        with open(self.path, "r") as fh:
            line_number = 0
            expecting_header = False
            for line in fh:
                line_number += 1
                vals = line.strip().split()
                if not vals:
                    continue

                if len(vals) == 1 and vals[0].isdigit():
                    current_index += 1
                    if current_index > max_requested:
                        break
                    self._n_atoms = int(vals[0])
                    current_selected = current_index in requested_set
                    current_summary = [self._n_atoms] if current_selected else None
                    atom_buf = []
                    atom_count = 0
                    current_atom_cols = None
                    expecting_header = True
                    continue

                if expecting_header:
                    name, iteration, numeric_values = _parse_xmolout_header(
                        line,
                        path=self.path,
                        frame_index=current_index,
                        line_number=line_number,
                    )
                    if not self.simulation_name:
                        self.simulation_name = name
                    if current_selected:
                        current_summary = [self._n_atoms, iteration, *numeric_values]
                    expecting_header = False
                    continue

                if not current_selected or not self._n_atoms or len(vals) < 4:
                    continue

                if current_atom_cols is None:
                    n_extras = max(0, len(vals) - 4)
                    if self._extra_atom_cols:
                        names = list(self._extra_atom_cols)[:n_extras]
                        if len(names) < n_extras:
                            names += [f"unknown_{i + 1}" for i in range(n_extras - len(names))]
                    else:
                        names = [f"unknown_{i + 1}" for i in range(n_extras)]
                    current_atom_cols = base_atom_cols + names

                base = [vals[0]] + list(map(float, vals[1:4]))
                expected_extras = len(current_atom_cols) - 4
                extras_vals = [float(x) for x in vals[4:4 + expected_extras]]
                extras_vals.extend([float("nan")] * (expected_extras - len(extras_vals)))
                atom_buf.append(base + extras_vals)
                atom_count += 1

                if atom_count == self._n_atoms:
                    if current_summary is not None:
                        records[current_index] = (
                            current_summary,
                            pd.DataFrame(atom_buf, columns=current_atom_cols),
                        )
                    if self._reporter:
                        done = sum(1 for i in requested if i in records)
                        self._reporter("load", done, len(requested), "Loading selected xmolout frames")
                    atom_buf = []
                    atom_count = 0
                    current_atom_cols = None
                    if len(records) == len(requested_set):
                        break

        source_indices = [i for i in requested if i in records]
        sim_rows = [records[i][0] for i in source_indices]
        frames = [records[i][1] for i in source_indices]
        df = pd.DataFrame(sim_rows, columns=sim_cols)

        if not df.empty and "iter" in df.columns:
            keep_idx = df.drop_duplicates("iter", keep="last").index.tolist()
            frames = [frames[i] for i in keep_idx]
            source_indices = [source_indices[i] for i in keep_idx]
            df = df.iloc[keep_idx].reset_index(drop=True)

        self._frames = frames
        meta: Dict[str, Any] = {
            "simulation_name": self.simulation_name,
            "n_atoms": self._n_atoms,
            "n_frames": len(frames),
            "has_time": False,
            "source_frame_indices": source_indices,
            "partial": True,
        }
        if self._reporter:
            self._reporter("load", len(source_indices), len(requested), "Finished loading selected xmolout frames")
        return df, meta

    def _count_lines(self) -> int:
        """Count lines."""
        with open(self.path, "r") as fh:
            return sum(1 for _ in fh)

    def stream_file_frames(self, *, coordinates_only: bool = False) -> Iterator[Dict[str, Any]]:
        """Yield coordinate frames directly from ``xmolout`` without caching them.

        Unlike :meth:`iter_frames`, this method does not call ``parse()`` and
        never populates ``self._frames``.  At most one atom table is retained
        while the caller consumes the iterator.  ``coordinates_only`` avoids
        building a pandas table and parsing unused per-atom columns.  It is
        intended for total electrostatics, which consumes only XYZ positions.
        """
        if self._frame_indices is not None:
            requested_order = list(self._frame_indices)
            store = self._frame_store(
                streaming=True,
                coordinates_only=coordinates_only,
            )
            if store is not None:
                initially_available = store.available_indices(requested_order)
                missing = [index for index in requested_order if index not in initially_available]
                through_index = max(missing, default=-1)
                coverage = store.get_coverage()
                if (
                        missing
                        and not initially_available
                        and coverage.next_frame_index == 0
                        and requested_order == sorted(requested_order)
                ):
                    yield from self._stream_selected_frames_one_pass(
                        store,
                        requested_order,
                        coordinates_only=coordinates_only,
                    )
                    return
                index_work = (
                    max(0, through_index - coverage.next_frame_index + 1)
                    if missing and not coverage.complete
                    else 0
                )
                total_work = index_work + len(requested_order)
                if callable(self._reporter):
                    self._reporter("stream", 0, total_work, "Reading xmolout frames")
                scan_stats = self._scan_offsets(
                    store,
                    through_index=through_index,
                    progress_stage="stream",
                    progress_total=total_work,
                    progress_message="Reading xmolout frames",
                ) if missing else {"indexed_frames": 0, "index_bytes": 0, "index_seconds": 0.0}
                offsets = store.get_offsets(missing)
                parsed_frames = 0
                source_bytes = 0
                pending: dict[int, dict[str, Any]] = {}
                with self.path.open("rb") as handle:
                    for batch_start in range(0, len(requested_order), _FRAME_CACHE_BATCH_SIZE):
                        batch = requested_order[batch_start:batch_start + _FRAME_CACHE_BATCH_SIZE]
                        cached_batch = store.get_frames(batch)
                        for batch_offset, source_index in enumerate(batch, start=1):
                            record = cached_batch.get(source_index)
                            if record is None:
                                offset = offsets.get(source_index)
                                if offset is None:
                                    continue
                                indexed_record = self._parse_indexed_frame(handle, offset)
                                summary = indexed_record["summary"]
                                frame = indexed_record["frame"]
                                record = {
                                    "source_index": source_index,
                                    "iter": int(summary[1]),
                                    "num_of_atoms": int(summary[0]),
                                    "potential_energy": float(summary[2]),
                                    "cell_lengths": list(summary[3:6]),
                                    "cell_angles": list(summary[6:9]),
                                    "simulation_name": str(
                                        indexed_record.get("simulation_name") or ""
                                    ),
                                }
                                if coordinates_only:
                                    record["coordinates"] = frame[["x", "y", "z"]].to_numpy(dtype=float)
                                    record["elements"] = frame["atom_type"].astype(str).tolist()
                                else:
                                    record["frame"] = frame
                                pending[source_index] = record
                                parsed_frames += 1
                                source_bytes += int(indexed_record.get("source_bytes", 0))
                            if not self.simulation_name:
                                self.simulation_name = str(record.get("simulation_name") or "")
                            emitted = batch_start + batch_offset
                            if callable(self._reporter):
                                self._reporter(
                                    "stream",
                                    index_work + emitted,
                                    total_work,
                                    "Reading xmolout frames",
                                )
                            output = dict(record)
                            output.pop("simulation_name", None)
                            yield output
                        if pending:
                            store.put_frames(pending)
                            pending.clear()
                self._frame_cache_stats = {
                    "requested": len(requested_order),
                    "hits": len(initially_available),
                    "misses": len(requested_order) - len(initially_available),
                    "parsed_frames": parsed_frames,
                    "source_bytes": source_bytes,
                    **scan_stats,
                }
                if callable(self._reporter):
                    self._reporter(
                        "stream",
                        total_work,
                        total_work,
                        "Read xmolout frames",
                    )
                return

        requested = set(self._frame_indices) if self._frame_indices is not None else None
        max_requested = max(requested, default=-1) if requested is not None else None
        source_index = -1
        emitted = 0

        with open(self.path, "r", encoding="utf-8") as fh:
            while True:
                count_line = next((raw.strip() for raw in fh if raw.strip()), None)
                if count_line is None:
                    break
                values = count_line.split()
                if len(values) != 1 or not values[0].isdigit():
                    continue

                source_index += 1
                if max_requested is not None and source_index > max_requested:
                    break
                n_atoms = int(values[0])
                header = next((raw.strip() for raw in fh if raw.strip()), None)
                if header is None:
                    break
                name, iteration, numeric_values = _parse_xmolout_header(
                    header,
                    path=self.path,
                    frame_index=source_index,
                    line_number=None,
                )
                selected = requested is None or source_index in requested

                atom_rows: list[list[Any]] = []
                coordinates = np.empty((n_atoms, 3), dtype=float) if coordinates_only else None
                elements: list[str] = []
                atom_columns: list[str] | None = None
                for atom_index in range(n_atoms):
                    atom_line = next((raw.strip() for raw in fh if raw.strip()), None)
                    if atom_line is None:
                        break
                    if not selected:
                        continue
                    atom_values = atom_line.split(None, 4) if coordinates_only else atom_line.split()
                    if len(atom_values) < 4:
                        continue
                    if coordinates_only:
                        elements.append(atom_values[0])
                        coordinates[atom_index] = (
                            float(atom_values[1]),
                            float(atom_values[2]),
                            float(atom_values[3]),
                        )
                        continue
                    if atom_columns is None:
                        n_extras = max(0, len(atom_values) - 4)
                        if self._extra_atom_cols:
                            extra_names = list(self._extra_atom_cols)[:n_extras]
                            extra_names.extend(
                                f"unknown_{i + 1}"
                                for i in range(len(extra_names), n_extras)
                            )
                        else:
                            extra_names = [f"unknown_{i + 1}" for i in range(n_extras)]
                        atom_columns = ["atom_type", "x", "y", "z", *extra_names]
                    extras = [float(value) for value in atom_values[4:len(atom_columns)]]
                    extras.extend([float("nan")] * (len(atom_columns) - 4 - len(extras)))
                    atom_rows.append(
                        [atom_values[0], *[float(value) for value in atom_values[1:4]], *extras]
                    )

                if not selected:
                    continue
                if not self.simulation_name:
                    self.simulation_name = name
                emitted += 1
                if callable(self._reporter):
                    total = len(requested) if requested is not None else 0
                    self._reporter("stream", emitted, total, "Streaming xmolout frames")
                record = {
                    "source_index": source_index,
                    "iter": iteration,
                    "num_of_atoms": n_atoms,
                    "potential_energy": numeric_values[0],
                    "cell_lengths": numeric_values[1:4],
                    "cell_angles": numeric_values[4:7],
                }
                if coordinates_only:
                    record["coordinates"] = coordinates
                    record["elements"] = elements
                else:
                    record["frame"] = pd.DataFrame(
                        atom_rows,
                        columns=atom_columns or ["atom_type", "x", "y", "z"],
                    )
                yield record

    # ---- disk-cache override (parquet + json) -------------------
    def _disk_cache_dir(self, key: str) -> Path:
        """Disk cache dir."""
        return self._cache_root() / key

    def _store_in_disk_cache(self, key: str, payload: bytes) -> None:
        """Store in disk cache."""
        super()._store_in_disk_cache(key, payload)

    def _load_from_disk_cache(self, key: str) -> bytes | None:
        """Load from disk cache."""
        return super()._load_from_disk_cache(key)

    # ---- Explicit, file-specific accessors (no generic get())
    def n_frames(self) -> int:
        """
        N frames.

        Returns
        -------
        int
            Return value description.

        """
        return int(self.metadata().get("n_frames", 0))

    def n_atoms(self) -> Optional[int]:
        """
        N atoms.

        Returns
        -------
        Optional[int]
            Return value description.

        """
        return self._n_atoms

    def frame(self, i: int) -> Dict[str, Any]:
        """
        Frame.

        Parameters
        ----------
        i : int
            Parameter description.

        Returns
        -------
        Dict[str, Any]
            Return value description.

        """
        df = self.dataframe()
        if i < 0 or i >= len(self._frames):
            raise IndexError(f"frame index {i} out of range [0, {len(self._frames) - 1}]")

        frame_df = self._frames[i]
        coords = frame_df[["x", "y", "z"]].to_numpy(dtype=float)
        atom_types = frame_df["atom_type"].astype(str).tolist()

        row = df.iloc[i]
        return {
            "index": i,
            "source_index": int(self._meta.get("source_frame_indices", [i] * len(self._frames))[i]),
            "iter": int(row["iter"]) if "iter" in df.columns else i,
            "coords": coords,
            "atom_types": atom_types,
        }

    def iter_frames(self, step: int = 1) -> Iterator[Dict[str, Any]]:
        """
        Iter frames.

        Parameters
        ----------
        step : int
            Parameter description.

        Yields
        -------
        Iterator[Dict[str, Any]]
            Return value description.

        """
        for i in range(0, self.n_frames(), max(1, int(step))):
            yield self.frame(i)
