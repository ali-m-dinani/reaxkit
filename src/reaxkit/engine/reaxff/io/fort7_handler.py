"""
ReaxFF connectivity (fort.7) file handler.

This module provides a handler for parsing ReaxFF ``fort.7`` files,
which store per-iteration atom connectivity, bond-order information,
and system-wide totals.

Typical use cases include:

- extracting per-atom bond-order features
- computing coordination statistics
- building molecule- and structure-level descriptors

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
import re
import shutil
from typing import List, Dict, Any, Iterator, Optional
import numpy as np
import pandas as pd

from reaxkit.core.platform.exceptions import ParseError
from reaxkit.core.storage.frame_store import FrameOffset, FrameStore, IndexCoverage
from reaxkit.engine.reaxff.io.base import BaseHandler
from reaxkit.engine.reaxff.io.frame_header_validation import validate_geometry_name

_FORT7_HEADER_RE = re.compile(
    r"^\s*(?P<num_atoms>\d+)\s+(?P<simulation_name>\S+)\s+Iteration:\s*(?P<iteration>\d+)\s+#Bonds:\s*(?P<num_bonds>\d+)\s*$"
)
_FORT7_FLOAT_FIELD_RE = re.compile(
    r"(?<!\S)[+-]?(?:\d*\.\d+|\d+\.?\d*[Ee][+-]?\d+)(?=\s|$)"
)


def _validate_fort7_header(
        match: re.Match[str],
        *,
        path: str | Path,
        frame_index: int,
        line_number: int | None,
        header: str,
) -> None:
    validate_geometry_name(
        match.group("simulation_name"),
        file_kind="fort.7",
        path=path,
        frame_index=frame_index,
        line_number=line_number,
        header=header,
    )


def _match_fort7_header(
        raw: str,
        *,
        path: str | Path,
        frame_index: int,
        line_number: int | None,
) -> re.Match[str] | None:
    """Match a fort.7 header and reject header-like malformed lines."""
    match = _FORT7_HEADER_RE.match(raw)
    if match is not None:
        _validate_fort7_header(
            match,
            path=path,
            frame_index=frame_index,
            line_number=line_number,
            header=raw,
        )
        return match
    if "Iteration:" not in raw and "#Bonds:" not in raw:
        return None

    location = f"frame {frame_index}"
    if line_number is not None:
        location += f", line {line_number}"
    raise ParseError(
        f"Malformed fort.7 frame header in '{Path(path)}' ({location}). "
        "Expected: atom_count geometry_name Iteration: iteration #Bonds: count. "
        "The geometry name must be a short token without spaces, quotes, or '='. "
        f"Header: {raw.strip()!r}"
    )


class Fort7Handler(BaseHandler):
    """
    Parser for ReaxFF connectivity output files (``fort.7``).

    This class parses ReaxFF ``fort.7`` files and exposes both
    iteration-level summaries and per-iteration atom connectivity
    tables as structured tabular data.

    Parsed Data
    -----------
    Summary table
        One row per iteration, returned by ``dataframe()``, with columns:
        ["iter", "num_of_atoms", "num_of_bonds",
         "total_BO", "total_LP", "total_BO_uncorrected", "total_charge"]

    Per-frame atom tables
        Stored in ``self._frames``, one table per iteration, where each
        frame is a ``pandas.DataFrame`` with columns:
        ["atom_num", "atom_type_num", "atom_cnn1..nb", "molecule_num",
         "BO1..nb", "sum_BOs", "num_LPs", "partial_charge", ...]

        Here, ``nb`` denotes the number of bonded neighbors in that frame,
        leading to variable-length connectivity and bond-order columns.

    Metadata
        Returned by ``metadata()``, containing:
        ["n_frames", "n_records", "simulation_name"]

    Notes
    -----
    - Duplicate iterations are resolved by keeping the last occurrence.
    - Connectivity and bond-order columns are inferred from the header.
    - Extra, file-dependent columns are preserved as ``unknown*`` fields.
    """

    def __init__(
            self,
            file_path: str | Path = "fort.7",
            reporter=None,
            *,
            frame_indices: Optional[list[int]] = None,
            input_cache: bool = True,
            frame_cache_root: str | Path | None = None,
    ):
        """Initialize a handler for a ReaxFF ``fort.7`` connectivity file.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Parameters
        ----------
        file_path : str or pathlib.Path, optional
            Path to the ``fort.7`` file to be parsed.

        Returns
        -------
        None
            Initializes the handler without parsing the file.
        """
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []
        self._sim_name: Optional[str] = None
        self._frame_indices = (
            tuple(dict.fromkeys(int(i) for i in frame_indices if int(i) >= 0))
            if frame_indices is not None
            else None
        )
        self._reporter = reporter
        self._input_cache = bool(input_cache)
        self._frame_cache_root = Path(frame_cache_root) if frame_cache_root else None
        self._frame_cache_stats: dict[str, Any] = {}

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

        sim_rows: List[List[Any]] = []
        frames: List[pd.DataFrame] = []
        totals: List[List[float]] = []

        cur_atoms_rows: List[List[float | int]] = []
        cur_totals: List[float] = []
        cur_num_particles: Optional[int] = None
        cur_nbonds: Optional[int] = None
        sim_name: str = ""
        warned_large_atom_count = False

        def _finalize_iteration() -> None:
            """Finalize iteration."""
            if cur_num_particles is None or cur_nbonds is None or not cur_atoms_rows:
                return
            nb = int(cur_nbonds)
            atom_cols = (
                    ["atom_num", "atom_type_num"]
                    + [f"atom_cnn{i}" for i in range(1, nb + 1)]
                    + ["molecule_num"]
                    + [f"BO{i}" for i in range(1, nb + 1)]
                    + ["sum_BOs", "num_LPs", "partial_charge"]
            )
            extra = max(0, len(cur_atoms_rows[0]) - len(atom_cols))
            if extra > 0:
                atom_cols += [f"unknown{i}" for i in range(1, extra + 1)]
            frames.append(pd.DataFrame(cur_atoms_rows, columns=atom_cols))
            totals.append(cur_totals[:] if cur_totals else [float("nan")] * 4)

        total_lines = self._count_lines()
        with open(self.path, "r") as fh:
            lines_read = 0
            for raw in fh:
                lines_read += 1
                if self._reporter and (lines_read % 5000 == 0 or lines_read == total_lines):
                    self._reporter("load", lines_read, total_lines, "Parsing fort.7")
                values = raw.split()
                # print(f"line {lines_read} has {len(values)} values")
                if not values:
                    continue

                header_match = _match_fort7_header(
                    raw,
                    path=self.path,
                    frame_index=len(sim_rows),
                    line_number=lines_read,
                )
                # Header. Some ReaxFF outputs omit the space after
                # "Iteration:" once iteration numbers grow large.
                if header_match:
                    if cur_atoms_rows:
                        _finalize_iteration()
                        cur_atoms_rows.clear()
                        cur_totals.clear()

                    cur_num_particles = int(header_match.group("num_atoms"))
                    if cur_num_particles > 9999 and not warned_large_atom_count:
                        warning_msg = (
                            "Warning: fort.7 reports > 9999 atoms. ReaxFF fixed-width atom-index fields "
                            "can overflow at this size, which may concatenate neighbor indices and corrupt "
                            "fort.7 connectivity parsing. Consider running 'repair_fort7' before analysis."
                        )
                        print(warning_msg)
                        if self._reporter:
                            try:
                                self._reporter("warn", lines_read, total_lines, warning_msg)
                            except TypeError:
                                # Backward-compatible fallback for reporters that only handle load events.
                                self._reporter("load", lines_read, total_lines, warning_msg)
                        warned_large_atom_count = True
                    sim_name = header_match.group("simulation_name")
                    iteration = int(header_match.group("iteration"))
                    cur_nbonds = int(header_match.group("num_bonds"))
                    sim_rows.append([iteration, cur_num_particles, cur_nbonds])

                # Totals
                elif len(values) < 6:
                    cur_totals.extend(map(float, values))

                # Atom line
                else:
                    nb = int(cur_nbonds)
                    int_part = list(map(int, values[0: nb + 3]))
                    float_part = list(map(float, values[nb + 3:]))
                    cur_atoms_rows.append(int_part + float_part)

        # Final iter
        if cur_atoms_rows:
            _finalize_iteration()

        # Summary dataframe
        sim_df = pd.DataFrame(sim_rows, columns=["iter", "num_of_atoms", "num_of_bonds"])
        totals_df = pd.DataFrame(
            totals,
            columns=["total_BO", "total_LP", "total_BO_uncorrected", "total_charge"]
            if totals and len(totals[0]) == 4
            else [f"total_val{i}" for i in range(1, (len(totals[0]) if totals else 0) + 1)]
        )
        if not totals_df.empty:
            totals_df = totals_df.iloc[: len(sim_df)].reset_index(drop=True)
            sim_df = pd.concat([sim_df.reset_index(drop=True), totals_df], axis=1)

        # Deduplicate
        if not sim_df.empty and "iter" in sim_df.columns:
            keep_idx = sim_df.drop_duplicates("iter", keep="last").index
            frames = [frames[i] for i in keep_idx]
            sim_df = sim_df.loc[keep_idx].reset_index(drop=True)

        self._frames = frames
        self._sim_name = sim_name

        meta: Dict[str, Any] = {
            "n_frames": len(frames),
            "n_records": len(sim_df),
            "simulation_name": sim_name,
        }
        if self._reporter:
            self._reporter("load", total_lines, total_lines, "Finished parsing fort.7")

        return sim_df, meta

    def _frame_store(self, *, charge_only: bool = False) -> FrameStore | None:
        if not self._input_cache:
            return None
        root = self._frame_cache_root
        if root is None:
            configured = os.environ.get("REAXKIT_FRAME_CACHE_DIR", "").strip()
            root = Path(configured) if configured else self._cache_root().parent
        try:
            representation = "fort7-charge-frame-v1" if charge_only else "fort7-full-frame-v1"
            capabilities = (
                ("charges", "atom-types", "totals")
                if charge_only
                else ("charges", "atom-types", "connectivity", "totals")
            )
            return FrameStore.for_source(
                root,
                self.path,
                engine="reaxff",
                source_kind="fort7",
                parser=f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                parser_version="3",
                representation=representation,
                capabilities=capabilities,
            )
        except (OSError, RuntimeError, ValueError):
            return None

    def _scan_offsets(self, store: FrameStore, *, through_index: int) -> dict[str, int]:
        """Extend the fort.7 header index and persist a safe resume point."""
        coverage = store.get_coverage()
        if coverage.complete or coverage.next_frame_index > through_index:
            return {"indexed_frames": 0, "index_bytes": 0}
        indexed = 0
        bytes_read = 0
        with self.path.open("rb") as handle:
            handle.seek(coverage.next_byte_offset)
            frame_index = coverage.next_frame_index
            current_start: int | None = None
            current_iteration: int | None = None
            current_atoms: int | None = None
            while True:
                line_start = handle.tell()
                raw_bytes = handle.readline()
                if not raw_bytes:
                    if current_start is not None:
                        byte_end = handle.tell()
                        store.put_offsets([
                            FrameOffset(
                                frame_index,
                                current_start,
                                byte_end,
                                current_iteration,
                                current_atoms,
                            )
                        ])
                        frame_index += 1
                        indexed += 1
                        bytes_read += byte_end - current_start
                    store.set_coverage(IndexCoverage(frame_index, handle.tell(), True))
                    break
                raw = raw_bytes.decode("utf-8")
                header = _match_fort7_header(
                    raw,
                    path=self.path,
                    frame_index=frame_index if current_start is None else frame_index + 1,
                    line_number=None,
                )
                if header is None:
                    continue
                if current_start is None:
                    current_start = line_start
                    current_iteration = int(header.group("iteration"))
                    current_atoms = int(header.group("num_atoms"))
                    continue
                store.put_offsets([
                    FrameOffset(
                        frame_index,
                        current_start,
                        line_start,
                        current_iteration,
                        current_atoms,
                    )
                ])
                indexed += 1
                bytes_read += line_start - current_start
                frame_index += 1
                store.set_coverage(IndexCoverage(frame_index, line_start, False))
                if frame_index > through_index:
                    break
                current_start = line_start
                current_iteration = int(header.group("iteration"))
                current_atoms = int(header.group("num_atoms"))
        return {"indexed_frames": indexed, "index_bytes": bytes_read}

    def _parse_indexed_frame(self, handle, offset: FrameOffset) -> dict[str, Any]:
        handle.seek(offset.byte_start)
        raw_frame = handle.read(offset.byte_end - offset.byte_start).decode("utf-8")
        lines = raw_frame.splitlines()
        if not lines:
            raise ParseError(f"Missing fort.7 frame {offset.frame_index} in '{self.path}'.")
        header = _match_fort7_header(
            lines[0],
            path=self.path,
            frame_index=offset.frame_index,
            line_number=None,
        )
        if header is None:
            raise ParseError(f"Malformed fort.7 frame {offset.frame_index} in '{self.path}'.")
        num_atoms = int(header.group("num_atoms"))
        num_bonds = int(header.group("num_bonds"))
        atom_rows: list[list[float | int]] = []
        totals: list[float] = []
        for raw in lines[1:]:
            values = raw.split()
            if not values:
                continue
            if len(atom_rows) >= num_atoms or len(values) < 6:
                totals.extend(map(float, values))
                continue
            integer_count = num_bonds + 3
            int_part = list(map(int, values[:integer_count]))
            float_part = list(map(float, values[integer_count:]))
            atom_rows.append(int_part + float_part)
        columns = (
                ["atom_num", "atom_type_num"]
                + [f"atom_cnn{i}" for i in range(1, num_bonds + 1)]
                + ["molecule_num"]
                + [f"BO{i}" for i in range(1, num_bonds + 1)]
                + ["sum_BOs", "num_LPs", "partial_charge"]
        )
        if atom_rows:
            columns.extend(
                f"unknown{i}"
                for i in range(1, max(0, len(atom_rows[0]) - len(columns)) + 1)
            )
        return {
            "source_index": offset.frame_index,
            "iter": int(header.group("iteration")),
            "num_of_atoms": num_atoms,
            "num_of_bonds": num_bonds,
            "simulation_name": header.group("simulation_name"),
            "totals": totals or [float("nan")] * 4,
            "connectivity_incomplete": False,
            "frame": pd.DataFrame(atom_rows, columns=columns),
            "source_bytes": int(offset.byte_end - offset.byte_start),
        }

    def _parse_indexed_charge_frame(self, handle, offset: FrameOffset) -> dict[str, Any]:
        """Parse compact charges while preserving fused fixed-width recovery."""
        handle.seek(offset.byte_start)
        raw_frame = handle.read(offset.byte_end - offset.byte_start).decode("utf-8")
        lines = raw_frame.splitlines()
        if not lines:
            raise ParseError(f"Missing fort.7 frame {offset.frame_index} in '{self.path}'.")
        header = _match_fort7_header(
            lines[0],
            path=self.path,
            frame_index=offset.frame_index,
            line_number=None,
        )
        if header is None:
            raise ParseError(f"Malformed fort.7 frame {offset.frame_index} in '{self.path}'.")
        num_atoms = int(header.group("num_atoms"))
        num_bonds = int(header.group("num_bonds"))
        atom_ids: list[int] = []
        atom_types: list[int] = []
        charges: list[float] = []
        totals: list[float] = []
        trailing_count: int | None = None
        for raw in lines[1:]:
            stripped = raw.strip()
            if not stripped:
                continue
            if len(charges) >= num_atoms:
                totals.extend(map(float, stripped.split()))
                continue
            leading_fields = stripped.split(None, 5)
            if len(leading_fields) < 6:
                totals.extend(map(float, leading_fields))
                continue
            if trailing_count is None:
                float_match = _FORT7_FLOAT_FIELD_RE.search(stripped)
                float_fields = stripped[float_match.start():].split() if float_match else []
                charge_offset = num_bonds + 2
                if len(float_fields) <= charge_offset:
                    raise ValueError("Could not recover partial charge from a fort.7 atom row.")
                trailing_count = len(float_fields) - charge_offset - 1
                charge_token = float_fields[charge_offset]
            else:
                ending = stripped.rsplit(None, trailing_count + 1)
                charge_token = ending[-trailing_count - 1]
            atom_ids.append(int(leading_fields[0]))
            atom_type_token = raw[5:10].strip()
            if not atom_type_token.lstrip("+-").isdigit():
                atom_type_token = leading_fields[1]
            atom_types.append(int(atom_type_token))
            charges.append(float(charge_token))
        return {
            "source_index": offset.frame_index,
            "iter": int(header.group("iteration")),
            "num_of_atoms": num_atoms,
            "num_of_bonds": num_bonds,
            "simulation_name": header.group("simulation_name"),
            "totals": totals,
            "connectivity_incomplete": True,
            "charge_atom_ids": np.asarray(atom_ids, dtype=int),
            "charge_atom_type_nums": np.asarray(atom_types, dtype=int),
            "charges": np.asarray(charges, dtype=float),
            "source_bytes": int(offset.byte_end - offset.byte_start),
        }

    def _load_indexed_records(
            self,
            store: FrameStore,
            requested: list[int],
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        records = store.get_frames(requested)
        missing = [index for index in requested if index not in records]
        scan_stats = self._scan_offsets(store, through_index=max(missing, default=-1)) if missing else {
            "indexed_frames": 0,
            "index_bytes": 0,
        }
        offsets = store.get_offsets(missing)
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
                    source_bytes += int(record["source_bytes"])
        store.put_frames(parsed)
        records.update(parsed)
        return records, {
            "requested": len(requested),
            "hits": len(requested) - len(missing),
            "misses": len(missing),
            "parsed_frames": len(parsed),
            "source_bytes": source_bytes,
            **scan_stats,
        }

    def _parse_selected_frames(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        requested = list(self._frame_indices or ())
        store = self._frame_store()
        if store is None:
            return self._parse_selected_frames_sequential()
        records, stats = self._load_indexed_records(store, requested)
        self._frame_cache_stats = stats
        source_indices = [index for index in requested if index in records]
        frames = [records[index]["frame"] for index in source_indices]
        sim_rows = [
            [
                records[index]["iter"],
                records[index]["num_of_atoms"],
                records[index]["num_of_bonds"],
            ]
            for index in source_indices
        ]
        totals = [records[index]["totals"] for index in source_indices]
        sim_df = pd.DataFrame(sim_rows, columns=["iter", "num_of_atoms", "num_of_bonds"])
        if totals:
            width = len(totals[0])
            total_cols = (
                ["total_BO", "total_LP", "total_BO_uncorrected", "total_charge"]
                if width == 4
                else [f"total_val{i}" for i in range(1, width + 1)]
            )
            sim_df = pd.concat([sim_df, pd.DataFrame(totals, columns=total_cols)], axis=1)
        if not sim_df.empty:
            keep_idx = sim_df.drop_duplicates("iter", keep="last").index.tolist()
            frames = [frames[index] for index in keep_idx]
            source_indices = [source_indices[index] for index in keep_idx]
            sim_df = sim_df.iloc[keep_idx].reset_index(drop=True)
        self._frames = frames
        self._sim_name = str(records[source_indices[0]]["simulation_name"]) if source_indices else ""
        meta: Dict[str, Any] = {
            "n_frames": len(frames),
            "n_records": len(sim_df),
            "simulation_name": self._sim_name,
            "source_frame_indices": source_indices,
            "partial": True,
            "frame_cache": stats,
        }
        if self._reporter:
            self._reporter(
                "load",
                len(source_indices),
                len(requested),
                f"fort.7 frame cache: {stats['hits']} hit, {stats['misses']} miss",
            )
        return sim_df, meta

    def _parse_selected_frames_sequential(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Parse only selected connectivity frames and stop after the last one."""
        requested = list(self._frame_indices or ())
        requested_set = set(requested)
        max_requested = max(requested, default=-1)
        records: dict[int, tuple[list[Any], pd.DataFrame, list[float]]] = {}

        current_index = -1
        current_selected = False
        current_row: list[Any] | None = None
        cur_atoms_rows: list[list[float | int]] = []
        cur_totals: list[float] = []
        cur_num_particles: int | None = None
        cur_nbonds: int | None = None
        sim_name = ""
        warned_large_atom_count = False

        def _finalize_iteration() -> None:
            if (
                    not current_selected
                    or current_row is None
                    or cur_nbonds is None
                    or not cur_atoms_rows
            ):
                return
            nb = int(cur_nbonds)
            atom_cols = (
                    ["atom_num", "atom_type_num"]
                    + [f"atom_cnn{i}" for i in range(1, nb + 1)]
                    + ["molecule_num"]
                    + [f"BO{i}" for i in range(1, nb + 1)]
                    + ["sum_BOs", "num_LPs", "partial_charge"]
            )
            extra = max(0, len(cur_atoms_rows[0]) - len(atom_cols))
            if extra:
                atom_cols += [f"unknown{i}" for i in range(1, extra + 1)]
            records[current_index] = (
                current_row,
                pd.DataFrame(cur_atoms_rows, columns=atom_cols),
                cur_totals[:] if cur_totals else [float("nan")] * 4,
            )
            if self._reporter:
                self._reporter(
                    "load",
                    len(records),
                    len(requested_set),
                    "Loading selected fort.7 frames",
                )

        with open(self.path, "r") as fh:
            line_number = 0
            for raw in fh:
                line_number += 1
                header_match = _match_fort7_header(
                    raw,
                    path=self.path,
                    frame_index=current_index + 1,
                    line_number=line_number,
                )
                if header_match:
                    _finalize_iteration()
                    cur_atoms_rows = []
                    cur_totals = []
                    current_index += 1
                    if current_index > max_requested:
                        break

                    cur_num_particles = int(header_match.group("num_atoms"))
                    cur_nbonds = int(header_match.group("num_bonds"))
                    sim_name = header_match.group("simulation_name")
                    current_selected = current_index in requested_set
                    current_row = (
                        [
                            int(header_match.group("iteration")),
                            cur_num_particles,
                            cur_nbonds,
                        ]
                        if current_selected
                        else None
                    )

                    if cur_num_particles > 9999 and not warned_large_atom_count:
                        warning_msg = (
                            "Warning: fort.7 reports > 9999 atoms. ReaxFF fixed-width atom-index fields "
                            "can overflow at this size, which may concatenate neighbor indices and corrupt "
                            "fort.7 connectivity parsing. Consider running 'repair_fort7' before analysis."
                        )
                        print(warning_msg)
                        warned_large_atom_count = True
                    continue

                if not current_selected:
                    continue
                values = raw.split()
                if not values:
                    continue
                if len(values) < 6:
                    cur_totals.extend(map(float, values))
                else:
                    if cur_nbonds is None:
                        continue
                    nb = int(cur_nbonds)
                    int_part = list(map(int, values[0:nb + 3]))
                    float_part = list(map(float, values[nb + 3:]))
                    cur_atoms_rows.append(int_part + float_part)
            else:
                _finalize_iteration()

        source_indices = [i for i in requested if i in records]
        sim_rows = [records[i][0] for i in source_indices]
        frames = [records[i][1] for i in source_indices]
        totals = [records[i][2] for i in source_indices]
        sim_df = pd.DataFrame(sim_rows, columns=["iter", "num_of_atoms", "num_of_bonds"])
        if totals:
            width = len(totals[0])
            total_cols = (
                ["total_BO", "total_LP", "total_BO_uncorrected", "total_charge"]
                if width == 4
                else [f"total_val{i}" for i in range(1, width + 1)]
            )
            sim_df = pd.concat(
                [sim_df.reset_index(drop=True), pd.DataFrame(totals, columns=total_cols)],
                axis=1,
            )

        if not sim_df.empty and "iter" in sim_df.columns:
            keep_idx = sim_df.drop_duplicates("iter", keep="last").index.tolist()
            frames = [frames[i] for i in keep_idx]
            source_indices = [source_indices[i] for i in keep_idx]
            sim_df = sim_df.iloc[keep_idx].reset_index(drop=True)

        self._frames = frames
        self._sim_name = sim_name
        meta: Dict[str, Any] = {
            "n_frames": len(frames),
            "n_records": len(sim_df),
            "simulation_name": sim_name,
            "source_frame_indices": source_indices,
            "partial": True,
        }
        if self._reporter:
            self._reporter("load", len(source_indices), len(requested), "Finished loading selected fort.7 frames")
        return sim_df, meta

    def _count_lines(self) -> int:
        """Count lines."""
        with open(self.path, "r") as fh:
            return sum(1 for _ in fh)

    def stream_file_frames(
            self,
            *,
            charges_only: bool = False,
            charge_arrays_only: bool = False,
            include_atom_types: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Yield ``fort.7`` frames without materializing the trajectory.

        The generator retains only the rows and totals for the current
        iteration.  It intentionally bypasses the handler parse/cache path so
        streaming analysis does not create a second full in-memory copy.

        When ``charges_only`` is true, rows whose large fixed-width neighbor
        ids have fused together are recovered without connectivity: atom id,
        atom type, bond-order values, and partial charge remain aligned, while
        unavailable neighbor ids are represented by zeros.  Total dipole,
        polarization, and charge analyses do not consume connectivity.

        ``charge_arrays_only`` is the low-overhead total-electrostatics path:
        it extracts the atom id and partial-charge field from each atom row
        without converting the unused connectivity/bond-order fields or
        constructing a pandas table.

        ``include_atom_types=False`` makes that path skip atom-type conversion
        as well. Total dipole and polarization calculations need only atom ids,
        coordinates, and charges, and skipping the field also avoids ambiguity
        when fixed-width neighbor ids above 9999 are concatenated to it.
        """
        if charge_arrays_only:
            charges_only = True
        if self._frame_indices is not None:
            requested_order = list(self._frame_indices)
            store = self._frame_store(charge_only=charge_arrays_only)
            if store is not None:
                rich_store = self._frame_store() if charge_arrays_only else None
                initially_available = store.available_indices(requested_order)
                if rich_store is not None:
                    initially_available |= rich_store.available_indices(requested_order)
                missing = [index for index in requested_order if index not in initially_available]
                scan_stats = self._scan_offsets(
                    store,
                    through_index=max(missing, default=-1),
                ) if missing else {"indexed_frames": 0, "index_bytes": 0}
                offsets = store.get_offsets(missing)
                parsed_frames = 0
                source_bytes = 0
                with self.path.open("rb") as handle:
                    for emitted, source_index in enumerate(requested_order, start=1):
                        cached = store.get_frames([source_index])
                        record = cached.get(source_index)
                        if record is None and rich_store is not None:
                            record = rich_store.get_frames([source_index]).get(source_index)
                        if record is None:
                            offset = offsets.get(source_index)
                            if offset is None:
                                continue
                            record = (
                                self._parse_indexed_charge_frame(handle, offset)
                                if charge_arrays_only
                                else self._parse_indexed_frame(handle, offset)
                            )
                            store.put_frames({source_index: record})
                            parsed_frames += 1
                            source_bytes += int(record.get("source_bytes", 0))
                        output = dict(record)
                        output.pop("source_bytes", None)
                        if charge_arrays_only:
                            frame = record.get("frame")
                            output.pop("frame", None)
                            if frame is not None:
                                output["charge_atom_ids"] = frame["atom_num"].to_numpy(dtype=int)
                                output["charge_atom_type_nums"] = frame["atom_type_num"].to_numpy(dtype=int)
                                output["charges"] = frame["partial_charge"].to_numpy(dtype=float)
                            if not include_atom_types:
                                output.pop("charge_atom_type_nums", None)
                        if callable(self._reporter):
                            self._reporter(
                                "stream",
                                emitted,
                                len(requested_order),
                                "Streaming fort.7 frames",
                            )
                        yield output
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
                        len(requested_order),
                        len(requested_order),
                        f"fort.7 frame cache: {len(initially_available)} hit, "
                        f"{len(requested_order) - len(initially_available)} miss",
                    )
                return
        requested = set(self._frame_indices) if self._frame_indices is not None else None
        max_requested = max(requested, default=-1) if requested is not None else None
        source_index = -1
        emitted = 0
        current: dict[str, Any] | None = None
        atom_rows: list[list[float | int]] = []
        charge_atom_ids: list[int] = []
        charge_atom_type_nums: list[int] = []
        charge_values: list[float] = []
        totals: list[float] = []

        def finalize() -> Dict[str, Any] | None:
            nonlocal emitted
            has_atoms = bool(charge_values) if charge_arrays_only else bool(atom_rows)
            if current is None or not current["selected"] or not has_atoms:
                return None
            num_bonds = int(current["num_of_bonds"])
            emitted += 1
            if callable(self._reporter):
                total = len(requested) if requested is not None else 0
                self._reporter("stream", emitted, total, "Streaming fort.7 frames")
            record = {
                "source_index": int(current["source_index"]),
                "iter": int(current["iter"]),
                "num_of_atoms": int(current["num_of_atoms"]),
                "num_of_bonds": num_bonds,
                "simulation_name": str(current["simulation_name"]),
                "totals": list(totals),
                "connectivity_incomplete": bool(current.get("connectivity_incomplete", False)),
            }
            if charge_arrays_only:
                record["charge_atom_ids"] = np.asarray(charge_atom_ids, dtype=int)
                if include_atom_types:
                    record["charge_atom_type_nums"] = np.asarray(
                        charge_atom_type_nums,
                        dtype=int,
                    )
                record["charges"] = np.asarray(charge_values, dtype=float)
            else:
                columns = (
                        ["atom_num", "atom_type_num"]
                        + [f"atom_cnn{i}" for i in range(1, num_bonds + 1)]
                        + ["molecule_num"]
                        + [f"BO{i}" for i in range(1, num_bonds + 1)]
                        + ["sum_BOs", "num_LPs", "partial_charge"]
                )
                extra = max(0, len(atom_rows[0]) - len(columns))
                columns.extend(f"unknown{i}" for i in range(1, extra + 1))
                record["frame"] = pd.DataFrame(atom_rows, columns=columns)
            return record

        with open(self.path, "r", encoding="utf-8") as fh:
            for line_number, raw in enumerate(fh, start=1):
                header = _match_fort7_header(
                    raw,
                    path=self.path,
                    frame_index=source_index + 1,
                    line_number=line_number,
                )
                if header:
                    record = finalize()
                    if record is not None:
                        yield record
                    source_index += 1
                    if max_requested is not None and source_index > max_requested:
                        return
                    atom_rows = []
                    charge_atom_ids = []
                    charge_atom_type_nums = []
                    charge_values = []
                    totals = []
                    current = {
                        "source_index": source_index,
                        "selected": requested is None or source_index in requested,
                        "iter": int(header.group("iteration")),
                        "num_of_atoms": int(header.group("num_atoms")),
                        "num_of_bonds": int(header.group("num_bonds")),
                        "simulation_name": header.group("simulation_name"),
                        "connectivity_incomplete": False,
                        "charge_trailing_fields": None,
                    }
                    continue
                if current is None or not current["selected"]:
                    continue
                if charge_arrays_only and len(charge_values) < int(current["num_of_atoms"]):
                    # Atom rows are the first num_of_atoms records after a
                    # header.  Skip the wide integer connectivity section and
                    # split only the short floating-point tail.  This also
                    # keeps partial_charge correctly positioned if a producer
                    # appends nonstandard extra fields after it.
                    stripped = raw.strip()
                    if stripped:
                        leading_fields = stripped.split(None, 5)
                        if len(leading_fields) < 6:
                            # A truncated frame can transition to its short
                            # totals row before the advertised atom count.
                            totals.extend(map(float, leading_fields))
                            continue
                        trailing_count = current["charge_trailing_fields"]
                        if trailing_count is None:
                            float_match = _FORT7_FLOAT_FIELD_RE.search(stripped)
                            float_fields = stripped[float_match.start():].split() if float_match else []
                            charge_offset = int(current["num_of_bonds"]) + 2
                            if len(float_fields) <= charge_offset:
                                raise ValueError("Could not recover partial charge from a fort.7 atom row.")
                            trailing_count = len(float_fields) - charge_offset - 1
                            current["charge_trailing_fields"] = trailing_count
                            charge_token = float_fields[charge_offset]
                        else:
                            ending = stripped.rsplit(None, int(trailing_count) + 1)
                            charge_token = ending[-int(trailing_count) - 1]
                        charge_atom_ids.append(int(leading_fields[0]))
                        if include_atom_types:
                            # The first two fort.7 fields are fixed-width. Once
                            # neighbor ids exceed 9999, whitespace splitting can
                            # merge the atom type with every full-width neighbor.
                            atom_type_token = raw[5:10].strip()
                            if not atom_type_token.lstrip("+-").isdigit():
                                atom_type_token = leading_fields[1]
                            charge_atom_type_nums.append(int(atom_type_token))
                        charge_values.append(float(charge_token))
                    continue
                values = raw.split()
                if not values:
                    continue
                if len(values) < 6:
                    totals.extend(map(float, values))
                else:
                    num_bonds = int(current["num_of_bonds"])
                    integer_count = num_bonds + 3
                    try:
                        int_part = list(map(int, values[0:integer_count]))
                        float_part = list(map(float, values[integer_count:]))
                    except ValueError:
                        if not charges_only:
                            raise
                        float_index = next(
                            (
                                index
                                for index, token in enumerate(values)
                                if any(marker in token.lower() for marker in (".", "e"))
                            ),
                            -1,
                        )
                        expected_float_count = num_bonds + 3
                        if (
                                float_index < 2
                                or len(values) - float_index < expected_float_count
                        ):
                            raise ValueError(
                                "Could not recover charge fields from a fused fort.7 atom row. "
                                "Run 'reaxkit repair_fort7' for this file."
                            )
                        atom_num = int(values[0])
                        atom_type_token = values[1]
                        if not atom_type_token or not atom_type_token[0].isdigit():
                            raise ValueError(
                                "Could not recover atom type from a fused fort.7 atom row."
                            )
                        atom_type_num = int(atom_type_token[0])
                        int_part = [atom_num, atom_type_num, *([0] * num_bonds), 0]
                        float_part = list(map(float, values[float_index:]))
                        current["connectivity_incomplete"] = True
                    atom_rows.append(int_part + float_part)

        record = finalize()
        if record is not None:
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

    # -------------------------------------------------------
    # Frame utilities (match XmoloutHandler API)
    # -------------------------------------------------------

    def n_frames(self) -> int:
        """
        Return the number of frames parsed from the ``fort.7`` file.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Returns
        -------
        int
            Number of parsed frames (iterations).
        """
        if not self._parsed:
            self.parse()
        return len(self._frames)

    def n_atoms(self, frame: int = 0) -> int:
        """
        Return the number of atoms in a given frame.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Parameters
        ----------
        frame : int, optional
            Frame index to query.

        Returns
        -------
        int
            Number of atoms in the selected frame.
        """
        if not hasattr(self, "_frames") or self.n_frames() == 0:
            return 0
        return len(self._frames[int(frame)])

    def frame(self, i: int):
        """Return a single frame as an atom-level connectivity table.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Parameters
        ----------
        i : int
            Frame index to retrieve.

        Returns
        -------
        pandas.DataFrame
            Atom-level table for the selected frame, including connectivity
            and bond-order columns.

        Examples
        --------
        >>> h = Fort7Handler("fort.7")
        >>> df = h.frame(0)
        """
        if not self._parsed:
            self.parse()
        return self._frames[int(i)]

    def iter_frames(self, step: int = 1):
        """Iterate over atom-level frames with optional subsampling.

        Works on
        --------
        Fort7Handler — ``fort.7``

        Parameters
        ----------
        step : int, optional
            Step size for subsampling frames (default: 1).

        Yields
        ------
        pandas.DataFrame
            Atom-level connectivity table for each yielded frame.

        Examples
        --------
        >>> h = Fort7Handler("fort.7")
        >>> for frame in h.iter_frames(step=10):
        ...     print(len(frame))
        """
        if not hasattr(self, "_frames"):
            return
        for i in range(0, self.n_frames(), max(1, int(step))):
            yield self._frames[i]
