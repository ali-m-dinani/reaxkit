"""Detect coarse ReaxFF isomer representatives for training workflows.

This module ports the file-based representative-selection workflow used by the
legacy ``isomer.cpp`` utility. It scans ``fort.7`` for molecules matching a
target formula from ``control_params``, classifies representative structures by
coarse bond-type count signatures, and writes representative coordinates from
``xmolout``.

The selection is intentionally coarser than full graph-isomorphism based isomer
enumeration. For ReaxFF training pipelines this is faster, cheaper, and usually
preferable because it reduces the number of downstream Jaguar (DFT) jobs while
still sampling distinct same-formula bonding environments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


_FORT7_HEADER_RE = re.compile(r"\s*(\d+)\D.*Iteration:\s*(\d+).*#Bonds:\s*(\d+)\s*")
_INTEGER_TOKEN_RE = re.compile(r"^[+-]?\d+$")


@dataclass(frozen=True)
class ReaxFFIsomerRepresentativeControl:
    """Parsed ``control_params`` settings for representative detection."""

    atom_map: dict[str, int]
    input_formula: dict[str, int]
    isomer_run: int = 1
    isomer_prefixname: str = ""

    @property
    def atom_type_to_symbol(self) -> dict[int, str]:
        """Return atom-type integer to element symbol lookup."""
        return {atom_type: symbol for symbol, atom_type in self.atom_map.items()}

    @property
    def atom_type_counts(self) -> dict[int, int]:
        """Return target formula counts keyed by ``fort.7`` atom type."""
        counts: dict[int, int] = {}
        for symbol, count in self.input_formula.items():
            if symbol not in self.atom_map:
                raise ValueError(f"input_formula element {symbol!r} is missing from atom_map.")
            counts[int(self.atom_map[symbol])] = int(count)
        return counts

    @property
    def total_atoms(self) -> int:
        """Return target molecule atom count."""
        return int(sum(self.input_formula.values()))


@dataclass(frozen=True)
class ReaxFFIsomerRepresentative:
    """One coarse isomer representative found in ``fort.7``."""

    isomer_index: int
    structure_name: str
    iteration: int
    molecule_no: int
    atom_count: int
    bond_type_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    bond_label_counts: dict[str, float] = field(default_factory=dict)


@dataclass
class ReaxFFIsomerRepresentativeResult:
    """Result metadata for ReaxFF isomer representative detection."""

    control: ReaxFFIsomerRepresentativeControl
    records: list[ReaxFFIsomerRepresentative]
    table: pd.DataFrame
    output_xmolout_isomers: Path
    isomer_dir: Path | None = None
    log_path: Path | None = None


def _parse_symbol_counts(raw: str, *, field_name: str) -> dict[str, int]:
    """Parse ``Element:count`` comma-separated control values."""
    out: dict[str, int] = {}
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid {field_name} token {token!r}; expected Element:value.")
        symbol, value = token.split(":", 1)
        symbol = symbol.strip()
        if not symbol:
            raise ValueError(f"Invalid {field_name} token {token!r}; missing element symbol.")
        out[symbol] = int(value.strip())
    if not out:
        raise ValueError(f"{field_name} cannot be empty.")
    return out


def parse_reaxff_isomer_representative_control(path: str | Path) -> ReaxFFIsomerRepresentativeControl:
    """Parse a legacy ``control_params`` file.

    Required keys are ``atom_map``, ``input_formula``, and
    ``isomer_prefixname``. ``isomer_run`` defaults to ``1``.
    """
    control_path = Path(path)
    if not control_path.is_file():
        raise FileNotFoundError(f"control_params file not found: {control_path}")

    values: dict[str, str] = {}
    for line in control_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid control_params line {line!r}; expected key=value.")
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()

    missing = [key for key in ("atom_map", "input_formula", "isomer_prefixname") if not values.get(key)]
    if missing:
        raise ValueError(f"control_params is missing required keys: {missing}")

    return ReaxFFIsomerRepresentativeControl(
        atom_map=_parse_symbol_counts(values["atom_map"], field_name="atom_map"),
        input_formula=_parse_symbol_counts(values["input_formula"], field_name="input_formula"),
        isomer_run=int(values.get("isomer_run", "1")),
        isomer_prefixname=values["isomer_prefixname"],
    )


def _format_bond_label_counts(
    bond_type_counts: dict[tuple[int, int], int],
    atom_type_to_symbol: dict[int, str],
) -> dict[str, float]:
    """Convert raw doubled bond counts to legacy human-readable bond labels."""
    out: dict[str, float] = {}
    for (first_type, second_type), count in sorted(bond_type_counts.items()):
        first = atom_type_to_symbol.get(first_type, str(first_type))
        second = atom_type_to_symbol.get(second_type, str(second_type))
        value = count / 2
        out[f"{first}-{second}"] = int(value) if float(value).is_integer() else float(value)
    return out


def _matches_formula(
    atom_ids: Iterable[int],
    atom_num_to_type: dict[int, int],
    target_type_counts: dict[int, int],
    total_atoms: int,
) -> bool:
    """Return whether a molecule atom list exactly matches the target formula."""
    atom_ids = list(atom_ids)
    if len(atom_ids) != total_atoms:
        return False
    counts: dict[int, int] = defaultdict(int)
    for atom_id in atom_ids:
        counts[int(atom_num_to_type[int(atom_id)])] += 1
    return dict(counts) == dict(target_type_counts)


def _bond_type_counts(
    atom_ids: Iterable[int],
    atom_num_to_type: dict[int, int],
    bonds: dict[int, list[int]],
) -> dict[tuple[int, int], int]:
    """Count molecule bonds by sorted atom-type pair using legacy semantics."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for atom_id in atom_ids:
        atom_type = int(atom_num_to_type[int(atom_id)])
        for neighbor_id in bonds.get(int(atom_id), []):
            neighbor_type = int(atom_num_to_type[int(neighbor_id)])
            key = tuple(sorted((atom_type, neighbor_type)))
            counts[key] += 1
    return dict(counts)


def _read_fort7_atom_line(line: str, *, bond_slots: int) -> tuple[int, int, list[int], int]:
    """Parse one atom row from a ``fort.7`` frame."""
    parts = line.split()
    minimum = 2 + bond_slots + 1
    if len(parts) < minimum:
        raise ValueError(f"Invalid fort.7 atom line; expected at least {minimum} columns: {line!r}")
    atom_id = int(parts[0])
    atom_type = int(parts[1])
    neighbors = [int(value) for value in parts[2 : 2 + bond_slots] if int(value) != 0]
    molecule_no = int(parts[2 + bond_slots])
    return atom_id, atom_type, neighbors, molecule_no


def scan_reaxff_isomer_representatives(
    fort7_path: str | Path,
    control: ReaxFFIsomerRepresentativeControl,
    *,
    max_representatives: int | None = None,
) -> list[ReaxFFIsomerRepresentative]:
    """Scan ``fort.7`` and return coarse matching isomer representatives."""
    path = Path(fort7_path)
    if not path.is_file():
        raise FileNotFoundError(f"fort.7 file not found: {path}")
    if max_representatives is not None and int(max_representatives) < 1:
        raise ValueError("max_representatives must be a positive integer when provided.")
    max_count = int(max_representatives) if max_representatives is not None else None

    target_type_counts = control.atom_type_counts
    records: list[ReaxFFIsomerRepresentative] = []
    seen_signatures: set[tuple[tuple[tuple[int, int], int], ...]] = set()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _FORT7_HEADER_RE.match(line)
            if not match:
                continue

            frame_atoms = int(match.group(1))
            iteration = int(match.group(2))
            bond_slots = int(match.group(3))
            atom_num_to_type: dict[int, int] = {}
            bonds: dict[int, list[int]] = {}
            molecule_to_atoms: dict[int, list[int]] = defaultdict(list)

            for _ in range(frame_atoms):
                atom_line = handle.readline()
                if atom_line == "":
                    raise ValueError(f"Unexpected end of fort.7 while reading iteration {iteration}.")
                atom_id, atom_type, neighbors, molecule_no = _read_fort7_atom_line(
                    atom_line,
                    bond_slots=bond_slots,
                )
                atom_num_to_type[atom_id] = atom_type
                bonds[atom_id] = neighbors
                molecule_to_atoms[molecule_no].append(atom_id)

            for molecule_no, atom_ids in molecule_to_atoms.items():
                if not _matches_formula(
                    atom_ids,
                    atom_num_to_type,
                    target_type_counts,
                    control.total_atoms,
                ):
                    continue
                bond_counts = _bond_type_counts(atom_ids, atom_num_to_type, bonds)
                signature = tuple(sorted(bond_counts.items()))
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                isomer_index = len(records)
                records.append(
                    ReaxFFIsomerRepresentative(
                        isomer_index=isomer_index,
                        structure_name=f"{control.isomer_prefixname}_{isomer_index}",
                        iteration=iteration,
                        molecule_no=int(molecule_no),
                        atom_count=control.total_atoms,
                        bond_type_counts=bond_counts,
                        bond_label_counts=_format_bond_label_counts(
                            bond_counts,
                            control.atom_type_to_symbol,
                        ),
                    )
                )
                if max_count is not None and len(records) >= max_count:
                    return records

    return records


def _parse_xmolout_iteration(comment_line: str) -> int | None:
    """Extract the iteration token from an ``xmolout`` comment line."""
    parts = comment_line.split()
    for token in parts[1:]:
        if _INTEGER_TOKEN_RE.match(token):
            return int(token)
    return None


def _strip_trailing_molecule_no(line: str) -> str:
    """Return an atom line without its final molecule-number column."""
    parts = line.rstrip("\n").rsplit(None, 1)
    return parts[0] if parts else ""


def _atom_line_molecule_no(line: str) -> int | None:
    """Return the final molecule-number token from an ``xmolout`` atom row."""
    parts = line.split()
    if not parts:
        return None
    token = parts[-1]
    return int(token) if _INTEGER_TOKEN_RE.match(token) else None


def extract_xmolout_isomer_structures(
    xmolout_path: str | Path,
    records: list[ReaxFFIsomerRepresentative],
    *,
    output_xmolout_isomers: str | Path,
    isomer_dir: str | Path | None = None,
) -> None:
    """Write representative isomer structures from ``xmolout``."""
    xmolout = Path(xmolout_path)
    if not xmolout.is_file():
        raise FileNotFoundError(f"xmolout file not found: {xmolout}")

    output_path = Path(output_xmolout_isomers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    isomer_base = Path(isomer_dir) if isomer_dir is not None else None
    if isomer_base is not None:
        isomer_base.mkdir(parents=True, exist_ok=True)

    pending_by_iteration: dict[int, list[ReaxFFIsomerRepresentative]] = defaultdict(list)
    for record in records:
        pending_by_iteration[int(record.iteration)].append(record)
    written: set[int] = set()

    with xmolout.open("r", encoding="utf-8", errors="replace") as src, output_path.open(
        "w",
        encoding="utf-8",
    ) as combined:
        while len(written) < len(records):
            atom_count_line = src.readline()
            if atom_count_line == "":
                break
            stripped_count = atom_count_line.strip()
            if not stripped_count:
                continue
            if not _INTEGER_TOKEN_RE.match(stripped_count):
                continue
            frame_atom_count = int(stripped_count)
            comment_line = src.readline()
            if comment_line == "":
                raise ValueError("Unexpected end of xmolout while reading frame comment line.")
            atom_lines = [src.readline() for _ in range(frame_atom_count)]
            if any(line == "" for line in atom_lines):
                raise ValueError("Unexpected end of xmolout while reading atom rows.")

            iteration = _parse_xmolout_iteration(comment_line)
            if iteration is None or iteration not in pending_by_iteration:
                continue

            for record in pending_by_iteration[iteration]:
                if record.isomer_index in written:
                    continue
                selected = [
                    line.rstrip("\n")
                    for line in atom_lines
                    if _atom_line_molecule_no(line) == int(record.molecule_no)
                ]
                if len(selected) != record.atom_count:
                    continue

                combined.write(f"{record.atom_count}\n")
                combined.write(f"{record.structure_name}\n")
                for line in selected:
                    combined.write(f"{line}\n")

                if isomer_base is not None:
                    structure_dir = isomer_base / record.structure_name
                    structure_dir.mkdir(parents=True, exist_ok=True)
                    with (structure_dir / "xmolout").open("w", encoding="utf-8") as out:
                        for line in selected:
                            out.write(f"{_strip_trailing_molecule_no(line)}\n")
                        out.write("&\n")

                written.add(record.isomer_index)

    missing = [record.structure_name for record in records if record.isomer_index not in written]
    if missing:
        raise ValueError(f"Could not extract xmolout structures for isomers: {missing}")


def _records_table(records: list[ReaxFFIsomerRepresentative]) -> pd.DataFrame:
    """Build a stable metadata table for detected isomers."""
    return pd.DataFrame(
        [
            {
                "isomer_index": record.isomer_index,
                "structure_name": record.structure_name,
                "iteration": record.iteration,
                "molecule_no": record.molecule_no,
                "atom_count": record.atom_count,
                "bond_signature": ";".join(
                    f"{label}:{count}" for label, count in record.bond_label_counts.items()
                ),
            }
            for record in records
        ],
        columns=[
            "isomer_index",
            "structure_name",
            "iteration",
            "molecule_no",
            "atom_count",
            "bond_signature",
        ],
    )


def write_reaxff_isomer_representative_log(
    records: list[ReaxFFIsomerRepresentative],
    log_path: str | Path,
) -> None:
    """Write a legacy-style isomer run log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(records)}\n")
        for record in records:
            handle.write(
                f"isomer: {record.structure_name} Iteration: {record.iteration} "
                f"Molecule No: {record.molecule_no}\n"
            )
            for label, count in record.bond_label_counts.items():
                handle.write(f"{label}: {count}\n")


def detect_reaxff_isomer_representatives(
    *,
    fort7_path: str | Path,
    xmolout_path: str | Path,
    control_path: str | Path,
    output_dir: str | Path,
    write_isomer_dirs: bool | None = None,
    max_representatives: int | None = None,
    output_name: str = "xmolout_isomers",
    log_name: str = "isomer_run_log.txt",
) -> ReaxFFIsomerRepresentativeResult:
    """Detect coarse ReaxFF isomer representatives and write outputs.

    Parameters
    -----
    fort7_path : str | Path
        Path to ``fort.7``.
    xmolout_path : str | Path
        Path to ``xmolout``.
    control_path : str | Path
        Path to ``control_params``.
    output_dir : str | Path
        Directory where ``xmolout_isomers`` and logs are written.
    write_isomer_dirs : bool | None
        If ``None``, follow legacy ``isomer_run == 2`` behavior. If true,
        write per-isomer folders under ``output_dir/isomers``.
    max_representatives : int | None
        Optional positive cap on representatives to extract. This is useful
        when only a smaller, cheaper set of Jaguar (DFT) jobs is needed.
    output_name : str
        Combined output filename.
    log_name : str
        Log filename.
    """
    control = parse_reaxff_isomer_representative_control(control_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xmolout = Path(xmolout_path)
    if not xmolout.is_file():
        raise FileNotFoundError(f"xmolout file not found: {xmolout}")

    records = scan_reaxff_isomer_representatives(
        fort7_path,
        control,
        max_representatives=max_representatives,
    )
    should_write_dirs = bool(control.isomer_run == 2) if write_isomer_dirs is None else bool(write_isomer_dirs)
    isomer_dir = out_dir / "isomers" if should_write_dirs else None
    output_xmolout = out_dir / output_name
    extract_xmolout_isomer_structures(
        xmolout,
        records,
        output_xmolout_isomers=output_xmolout,
        isomer_dir=isomer_dir,
    )

    log_path = out_dir / log_name
    write_reaxff_isomer_representative_log(records, log_path)

    return ReaxFFIsomerRepresentativeResult(
        control=control,
        records=records,
        table=_records_table(records),
        output_xmolout_isomers=output_xmolout,
        isomer_dir=isomer_dir,
        log_path=log_path,
    )


__all__ = [
    "ReaxFFIsomerRepresentativeControl",
    "ReaxFFIsomerRepresentativeResult",
    "ReaxFFIsomerRepresentative",
    "detect_reaxff_isomer_representatives",
    "extract_xmolout_isomer_structures",
    "parse_reaxff_isomer_representative_control",
    "scan_reaxff_isomer_representatives",
    "write_reaxff_isomer_representative_log",
]
