"""
GEO/XTLGRF file generation utilities.

This module contains GEO-specific text/file generation helpers and re-exports
general structure I/O and transform helpers from the split geometry modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd

from reaxkit.engine.common.io.geo_io import read_structure, write_structure
from reaxkit.engine.common.generators.structure_transformers import (
    build_surface,
    make_supercell,
    orthogonalize_hexagonal_cell,
    place2,
)
from reaxkit.engine.reaxff.io.geo_handler import GeoHandler


SortKey = Literal["x", "y", "z", "atom_type"]
GeoSortKey = Literal["m", "x", "y", "z", "atom_type"]

__all__ = [
    "SortKey",
    "GeoSortKey",
    "xtob",
    "sort_geo",
    "add_restraints_to_geo",
    "_format_crystx",
    "_format_hetatm_line",
    "read_structure",
    "write_structure",
    "build_surface",
    "make_supercell",
    "orthogonalize_hexagonal_cell",
    "place2",
]


def _read_xyz(xyz_path: str | Path) -> Tuple[str, pd.DataFrame]:
    """
    Read a simple XYZ file and return (descriptor, atoms_df).
    """
    xyz_path = Path(xyz_path)

    with xyz_path.open("r", encoding="utf-8") as fh:
        first = ""
        while first == "":
            first = fh.readline()
            if not first:
                raise ValueError(f"{xyz_path} appears to be empty.")
            first = first.strip()

        try:
            nat_expected = int(first.split()[0])
        except ValueError as exc:
            raise ValueError(f"First line of {xyz_path} is not a valid atom count: {first!r}") from exc

        second = ""
        while second == "":
            second = fh.readline()
            if not second:
                raise ValueError(f"{xyz_path} ended before descriptor line.")
            second = second.strip()

        descriptor_tokens = second.split()
        descriptor = descriptor_tokens[0] if descriptor_tokens else ""

        records: List[Dict[str, Any]] = []
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue

            symbol = parts[0]
            if symbol.upper().startswith("VEC"):
                continue
            if not symbol[0].isalpha():
                continue

            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                continue

            records.append({"atom_type": symbol, "x": x, "y": y, "z": z})

    atoms_df = pd.DataFrame(records, columns=["atom_type", "x", "y", "z"])
    if len(atoms_df) != nat_expected:
        raise ValueError(
            f"Number of atoms in XYZ header ({nat_expected}) does not match coordinate lines found ({len(atoms_df)})."
        )
    return descriptor, atoms_df


def _sort_atoms(
    atoms: pd.DataFrame,
    sort_by: Optional[SortKey] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    if sort_by is None:
        return atoms
    if sort_by not in atoms.columns:
        raise ValueError(f"sort_by must be one of {list(atoms.columns)!r}, got {sort_by!r}.")
    return atoms.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)


def _format_crystx(
    box_lengths: Iterable[float],
    box_angles: Iterable[float],
) -> str:
    a, b, c = list(box_lengths)
    alpha, beta, gamma = list(box_angles)
    nums = (a, b, c, alpha, beta, gamma)
    return "CRYSTX" + "".join(f"{value:11.5f}" for value in nums)


def _format_hetatm_line(atom_id: int, atom_type: str, x: float, y: float, z: float) -> str:
    at2 = atom_type.strip()[:2]
    at5 = atom_type.strip()[:5]
    return (
        "HETATM"
        f" {atom_id:5d}"
        f" {at2:2s}"
        "   "
        " "
        "   "
        " "
        " "
        " "
        "     "
        f"{x:10.5f}{y:10.5f}{z:10.5f}"
        f" {at5:5s}"
        f"{0:3d}{0:2d}"
        f" {0.0:8.5f}"
    )


def _generate_geo_text(
    descriptor: str,
    atoms: pd.DataFrame,
    box_lengths: Iterable[float],
    box_angles: Iterable[float],
    sort_by: Optional[SortKey] = None,
    ascending: bool = True,
) -> str:
    box_lengths = list(box_lengths)
    box_angles = list(box_angles)
    if len(box_lengths) != 3 or len(box_angles) != 3:
        raise ValueError("box_lengths and box_angles must each contain exactly 3 values.")

    atoms_sorted = _sort_atoms(atoms, sort_by=sort_by, ascending=ascending).reset_index(drop=True)
    atoms_sorted["atom_id"] = atoms_sorted.index + 1

    sort_remark = None
    if sort_by is not None:
        direction = "ascending" if ascending else "descending"
        coord_label = "atom type" if sort_by == "atom_type" else f"{sort_by}-coordinate"
        sort_remark = f"REMARK Structure sorted by {coord_label} ({direction})"

    lines = [
        "XTLGRF 200",
        f"DESCRP  {descriptor}",
        "REMARK .bgf-file generated by xtob-python",
    ]
    if sort_remark:
        lines.append(sort_remark)
    lines.append(_format_crystx(box_lengths, box_angles))
    lines.append("FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)")

    for row in atoms_sorted.itertuples(index=False):
        lines.append(_format_hetatm_line(row.atom_id, row.atom_type, row.x, row.y, row.z))
    lines.append("END")
    return "\n".join(lines) + "\n"


def xtob(
    xyz_file: str | Path,
    geo_file: str | Path = "geo",
    box_lengths: Iterable[float] = (1.0, 1.0, 1.0),
    box_angles: Iterable[float] = (90.0, 90.0, 90.0),
    sort_by: Optional[SortKey] = None,
    ascending: bool = True,
) -> Path:
    """
    Convert an XYZ file to ReaxFF GEO/XTLGRF format.
    """
    descriptor, atoms = _read_xyz(xyz_file)
    text = _generate_geo_text(
        descriptor=descriptor,
        atoms=atoms,
        box_lengths=box_lengths,
        box_angles=box_angles,
        sort_by=sort_by,
        ascending=ascending,
    )
    geo_file = Path(geo_file)
    geo_file.parent.mkdir(parents=True, exist_ok=True)
    geo_file.write_text(text, encoding="utf-8")
    return geo_file


def sort_geo(
    *,
    input_geo: str | Path,
    output_geo: str | Path,
    sort_by: GeoSortKey,
    descending: bool = False,
) -> Path:
    """
    Sort atoms in an existing GEO file and write a new GEO file.
    """
    in_path = Path(input_geo)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input GEO file not found: {in_path}")

    handler = GeoHandler(in_path)
    df = handler.dataframe().copy()
    meta = handler.metadata()

    sort_map = {"m": "atom_id", "x": "x", "y": "y", "z": "z", "atom_type": "atom_type"}
    sort_col = sort_map[sort_by]
    df_sorted = df.sort_values(by=sort_col, ascending=not descending).reset_index(drop=True)
    df_sorted["atom_id"] = df_sorted.index + 1

    descriptor = meta.get("descriptor") or ""
    remark = meta.get("remark")
    cell_lengths = meta.get("cell_lengths")
    cell_angles = meta.get("cell_angles")

    out_path = Path(output_geo)
    direction = "descending" if descending else "ascending"
    sort_label_map = {
        "m": "atom index",
        "x": "x-coordinate",
        "y": "y-coordinate",
        "z": "z-coordinate",
        "atom_type": "atom type",
    }
    sort_label = sort_label_map[sort_by]

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("XTLGRF 200\n")
        if descriptor:
            fh.write(f"DESCRP  {descriptor}\n")
        if remark:
            fh.write(f"REMARK {remark}\n")
        fh.write(f"REMARK Structure sorted by {sort_label} ({direction})\n")
        if cell_lengths and cell_angles:
            try:
                lengths = [cell_lengths.get(k) for k in ("a", "b", "c")]
                angles = [cell_angles.get(k) for k in ("alpha", "beta", "gamma")]
                if all(v is not None for v in lengths + angles):
                    fh.write(_format_crystx(lengths, angles) + "\n")
            except Exception:
                pass
        fh.write("FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)\n")
        for row in df_sorted.itertuples(index=False):
            fh.write(_format_hetatm_line(row.atom_id, row.atom_type, row.x, row.y, row.z) + "\n")
        fh.write("END\n")
    return out_path


def add_restraints_to_geo(
    geo_file: str | Path,
    *,
    out_file: str | Path | None = None,
    kinds: Sequence[str],
    params: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Insert sample restraint blocks into a GEO/XTLGRF file.
    """
    geo_file = Path(geo_file)
    if not geo_file.is_file():
        raise FileNotFoundError(f"Input GEO file not found: {geo_file}")

    out_path = Path(out_file) if out_file is not None else geo_file
    params = params or {}

    wanted = [str(kind).strip().upper() for kind in kinds if str(kind).strip()]
    if not wanted:
        raise ValueError("No restraint kinds provided (kinds is empty).")

    order = ["BOND", "ANGLE", "TORSION", "MASCEN"]
    wanted_sorted = [kind for kind in order if kind in set(wanted)]

    default_params: Dict[str, str] = {
        "BOND": "1   2  1.0900 7500.00 0.25000 0.0000000",
        "ANGLE": "1   2   3 109.5000 600.00 0.25000 0.0000000",
        "TORSION": "1   2   3   4 180.0000 100.00 0.25000 0.0000000",
        "MASCEN": "1   0.0000 0.0000 0.0000 500.00 0.25000 0.0000000",
    }

    format_lines: Dict[str, str] = {
        "BOND": "FORMAT BOND RESTRAINT (15x,2i4,f8.4,f8.2,f8.5,f10.7)",
        "ANGLE": "FORMAT ANGLE RESTRAINT (15x,3i4,f8.3,f8.2,f8.5,f10.7)",
        "TORSION": "FORMAT TORSION RESTRAINT (15x,4i4,f8.3,f8.2,f8.5,f10.7)",
        "MASCEN": "FORMAT MASCEN RESTRAINT (15x,i4,3f10.4,f8.2,f8.5,f10.7)",
    }

    guide_lines: Dict[str, str] = {
        "BOND": "#                                 At1 At2  R12     Force1  Force2  dR12/dIteration(MD only)",
        "ANGLE": "#                                 At1 At2 At3  A123    Force1  Force2  dA123/dIteration(MD only)",
        "TORSION": "#                                 At1 At2 At3 At4  T1234   Force1  Force2  dT1234/dIteration(MD only)",
        "MASCEN": "#                                 At1    X          Y          Z          Force1  Force2  dR/dIteration(MD only)",
    }

    token_layout: Dict[str, List[Tuple[str, str, int]]] = {
        "BOND": [
            ("At1", "i", 3),
            ("At2", "i", 3),
            ("R12", "f4", 7),
            ("Force1", "f2", 7),
            ("Force2", "f5", 7),
            ("dR12/dIteration(MD only)", "f7", 10),
        ],
        "ANGLE": [
            ("At1", "i", 3),
            ("At2", "i", 3),
            ("At3", "i", 3),
            ("A123", "f3", 7),
            ("Force1", "f2", 7),
            ("Force2", "f5", 7),
            ("dA123/dIteration(MD only)", "f7", 10),
        ],
        "TORSION": [
            ("At1", "i", 3),
            ("At2", "i", 3),
            ("At3", "i", 3),
            ("At4", "i", 3),
            ("T1234", "f3", 7),
            ("Force1", "f2", 7),
            ("Force2", "f5", 7),
            ("dT1234/dIteration(MD only)", "f7", 10),
        ],
        "MASCEN": [
            ("At1", "i", 3),
            ("X", "f4_10", 10),
            ("Y", "f4_10", 10),
            ("Z", "f4_10", 10),
            ("Force1", "f2", 7),
            ("Force2", "f5", 7),
            ("dR/dIteration(MD only)", "f7", 10),
        ],
    }

    def _format_value(token: str, fmt: str) -> str:
        if fmt == "i":
            return str(int(float(token)))
        if fmt == "f2":
            return f"{float(token):.2f}"
        if fmt == "f3":
            return f"{float(token):.3f}"
        if fmt == "f4":
            return f"{float(token):.4f}"
        if fmt == "f5":
            return f"{float(token):.5f}"
        if fmt == "f7":
            return f"{float(token):.7f}"
        if fmt == "f4_10":
            return f"{float(token):.4f}"
        return token

    def _token_starts(guide: str, names: List[str]) -> List[int]:
        starts: List[int] = []
        cursor = 0
        for name in names:
            index = guide.find(name, cursor)
            if index < 0:
                index = (starts[-1] + 4) if starts else guide.find("#") + 2
            starts.append(index)
            cursor = index + len(name)
        return starts

    def _build_aligned_data_line(kind: str, param_str: str) -> str:
        guide = guide_lines[kind]
        layout = token_layout[kind]
        names = [item[0] for item in layout]
        starts = _token_starts(guide, names)

        tokens = [token for token in param_str.split() if token.strip()]
        need = len(layout)
        if len(tokens) < need:
            raise ValueError(f"{kind} params need {need} tokens but got {len(tokens)}: {tokens}")
        tokens = tokens[:need]

        label = f"{kind} RESTRAINT"
        line = label
        if len(line) < starts[0]:
            line += " " * (starts[0] - len(line))
        else:
            line += " "

        for idx, ((_, fmt, min_width), start, token) in enumerate(zip(layout, starts, tokens)):
            value = _format_value(token, fmt)
            if len(line) < start:
                line += " " * (start - len(line))
            if idx < len(starts) - 1:
                width = max(min_width, starts[idx + 1] - start - 1)
            else:
                width = max(min_width, len(value))
            line += value.ljust(width) + " "

        return line.rstrip()

    lines = geo_file.read_text(encoding="utf-8").splitlines()

    insert_idx: Optional[int] = None
    triggers = ("CRYSTX", "FORMAT ATOM", "HETATM", "ATOM")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if any(stripped.startswith(trigger) for trigger in triggers):
            insert_idx = i
            break
    if insert_idx is None:
        for i, line in enumerate(lines):
            if line.strip() == "END":
                insert_idx = i
                break
    if insert_idx is None:
        insert_idx = len(lines)

    block = ["REMARK Restraints added by ReaxKit (sample lines; edit as needed)"]
    for kind in wanted_sorted:
        block.append(format_lines[kind])
        block.append(guide_lines[kind])
        param_string = (params.get(kind) or "").strip() or default_params[kind]
        block.append(_build_aligned_data_line(kind, param_string))

    new_lines = lines[:insert_idx] + block + lines[insert_idx:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return out_path
