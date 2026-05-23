"""
ReaxFF ``ffield`` merge utilities.

This module merges atom-type parameter blocks from a source ``ffield`` into a
destination ``ffield`` and writes a merged output file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from reaxkit.engine.reaxff.io.ffield_handler import FFieldHandler


SECTION_ATOM_COLS: dict[str, tuple[str, ...]] = {
    "bond": ("i", "j"),
    "off_diagonal": ("i", "j"),
    "angle": ("i", "j", "k"),
    "torsion": ("i", "j", "k", "l"),
    "hbond": ("i", "j", "k"),
}

FIELD_ALIASES: dict[str, str] = {
    "atom": "atom",
    "atoms": "atom",
    "bond": "bond",
    "bonds": "bond",
    "off": "off_diagonal",
    "off_diagonal": "off_diagonal",
    "off-diagonal": "off_diagonal",
    "angle": "angle",
    "angles": "angle",
    "torsion": "torsion",
    "torsions": "torsion",
    "hbond": "hbond",
    "hbonds": "hbond",
    "hydrogen_bond": "hbond",
    "hydrogen-bond": "hbond",
}

SUPPORTED_FIELDS: tuple[str, ...] = ("atom", "bond", "off_diagonal", "angle", "torsion", "hbond")


@dataclass(frozen=True)
class FFieldMergeSummary:
    """Summary of a merge operation."""

    output_path: Path
    atom_types_merged: tuple[str, ...]
    fields: tuple[str, ...]
    appended: dict[str, int]
    updated: dict[str, int]
    skipped_existing: dict[str, int]
    skipped_incompatible: dict[str, int]
    source_labels: dict[str, list[str]]
    source_blocks: dict[str, list[str]]
    destination_labels: dict[str, list[str]]
    destination_blocks: dict[str, list[str]]


def _normalize_fields(fields: Iterable[str] | None) -> tuple[str, ...]:
    if fields is None:
        return SUPPORTED_FIELDS
    out: list[str] = []
    for raw in fields:
        key = str(raw).strip().lower()
        if not key:
            continue
        canonical = FIELD_ALIASES.get(key)
        if canonical is None:
            raise ValueError(f"Unsupported field {raw!r}. Supported: {', '.join(SUPPORTED_FIELDS)}")
        if canonical not in out:
            out.append(canonical)
    if not out:
        raise ValueError("No valid fields were provided.")
    return tuple(out)


def _normalize_atom_types(atom_types: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in atom_types:
        token = str(raw).strip()
        if token and token not in out:
            out.append(token)
    if not out:
        raise ValueError("At least one atom type is required.")
    return tuple(out)


def _safe_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _atom_maps(atom_df: pd.DataFrame) -> tuple[dict[int, str], dict[str, int]]:
    if atom_df.empty or "symbol" not in atom_df.columns:
        raise ValueError("Atom section must be non-empty and include 'symbol'.")
    idx_to_sym: dict[int, str] = {}
    sym_to_idx: dict[str, int] = {}
    for idx, row in atom_df.iterrows():
        atom_idx = int(idx)
        symbol = str(row["symbol"]).strip()
        idx_to_sym[atom_idx] = symbol
        sym_to_idx[symbol] = atom_idx
    return idx_to_sym, sym_to_idx


def _format_general(general_df: pd.DataFrame, names: list[str]) -> str:
    lines: list[str] = []
    for idx in general_df.index.tolist():
        value = float(general_df.loc[idx, "value"])
        comment = ""
        if "raw_comment" in general_df.columns:
            comment = str(general_df.loc[idx, "raw_comment"]).strip()
        if not comment and (idx - 1) < len(names):
            comment = names[idx - 1]
        lines.append(f"{value:10.4f} !{comment}".rstrip())
    return "\n".join(lines) + ("\n" if lines else "")


def _format_atom(atom_df: pd.DataFrame, names: list[str]) -> str:
    lines: list[str] = []
    for idx in atom_df.index.tolist():
        symbol = str(atom_df.loc[idx, "symbol"]).strip()
        values = [float(atom_df.loc[idx, name]) for name in names]
        for block in range(4):
            start = block * 8
            end = start + 8
            lead = f"{int(idx):3d} {symbol:<2} " if block == 0 else "       "
            tail = "".join(f"{value:9.4f}" for value in values[start:end])
            lines.append(f"{lead}{tail}".rstrip())
    return "\n".join(lines) + ("\n" if lines else "")


def _format_two_line(df: pd.DataFrame, atom_cols: tuple[str, str], param_names: list[str]) -> str:
    lines: list[str] = []
    for idx in df.index.tolist():
        i_val = int(df.loc[idx, atom_cols[0]])
        j_val = int(df.loc[idx, atom_cols[1]])
        values = [float(df.loc[idx, name]) for name in param_names]
        lines.append(f"{i_val:3d}{j_val:3d}{''.join(f'{value:9.4f}' for value in values[0:8])}".rstrip())
        lines.append(f"      {''.join(f'{value:9.4f}' for value in values[8:16])}".rstrip())
    return "\n".join(lines) + ("\n" if lines else "")


def _format_one_line(df: pd.DataFrame, atom_cols: tuple[str, ...], param_names: list[str]) -> str:
    lines: list[str] = []
    for idx in df.index.tolist():
        atom_part = "".join(f"{int(df.loc[idx, col]):3d}" for col in atom_cols)
        values = "".join(f"{float(df.loc[idx, name]):9.4f}" for name in param_names)
        lines.append(f"{atom_part}{values}".rstrip())
    return "\n".join(lines) + ("\n" if lines else "")


def _format_section_rows(section: str, df: pd.DataFrame, row_indices: list[int], handler: FFieldHandler) -> list[str]:
    if not row_indices:
        return []
    sub = df.loc[row_indices]
    if section == "atom":
        atom_names = handler._number_unused_titles(handler._ATOM_PARAM_NAMES_BASE)
        txt = _format_atom(sub, atom_names)
    elif section == "bond":
        bond_names = handler._number_unused_titles(handler._BOND_PARAM_NAMES_BASE)
        txt = _format_two_line(sub, ("i", "j"), bond_names)
    elif section == "off_diagonal":
        txt = _format_one_line(sub, ("i", "j"), list(handler._OFF_DIAGONAL_PARAM_NAMES))
    elif section == "angle":
        txt = _format_one_line(sub, ("i", "j", "k"), list(handler._ANGLE_PARAM_NAMES))
    elif section == "torsion":
        torsion_names = handler._number_unused_titles(handler._TORSION_PARAM_NAMES_BASE)
        txt = _format_one_line(sub, ("i", "j", "k", "l"), torsion_names)
    elif section == "hbond":
        txt = _format_one_line(sub, ("i", "j", "k"), list(handler._HBOND_PARAM_NAMES))
    else:
        return []
    blocks: list[str] = []
    current: list[str] = []
    for line in txt.strip("\n").splitlines():
        if section in {"atom", "bond"}:
            if section == "atom" and line[:3].strip():
                if current:
                    blocks.append("\n".join(current))
                    current = []
            if section == "bond" and len(line) >= 6 and line[:6].strip():
                if current:
                    blocks.append("\n".join(current))
                    current = []
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _labels_for_rows(
    section: str,
    df: pd.DataFrame,
    row_indices: list[int],
    idx_to_sym: dict[int, str],
) -> list[str]:
    labels: list[str] = []
    for ridx in row_indices:
        row = df.loc[ridx]
        if section == "atom":
            labels.append(str(row["symbol"]).strip())
            continue
        cols = SECTION_ATOM_COLS[section]
        atoms: list[str] = []
        for col in cols:
            iv = _safe_int(row[col])
            if iv == 0:
                atoms.append("A")
            elif iv is None:
                atoms.append("?")
            else:
                atoms.append(idx_to_sym.get(iv, str(iv)))
        labels.append("-".join(atoms))
    return labels


def _mapped_row_for_destination(
    src_row: pd.Series,
    atom_cols: tuple[str, ...],
    mapped_atoms: list[int],
    dst_param_cols: list[str],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for col, mapped in zip(atom_cols, mapped_atoms):
        payload[col] = mapped
    for col in dst_param_cols:
        payload[col] = src_row[col] if col in src_row.index else float("nan")
    return payload


def _write_ffield_sections(path: str | Path, sections: dict[str, pd.DataFrame], handler: FFieldHandler) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    description = "Merged force field generated by reaxkit."
    general_df = sections["general"]
    atom_df = sections["atom"]
    bond_df = sections["bond"]
    off_df = sections["off_diagonal"]
    angle_df = sections["angle"]
    torsion_df = sections["torsion"]
    hbond_df = sections["hbond"]

    atom_names = handler._number_unused_titles(handler._ATOM_PARAM_NAMES_BASE)
    bond_names = handler._number_unused_titles(handler._BOND_PARAM_NAMES_BASE)
    torsion_names = handler._number_unused_titles(handler._TORSION_PARAM_NAMES_BASE)

    text = []
    text.append(description)
    text.append(f"{len(general_df):3d}       ! Number of general parameters")
    text.append(_format_general(general_df, handler._GENERAL_PARAM_NAMES).rstrip("\n"))

    text.append(f"{len(atom_df):3d}    ! Nr of atoms; atomID; symbol; parameters")
    text.append("            atom parameters line 2")
    text.append("            atom parameters line 3")
    text.append("            atom parameters line 4")
    text.append(_format_atom(atom_df, atom_names).rstrip("\n"))

    text.append(f"{len(bond_df):3d}      ! Nr of bonds; at1;at2; bond parameters")
    text.append("                      bond parameters line 2")
    text.append(_format_two_line(bond_df, ("i", "j"), bond_names).rstrip("\n"))

    text.append(f"{len(off_df):3d}    ! Nr of off-diagonal terms")
    text.append(_format_one_line(off_df, ("i", "j"), list(handler._OFF_DIAGONAL_PARAM_NAMES)).rstrip("\n"))

    text.append(f"{len(angle_df):3d}    ! Nr of angles")
    text.append(_format_one_line(angle_df, ("i", "j", "k"), list(handler._ANGLE_PARAM_NAMES)).rstrip("\n"))

    text.append(f"{len(torsion_df):3d}    ! Nr of torsions")
    text.append(_format_one_line(torsion_df, ("i", "j", "k", "l"), torsion_names).rstrip("\n"))

    text.append(f"{len(hbond_df):3d}    ! Nr of hydrogen bonds")
    text.append(_format_one_line(hbond_df, ("i", "j", "k"), list(handler._HBOND_PARAM_NAMES)).rstrip("\n"))

    out_path.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")
    return out_path


def merge_ffields(
    source: str | Path,
    destination: str | Path,
    output: str | Path,
    atom_types: Iterable[str],
    *,
    fields: Iterable[str] | None = None,
    replace_existing: bool = False,
    allow_torsion_wildcard: bool = True,
) -> FFieldMergeSummary:
    """
    Merge selected atom-type parameters from source ``ffield`` into destination.
    """
    wanted_atoms = _normalize_atom_types(atom_types)
    wanted_fields = _normalize_fields(fields)

    src_handler = FFieldHandler(source)
    dst_handler = FFieldHandler(destination)
    src_sections = src_handler.sections
    dst_sections = {name: df.copy() for name, df in dst_handler.sections.items()}

    src_atom_df = src_sections["atom"].copy()
    dst_atom_df = dst_sections["atom"].copy()

    src_idx_to_sym, src_sym_to_idx = _atom_maps(src_atom_df)
    _, dst_sym_to_idx = _atom_maps(dst_atom_df)

    missing_in_source = [sym for sym in wanted_atoms if sym not in src_sym_to_idx]
    if missing_in_source:
        raise KeyError(f"Source ffield missing atom type(s): {', '.join(missing_in_source)}")

    source_selected_indices = {src_sym_to_idx[sym] for sym in wanted_atoms}
    new_index_by_symbol: dict[str, int] = {}

    appended = {field: 0 for field in SUPPORTED_FIELDS}
    updated = {field: 0 for field in SUPPORTED_FIELDS}
    skipped_existing = {field: 0 for field in SUPPORTED_FIELDS}
    skipped_incompatible = {field: 0 for field in SUPPORTED_FIELDS}
    appended_indices: dict[str, list[int]] = {field: [] for field in SUPPORTED_FIELDS}
    source_rows_used: dict[str, list[pd.Series]] = {field: [] for field in SUPPORTED_FIELDS}

    if "atom" in wanted_fields:
        next_idx = int(max(dst_atom_df.index.tolist())) + 1 if len(dst_atom_df.index) else 1
        for symbol in wanted_atoms:
            src_row = src_atom_df.loc[src_sym_to_idx[symbol]].copy()
            if symbol in dst_sym_to_idx:
                if replace_existing:
                    dst_idx = dst_sym_to_idx[symbol]
                    for col in dst_atom_df.columns:
                        if col in src_row.index:
                            dst_atom_df.loc[dst_idx, col] = src_row[col]
                    updated["atom"] += 1
                else:
                    skipped_existing["atom"] += 1
                continue

            src_row["symbol"] = symbol
            dst_atom_df.loc[next_idx] = src_row
            dst_sym_to_idx[symbol] = next_idx
            new_index_by_symbol[symbol] = next_idx
            appended["atom"] += 1
            appended_indices["atom"].append(next_idx)
            source_rows_used["atom"].append(src_row.copy())
            next_idx += 1
        dst_atom_df = dst_atom_df.sort_index()
        dst_sections["atom"] = dst_atom_df

    def _map_index(section: str, idx_val: int | None) -> int | None:
        if idx_val is None:
            return None
        if idx_val == 0:
            if section == "torsion" and allow_torsion_wildcard:
                return 0
            return None
        symbol = src_idx_to_sym.get(idx_val)
        if symbol is None:
            return None
        mapped = dst_sym_to_idx.get(symbol)
        if mapped is not None:
            return int(mapped)
        mapped_new = new_index_by_symbol.get(symbol)
        if mapped_new is not None:
            return int(mapped_new)
        return None

    for section, atom_cols in SECTION_ATOM_COLS.items():
        if section not in wanted_fields:
            continue
        src_df = src_sections[section]
        dst_df = dst_sections[section].copy()
        param_cols = [col for col in dst_df.columns if col not in atom_cols]

        for _, src_row in src_df.iterrows():
            src_atoms = [_safe_int(src_row[col]) for col in atom_cols]
            if not any(atom in source_selected_indices for atom in src_atoms if atom is not None):
                continue

            mapped_atoms: list[int] = []
            compatible = True
            for atom_idx in src_atoms:
                mapped = _map_index(section, atom_idx)
                if mapped is None:
                    compatible = False
                    break
                mapped_atoms.append(mapped)
            if not compatible:
                skipped_incompatible[section] += 1
                continue

            existing_mask = pd.Series([True] * len(dst_df), index=dst_df.index)
            for col, mapped in zip(atom_cols, mapped_atoms):
                existing_mask &= (pd.to_numeric(dst_df[col], errors="coerce").astype("Int64") == int(mapped))
            existing_rows = dst_df.loc[existing_mask]

            if not existing_rows.empty:
                if replace_existing:
                    target_idx = existing_rows.index[0]
                    for col in param_cols:
                        if col in src_row.index:
                            dst_df.loc[target_idx, col] = src_row[col]
                    updated[section] += 1
                else:
                    skipped_existing[section] += 1
                continue

            row_payload = _mapped_row_for_destination(src_row, atom_cols, mapped_atoms, param_cols)
            new_idx = int(max(dst_df.index.tolist())) + 1 if len(dst_df.index) else 1
            dst_df.loc[new_idx] = row_payload
            appended[section] += 1
            appended_indices[section].append(new_idx)
            source_rows_used[section].append(src_row.copy())

        dst_sections[section] = dst_df.sort_index()

    out_path = _write_ffield_sections(output, dst_sections, dst_handler)
    dst_idx_to_sym, _ = _atom_maps(dst_sections["atom"])
    destination_labels: dict[str, list[str]] = {}
    destination_blocks: dict[str, list[str]] = {}
    for section in SUPPORTED_FIELDS:
        destination_labels[section] = _labels_for_rows(
            section,
            dst_sections[section],
            appended_indices[section],
            dst_idx_to_sym,
        )
        destination_blocks[section] = _format_section_rows(
            section,
            dst_sections[section],
            appended_indices[section],
            dst_handler,
        )

    source_labels: dict[str, list[str]] = {field: [] for field in SUPPORTED_FIELDS}
    source_blocks: dict[str, list[str]] = {field: [] for field in SUPPORTED_FIELDS}
    src_format_helper = FFieldHandler(source)
    for section in SUPPORTED_FIELDS:
        rows = source_rows_used[section]
        if not rows:
            continue
        if section == "atom":
            temp_df = pd.DataFrame(rows)
            if "symbol" not in temp_df.columns:
                temp_df["symbol"] = None
            temp_df.index = list(range(1, len(temp_df) + 1))
            source_labels[section] = [str(temp_df.loc[i, "symbol"]).strip() for i in temp_df.index]
            source_blocks[section] = _format_section_rows(section, temp_df, temp_df.index.tolist(), src_format_helper)
            continue

        atom_cols = SECTION_ATOM_COLS[section]
        temp_df = pd.DataFrame(rows)
        temp_df.index = list(range(1, len(temp_df) + 1))
        source_labels[section] = _labels_for_rows(section, temp_df, temp_df.index.tolist(), src_idx_to_sym)
        source_blocks[section] = _format_section_rows(section, temp_df, temp_df.index.tolist(), src_format_helper)

    return FFieldMergeSummary(
        output_path=out_path,
        atom_types_merged=wanted_atoms,
        fields=wanted_fields,
        appended=appended,
        updated=updated,
        skipped_existing=skipped_existing,
        skipped_incompatible=skipped_incompatible,
        source_labels=source_labels,
        source_blocks=source_blocks,
        destination_labels=destination_labels,
        destination_blocks=destination_blocks,
    )
