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

try:
    from pymatgen.core.periodic_table import Element as _PMElement
except Exception:  # pragma: no cover - optional dependency in some environments
    _PMElement = None


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

SIMILARITY_ALIASES: dict[str, str] = {
    "family": "family",
    "group": "group",
    # backward-compatible aliases
    "column": "group",
    "periodic-column": "group",
    "periodic_column": "group",
    "radius": "radius",
    "radii": "radius",
}

RADIUS_KEYS: tuple[str, ...] = (
    "atomic_radius",
    "covalent_radius",
    "van_der_waals_radius",
    "atomic_radius_calculated",
    "average_ionic_radius",
    "average_cationic_radius",
    "average_anionic_radius",
)


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


@dataclass(frozen=True)
class FFieldAddElementSummary:
    """Summary of a single-element add operation."""

    output_path: Path
    element: str
    template_atom: str
    similarity_mode: str
    fields: tuple[str, ...]
    appended: dict[str, int]
    updated: dict[str, int]
    skipped_existing: dict[str, int]
    skipped_incompatible: dict[str, int]
    template_labels: dict[str, list[str]]
    template_blocks: dict[str, list[str]]
    destination_labels: dict[str, list[str]]
    destination_blocks: dict[str, list[str]]
    similarity_details: dict[str, object]


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


def _normalize_similarity_mode(mode: str | None) -> str:
    token = str(mode or "group").strip().lower()
    canonical = SIMILARITY_ALIASES.get(token)
    if canonical is None:
        choices = ", ".join(sorted(set(SIMILARITY_ALIASES.values())))
        raise ValueError(f"Unsupported similarity mode {mode!r}. Supported: {choices}")
    return canonical


def _normalize_radius_metrics(metrics: Iterable[str] | None) -> tuple[str, ...]:
    if metrics is None:
        return ("atomic_radius", "covalent_radius", "van_der_waals_radius")
    raw_tokens: list[str] = []
    for token in metrics:
        part = str(token).strip()
        if part:
            raw_tokens.append(part)
    if len(raw_tokens) == 1 and raw_tokens[0].lower() == "all":
        return RADIUS_KEYS
    out: list[str] = []
    for token in raw_tokens:
        key = token.strip().lower().replace(" ", "_").replace("-", "_")
        if key == "vdw_radius":
            key = "van_der_waals_radius"
        if key not in RADIUS_KEYS:
            raise ValueError(f"Unsupported radius metric {token!r}. Supported: {', '.join(RADIUS_KEYS)} or 'all'.")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("At least one radius metric is required.")
    return tuple(out)


def _safe_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


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


def _element_family(symbol: str) -> str | None:
    if _PMElement is None:
        return None
    try:
        elem = _PMElement(symbol)
    except Exception:
        return None
    if bool(getattr(elem, "is_transition_metal", False)):
        return "transition_metal"
    if bool(getattr(elem, "is_lanthanoid", False)):
        return "lanthanoid"
    if bool(getattr(elem, "is_actinoid", False)):
        return "actinoid"
    if bool(getattr(elem, "is_alkali", False)):
        return "alkali_metal"
    if bool(getattr(elem, "is_alkaline", False)):
        return "alkaline_earth_metal"
    if bool(getattr(elem, "is_halogen", False)):
        return "halogen"
    if bool(getattr(elem, "is_noble_gas", False)):
        return "noble_gas"
    if bool(getattr(elem, "is_metalloid", False)):
        return "metalloid"
    if bool(getattr(elem, "is_post_transition_metal", False)):
        return "post_transition_metal"
    return "other"


def _element_metadata(symbol: str, *, strict: bool = False) -> dict[str, object]:
    if _PMElement is None:
        return {"symbol": symbol, "group": None, "family": None, "atomic_radius": None, "covalent_radius": None, "van_der_waals_radius": None}
    try:
        elem = _PMElement(symbol)
    except Exception as exc:
        if strict:
            raise ValueError(f"Unknown or unsupported element symbol {symbol!r}.") from exc
        return {"symbol": symbol, "group": None, "family": None, "atomic_radius": None, "covalent_radius": None, "van_der_waals_radius": None}
    return {
        "symbol": symbol,
        "group": _safe_int(getattr(elem, "group", None)),
        "family": _element_family(symbol),
        # pymatgen does not expose a direct covalent radius consistently; atomic_radius_calculated is a close proxy.
        "atomic_radius": _safe_float(getattr(elem, "atomic_radius", None)),
        "covalent_radius": _safe_float(getattr(elem, "atomic_radius_calculated", None)),
        "van_der_waals_radius": _safe_float(getattr(elem, "van_der_waals_radius", None)),
    }


def _radius_distance(a: dict[str, object], b: dict[str, object], radius_metrics: tuple[str, ...]) -> float:
    diffs: list[float] = []
    for key in radius_metrics:
        av = _safe_float(a.get(key))
        bv = _safe_float(b.get(key))
        if av is None or bv is None:
            continue
        diffs.append(abs(av - bv))
    if not diffs:
        return float("inf")
    return float(sum(diffs) / len(diffs))


def _selection_sort_key(
    *,
    mode: str,
    target: dict[str, object],
    candidate: dict[str, object],
    radius_metrics: tuple[str, ...],
) -> tuple[object, ...]:
    tgt_group = _safe_int(target.get("group"))
    cand_group = _safe_int(candidate.get("group"))
    tgt_family = target.get("family")
    cand_family = candidate.get("family")
    same_group = int(not (tgt_family is not None and tgt_family == cand_family))
    same_column = int(not (tgt_group is not None and cand_group is not None and tgt_group == cand_group))
    group_gap = abs(tgt_group - cand_group) if (tgt_group is not None and cand_group is not None) else 999
    rdist = _radius_distance(target, candidate, radius_metrics)
    symbol = str(candidate.get("symbol", ""))

    if mode == "family":
        return same_group, same_column, group_gap, rdist, symbol
    if mode == "group":
        return same_column, group_gap, same_group, rdist, symbol
    return rdist, same_column, group_gap, same_group, symbol


def _pick_template_atom(
    *,
    target_symbol: str,
    available_symbols: Iterable[str],
    mode: str,
    manual_template: str | None,
    radius_metrics: tuple[str, ...],
) -> tuple[str, dict[str, object]]:
    candidates = [str(s).strip() for s in available_symbols if str(s).strip()]
    if not candidates:
        raise ValueError("Destination ffield has no atom symbols available for template matching.")

    target_md = _element_metadata(target_symbol, strict=True)

    if manual_template:
        picked = str(manual_template).strip()
        if picked not in candidates:
            raise ValueError(f"Requested template atom {picked!r} is not present in destination ffield.")
        picked_md = _element_metadata(picked)
        return picked, {
            "mode": "manual",
            "target": target_md,
            "chosen": picked_md,
            "candidates": [{"symbol": c, "selected": bool(c == picked)} for c in sorted(candidates)],
        }

    scored: list[tuple[tuple[object, ...], str, dict[str, object]]] = []
    for symbol in sorted(candidates):
        md = _element_metadata(symbol)
        scored.append(
            (
                _selection_sort_key(mode=mode, target=target_md, candidate=md, radius_metrics=radius_metrics),
                symbol,
                md,
            )
        )
    scored.sort(key=lambda x: x[0])
    chosen_symbol = scored[0][1]
    chosen_md = scored[0][2]

    details = []
    for key, symbol, md in scored:
        _ = key
        details.append(
            {
                "symbol": symbol,
                "group": md.get("group"),
                "family": md.get("family"),
                "radius_distance": _radius_distance(target_md, md, radius_metrics),
                "selected": bool(symbol == chosen_symbol),
            }
        )
    return chosen_symbol, {"mode": mode, "target": target_md, "chosen": chosen_md, "candidates": details}


def _write_ffield_sections(
    path: str | Path,
    sections: dict[str, pd.DataFrame],
    handler: FFieldHandler,
    *,
    description: str = "Merged force field generated by reaxkit.",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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


def _atom_tuple_exists(df: pd.DataFrame, atom_cols: tuple[str, ...], atom_values: list[int]) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    for col, value in zip(atom_cols, atom_values):
        mask &= (pd.to_numeric(df[col], errors="coerce").astype("Int64") == int(value))
    return df.loc[mask]


def _replacement_variants(
    atoms: list[int],
    *,
    src_atom_idx: int,
    dst_atom_idx: int,
) -> list[list[int]]:
    positions = [i for i, val in enumerate(atoms) if int(val) == int(src_atom_idx)]
    if not positions:
        return []
    variants: list[list[int]] = []
    n = len(positions)
    for mask in range(1, 2 ** n):
        mapped = list(atoms)
        for bit in range(n):
            if mask & (1 << bit):
                mapped[positions[bit]] = int(dst_atom_idx)
        variants.append(mapped)
    return variants


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


def add_element_to_ffield(
    destination: str | Path,
    output: str | Path,
    element: str,
    *,
    fields: Iterable[str] | None = None,
    similarity_mode: str = "group",
    closest_atom: str | None = None,
    radius_metrics: Iterable[str] | None = None,
    replace_existing: bool = False,
    allow_torsion_wildcard: bool = True,
) -> FFieldAddElementSummary:
    """
    Add one element to a destination ``ffield`` by cloning parameters from the closest existing atom type.
    """
    target_symbol = str(element).strip()
    if not target_symbol:
        raise ValueError("Element symbol must be non-empty.")
    wanted_fields = _normalize_fields(fields)
    mode = _normalize_similarity_mode(similarity_mode)
    chosen_radius_metrics = _normalize_radius_metrics(radius_metrics)

    dst_handler = FFieldHandler(destination)
    dst_sections = {name: df.copy() for name, df in dst_handler.sections.items()}
    dst_atom_df = dst_sections["atom"].copy()
    dst_idx_to_sym, dst_sym_to_idx = _atom_maps(dst_atom_df)

    if target_symbol in dst_sym_to_idx and not replace_existing:
        raise ValueError(
            f"Atom {target_symbol!r} already exists in destination ffield. "
            "Use --replace-existing to refresh atom parameters and projected terms."
        )

    candidate_symbols = [sym for sym in dst_sym_to_idx.keys() if sym != target_symbol]
    template_symbol, similarity_details = _pick_template_atom(
        target_symbol=target_symbol,
        available_symbols=candidate_symbols,
        mode=mode,
        manual_template=closest_atom,
        radius_metrics=chosen_radius_metrics,
    )
    if template_symbol not in dst_sym_to_idx:
        raise ValueError(f"Template atom {template_symbol!r} was not found in destination ffield.")
    template_idx = int(dst_sym_to_idx[template_symbol])

    appended = {field: 0 for field in SUPPORTED_FIELDS}
    updated = {field: 0 for field in SUPPORTED_FIELDS}
    skipped_existing = {field: 0 for field in SUPPORTED_FIELDS}
    skipped_incompatible = {field: 0 for field in SUPPORTED_FIELDS}
    appended_indices: dict[str, list[int]] = {field: [] for field in SUPPORTED_FIELDS}
    template_rows_used: dict[str, list[pd.Series]] = {field: [] for field in SUPPORTED_FIELDS}

    target_idx = dst_sym_to_idx.get(target_symbol)
    if target_idx is None:
        target_idx = int(max(dst_atom_df.index.tolist())) + 1 if len(dst_atom_df.index) else 1
        if "atom" in wanted_fields:
            src_row = dst_atom_df.loc[template_idx].copy()
            src_row["symbol"] = target_symbol
            dst_atom_df.loc[target_idx] = src_row
            appended["atom"] += 1
            appended_indices["atom"].append(target_idx)
            template_rows_used["atom"].append(dst_atom_df.loc[template_idx].copy())
        dst_sym_to_idx[target_symbol] = target_idx
        dst_idx_to_sym[target_idx] = target_symbol
    elif "atom" in wanted_fields and replace_existing:
        src_row = dst_atom_df.loc[template_idx].copy()
        for col in dst_atom_df.columns:
            if col == "symbol":
                continue
            if col in src_row.index:
                dst_atom_df.loc[target_idx, col] = src_row[col]
        updated["atom"] += 1
        template_rows_used["atom"].append(dst_atom_df.loc[template_idx].copy())
    elif "atom" in wanted_fields:
        skipped_existing["atom"] += 1

    dst_atom_df = dst_atom_df.sort_index()
    dst_sections["atom"] = dst_atom_df

    for section, atom_cols in SECTION_ATOM_COLS.items():
        if section not in wanted_fields:
            continue
        src_df = dst_sections[section].copy()
        dst_df = dst_sections[section].copy()
        param_cols = [col for col in dst_df.columns if col not in atom_cols]

        for _, src_row in src_df.iterrows():
            atoms = [_safe_int(src_row[col]) for col in atom_cols]
            if any(v is None for v in atoms):
                skipped_incompatible[section] += 1
                continue
            atom_ints = [int(v) for v in atoms if v is not None]
            if section == "torsion" and not allow_torsion_wildcard and any(v == 0 for v in atom_ints):
                skipped_incompatible[section] += 1
                continue

            variants = _replacement_variants(atom_ints, src_atom_idx=template_idx, dst_atom_idx=target_idx)
            if not variants:
                continue

            for mapped_atoms in variants:
                existing_rows = _atom_tuple_exists(dst_df, atom_cols, mapped_atoms)
                if not existing_rows.empty:
                    if replace_existing:
                        target_row = existing_rows.index[0]
                        for col in param_cols:
                            if col in src_row.index:
                                dst_df.loc[target_row, col] = src_row[col]
                        updated[section] += 1
                        template_rows_used[section].append(src_row.copy())
                    else:
                        skipped_existing[section] += 1
                    continue

                row_payload = _mapped_row_for_destination(src_row, atom_cols, mapped_atoms, param_cols)
                new_idx = int(max(dst_df.index.tolist())) + 1 if len(dst_df.index) else 1
                dst_df.loc[new_idx] = row_payload
                appended[section] += 1
                appended_indices[section].append(new_idx)
                template_rows_used[section].append(src_row.copy())

        dst_sections[section] = dst_df.sort_index()

    out_path = _write_ffield_sections(
        output,
        dst_sections,
        dst_handler,
        description=f"Force field expanded with {target_symbol} from template {template_symbol} generated by reaxkit.",
    )

    destination_labels: dict[str, list[str]] = {}
    destination_blocks: dict[str, list[str]] = {}
    for section in SUPPORTED_FIELDS:
        destination_labels[section] = _labels_for_rows(section, dst_sections[section], appended_indices[section], dst_idx_to_sym)
        destination_blocks[section] = _format_section_rows(section, dst_sections[section], appended_indices[section], dst_handler)

    template_labels: dict[str, list[str]] = {field: [] for field in SUPPORTED_FIELDS}
    template_blocks: dict[str, list[str]] = {field: [] for field in SUPPORTED_FIELDS}
    template_idx_to_sym = dict(dst_idx_to_sym)
    template_format_helper = FFieldHandler(destination)
    for section in SUPPORTED_FIELDS:
        rows = template_rows_used[section]
        if not rows:
            continue
        if section == "atom":
            temp_df = pd.DataFrame(rows)
            if "symbol" not in temp_df.columns:
                temp_df["symbol"] = None
            temp_df.index = list(range(1, len(temp_df) + 1))
            template_labels[section] = [str(temp_df.loc[i, "symbol"]).strip() for i in temp_df.index]
            template_blocks[section] = _format_section_rows(section, temp_df, temp_df.index.tolist(), template_format_helper)
            continue
        temp_df = pd.DataFrame(rows)
        temp_df.index = list(range(1, len(temp_df) + 1))
        template_labels[section] = _labels_for_rows(section, temp_df, temp_df.index.tolist(), template_idx_to_sym)
        template_blocks[section] = _format_section_rows(section, temp_df, temp_df.index.tolist(), template_format_helper)

    return FFieldAddElementSummary(
        output_path=out_path,
        element=target_symbol,
        template_atom=template_symbol,
        similarity_mode="manual" if closest_atom else mode,
        fields=wanted_fields,
        appended=appended,
        updated=updated,
        skipped_existing=skipped_existing,
        skipped_incompatible=skipped_incompatible,
        template_labels=template_labels,
        template_blocks=template_blocks,
        destination_labels=destination_labels,
        destination_blocks=destination_blocks,
        similarity_details=similarity_details,
    )
