"""
ReaxFF ``ffield`` merge utilities.

This module merges atom-type parameter blocks from a source ``ffield`` into a
destination ``ffield`` and writes a merged output file.

**Usage context**

- Template generation: Produce canonical text payloads for ReaxFF artifacts.
- File writing: Persist generated outputs to disk with stable formatting.
- Workflow integration: Support higher-level ReaxKit workflow commands.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from reaxkit.engine.common.io.ffield_handler import FFieldHandler

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
    """Represent FFieldMergeSummary.

    Public class used by ReaxFF generator components.

    Fields
    ------
    output_path : Path
        Dataclass field.
    atom_types_merged : tuple[str, ...]
        Dataclass field.
    fields : tuple[str, ...]
        Dataclass field.
    appended : dict[str, int]
        Dataclass field.
    updated : dict[str, int]
        Dataclass field.
    skipped_existing : dict[str, int]
        Dataclass field.
    skipped_incompatible : dict[str, int]
        Dataclass field.
    source_labels : dict[str, list[str]]
        Dataclass field.
    source_blocks : dict[str, list[str]]
        Dataclass field.
    destination_labels : dict[str, list[str]]
        Dataclass field.
    destination_blocks : dict[str, list[str]]
        Dataclass field.
    skipped_existing_labels : dict[str, list[str]]
        Dataclass field.
    skipped_existing_blocks : dict[str, list[str]]
        Dataclass field.
    template_labels : dict[str, list[str]]
        Dataclass field.
    template_blocks : dict[str, list[str]]
        Dataclass field.
    template_generated : dict[str, int]
        Dataclass field.
    template_choices : dict[str, str]
        Dataclass field.
    template_similarity_details : dict[str, dict[str, object]]
        Dataclass field.
    """

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
    skipped_existing_labels: dict[str, list[str]]
    skipped_existing_blocks: dict[str, list[str]]
    template_labels: dict[str, list[str]]
    template_blocks: dict[str, list[str]]
    template_generated: dict[str, int]
    template_choices: dict[str, str]
    template_similarity_details: dict[str, dict[str, object]]


@dataclass(frozen=True)
class FFieldAddElementSummary:
    """Represent FFieldAddElementSummary.

    Public class used by ReaxFF generator components.

    Fields
    ------
    output_path : Path
        Dataclass field.
    element : str
        Dataclass field.
    template_atom : str
        Dataclass field.
    similarity_mode : str
        Dataclass field.
    fields : tuple[str, ...]
        Dataclass field.
    appended : dict[str, int]
        Dataclass field.
    updated : dict[str, int]
        Dataclass field.
    skipped_existing : dict[str, int]
        Dataclass field.
    skipped_incompatible : dict[str, int]
        Dataclass field.
    template_labels : dict[str, list[str]]
        Dataclass field.
    template_blocks : dict[str, list[str]]
        Dataclass field.
    destination_labels : dict[str, list[str]]
        Dataclass field.
    destination_blocks : dict[str, list[str]]
        Dataclass field.
    similarity_details : dict[str, object]
        Dataclass field.
    """

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


@dataclass(frozen=True)
class FFieldAddTermSummary:
    """Represent FFieldAddTermSummary.

    Public class used by ReaxFF generator components.

    Fields
    ------
    output_path : Path
        Dataclass field.
    field : str
        Dataclass field.
    term : str
        Dataclass field.
    template_term : str | None
        Dataclass field.
    template_atoms : dict[str, str]
        Dataclass field.
    similarity_mode : str
        Dataclass field.
    appended : int
        Dataclass field.
    updated : int
        Dataclass field.
    skipped_existing : int
        Dataclass field.
    similarity_details : dict[str, object]
        Dataclass field.
    """

    output_path: Path
    field: str
    term: str
    template_term: str | None
    template_atoms: dict[str, str]
    similarity_mode: str
    appended: int
    updated: int
    skipped_existing: int
    similarity_details: dict[str, object]


def _normalize_fields(fields: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize fields."""
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


def _parse_term_atoms(term: str) -> tuple[str, ...]:
    """Parse term atoms."""
    parts = [p.strip() for p in str(term).replace(",", "-").split("-") if p.strip()]
    if not parts:
        raise ValueError("Term must include atom symbols separated by '-'; for example: Al-N-Al")
    return tuple(parts)


def _field_for_term_size(field: str, n_atoms: int) -> None:
    """Field for term size."""
    expected = len(SECTION_ATOM_COLS[field])
    if n_atoms != expected:
        raise ValueError(f"Field {field!r} expects {expected} atoms, got {n_atoms}.")


def _normalize_atom_types(atom_types: Iterable[str]) -> tuple[str, ...]:
    """Normalize atom types."""
    out: list[str] = []
    for raw in atom_types:
        token = str(raw).strip()
        if token and token not in out:
            out.append(token)
    if not out:
        raise ValueError("At least one atom type is required.")
    return tuple(out)


def _normalize_similarity_mode(mode: str | None) -> str:
    """Normalize similarity mode."""
    token = str(mode or "group").strip().lower()
    canonical = SIMILARITY_ALIASES.get(token)
    if canonical is None:
        choices = ", ".join(sorted(set(SIMILARITY_ALIASES.values())))
        raise ValueError(f"Unsupported similarity mode {mode!r}. Supported: {choices}")
    return canonical


def _normalize_radius_metrics(metrics: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize radius metrics."""
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
    """Safe int."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    """Safe float."""
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _is_float_token(value: str) -> bool:
    """Return whether a token can be parsed as a floating-point value."""
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _atom_maps(atom_df: pd.DataFrame) -> tuple[dict[int, str], dict[str, int]]:
    """Atom maps."""
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
    """Format general."""
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
    """Format atom."""
    lines: list[str] = []
    for idx in atom_df.index.tolist():
        symbol = str(atom_df.loc[idx, "symbol"]).strip()
        values = [float(atom_df.loc[idx, name]) for name in names]
        for block in range(4):
            start = block * 8
            end = start + 8
            lead = f" {symbol:<2} " if block == 0 else "    "
            tail = "".join(f"{value:9.4f}" for value in values[start:end])
            lines.append(f"{lead}{tail}".rstrip())
    return "\n".join(lines) + ("\n" if lines else "")


def _format_two_line(df: pd.DataFrame, atom_cols: tuple[str, str], param_names: list[str]) -> str:
    """Format two line."""
    lines: list[str] = []
    for idx in df.index.tolist():
        i_val = int(df.loc[idx, atom_cols[0]])
        j_val = int(df.loc[idx, atom_cols[1]])
        values = [float(df.loc[idx, name]) for name in param_names]
        lines.append(f"{i_val:3d}{j_val:3d}{''.join(f'{value:9.4f}' for value in values[0:8])}".rstrip())
        lines.append(f"      {''.join(f'{value:9.4f}' for value in values[8:16])}".rstrip())
    return "\n".join(lines) + ("\n" if lines else "")


def _format_one_line(df: pd.DataFrame, atom_cols: tuple[str, ...], param_names: list[str]) -> str:
    """Format one line."""
    lines: list[str] = []
    for idx in df.index.tolist():
        atom_part = "".join(f"{int(df.loc[idx, col]):3d}" for col in atom_cols)
        values = "".join(f"{float(df.loc[idx, name]):9.4f}" for name in param_names)
        lines.append(f"{atom_part}{values}".rstrip())
    return "\n".join(lines) + ("\n" if lines else "")


def _format_section_rows(section: str, df: pd.DataFrame, row_indices: list[int], handler: FFieldHandler) -> list[str]:
    """Format section rows."""
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
            if section == "atom" and not _is_float_token(line.split()[0]):
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
    """Labels for rows."""
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
    """Mapped row for destination."""
    payload: dict[str, object] = {}
    for col, mapped in zip(atom_cols, mapped_atoms):
        payload[col] = mapped
    for col in dst_param_cols:
        payload[col] = src_row[col] if col in src_row.index else float("nan")
    return payload


def _element_family(symbol: str) -> str | None:
    """Element family."""
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
    """Element metadata."""
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
    """Radius distance."""
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
    """Selection sort key."""
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
    """Pick template atom."""
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
    """Write ffield sections."""
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
    """Atom tuple exists."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col, value in zip(atom_cols, atom_values):
        mask &= (pd.to_numeric(df[col], errors="coerce").astype("Int64") == int(value))
    return df.loc[mask]


def _canonical_atom_tuple(section: str, atoms: list[int]) -> list[int]:
    """Canonical atom tuple."""
    if section in {"bond", "off_diagonal"} and len(atoms) == 2:
        a, b = int(atoms[0]), int(atoms[1])
        return [a, b] if a <= b else [b, a]
    if section in {"angle", "hbond"} and len(atoms) == 3:
        i, j, k = int(atoms[0]), int(atoms[1]), int(atoms[2])
        return [i, j, k] if i <= k else [k, j, i]
    if section == "torsion" and len(atoms) == 4:
        fwd = tuple(int(v) for v in atoms)
        rev = (fwd[3], fwd[2], fwd[1], fwd[0])
        return list(fwd if fwd <= rev else rev)
    return [int(v) for v in atoms]


def _equivalent_atom_tuples(section: str, atoms: list[int]) -> list[list[int]]:
    """Equivalent atom tuples."""
    canonical = _canonical_atom_tuple(section, atoms)
    out = [canonical]
    if section in {"bond", "off_diagonal"} and len(canonical) == 2:
        rev = [canonical[1], canonical[0]]
        if rev != canonical:
            out.append(rev)
    elif section in {"angle", "hbond"} and len(canonical) == 3:
        rev = [canonical[2], canonical[1], canonical[0]]
        if rev != canonical:
            out.append(rev)
    elif section == "torsion" and len(canonical) == 4:
        rev = [canonical[3], canonical[2], canonical[1], canonical[0]]
        if rev != canonical:
            out.append(rev)
    return out


def _atom_tuple_exists_equivalent(
    section: str,
    df: pd.DataFrame,
    atom_cols: tuple[str, ...],
    atom_values: list[int],
) -> pd.DataFrame:
    """Atom tuple exists equivalent."""
    candidates = _equivalent_atom_tuples(section, atom_values)
    hits: list[pd.DataFrame] = []
    for candidate in candidates:
        found = _atom_tuple_exists(df, atom_cols, candidate)
        if not found.empty:
            hits.append(found)
    if not hits:
        return df.iloc[0:0]
    return pd.concat(hits).loc[lambda x: ~x.index.duplicated(keep="first")]


def _equality_pattern(atoms: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Equality pattern."""
    pairs: list[tuple[int, int]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            pairs.append((1 if atoms[i] == atoms[j] else 0, j - i))
    return tuple(pairs)


def _atom_similarity_vector(target_symbol: str, candidate_symbol: str, mode: str, radius_metrics: tuple[str, ...]) -> tuple[float, float, float, float]:
    """Atom similarity vector."""
    tmd = _element_metadata(target_symbol, strict=True)
    cmd = _element_metadata(candidate_symbol, strict=True)
    same_family = 0.0 if (tmd.get("family") is not None and tmd.get("family") == cmd.get("family")) else 1.0
    tgrp = _safe_int(tmd.get("group"))
    cgrp = _safe_int(cmd.get("group"))
    same_group = 0.0 if (tgrp is not None and cgrp is not None and tgrp == cgrp) else 1.0
    group_gap = float(abs(tgrp - cgrp)) if (tgrp is not None and cgrp is not None) else 999.0
    rdist = float(_radius_distance(tmd, cmd, radius_metrics))
    if mode == "family":
        return same_family, same_group, group_gap, rdist
    if mode == "group":
        return same_group, group_gap, same_family, rdist
    return rdist, same_group, group_gap, same_family


def _best_unique_mapping_score(
    target_unique: tuple[str, ...],
    candidate_unique: tuple[str, ...],
    mode: str,
    radius_metrics: tuple[str, ...],
) -> tuple[tuple[float, ...], dict[str, str]]:
    """Best unique mapping score."""
    if not target_unique or not candidate_unique:
        raise ValueError("Target/candidate unique atom symbol lists must be non-empty.")
    n = len(target_unique)
    m = len(candidate_unique)
    best_score: tuple[float, ...] | None = None
    best_map: dict[str, str] = {}
    if m >= n:
        iter_maps = itertools.permutations(candidate_unique, n)
    else:
        iter_maps = itertools.product(candidate_unique, repeat=n)
    for assigned in iter_maps:
        comp = [0.0, 0.0, 0.0, 0.0]
        local_map: dict[str, str] = {}
        for src, dst in zip(target_unique, assigned):
            local_map[src] = dst
            v = _atom_similarity_vector(src, dst, mode, radius_metrics)
            comp = [comp[i] + float(v[i]) for i in range(4)]
        size_gap = float(abs(m - n))
        score = (size_gap, *comp)
        if best_score is None or score < best_score:
            best_score = score
            best_map = local_map
    return (best_score if best_score is not None else (9999.0, 9999.0, 9999.0, 9999.0, 9999.0)), best_map


def _replacement_variants(
    atoms: list[int],
    *,
    src_atom_idx: int,
    dst_atom_idx: int,
) -> list[list[int]]:
    """Replacement variants."""
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
    fill_missing_with_template: bool = False,
    template_similarity_mode: str = "group",
    template_closest_atom: str | None = None,
    template_radius_metrics: Iterable[str] | None = None,
) -> FFieldMergeSummary:
    """Merge ffields.

    Parameters
    ----------
    source : str | Path
        Input parameter.
    destination : str | Path
        Input parameter.
    output : str | Path
        Input parameter.
    atom_types : Iterable[str]
        Input parameter.
    fields : Iterable[str] | None, optional
        Keyword-only parameter.
    replace_existing : bool, optional
        Keyword-only parameter.
    allow_torsion_wildcard : bool, optional
        Keyword-only parameter.
    fill_missing_with_template : bool, optional
        Keyword-only parameter.
    template_similarity_mode : str, optional
        Keyword-only parameter.
    template_closest_atom : str | None, optional
        Keyword-only parameter.
    template_radius_metrics : Iterable[str] | None, optional
        Keyword-only parameter.

    Returns
    -------
    FFieldMergeSummary
        Return value.

    Examples
    --------
    ```python
    # Example
    merge_ffields(...)
    ```
    """
    wanted_atoms = _normalize_atom_types(atom_types)
    wanted_fields = _normalize_fields(fields)

    src_handler = FFieldHandler(source)
    dst_handler = FFieldHandler(destination)
    src_sections = src_handler.sections
    dst_sections = {name: df.copy() for name, df in dst_handler.sections.items()}

    src_atom_df = src_sections["atom"].copy()
    dst_atom_df = dst_sections["atom"].copy()
    base_dst_sections = {name: df.copy() for name, df in dst_sections.items()}

    src_idx_to_sym, src_sym_to_idx = _atom_maps(src_atom_df)
    _, dst_sym_to_idx = _atom_maps(dst_atom_df)
    base_dst_idx_to_sym, base_dst_sym_to_idx = _atom_maps(base_dst_sections["atom"])

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
    skipped_rows_projected: dict[str, list[dict[str, object]]] = {field: [] for field in SUPPORTED_FIELDS}
    template_rows_used: dict[str, list[pd.Series]] = {field: [] for field in SUPPORTED_FIELDS}
    template_generated = {field: 0 for field in SUPPORTED_FIELDS}
    template_choices: dict[str, str] = {}
    template_similarity_details: dict[str, dict[str, object]] = {}

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
                    skipped_atom_payload = {col: (src_row[col] if col in src_row.index else float("nan")) for col in dst_atom_df.columns}
                    skipped_atom_payload["symbol"] = symbol
                    skipped_rows_projected["atom"].append(skipped_atom_payload)
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
        """Map index."""
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
            mapped_atoms = _canonical_atom_tuple(section, mapped_atoms)

            existing_rows = _atom_tuple_exists_equivalent(section, dst_df, atom_cols, mapped_atoms)

            if not existing_rows.empty:
                if replace_existing:
                    target_idx = existing_rows.index[0]
                    for col in param_cols:
                        if col in src_row.index:
                            dst_df.loc[target_idx, col] = src_row[col]
                    updated[section] += 1
                else:
                    skipped_payload = _mapped_row_for_destination(src_row, atom_cols, mapped_atoms, param_cols)
                    skipped_rows_projected[section].append(skipped_payload)
                    skipped_existing[section] += 1
                continue

            row_payload = _mapped_row_for_destination(src_row, atom_cols, mapped_atoms, param_cols)
            new_idx = int(max(dst_df.index.tolist())) + 1 if len(dst_df.index) else 1
            dst_df.loc[new_idx] = row_payload
            appended[section] += 1
            appended_indices[section].append(new_idx)
            source_rows_used[section].append(src_row.copy())

        dst_sections[section] = dst_df.sort_index()

    if fill_missing_with_template:
        t_mode = _normalize_similarity_mode(template_similarity_mode)
        t_radius_metrics = _normalize_radius_metrics(template_radius_metrics)
        base_candidate_symbols = sorted(set(base_dst_sym_to_idx.keys()))

        for merged_symbol in wanted_atoms:
            target_idx = dst_sym_to_idx.get(merged_symbol)
            if target_idx is None:
                continue
            available = [s for s in base_candidate_symbols if s != merged_symbol]
            if not available:
                continue
            chosen_template, _details = _pick_template_atom(
                target_symbol=merged_symbol,
                available_symbols=available,
                mode=t_mode,
                manual_template=template_closest_atom,
                radius_metrics=t_radius_metrics,
            )
            template_choices[merged_symbol] = chosen_template
            template_similarity_details[merged_symbol] = _details
            template_idx = base_dst_sym_to_idx[chosen_template]

            for section, atom_cols in SECTION_ATOM_COLS.items():
                if section not in wanted_fields:
                    continue
                template_df = base_dst_sections[section]
                dst_df = dst_sections[section].copy()
                param_cols = [col for col in dst_df.columns if col not in atom_cols]

                for _, tpl_row in template_df.iterrows():
                    tpl_atoms = [_safe_int(tpl_row[col]) for col in atom_cols]
                    if any(v is None for v in tpl_atoms):
                        continue
                    tpl_atom_ints = [int(v) for v in tpl_atoms if v is not None]
                    if section == "torsion" and not allow_torsion_wildcard and any(v == 0 for v in tpl_atom_ints):
                        continue
                    variants = _replacement_variants(tpl_atom_ints, src_atom_idx=int(template_idx), dst_atom_idx=int(target_idx))
                    if not variants:
                        continue

                    for mapped_atoms in variants:
                        mapped_atoms = _canonical_atom_tuple(section, mapped_atoms)
                        row_payload = _mapped_row_for_destination(tpl_row, atom_cols, mapped_atoms, param_cols)
                        existing_rows = _atom_tuple_exists_equivalent(section, dst_df, atom_cols, mapped_atoms)
                        if not existing_rows.empty:
                            skipped_rows_projected[section].append(row_payload)
                            continue
                        new_idx = int(max(dst_df.index.tolist())) + 1 if len(dst_df.index) else 1
                        dst_df.loc[new_idx] = row_payload
                        appended[section] += 1
                        template_generated[section] += 1
                        appended_indices[section].append(new_idx)
                        template_rows_used[section].append(tpl_row.copy())

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

    template_labels: dict[str, list[str]] = {field: [] for field in SUPPORTED_FIELDS}
    template_blocks: dict[str, list[str]] = {field: [] for field in SUPPORTED_FIELDS}
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
        template_labels[section] = _labels_for_rows(section, temp_df, temp_df.index.tolist(), base_dst_idx_to_sym)
        template_blocks[section] = _format_section_rows(section, temp_df, temp_df.index.tolist(), template_format_helper)

    skipped_existing_labels: dict[str, list[str]] = {field: [] for field in SUPPORTED_FIELDS}
    skipped_existing_blocks: dict[str, list[str]] = {field: [] for field in SUPPORTED_FIELDS}
    for section in SUPPORTED_FIELDS:
        rows = skipped_rows_projected[section]
        if not rows:
            continue
        temp_df = pd.DataFrame(rows)
        if temp_df.empty:
            continue
        if section == "atom":
            if "symbol" not in temp_df.columns:
                temp_df["symbol"] = None
            temp_df.index = list(range(1, len(temp_df) + 1))
            skipped_existing_labels[section] = [str(temp_df.loc[i, "symbol"]).strip() for i in temp_df.index]
            skipped_existing_blocks[section] = _format_section_rows(section, temp_df, temp_df.index.tolist(), dst_handler)
            continue
        temp_df.index = list(range(1, len(temp_df) + 1))
        skipped_existing_labels[section] = _labels_for_rows(section, temp_df, temp_df.index.tolist(), dst_idx_to_sym)
        skipped_existing_blocks[section] = _format_section_rows(section, temp_df, temp_df.index.tolist(), dst_handler)

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
        skipped_existing_labels=skipped_existing_labels,
        skipped_existing_blocks=skipped_existing_blocks,
        template_labels=template_labels,
        template_blocks=template_blocks,
        template_generated=template_generated,
        template_choices=template_choices,
        template_similarity_details=template_similarity_details,
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
    """Add element to ffield.

    Parameters
    ----------
    destination : str | Path
        Input parameter.
    output : str | Path
        Input parameter.
    element : str
        Input parameter.
    fields : Iterable[str] | None, optional
        Keyword-only parameter.
    similarity_mode : str, optional
        Keyword-only parameter.
    closest_atom : str | None, optional
        Keyword-only parameter.
    radius_metrics : Iterable[str] | None, optional
        Keyword-only parameter.
    replace_existing : bool, optional
        Keyword-only parameter.
    allow_torsion_wildcard : bool, optional
        Keyword-only parameter.

    Returns
    -------
    FFieldAddElementSummary
        Return value.

    Examples
    --------
    ```python
    # Example
    add_element_to_ffield(...)
    ```
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
                mapped_atoms = _canonical_atom_tuple(section, mapped_atoms)
                existing_rows = _atom_tuple_exists_equivalent(section, dst_df, atom_cols, mapped_atoms)
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


def add_term_to_ffield(
    destination: str | Path,
    output: str | Path,
    *,
    field: str,
    term: str,
    closest_term: str | None = None,
    template_atom_map: Mapping[str, str] | None = None,
    similarity_mode: str = "group",
    radius_metrics: Iterable[str] | None = None,
    same_general_order: bool = False,
    replace_existing: bool = False,
) -> FFieldAddTermSummary:
    """Add term to ffield.

    Parameters
    ----------
    destination : str | Path
        Input parameter.
    output : str | Path
        Input parameter.
    field : str
        Keyword-only parameter.
    term : str
        Keyword-only parameter.
    closest_term : str | None, optional
        Keyword-only parameter.
    template_atom_map : Mapping[str, str] | None, optional
        Keyword-only parameter.
    similarity_mode : str, optional
        Keyword-only parameter.
    radius_metrics : Iterable[str] | None, optional
        Keyword-only parameter.
    same_general_order : bool, optional
        Keyword-only parameter.
    replace_existing : bool, optional
        Keyword-only parameter.

    Returns
    -------
    FFieldAddTermSummary
        Return value.

    Examples
    --------
    ```python
    # Example
    add_term_to_ffield(...)
    ```
    """
    wanted_field = _normalize_fields([field])[0]
    if wanted_field == "atom":
        raise ValueError("add_term_to_ffield does not support 'atom'. Use add_element_to_ffield instead.")
    mode = _normalize_similarity_mode(similarity_mode)
    chosen_radius_metrics = _normalize_radius_metrics(radius_metrics)
    requested_atoms = _parse_term_atoms(term)
    _field_for_term_size(wanted_field, len(requested_atoms))

    dst_handler = FFieldHandler(destination)
    dst_sections = {name: df.copy() for name, df in dst_handler.sections.items()}
    atom_df = dst_sections["atom"]
    _, sym_to_idx = _atom_maps(atom_df)
    atom_cols = SECTION_ATOM_COLS[wanted_field]
    section_df = dst_sections[wanted_field].copy()
    param_cols = [col for col in section_df.columns if col not in atom_cols]

    missing = [s for s in requested_atoms if s not in sym_to_idx]
    if missing:
        raise ValueError(
            f"All term atoms must exist in destination ffield. Missing: {', '.join(sorted(set(missing)))}"
        )

    requested_idxs = _canonical_atom_tuple(wanted_field, [int(sym_to_idx[s]) for s in requested_atoms])
    existing = _atom_tuple_exists_equivalent(wanted_field, section_df, atom_cols, requested_idxs)

    manual_map = {str(k).strip(): str(v).strip() for k, v in (template_atom_map or {}).items() if str(k).strip()}
    requested_pattern = _equality_pattern(requested_atoms)
    requested_unique = tuple(sorted(set(requested_atoms)))
    atom_template_map: dict[str, str] = {}
    similarity_details: dict[str, object] = {
        "mode": "manual_term" if closest_term else ("manual" if manual_map else mode),
        "same_general_order": bool(same_general_order),
        "candidate_count": int(len(section_df)),
        "ranked_candidates": [],
    }

    template_row = None
    template_term_symbols: tuple[str, ...] | None = None
    if closest_term:
        closest_atoms = _parse_term_atoms(closest_term)
        _field_for_term_size(wanted_field, len(closest_atoms))
        missing_closest = [s for s in closest_atoms if s not in sym_to_idx]
        if missing_closest:
            raise ValueError(
                f"All --closest-term atoms must exist in destination ffield. Missing: "
                f"{', '.join(sorted(set(missing_closest)))}"
            )
        closest_idxs = _canonical_atom_tuple(wanted_field, [int(sym_to_idx[s]) for s in closest_atoms])
        closest_existing = _atom_tuple_exists_equivalent(wanted_field, section_df, atom_cols, closest_idxs)
        if closest_existing.empty:
            raise ValueError(
                f"Requested --closest-term {closest_term!r} was not found in destination field section {wanted_field!r}."
            )
        template_row = section_df.loc[closest_existing.index[0]]
        template_term_symbols = tuple(closest_atoms)
        for src, dst in zip(requested_atoms, closest_atoms):
            atom_template_map[src] = dst
    elif manual_map:
        for symbol, mapped in manual_map.items():
            if symbol not in set(requested_atoms):
                continue
            if mapped not in sym_to_idx:
                raise ValueError(f"Template atom {mapped!r} for {symbol!r} is not in destination ffield.")
        atom_template_map = dict(manual_map)
        template_idxs = _canonical_atom_tuple(
            wanted_field,
            [int(sym_to_idx[atom_template_map.get(s, s)]) for s in requested_atoms],
        )
        template_existing = _atom_tuple_exists_equivalent(wanted_field, section_df, atom_cols, template_idxs)
        if template_existing.empty:
            raise ValueError(
                "No template term was found for the selected manual mapping. "
                f"Requested term: {'-'.join(requested_atoms)} ; template term: "
                f"{'-'.join(atom_template_map.get(s, s) for s in requested_atoms)}"
            )
        template_row = section_df.loc[template_existing.index[0]]
        template_term_symbols = tuple(atom_template_map.get(s, s) for s in requested_atoms)
    else:
        if section_df.empty:
            raise ValueError(f"Destination field section {wanted_field!r} has no rows to use as template.")
        idx_to_sym, _ = _atom_maps(atom_df)
        ranked: list[tuple[tuple[float, ...], int, tuple[str, ...], dict[str, str]]] = []
        for ridx, row in section_df.iterrows():
            raw_atoms = [_safe_int(row[c]) for c in atom_cols]
            if any(v is None for v in raw_atoms):
                continue
            cand_idxs = _canonical_atom_tuple(wanted_field, [int(v) for v in raw_atoms if v is not None])
            cand_symbols = tuple(idx_to_sym.get(v, str(v)) for v in cand_idxs)
            cand_pattern = _equality_pattern(cand_symbols)
            if same_general_order and cand_pattern != requested_pattern:
                continue
            pattern_penalty = 0.0 if cand_pattern == requested_pattern else 1.0
            cand_unique = tuple(sorted(set(cand_symbols)))
            sim_score, map_unique = _best_unique_mapping_score(requested_unique, cand_unique, mode, chosen_radius_metrics)
            tuple_score = (pattern_penalty, *sim_score)
            ranked.append((tuple_score, int(ridx), cand_symbols, map_unique))
            similarity_details["ranked_candidates"].append(
                {
                    "row_index": int(ridx),
                    "term": "-".join(cand_symbols),
                    "pattern_match": bool(cand_pattern == requested_pattern),
                    "score": list(tuple_score),
                }
            )
        if not ranked:
            if same_general_order:
                raise ValueError("No candidate terms satisfy --same-general-order in the selected field section.")
            raise ValueError("No candidate terms available for similarity-based term templating.")
        ranked.sort(key=lambda x: (x[0], "-".join(x[2])))
        best_score, best_idx, best_symbols, best_map = ranked[0]
        template_row = section_df.loc[best_idx]
        template_term_symbols = best_symbols
        atom_template_map = best_map
        similarity_details["selected_score"] = list(best_score)
        similarity_details["selected_row_index"] = int(best_idx)

    row_payload = _mapped_row_for_destination(template_row, atom_cols, requested_idxs, param_cols)

    appended = 0
    updated = 0
    skipped_existing = 0
    if not existing.empty and not replace_existing:
        skipped_existing = 1
    elif not existing.empty:
        target_idx = existing.index[0]
        for col in param_cols:
            if col in row_payload:
                section_df.loc[target_idx, col] = row_payload[col]
        updated = 1
    else:
        new_idx = int(max(section_df.index.tolist())) + 1 if len(section_df.index) else 1
        section_df.loc[new_idx] = row_payload
        appended = 1
    dst_sections[wanted_field] = section_df.sort_index()

    out_path = _write_ffield_sections(
        output,
        dst_sections,
        dst_handler,
        description=(
            "Force field with explicit term added/generated by reaxkit. "
            f"field={wanted_field}, term={'-'.join(requested_atoms)}"
        ),
    )
    similarity_details["template_term"] = "-".join(template_term_symbols or ())
    return FFieldAddTermSummary(
        output_path=out_path,
        field=wanted_field,
        term="-".join(requested_atoms),
        template_term=str(similarity_details["template_term"] or ""),
        template_atoms=atom_template_map,
        similarity_mode="manual" if manual_map else mode,
        appended=appended,
        updated=updated,
        skipped_existing=skipped_existing,
        similarity_details=similarity_details,
    )
