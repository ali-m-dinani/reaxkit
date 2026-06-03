"""Engine-agnostic edit operations for ROMP active-control examples."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MoleculeRecord:
    molecule_id: int
    atom_ids: tuple[int, ...]
    formula: str
    element_counts: dict[str, int]


@dataclass(frozen=True)
class MoleculeDetectionResult:
    molecule_table: pd.DataFrame
    formula_table: pd.DataFrame


def _as_tuple_or_none(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    out = tuple(str(v).strip() for v in values if str(v).strip())
    return out or None


def _parse_atom_ids_field(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return tuple()
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def _formula_from_counts(counts: dict[str, int]) -> str:
    return "".join(f"{element}{int(counts[element])}" for element in sorted(counts))


def _connected_components(atom_ids: list[int], connectivity: pd.DataFrame) -> list[tuple[int, ...]]:
    adjacency = {int(atom_id): set() for atom_id in atom_ids}
    if not connectivity.empty:
        for row in connectivity[["source_atom_id", "target_atom_id"]].itertuples(index=False):
            source = int(row.source_atom_id)
            target = int(row.target_atom_id)
            if source in adjacency and target in adjacency:
                adjacency[source].add(target)
                adjacency[target].add(source)

    visited: set[int] = set()
    components: list[tuple[int, ...]] = []
    for atom_id in sorted(adjacency):
        if atom_id in visited:
            continue
        stack = [atom_id]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set()), reverse=True):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return components


def analyze_molecules_with_components(
    coordinates: pd.DataFrame,
    connectivity: pd.DataFrame,
    *,
    include_any_elements: list[str] | tuple[str, ...] | None = None,
    include_all_elements: list[str] | tuple[str, ...] | None = None,
    exclude_any_elements: list[str] | tuple[str, ...] | None = None,
    include_motifs: list[str] | tuple[str, ...] | None = None,
    exclude_motifs: list[str] | tuple[str, ...] | None = None,
) -> MoleculeDetectionResult:
    include_any = set(_as_tuple_or_none(include_any_elements) or ())
    include_all = set(_as_tuple_or_none(include_all_elements) or ())
    exclude_any = set(_as_tuple_or_none(exclude_any_elements) or ())
    include_motifs_norm = list(_as_tuple_or_none(include_motifs) or ())
    exclude_motifs_norm = list(_as_tuple_or_none(exclude_motifs) or ())

    element_by_id = {
        int(row.atom_id): str(row.atom_type)
        for row in coordinates[["atom_id", "atom_type"]].itertuples(index=False)
    }
    rows = []
    for molecule_id, atom_ids in enumerate(
        _connected_components(list(element_by_id.keys()), connectivity),
        start=1,
    ):
        counts: dict[str, int] = {}
        for atom_id in atom_ids:
            element = element_by_id[atom_id]
            counts[element] = counts.get(element, 0) + 1
        elems = set(counts)
        if include_any and not (elems & include_any):
            continue
        if include_all and not include_all.issubset(elems):
            continue
        if exclude_any and (elems & exclude_any):
            continue
        if include_motifs_norm and not any(_matches_formula_motif(counts, motif) for motif in include_motifs_norm):
            continue
        if exclude_motifs_norm and any(_matches_formula_motif(counts, motif) for motif in exclude_motifs_norm):
            continue

        rows.append(
            {
                "molecule_id": int(molecule_id),
                "formula": _formula_from_counts(counts),
                "atom_ids": ",".join(str(atom_id) for atom_id in atom_ids),
            }
        )

    molecule_table = pd.DataFrame(rows, columns=["molecule_id", "formula", "atom_ids"])
    if molecule_table.empty:
        formula_table = pd.DataFrame(columns=["formula", "molecule_count", "isomer_count"])
    else:
        formula_table = (
            molecule_table.groupby("formula", as_index=False)
            .agg(molecule_count=("molecule_id", "count"))
            .sort_values("formula", kind="stable")
            .reset_index(drop=True)
        )
        formula_table["isomer_count"] = 1
    return MoleculeDetectionResult(molecule_table=molecule_table, formula_table=formula_table)


def molecules_from_detection_result(
    detection: MoleculeDetectionResult,
    coordinates: pd.DataFrame,
) -> list[MoleculeRecord]:
    if detection.molecule_table.empty:
        return []

    element_by_id = {
        int(row.atom_id): str(row.atom_type)
        for row in coordinates[["atom_id", "atom_type"]].itertuples(index=False)
    }
    out: list[MoleculeRecord] = []
    table = detection.molecule_table.sort_values("molecule_id", kind="stable").reset_index(drop=True)
    for row in table.itertuples(index=False):
        atom_ids = _parse_atom_ids_field(row.atom_ids)
        counts: dict[str, int] = {}
        for atom_id in atom_ids:
            element = element_by_id.get(atom_id)
            if element is None:
                continue
            counts[element] = counts.get(element, 0) + 1
        out.append(
            MoleculeRecord(
                molecule_id=int(row.molecule_id),
                atom_ids=atom_ids,
                formula=str(row.formula),
                element_counts=counts,
            )
        )
    return out


def formula_count_table(detection: MoleculeDetectionResult) -> pd.DataFrame:
    if detection.formula_table.empty:
        return pd.DataFrame(columns=["formula", "molecule_count", "isomer_count"])
    return detection.formula_table.copy().reset_index(drop=True)


def connectivity_groups_from_detection(detection: MoleculeDetectionResult) -> list[tuple[int, ...]]:
    if detection.molecule_table.empty:
        return []
    groups: list[tuple[int, ...]] = []
    for value in detection.molecule_table["atom_ids"].tolist():
        groups.append(_parse_atom_ids_field(value))
    return groups


def molecules_from_connectivity(
    coordinates: pd.DataFrame,
    connectivity: pd.DataFrame,
) -> list[MoleculeRecord]:
    detection = analyze_molecules_with_components(coordinates, connectivity)
    return molecules_from_detection_result(detection, coordinates)


def _pick_by_strategy(
    records: list[MoleculeRecord],
    *,
    max_molecules: int | None,
    selection_strategy: str,
    rng: np.random.Generator | None,
) -> list[MoleculeRecord]:
    if max_molecules is None:
        return records

    n = max(0, int(max_molecules))
    if n == 0 or not records:
        return []
    if n >= len(records):
        return records

    strategy = str(selection_strategy).strip().lower()
    if strategy == "random":
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(records), size=n, replace=False)
        idx_set = set(int(i) for i in idx.tolist())
        return [rec for i, rec in enumerate(records) if i in idx_set]
    return records[:n]


def select_atoms_to_remove_by_formula(
    molecules: list[MoleculeRecord],
    target_formulas: list[str] | tuple[str, ...],
    max_molecules: int | None = None,
    selection_strategy: str = "first",
    rng: np.random.Generator | None = None,
) -> tuple[set[int], list[MoleculeRecord]]:
    target = {str(formula).strip() for formula in target_formulas if str(formula).strip()}
    matched = [m for m in molecules if m.formula in target]
    matched = _pick_by_strategy(
        matched,
        max_molecules=max_molecules,
        selection_strategy=selection_strategy,
        rng=rng,
    )
    atom_ids: set[int] = set()
    for molecule in matched:
        atom_ids.update(molecule.atom_ids)
    return atom_ids, matched


_MOTIF_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d+|x)?")


def _as_symbol_set(values: list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()
    return {str(v).strip() for v in values if str(v).strip()}


def _parse_formula_motif(motif: str) -> dict[str, tuple[str, int]]:
    text = str(motif).strip()
    if not text:
        return {}

    parsed: dict[str, tuple[str, int]] = {}
    pos = 0
    for match in _MOTIF_TOKEN_RE.finditer(text):
        if match.start() != pos:
            raise ValueError(f"Invalid motif syntax '{motif}' near index {pos}.")
        element = match.group(1)
        raw_count = match.group(2)
        if raw_count is None or raw_count.lower() == "x":
            parsed[element] = (">=", 1)
        else:
            parsed[element] = ("==", int(raw_count))
        pos = match.end()
    if pos != len(text):
        raise ValueError(f"Invalid motif syntax '{motif}' near index {pos}.")
    return parsed


def _matches_formula_motif(element_counts: dict[str, int], motif: str) -> bool:
    rules = _parse_formula_motif(motif)
    if not rules:
        return False
    for element, (op, value) in rules.items():
        count = int(element_counts.get(element, 0))
        if op == "==" and count != value:
            return False
        if op == ">=" and count < value:
            return False
    return True


def select_atoms_to_remove_by_rules(
    molecules: list[MoleculeRecord],
    *,
    include_any_elements: list[str] | tuple[str, ...] | None = None,
    include_all_elements: list[str] | tuple[str, ...] | None = None,
    exclude_any_elements: list[str] | tuple[str, ...] | None = None,
    allowed_elements: list[str] | tuple[str, ...] | None = None,
    include_motifs: list[str] | tuple[str, ...] | None = None,
    exclude_motifs: list[str] | tuple[str, ...] | None = None,
    max_molecules: int | None = None,
    selection_strategy: str = "first",
    rng: np.random.Generator | None = None,
) -> tuple[set[int], list[MoleculeRecord]]:
    include_any = _as_symbol_set(include_any_elements)
    include_all = _as_symbol_set(include_all_elements)
    exclude_any = _as_symbol_set(exclude_any_elements)
    allowed = _as_symbol_set(allowed_elements)
    include_motifs_norm = [str(m).strip() for m in (include_motifs or []) if str(m).strip()]
    exclude_motifs_norm = [str(m).strip() for m in (exclude_motifs or []) if str(m).strip()]
    has_criteria = any(
        [
            bool(include_any),
            bool(include_all),
            bool(exclude_any),
            bool(allowed),
            bool(include_motifs_norm),
            bool(exclude_motifs_norm),
        ]
    )
    if not has_criteria:
        return set(), []

    matched: list[MoleculeRecord] = []
    for molecule in molecules:
        elems = set(molecule.element_counts.keys())
        if include_any and not (elems & include_any):
            continue
        if include_all and not include_all.issubset(elems):
            continue
        if exclude_any and (elems & exclude_any):
            continue
        if allowed and not elems.issubset(allowed):
            continue
        if include_motifs_norm and not any(
            _matches_formula_motif(molecule.element_counts, motif) for motif in include_motifs_norm
        ):
            continue
        if exclude_motifs_norm and any(
            _matches_formula_motif(molecule.element_counts, motif) for motif in exclude_motifs_norm
        ):
            continue
        matched.append(molecule)

    matched = _pick_by_strategy(
        matched,
        max_molecules=max_molecules,
        selection_strategy=selection_strategy,
        rng=rng,
    )
    atom_ids: set[int] = set()
    for molecule in matched:
        atom_ids.update(molecule.atom_ids)
    return atom_ids, matched


def select_molecule_ids_for_deletion(
    molecules: list[MoleculeRecord],
    *,
    mode: str = "formula",
    target_formulas: list[str] | tuple[str, ...] | None = None,
    include_any_elements: list[str] | tuple[str, ...] | None = None,
    include_all_elements: list[str] | tuple[str, ...] | None = None,
    exclude_any_elements: list[str] | tuple[str, ...] | None = None,
    allowed_elements: list[str] | tuple[str, ...] | None = None,
    include_motifs: list[str] | tuple[str, ...] | None = None,
    exclude_motifs: list[str] | tuple[str, ...] | None = None,
    max_molecules: int | None = None,
    selection_strategy: str = "first",
    rng: np.random.Generator | None = None,
) -> list[int]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "rules":
        _atom_ids, matched = select_atoms_to_remove_by_rules(
            molecules,
            include_any_elements=include_any_elements,
            include_all_elements=include_all_elements,
            exclude_any_elements=exclude_any_elements,
            allowed_elements=allowed_elements,
            include_motifs=include_motifs,
            exclude_motifs=exclude_motifs,
            max_molecules=max_molecules,
            selection_strategy=selection_strategy,
            rng=rng,
        )
    else:
        _atom_ids, matched = select_atoms_to_remove_by_formula(
            molecules,
            target_formulas=target_formulas or [],
            max_molecules=max_molecules,
            selection_strategy=selection_strategy,
            rng=rng,
        )
    return [int(m.molecule_id) for m in matched]


def remove_atoms_by_id(coordinates: pd.DataFrame, atom_ids_to_remove: set[int]) -> pd.DataFrame:
    if not atom_ids_to_remove:
        return coordinates.copy()
    keep = ~coordinates["atom_id"].astype(int).isin(set(int(x) for x in atom_ids_to_remove))
    return coordinates.loc[keep].copy().reset_index(drop=True)


def choose_anchor_and_nearest(
    coordinates: pd.DataFrame,
    *,
    anchor_element: str = "Ru",
    nearest_element: str = "C",
    rng: np.random.Generator | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    if rng is None:
        rng = np.random.default_rng()

    anchors = coordinates.loc[coordinates["atom_type"].astype(str) == str(anchor_element)].copy()
    if anchors.empty:
        raise ValueError(f"No anchor atoms found for element '{anchor_element}'.")
    anchor_idx = int(rng.integers(0, len(anchors)))
    anchor = anchors.iloc[anchor_idx]

    candidates = coordinates.loc[
        coordinates["atom_type"].astype(str) == str(nearest_element)
    ].copy()
    if candidates.empty:
        return anchor, None

    anchor_xyz = np.asarray([float(anchor["x"]), float(anchor["y"]), float(anchor["z"])], dtype=float)
    cand_xyz = candidates[["x", "y", "z"]].to_numpy(dtype=float)
    distances = np.linalg.norm(cand_xyz - anchor_xyz, axis=1)
    nearest_idx = int(np.argmin(distances))
    return anchor, candidates.iloc[nearest_idx]


def monomer_relative_to_com(monomer_coordinates: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    symbols = monomer_coordinates["atom_type"].astype(str).tolist()
    xyz = monomer_coordinates[["x", "y", "z"]].to_numpy(dtype=float)
    com = xyz.mean(axis=0)
    return symbols, xyz - com


def sample_candidate_points_around_anchor(
    anchor_xyz: np.ndarray,
    *,
    radius: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    points = np.empty((n_samples, 3), dtype=float)
    for i in range(n_samples):
        r = float(rng.uniform(0.0, radius))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        phi = float(rng.uniform(0.0, np.pi))
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        points[i, :] = anchor_xyz + np.asarray([x, y, z], dtype=float)
    return points


def find_nonoverlapping_point(
    existing_xyz: np.ndarray,
    monomer_relative_xyz: np.ndarray,
    anchor_xyz: np.ndarray,
    *,
    radius: float,
    min_distance: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    candidate_points = sample_candidate_points_around_anchor(
        anchor_xyz,
        radius=radius,
        n_samples=n_samples,
        rng=rng,
    )

    for point in candidate_points:
        placed = monomer_relative_xyz + point
        ok = True
        for pos in placed:
            dists = np.linalg.norm(existing_xyz - pos, axis=1)
            if float(dists.min()) <= float(min_distance):
                ok = False
                break
        if ok:
            return point
    return None


def append_monomer_at_point(
    coordinates: pd.DataFrame,
    monomer_symbols: list[str],
    monomer_relative_xyz: np.ndarray,
    placement_point: np.ndarray,
) -> pd.DataFrame:
    max_id = int(coordinates["atom_id"].astype(int).max()) if not coordinates.empty else 0
    placed_xyz = monomer_relative_xyz + placement_point
    new_rows = []
    for i, (symbol, xyz) in enumerate(zip(monomer_symbols, placed_xyz), start=1):
        new_rows.append(
            {
                "atom_id": max_id + i,
                "atom_type": str(symbol),
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            }
        )
    appended = pd.concat([coordinates, pd.DataFrame(new_rows)], ignore_index=True)
    return appended
