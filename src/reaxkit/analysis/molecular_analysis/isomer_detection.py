"""Detect molecular formulas and isomer groups from structure connectivity.

This module identifies molecules, groups them by formula, and derives isomer
partitions from single-structure connectivity representations. It is scoped to
graph-based molecule/isomer detection and does not perform time-resolved
molecular tracking.

**Usage context**

- Structure decomposition: Enumerate molecules present in one geometry.
- Isomer counting: Split molecules into isomer classes per formula.
- Topology summaries: Export formula/isomer tables for structural reports.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field as dc_field
import re
from typing import Any

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import GeometryData
from reaxkit.presentation.specs import PresentationSpec


def _empty_formula_table() -> pd.DataFrame:
    """Return an empty formula summary table with stable output columns."""
    return pd.DataFrame(columns=["formula", "molecule_count", "isomer_count"])


def _empty_isomer_table() -> pd.DataFrame:
    """Return an empty isomer summary table with stable output columns."""
    return pd.DataFrame(
        columns=[
            "formula",
            "isomer_id",
            "molecule_count",
            "atom_count",
            "representative_molecule_id",
        ]
    )


def _empty_molecule_table() -> pd.DataFrame:
    """Return an empty per-molecule assignment table with stable columns."""
    return pd.DataFrame(
        columns=[
            "molecule_id",
            "formula",
            "isomer_id",
            "atom_count",
            "atom_ids",
        ]
    )


def _coordinates_table(data: GeometryData) -> pd.DataFrame:
    """Normalize geometry coordinate rows into unique ``atom_id``/``atom_type`` pairs."""
    coordinates = data.coordinates.copy()
    if coordinates.empty:
        return pd.DataFrame(columns=["atom_id", "atom_type"])
    required = {"atom_id", "atom_type"}
    missing = required.difference(set(coordinates.columns))
    if missing:
        raise ValueError(f"GeometryData.coordinates is missing required columns: {sorted(missing)}")
    out = coordinates.loc[:, ["atom_id", "atom_type"]].copy()
    out["atom_id"] = pd.to_numeric(out["atom_id"], errors="coerce")
    out = out.dropna(subset=["atom_id"]).copy()
    out["atom_id"] = out["atom_id"].astype(int)
    out["atom_type"] = out["atom_type"].astype(str)
    out = out.drop_duplicates(subset=["atom_id"], keep="first").sort_values("atom_id", kind="stable")
    return out.reset_index(drop=True)


def _connectivity_table(data: GeometryData) -> pd.DataFrame:
    """Normalize geometry connectivity rows into integer source/target edges."""
    connectivity = data.connectivity.copy()
    if connectivity.empty:
        return pd.DataFrame(columns=["source_atom_id", "target_atom_id"])
    required = {"source_atom_id", "target_atom_id"}
    missing = required.difference(set(connectivity.columns))
    if missing:
        raise ValueError(f"GeometryData.connectivity is missing required columns: {sorted(missing)}")
    out = connectivity.loc[:, ["source_atom_id", "target_atom_id"]].copy()
    out["source_atom_id"] = pd.to_numeric(out["source_atom_id"], errors="coerce")
    out["target_atom_id"] = pd.to_numeric(out["target_atom_id"], errors="coerce")
    out = out.dropna(subset=["source_atom_id", "target_atom_id"]).copy()
    out["source_atom_id"] = out["source_atom_id"].astype(int)
    out["target_atom_id"] = out["target_atom_id"].astype(int)
    return out.reset_index(drop=True)


def _adjacency_from_edges(atom_ids: list[int], connectivity: pd.DataFrame) -> dict[int, set[int]]:
    """Build an undirected adjacency map from connectivity edge rows."""
    atom_set = set(atom_ids)
    adjacency: dict[int, set[int]] = {atom_id: set() for atom_id in atom_ids}

    for row in connectivity.itertuples(index=False):
        src = int(row.source_atom_id)
        dst = int(row.target_atom_id)
        if src not in atom_set or dst not in atom_set:
            continue
        if src == dst:
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    return adjacency


def _connected_components(atom_ids: list[int], adjacency: dict[int, set[int]]) -> list[tuple[int, ...]]:
    """Compute connected components over atom ids using breadth-first traversal."""
    visited: set[int] = set()
    components: list[tuple[int, ...]] = []

    for start in sorted(atom_ids):
        if start in visited:
            continue
        queue = deque([start])
        component: list[int] = []
        while queue:
            atom_id = queue.popleft()
            if atom_id in visited:
                continue
            visited.add(atom_id)
            component.append(atom_id)
            for neighbor in adjacency.get(atom_id, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return components


def _formula(atom_ids: tuple[int, ...], element_by_atom_id: dict[int, str]) -> str:
    """Build a canonical alphabetical formula string for a molecule component."""
    counts: dict[str, int] = {}
    for atom_id in atom_ids:
        element = element_by_atom_id[atom_id]
        counts[element] = counts.get(element, 0) + 1
    return "".join(f"{element}{counts[element]}" for element in sorted(counts))


def _component_adjacency(component: tuple[int, ...], adjacency: dict[int, set[int]]) -> dict[int, set[int]]:
    """Restrict full adjacency to nodes belonging to one connected component."""
    component_set = set(component)
    return {
        atom_id: {neighbor for neighbor in adjacency.get(atom_id, set()) if neighbor in component_set}
        for atom_id in component
    }


def _element_counts(atom_ids: tuple[int, ...], element_by_atom_id: dict[int, str]) -> dict[str, int]:
    """Count per-element atom occurrences for one molecule component."""
    counts: dict[str, int] = {}
    for atom_id in atom_ids:
        element = element_by_atom_id[atom_id]
        counts[element] = counts.get(element, 0) + 1
    return counts


def _normalize_element_symbol(value: str) -> str:
    """Normalize an element token to standard symbol capitalization."""
    token = str(value).strip()
    if not token:
        return ""
    if len(token) == 1:
        return token.upper()
    return token[0].upper() + token[1:].lower()


_MOTIF_ALIASES: dict[str, str] = {
    "sfx": "only:F,S;S==1;F>=1",
}


def _parse_motif(motif: str) -> dict[str, Any]:
    """Parse a motif filter expression into normalized matching constraints."""
    raw = str(motif).strip()
    if not raw:
        raise ValueError("Motif cannot be empty.")
    expanded = _MOTIF_ALIASES.get(raw.lower(), raw)
    parts = [part.strip() for part in expanded.split(";") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid motif specification: {motif!r}")

    only_elements: set[str] | None = None
    constraints: list[tuple[str, str, int]] = []
    for part in parts:
        if part.lower().startswith("only:"):
            values = [v.strip() for v in part.split(":", 1)[1].split(",") if v.strip()]
            normalized = {_normalize_element_symbol(v) for v in values if _normalize_element_symbol(v)}
            if not normalized:
                raise ValueError(f"Invalid 'only' motif clause: {part!r}")
            only_elements = normalized
            continue

        match = re.match(r"^([A-Za-z]+)\s*(==|>=|<=|>|<)\s*(\d+)$", part)
        if not match:
            raise ValueError(
                "Invalid motif clause. Expected 'only:E1,E2' or 'El==n' / 'El>=n' / 'El<=n' / 'El>n' / 'El<n'. "
                f"Got: {part!r}"
            )
        element = _normalize_element_symbol(match.group(1))
        op = match.group(2)
        value = int(match.group(3))
        constraints.append((element, op, value))

    return {
        "raw": raw,
        "only": only_elements,
        "constraints": constraints,
    }


def _motif_matches(counts: dict[str, int], parsed_motif: dict[str, Any]) -> bool:
    """Check whether per-element counts satisfy a parsed motif definition."""
    only = parsed_motif["only"]
    if only is not None and set(counts.keys()) != set(only):
        return False

    for element, op, value in parsed_motif["constraints"]:
        count = int(counts.get(element, 0))
        if op == "==" and not (count == value):
            return False
        if op == ">=" and not (count >= value):
            return False
        if op == "<=" and not (count <= value):
            return False
        if op == ">" and not (count > value):
            return False
        if op == "<" and not (count < value):
            return False
    return True


def _isomorphic_component(
    molecule_a: dict[str, Any],
    molecule_b: dict[str, Any],
) -> bool:
    """Determine graph isomorphism between two same-formula molecule components."""
    nodes_a = tuple(molecule_a["atom_ids"])
    nodes_b = tuple(molecule_b["atom_ids"])
    if len(nodes_a) != len(nodes_b):
        return False

    elem_a: dict[int, str] = molecule_a["element_by_atom"]
    elem_b: dict[int, str] = molecule_b["element_by_atom"]
    if sorted(elem_a.values()) != sorted(elem_b.values()):
        return False

    graph_a: dict[int, set[int]] = molecule_a["adjacency"]
    graph_b: dict[int, set[int]] = molecule_b["adjacency"]

    degree_a = {node: len(graph_a[node]) for node in nodes_a}
    degree_b = {node: len(graph_b[node]) for node in nodes_b}

    buckets_a: dict[tuple[str, int], list[int]] = defaultdict(list)
    buckets_b: dict[tuple[str, int], list[int]] = defaultdict(list)
    for node in nodes_a:
        buckets_a[(elem_a[node], degree_a[node])].append(node)
    for node in nodes_b:
        buckets_b[(elem_b[node], degree_b[node])].append(node)

    if set(buckets_a) != set(buckets_b):
        return False
    for key in buckets_a:
        if len(buckets_a[key]) != len(buckets_b[key]):
            return False

    candidates: dict[int, set[int]] = {}
    for node in nodes_a:
        candidates[node] = set(buckets_b[(elem_a[node], degree_a[node])])

    mapping: dict[int, int] = {}
    used_nodes_b: set[int] = set()
    order = sorted(
        nodes_a,
        key=lambda node: (len(candidates[node]), -degree_a[node], elem_a[node], node),
    )

    def _compatible(node_a: int, node_b: int) -> bool:
        for mapped_node_a, mapped_node_b in mapping.items():
            edge_a = mapped_node_a in graph_a[node_a]
            edge_b = mapped_node_b in graph_b[node_b]
            if edge_a != edge_b:
                return False
        return True

    def _backtrack(depth: int) -> bool:
        if depth == len(order):
            return True
        node_a = order[depth]
        for node_b in candidates[node_a]:
            if node_b in used_nodes_b:
                continue
            if not _compatible(node_a, node_b):
                continue
            mapping[node_a] = node_b
            used_nodes_b.add(node_b)
            if _backtrack(depth + 1):
                return True
            used_nodes_b.remove(node_b)
            del mapping[node_a]
        return False

    return _backtrack(0)


def _build_tables(molecules: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build formula-, isomer-, and molecule-level summary tables."""
    if not molecules:
        return _empty_formula_table(), _empty_isomer_table(), _empty_molecule_table()

    by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for molecule in molecules:
        by_formula[molecule["formula"]].append(molecule)

    formula_rows: list[dict[str, Any]] = []
    isomer_rows: list[dict[str, Any]] = []
    molecule_rows: list[dict[str, Any]] = []

    for formula in sorted(by_formula):
        formula_molecules = by_formula[formula]
        isomers: list[dict[str, Any]] = []
        for molecule in formula_molecules:
            matched = False
            for isomer in isomers:
                if _isomorphic_component(molecule, isomer["representative"]):
                    isomer["members"].append(molecule)
                    matched = True
                    break
            if not matched:
                isomers.append({"representative": molecule, "members": [molecule]})

        formula_rows.append(
            {
                "formula": formula,
                "molecule_count": int(len(formula_molecules)),
                "isomer_count": int(len(isomers)),
            }
        )

        for isomer_id, isomer in enumerate(isomers, start=1):
            members = isomer["members"]
            representative = isomer["representative"]
            isomer_rows.append(
                {
                    "formula": formula,
                    "isomer_id": int(isomer_id),
                    "molecule_count": int(len(members)),
                    "atom_count": int(representative["atom_count"]),
                    "representative_molecule_id": int(representative["molecule_id"]),
                }
            )
            for molecule in members:
                molecule_rows.append(
                    {
                        "molecule_id": int(molecule["molecule_id"]),
                        "formula": formula,
                        "isomer_id": int(isomer_id),
                        "atom_count": int(molecule["atom_count"]),
                        "atom_ids": ",".join(str(atom_id) for atom_id in molecule["atom_ids"]),
                    }
                )

    formula_table = pd.DataFrame(formula_rows).sort_values(["formula"], kind="stable").reset_index(drop=True)
    isomer_table = (
        pd.DataFrame(isomer_rows)
        .sort_values(["formula", "isomer_id"], kind="stable")
        .reset_index(drop=True)
    )
    molecule_table = (
        pd.DataFrame(molecule_rows)
        .sort_values(["molecule_id"], kind="stable")
        .reset_index(drop=True)
    )
    return formula_table, isomer_table, molecule_table


@dataclass
class MoleculeIsomerDetectionRequest(BaseRequest):
    """Request payload for single-structure molecule/isomer detection.

    This request controls component-size thresholds and optional element/motif
    inclusion or exclusion filters applied before formula/isomer grouping.

    Fields
    -----
    min_atoms_per_molecule : int
        Minimum connected-component atom count to retain as a molecule.
    include_any_elements : tuple[str, ...] | None
        Keep molecules containing at least one listed element.
    include_all_elements : tuple[str, ...] | None
        Keep molecules containing all listed elements.
    exclude_any_elements : tuple[str, ...] | None
        Exclude molecules containing any listed element.
    include_motifs : tuple[str, ...] | None
        Keep molecules that match at least one motif expression.
    exclude_motifs : tuple[str, ...] | None
        Exclude molecules that match any motif expression.

    Examples
    -----
    ```python
    request = MoleculeIsomerDetectionRequest(
        min_atoms_per_molecule=2,
        include_motifs=("SFx",),
        exclude_any_elements=("H",),
    )
    ```
    The request keeps sulfur/fluorine motif matches while excluding hydrogen-containing molecules.
    """
    min_atoms_per_molecule: int = dc_field(
        default=1,
        metadata={
            "label": "Min Atoms",
            "help": "Minimum component size retained as a molecule.",
            "min": 1,
        },
    )
    include_any_elements: tuple[str, ...] | None = dc_field(
        default=None,
        metadata={
            "label": "Include Any Elements",
            "help": "Keep molecules containing at least one of these elements.",
        },
    )
    include_all_elements: tuple[str, ...] | None = dc_field(
        default=None,
        metadata={
            "label": "Include All Elements",
            "help": "Keep molecules containing all listed elements.",
        },
    )
    exclude_any_elements: tuple[str, ...] | None = dc_field(
        default=None,
        metadata={
            "label": "Exclude Any Elements",
            "help": "Exclude molecules containing any listed element.",
        },
    )
    include_motifs: tuple[str, ...] | None = dc_field(
        default=None,
        metadata={
            "label": "Include Motifs",
            "help": "Keep molecules matching at least one motif rule (e.g. 'SFx' or 'only:F,S;S==1;F>=1').",
        },
    )
    exclude_motifs: tuple[str, ...] | None = dc_field(
        default=None,
        metadata={
            "label": "Exclude Motifs",
            "help": "Exclude molecules matching any motif rule (e.g. 'SFx' or 'only:F,S;S==1;F>=1').",
        },
    )


@dataclass
class MoleculeIsomerDetectionResult(BaseResult):
    """Result payload for molecule and isomer detection outputs.

    The analyzer returns three related summary tables over one geometry:
    formula-level counts, isomer-level groupings, and per-molecule assignments.

    Fields
    -----
    table : pd.DataFrame
        Default table alias equal to ``isomer_table`` for compatibility with
        table-oriented analyzers and exporters.
    formula_table : pd.DataFrame
        Formula-level counts with columns ``formula``, ``molecule_count``,
        and ``isomer_count``.
    isomer_table : pd.DataFrame
        Isomer-level summary with columns ``formula``, ``isomer_id``,
        ``molecule_count``, ``atom_count``, and
        ``representative_molecule_id``.
    molecule_table : pd.DataFrame
        Per-molecule assignments with columns ``molecule_id``, ``formula``,
        ``isomer_id``, ``atom_count``, and ``atom_ids``.
    request : MoleculeIsomerDetectionRequest
        Request object used to generate this result.

    Examples
    -----
    ```python
    isomer_row = {
        "formula": "C2H6O1",
        "isomer_id": 1,
        "molecule_count": 2,
        "atom_count": 9,
        "representative_molecule_id": 3,
    }
    ```
    The sample row summarizes one detected isomer class for a formula.
    """

    table: pd.DataFrame
    formula_table: pd.DataFrame
    isomer_table: pd.DataFrame
    molecule_table: pd.DataFrame
    request: MoleculeIsomerDetectionRequest


@register_task("molecule_isomer_detection", label="Molecule Isomer Detection")
class MoleculeIsomerDetectionTask(AnalysisTask):
    """Connected-components molecule detection + formula + graph isomer grouping."""

    required_data = GeometryData

    @staticmethod
    def recommended_presentations(
        _result: MoleculeIsomerDetectionResult,
        _payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Recommend the default isomer summary table presentation.

        The isomer table is the canonical human-readable summary and is exposed
        as the default view for this analyzer.

        Works on
        Analyzer task output for ``molecule_isomer_detection``.

        Parameters
        -----
        _result : MoleculeIsomerDetectionResult
            Typed analyzer result instance (unused by current logic).
        _payload : dict[str, Any]
            Serialized payload (unused for this fixed recommendation).

        Returns
        -----
        list[PresentationSpec]
            Single table presentation specification.

        Examples
        -----
        ```python
        specs = MoleculeIsomerDetectionTask.recommended_presentations(_result, {})
        ```
        The returned list contains one table presentation spec.
        """
        return [
            PresentationSpec(renderer="table", label="Isomer Table", view_type="table"),
        ]

    def run(
        self,
        data: GeometryData,
        request: MoleculeIsomerDetectionRequest,
        reporter=None,
    ) -> MoleculeIsomerDetectionResult:
        """Run molecule detection, formula grouping, and isomer partitioning.

        Builds molecular connected components from geometry connectivity, applies
        request filters, groups molecules by formula, and splits each formula
        into graph-isomorphic isomer classes.

        Works on
        ``GeometryData`` representing one structure with coordinates and connectivity.

        Parameters
        -----
        data : GeometryData
            Parsed geometry model containing atom and connectivity tables.
        request : MoleculeIsomerDetectionRequest
            Component threshold and element/motif filter configuration.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks.

        Returns
        -----
        MoleculeIsomerDetectionResult
            Result containing formula, isomer, and molecule summary tables.

        Examples
        -----
        ```python
        result = MoleculeIsomerDetectionTask().run(
            data,
            MoleculeIsomerDetectionRequest(min_atoms_per_molecule=2),
        )
        ```
        ``result.isomer_table`` contains grouped isomer classes for retained molecules.
        """
        if reporter:
            reporter("analyze", 0, 4, "Preparing molecule/isomer detection")

        min_atoms = max(1, int(request.min_atoms_per_molecule))
        include_any_elements = (
            {_normalize_element_symbol(v) for v in request.include_any_elements if _normalize_element_symbol(v)}
            if request.include_any_elements
            else set()
        )
        include_all_elements = (
            {_normalize_element_symbol(v) for v in request.include_all_elements if _normalize_element_symbol(v)}
            if request.include_all_elements
            else set()
        )
        exclude_any_elements = (
            {_normalize_element_symbol(v) for v in request.exclude_any_elements if _normalize_element_symbol(v)}
            if request.exclude_any_elements
            else set()
        )
        include_motifs = [_parse_motif(v) for v in request.include_motifs] if request.include_motifs else []
        exclude_motifs = [_parse_motif(v) for v in request.exclude_motifs] if request.exclude_motifs else []

        coordinates = _coordinates_table(data)
        if coordinates.empty:
            if reporter:
                reporter("analyze", 4, 4, "Finished molecule/isomer detection")
            empty_formula = _empty_formula_table()
            empty_isomer = _empty_isomer_table()
            empty_molecule = _empty_molecule_table()
            return MoleculeIsomerDetectionResult(
                table=empty_isomer,
                formula_table=empty_formula,
                isomer_table=empty_isomer,
                molecule_table=empty_molecule,
                request=request,
            )

        atom_ids = coordinates["atom_id"].astype(int).tolist()
        element_by_atom_id = {
            int(row.atom_id): str(row.atom_type) for row in coordinates.itertuples(index=False)
        }
        connectivity = _connectivity_table(data)
        adjacency = _adjacency_from_edges(atom_ids, connectivity)
        if reporter:
            reporter("analyze", 1, 4, "Detecting connected components")

        components = _connected_components(atom_ids, adjacency)
        molecules: list[dict[str, Any]] = []
        for component in components:
            if len(component) < min_atoms:
                continue
            counts = _element_counts(component, element_by_atom_id)
            elements_present = set(counts.keys())

            if include_any_elements and elements_present.isdisjoint(include_any_elements):
                continue
            if include_all_elements and not include_all_elements.issubset(elements_present):
                continue
            if exclude_any_elements and elements_present.intersection(exclude_any_elements):
                continue
            if include_motifs and not any(_motif_matches(counts, motif) for motif in include_motifs):
                continue
            if exclude_motifs and any(_motif_matches(counts, motif) for motif in exclude_motifs):
                continue
            molecule_id = len(molecules) + 1
            component_elements = {atom_id: element_by_atom_id[atom_id] for atom_id in component}
            molecules.append(
                {
                    "molecule_id": int(molecule_id),
                    "atom_ids": component,
                    "atom_count": int(len(component)),
                    "formula": _formula(component, element_by_atom_id),
                    "element_by_atom": component_elements,
                    "adjacency": _component_adjacency(component, adjacency),
                }
            )
        if reporter:
            reporter("analyze", 2, 4, "Grouping formulas")

        formula_table, isomer_table, molecule_table = _build_tables(molecules)
        if reporter:
            reporter("analyze", 4, 4, "Finished molecule/isomer detection")

        return MoleculeIsomerDetectionResult(
            table=isomer_table,
            formula_table=formula_table,
            isomer_table=isomer_table,
            molecule_table=molecule_table,
            request=request,
        )


__all__ = [
    "MoleculeIsomerDetectionRequest",
    "MoleculeIsomerDetectionResult",
    "MoleculeIsomerDetectionTask",
]
