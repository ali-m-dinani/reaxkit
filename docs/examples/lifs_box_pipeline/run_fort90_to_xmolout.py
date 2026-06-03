"""Example runner: fort.90 -> Li/S box preparation -> output.xyz write."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from reaxkit.domain.data_models import TrajectoryData
from reaxkit.engine.common.generators.xyz_generator import write_xyz_trajectory
from reaxkit.engine.reaxff.io.geo_handler import GeoHandler

from prepare_lis_box import (
    LiSBoxPreparationConfig,
    prepare_lis_box_from_total_s,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Li/S packed box from fort.90 and write XYZ outputs."
    )
    parser.add_argument("--fort90", required=True, help="Path to input fort.90")
    parser.add_argument("--output", default="output.xyz", help="Final packed XYZ file")
    parser.add_argument(
        "--isomer-xyz-dir",
        default=None,
        help="Directory for representative isomer XYZ files (default: output parent).",
    )

    parser.add_argument("--density", type=float, default=0.007645, help="Target density in g/Angstrom^3")
    parser.add_argument("--xS", type=float, default=0.12, help="Sulfur monoatomic fraction")
    parser.add_argument("--tolerance", type=float, default=4.0, help="Minimum atom-atom distance in Angstrom")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for placement")
    parser.add_argument("--max-attempts", type=int, default=1000, help="Max placement attempts per fragment")

    parser.add_argument("--li-per-s", type=float, default=2.0, help="Li:S stoichiometric multiplier")
    parser.add_argument("--mass-s", type=float, default=32.0, help="Sulfur atomic mass used in density sizing")
    parser.add_argument("--mass-li", type=float, default=7.0, help="Lithium atomic mass used in density sizing")
    parser.add_argument("--s2-bond-length", type=float, default=1.9, help="S2 bond length in Angstrom")
    return parser


def _read_geo_graph(path: str | Path) -> tuple[dict[int, str], dict[int, np.ndarray], dict[int, set[int]]]:
    """Read XTLGRF atom/connectivity data through ReaxKit's GeoHandler."""
    handler = GeoHandler(path)
    atoms_df = handler.coordinates()
    connectivity_df = handler.connectivity()

    atom_elements = {
        int(row.atom_id): str(row.atom_type)
        for row in atoms_df.itertuples(index=False)
    }
    atom_coords = {
        int(row.atom_id): np.asarray([float(row.x), float(row.y), float(row.z)], dtype=float)
        for row in atoms_df.itertuples(index=False)
    }
    adjacency: dict[int, set[int]] = {}
    for atom_id in atom_elements:
        adjacency.setdefault(atom_id, set())
    for row in connectivity_df.itertuples(index=False):
        source = int(row.source_atom_id)
        target = int(row.target_atom_id)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return atom_elements, atom_coords, adjacency


def _connected_components(atom_ids: list[int], adjacency: dict[int, set[int]]) -> list[list[int]]:
    visited: set[int] = set()
    components: list[list[int]] = []
    for atom_id in sorted(atom_ids):
        if atom_id in visited:
            continue
        queue = deque([atom_id])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _element_counts(component: list[int], atom_elements: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom_id in component:
        symbol = atom_elements[atom_id]
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _formula_from_counts(counts: dict[str, int]) -> str:
    return "".join(f"{element}{counts[element]}" for element in sorted(counts))


def _is_bare_sfx(counts: dict[str, int]) -> bool:
    return set(counts.keys()) == {"S", "F"} and int(counts.get("S", 0)) == 1


def _induced_subgraph(component: list[int], adjacency: dict[int, set[int]]) -> dict[int, set[int]]:
    comp_set = set(component)
    return {atom_id: {n for n in adjacency.get(atom_id, set()) if n in comp_set} for atom_id in component}


def _are_isomorphic(
    atom_ids_a: tuple[int, ...],
    atom_ids_b: tuple[int, ...],
    elem_counts_a: dict[str, int],
    elem_counts_b: dict[str, int],
    graph_a: dict[int, set[int]],
    graph_b: dict[int, set[int]],
    atom_elements: dict[int, str],
) -> bool:
    if len(atom_ids_a) != len(atom_ids_b):
        return False
    if elem_counts_a != elem_counts_b:
        return False

    degree_a = {n: len(graph_a[n]) for n in atom_ids_a}
    degree_b = {n: len(graph_b[n]) for n in atom_ids_b}

    buckets_a: dict[tuple[str, int], list[int]] = defaultdict(list)
    buckets_b: dict[tuple[str, int], list[int]] = defaultdict(list)
    for n in atom_ids_a:
        buckets_a[(atom_elements[n], degree_a[n])].append(n)
    for n in atom_ids_b:
        buckets_b[(atom_elements[n], degree_b[n])].append(n)
    if set(buckets_a.keys()) != set(buckets_b.keys()):
        return False
    for key in buckets_a:
        if len(buckets_a[key]) != len(buckets_b[key]):
            return False

    candidates = {n: set(buckets_b[(atom_elements[n], degree_a[n])]) for n in atom_ids_a}
    mapping: dict[int, int] = {}
    used_b: set[int] = set()
    order = sorted(atom_ids_a, key=lambda n: (len(candidates[n]), -degree_a[n], atom_elements[n], n))

    def compatible(a: int, b: int) -> bool:
        for ma, mb in mapping.items():
            if ((ma in graph_a[a]) != (mb in graph_b[b])):
                return False
        return True

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        a = order[i]
        for b in candidates[a]:
            if b in used_b:
                continue
            if not compatible(a, b):
                continue
            mapping[a] = b
            used_b.add(b)
            if backtrack(i + 1):
                return True
            used_b.remove(b)
            del mapping[a]
        return False

    return backtrack(0)


def _extract_lifs_isomer_rows(
    fort90_path: str | Path,
    isomer_dir: str | Path,
) -> tuple[int, list[dict[str, object]]]:
    atom_elements, atom_coords, adjacency = _read_geo_graph(fort90_path)
    components = _connected_components(list(atom_elements.keys()), adjacency)

    molecules: list[dict[str, object]] = []
    for component in components:
        counts = _element_counts(component, atom_elements)
        if counts.get("S", 0) < 1 or _is_bare_sfx(counts):
            continue
        molecules.append(
            {
                "atom_ids": tuple(component),
                "element_counts": counts,
                "formula": _formula_from_counts(counts),
                "graph": _induced_subgraph(component, adjacency),
            }
        )

    by_formula: dict[str, list[dict[str, object]]] = defaultdict(list)
    for molecule in molecules:
        by_formula[molecule["formula"]].append(molecule)

    out = Path(isomer_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_s = 0
    rows: list[dict[str, object]] = []

    for formula in sorted(by_formula):
        buckets: list[dict[str, object]] = []
        for molecule in by_formula[formula]:
            matched = False
            for bucket in buckets:
                rep = bucket["representative"]
                if _are_isomorphic(
                    molecule["atom_ids"],
                    rep["atom_ids"],
                    molecule["element_counts"],
                    rep["element_counts"],
                    molecule["graph"],
                    rep["graph"],
                    atom_elements,
                ):
                    bucket["count"] = int(bucket["count"]) + 1
                    matched = True
                    break
            if not matched:
                buckets.append({"representative": molecule, "count": 1})

        for isomer_id, bucket in enumerate(buckets, start=1):
            rep = bucket["representative"]
            atom_ids = list(rep["atom_ids"])
            origin = atom_coords[atom_ids[0]]
            shifted = np.asarray([atom_coords[aid] - origin for aid in atom_ids], dtype=float)
            filename = f"{formula}_{isomer_id}.xyz"
            path = out / filename
            write_xyz_trajectory(
                TrajectoryData(
                    positions=shifted[np.newaxis, :, :],
                    elements=[atom_elements[aid] for aid in atom_ids],
                    atom_ids=atom_ids,
                    iterations=np.asarray([0], dtype=int),
                ),
                path,
            )

            count = int(bucket["count"])
            total_s += count * int(rep["element_counts"].get("S", 0))
            rows.append(
                {
                    "formula": formula,
                    "isomer_id": isomer_id,
                    "count": count,
                    "filename": str(path),
                }
            )
    return total_s, rows


def main() -> int:
    args = _build_parser().parse_args()

    cfg = LiSBoxPreparationConfig(
        desired_density_g_per_a3=float(args.density),
        sulfur_mono_fraction=float(args.xS),
        lithium_per_sulfur=float(args.li_per_s),
        sulfur_mass_amu=float(args.mass_s),
        lithium_mass_amu=float(args.mass_li),
        s2_bond_length_a=float(args.s2_bond_length),
        tolerance_a=float(args.tolerance),
        max_attempts_per_fragment=int(args.max_attempts),
        seed=int(args.seed),
    )

    isomer_dir = Path(args.isomer_xyz_dir) if args.isomer_xyz_dir else Path(args.output).parent
    total_s, isomer_rows = _extract_lifs_isomer_rows(args.fort90, isomer_dir)
    result = prepare_lis_box_from_total_s(total_s, cfg=cfg)
    atom_types = [str(atom_type) for atom_type in result["atom_types"]]
    coords = np.asarray(result["coords"], dtype=float)
    output_path = write_xyz_trajectory(
        TrajectoryData(
            positions=coords[np.newaxis, :, :],
            elements=atom_types,
            atom_ids=list(range(1, len(atom_types) + 1)),
            iterations=np.asarray([0], dtype=int),
        ),
        args.output,
    )

    for row in isomer_rows:
        print(row["formula"])
        print(row["count"])

    species = result["species_counts"]
    print(f"Total S (filtered from fort.90): {result['total_s']}")
    print(f"Li required: {result['li_required']}")
    print(f"Species counts: S2={species['S2']} S={species['S']} Li={species['Li']}")
    print(f"Box size (A): {result['box_size_a']:.6f}")
    print(f"Total atoms placed: {len(result['atom_types'])}")
    print(f"Wrote packed box XYZ: {output_path}")
    print(f"Wrote isomer reference XYZ files to: {isomer_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
