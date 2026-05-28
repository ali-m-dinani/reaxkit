"""Provide ring and defect classification helpers for active-site analyzers.

This module includes primitive-ring detection, ring-membership summaries, and
defect motif classification utilities used by active-site structural analysis.
It is scoped to graph/ring-defect logic and does not perform trajectory parsing.

**Usage context**

- Ring analysis: Enumerate primitive rings from bond graphs.
- Defect typing: Classify ring-cluster motifs into TRACT-aligned defect labels.
- Per-atom mapping: Project ring-cluster labels onto atom-level defect types.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import warnings

try:
    import igraph as ig  # type: ignore

    _HAS_IGRAPH = True
except Exception:
    ig = None  # type: ignore
    _HAS_IGRAPH = False

DEFECT_TYPE_PRIORITY: tuple[str, ...] = (
    "SW_5775",
    "SV_5_9",
    "DV_5_8_5",
    "DV_555_777",
    "GB_chain_5_7",
    "EDGE_reczag_57",
    "haeckelite_like",
    "non6_cluster",
)

DEFECT_TYPE_SCHEMA: tuple[str, ...] = DEFECT_TYPE_PRIORITY + ("none",)


def normalize_defect_type(label: str) -> str:
    """Normalize arbitrary defect labels to the supported TRACT schema.

    Parameters
    -----
    label : str
        Candidate defect label.

    Returns
    -----
    str
        Canonical defect label from `DEFECT_TYPE_SCHEMA`, defaulting to
        `"non6_cluster"` when unknown.

    Examples
    -----
    ```python
    normalize_defect_type("DV_5_8_5")
    ```
    Sample output:
    `"DV_5_8_5"`
    Meaning:
    Known labels pass through unchanged; unknown labels are canonicalized.
    """
    if label in DEFECT_TYPE_SCHEMA:
        return str(label)
    return "non6_cluster"


def prefer_defect_type(current: str, incoming: str) -> str:
    """Resolve competing defect labels using TRACT priority ordering.

    Parameters
    -----
    current : str
        Existing defect label.
    incoming : str
        New candidate defect label.

    Returns
    -----
    str
        Higher-priority canonical label according to `DEFECT_TYPE_PRIORITY`.

    Examples
    -----
    ```python
    prefer_defect_type("non6_cluster", "SW_5775")
    ```
    Sample output:
    `"SW_5775"`
    Meaning:
    Higher-priority motif labels replace weaker/unknown assignments.
    """
    cur = normalize_defect_type(current)
    new = normalize_defect_type(incoming)
    if cur == "none":
        return new
    if new == "none":
        return cur
    pidx = {name: i for i, name in enumerate(DEFECT_TYPE_PRIORITY)}
    return new if pidx.get(new, len(DEFECT_TYPE_PRIORITY)) < pidx.get(cur, len(DEFECT_TYPE_PRIORITY)) else cur


def _canonical_cycle(nodes: list[int]) -> tuple[int, ...]:
    if len(nodes) > 1 and nodes[0] == nodes[-1]:
        nodes = nodes[:-1]
    n = len(nodes)
    if n == 0:
        return tuple()
    k = min(range(n), key=lambda i: nodes[i])
    seq1 = tuple(nodes[(k + i) % n] for i in range(n))
    seq2 = tuple(reversed(seq1))
    return min(seq1, seq2)


def find_primitive_rings(graph: nx.Graph) -> list[tuple[int, ...]]:
    """Find primitive rings using shortest-path Franzblau-style logic.

    Parameters
    -----
    graph : nx.Graph
        Undirected bond graph.

    Returns
    -----
    list[tuple[int, ...]]
        Canonical primitive ring node cycles.

    Examples
    -----
    ```python
    rings = find_primitive_rings(graph)
    ```
    Sample output:
    `[(0, 1, 2, 3, 4, 5), ...]`
    Meaning:
    Each tuple encodes one primitive ring with canonicalized node ordering.
    """
    if _HAS_IGRAPH:
        return _sp_primitive_rings_igraph(graph)
    return _sp_primitive_rings_networkx(graph)


def _sp_primitive_rings_igraph(graph: nx.Graph) -> list[tuple[int, ...]]:
    """Franzblau shortest-path primitive rings using igraph."""
    nodes = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_nx = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]

    g = ig.Graph(n=len(nodes), edges=edges_nx)
    g.es["w"] = [1.0] * g.ecount()

    seen: set[tuple[int, ...]] = set()
    rings: list[tuple[int, ...]] = []
    for eid in range(g.ecount()):
        u_local = int(g.es[eid].source)
        v_local = int(g.es[eid].target)
        g.es[eid]["w"] = 1.0e18
        try:
            paths = g.get_shortest_paths(u_local, to=v_local, weights="w", output="vpath")
        except Exception as exc:
            warnings.warn(f"igraph shortest-path ring step failed at edge {eid}: {exc}", stacklevel=2)
            paths = []
        g.es[eid]["w"] = 1.0

        if not paths or not paths[0]:
            continue
        path_local = [int(v) for v in paths[0]]
        if len(path_local) < 3:
            continue
        path_orig = [nodes[idx] for idx in path_local]
        cyc = _canonical_cycle(path_orig)
        if not cyc:
            continue
        if cyc not in seen:
            seen.add(cyc)
            rings.append(cyc)
    return rings


def _sp_primitive_rings_networkx(graph: nx.Graph) -> list[tuple[int, ...]]:
    """Franzblau shortest-path primitive rings using networkx fallback."""
    seen: set[tuple[int, ...]] = set()
    rings: list[tuple[int, ...]] = []
    for u, v in graph.edges():
        g2 = graph.copy()
        g2.remove_edge(u, v)
        try:
            path = nx.shortest_path(g2, u, v)
        except nx.NetworkXNoPath:
            continue
        if len(path) < 3:
            continue
        cyc = _canonical_cycle(path)
        if not cyc:
            continue
        if cyc not in seen:
            seen.add(cyc)
            rings.append(cyc)
    return rings


def ring_membership(n_atoms: int, rings: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-atom ring-size extrema and non-hex membership flags.

    Parameters
    -----
    n_atoms : int
        Number of atoms in analyzed graph.
    rings : list[tuple[int, ...]]
        Primitive ring cycles.

    Returns
    -----
    tuple[np.ndarray, np.ndarray, np.ndarray]
        `(min_size, max_size, in_non6)` arrays per atom.

    Examples
    -----
    ```python
    min_size, max_size, in_non6 = ring_membership(n_atoms, rings)
    ```
    Sample output:
    Arrays of length `n_atoms`.
    Meaning:
    Atoms receive ring-size summary bounds and non-hex membership status.
    """
    min_size = np.full(n_atoms, -1, dtype=int)
    max_size = np.full(n_atoms, -1, dtype=int)
    in_non6 = np.zeros(n_atoms, dtype=bool)
    for cyc in rings:
        size = len(cyc)
        for a in cyc:
            if min_size[a] < 0 or size < min_size[a]:
                min_size[a] = size
            if max_size[a] < 0 or size > max_size[a]:
                max_size[a] = size
            if size != 6:
                in_non6[a] = True
    return min_size, max_size, in_non6


def _ring_edges(cyc: tuple[int, ...]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for a, b in zip(cyc, cyc[1:] + cyc[:1]):
        if a > b:
            a, b = b, a
        out.add((a, b))
    return out


def _build_ring_graph(
    rings: list[tuple[int, ...]],
    boundary_nodes: Optional[set[int]] = None,
) -> tuple[list[int], nx.Graph, list[float]]:
    sizes = [len(r) for r in rings]
    r_edges = [_ring_edges(r) for r in rings]
    r_atoms = [set(r) for r in rings]
    adj = nx.Graph()
    adj.add_nodes_from(range(len(rings)))
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if r_edges[i] & r_edges[j]:
                adj.add_edge(i, j)
    if boundary_nodes is None:
        boundary_nodes = set()
    bfrac = []
    for atoms in r_atoms:
        if not atoms:
            bfrac.append(0.0)
        else:
            bfrac.append(float(len(atoms & boundary_nodes)) / float(len(atoms)))
    return sizes, adj, bfrac


def _is_adjacent(graph: nx.Graph, i: int, j: int) -> bool:
    return graph.has_edge(i, j)


def _find_cycle_with_sizes(graph: nx.Graph, wanted_sizes: list[int], ring_sizes: list[int]) -> bool:
    target = sorted(wanted_sizes)
    for cyc in nx.cycle_basis(graph):
        if len(cyc) != len(wanted_sizes):
            continue
        got = sorted([ring_sizes[k] for k in cyc])
        if got == target:
            return True
    return False


def classify_defect_clusters(
    ring_sizes: list[int],
    ring_adj: nx.Graph,
    boundary_frac: list[float],
    min_gb_len: int = 4,
) -> dict[int, str]:
    """Classify connected non-hex ring clusters into canonical defect motifs.

    Parameters
    -----
    ring_sizes : list[int]
        Ring size per ring index.
    ring_adj : nx.Graph
        Ring adjacency graph (rings are nodes; shared edges define adjacency).
    boundary_frac : list[float]
        Boundary participation fraction per ring.
    min_gb_len : int, optional
        Minimum alternating 5-7 chain length for grain-boundary labeling.

    Returns
    -----
    dict[int, str]
        Mapping from ring index to canonical defect motif label.

    Examples
    -----
    ```python
    labels = classify_defect_clusters(ring_sizes, ring_adj, boundary_frac)
    ```
    Sample output:
    `{0: "SW_5775", 3: "GB_chain_5_7"}`
    Meaning:
    Defect motifs are assigned at ring-cluster resolution.
    """
    non6 = [i for i, s in enumerate(ring_sizes) if s != 6]
    h = ring_adj.subgraph(non6).copy()
    labels: dict[int, str] = {}

    def mark(nodes: list[int], label: str) -> None:
        for node in nodes:
            labels[node] = label

    for comp_set in nx.connected_components(h):
        comp = list(comp_set)
        sizes = [ring_sizes[u] for u in comp]
        mult = {k: sizes.count(k) for k in set(sizes)}

        if len(comp) == 4 and mult.get(5, 0) == 2 and mult.get(7, 0) == 2:
            if _find_cycle_with_sizes(h.subgraph(comp), [5, 7, 7, 5], ring_sizes):
                mark(comp, "SW_5775")
            else:
                mark(comp, "non6_cluster")
            continue

        if len(comp) == 2 and set(sizes) == {5, 9} and _is_adjacent(h, comp[0], comp[1]):
            mark(comp, "SV_5_9")
            continue

        if len(comp) == 3 and mult.get(5, 0) == 2 and mult.get(8, 0) == 1:
            sub = h.subgraph(comp)
            idx8 = [u for u in comp if ring_sizes[u] == 8][0]
            neigh = list(sub.neighbors(idx8))
            if len(neigh) == 2 and all(ring_sizes[v] == 5 for v in neigh):
                mark(comp, "DV_5_8_5")
                continue

        if mult.get(5, 0) == 3 and mult.get(7, 0) == 3 and len(comp) <= 8:
            sub = h.subgraph(comp)
            alt_ok = False
            for cyc in nx.cycle_basis(sub):
                if len(cyc) != 6:
                    continue
                rs = [ring_sizes[k] for k in cyc]
                if all(
                    (rs[i] == 5 and rs[(i + 1) % 6] == 7)
                    or (rs[i] == 7 and rs[(i + 1) % 6] == 5)
                    for i in range(6)
                ):
                    alt_ok = True
                    break
            if alt_ok:
                mark(comp, "DV_555_777")
            else:
                mark(comp, "non6_cluster")
            continue

        if (mult.get(5, 0) + mult.get(7, 0)) >= max(2, len(comp) - 1):
            bf = float(np.mean([boundary_frac[u] for u in comp])) if comp else 0.0
            if bf > 0.5:
                mark(comp, "EDGE_reczag_57")
                continue

        if mult.get(5, 0) > 0 and mult.get(7, 0) > 0 and mult.get(8, 0) == 0 and mult.get(9, 0) == 0:
            sub = h.subgraph(comp).copy()
            seeds = [u for u in comp if sub.degree(u) == 1] or [comp[0]]
            best_len = 0
            for s in seeds:
                for t in comp:
                    if s == t:
                        continue
                    try:
                        p = nx.shortest_path(sub, s, t)
                    except nx.NetworkXNoPath:
                        continue
                    if all(ring_sizes[a] != ring_sizes[b] for a, b in zip(p, p[1:])):
                        best_len = max(best_len, len(p))
            if best_len >= int(min_gb_len):
                mark(comp, "GB_chain_5_7")
                continue

        if (mult.get(5, 0) + mult.get(7, 0)) >= max(3, int(0.6 * len(comp))):
            mark(comp, "haeckelite_like")
            continue

        mark(comp, "non6_cluster")

    return labels


def per_atom_defect_types(
    n_atoms: int,
    rings: list[tuple[int, ...]],
    boundary_nodes: Optional[set[int]] = None,
) -> np.ndarray:
    """Project ring-cluster defect labels to per-atom defect types.

    Parameters
    -----
    n_atoms : int
        Number of atoms in analyzed graph.
    rings : list[tuple[int, ...]]
        Primitive rings used for defect classification.
    boundary_nodes : Optional[set[int]], optional
        Optional boundary-node set used by edge-related motif logic.

    Returns
    -----
    np.ndarray
        Per-atom defect label array following `DEFECT_TYPE_SCHEMA`.

    Examples
    -----
    ```python
    labels = per_atom_defect_types(n_atoms, rings, boundary_nodes=boundary_nodes)
    ```
    Sample output:
    `array(["none", "SW_5775", ...], dtype=object)`
    Meaning:
    Each atom receives one canonical defect label for downstream labeling.
    """
    out = np.full(n_atoms, "none", dtype=object)
    if not rings:
        return out

    sizes, adj, bfrac = _build_ring_graph(rings, boundary_nodes=boundary_nodes)
    labels = classify_defect_clusters(sizes, adj, bfrac)
    if not labels:
        return out

    # TRACT parity: per atom, use the first labeled ring (by ring-index order).
    atom_rings: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for ring_id, cyc in enumerate(rings):
        for atom in cyc:
            ai = int(atom)
            if 0 <= ai < n_atoms:
                atom_rings[ai].append(int(ring_id))

    for atom_i, ri_list in enumerate(atom_rings):
        defect_type = "none"
        for ri in ri_list:
            if ri in labels:
                defect_type = normalize_defect_type(labels[ri])
                break
        out[atom_i] = defect_type
    return out
