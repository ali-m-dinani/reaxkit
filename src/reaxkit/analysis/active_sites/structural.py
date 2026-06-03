"""Run active-site structural analysis with phase-2 edge-typing parity outputs.

This module computes local structural descriptors, neighborhood features, and
site-label artifacts used by the active-sites analysis flow. It is scoped to
structural characterization and table export, not event chronology detection.

**Usage context**

- Site typing: Classify local environments around candidate active atoms.
- Defect-aware analysis: Combine ring/defect cues with connectivity metadata.
- TrACT parity runs: Produce structural outputs aligned with TrACT conventions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import warnings

from reaxkit.analysis.active_sites.classification import site_labels_tract_parity
from reaxkit.analysis.active_sites.defects import find_primitive_rings, per_atom_defect_types, ring_membership
from reaxkit.analysis.active_sites.models import ActiveSiteStructuralRequest, ActiveSiteStructuralResult
from reaxkit.analysis.active_sites.pbc import frame_cell_matrix, minimum_image_vector
from reaxkit.analysis.active_sites.tract_compat import to_tract_structural_table
from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.data_models import ConnectivityTrajectoryData, TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

try:
    from scipy.spatial import Delaunay

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _as_dense_frame(frame_obj: Any) -> np.ndarray:
    if isinstance(frame_obj, np.ndarray):
        return frame_obj.astype(float)
    if hasattr(frame_obj, "toarray"):
        return np.asarray(frame_obj.toarray(), dtype=float)
    if hasattr(frame_obj, "todense"):
        return np.asarray(frame_obj.todense(), dtype=float)
    return np.asarray(frame_obj, dtype=float)


def _frame_from_series(series: Any, frame_index: int) -> Any:
    if isinstance(series, np.ndarray):
        return series[frame_index]
    if isinstance(series, (list, tuple)):
        return series[frame_index]
    return None


def _ragged_bond_order_frame(data: ConnectivityTrajectoryData, frame_index: int, bo_frame: np.ndarray) -> np.ndarray:
    """Expand per-atom neighbor-list bond orders into a dense frame matrix."""
    n_atoms = len(data.trajectory.atom_ids)
    conn_raw = _frame_from_series(data.connectivity.connectivity, frame_index)
    if conn_raw is None:
        raise ValueError("Ragged bond-order frames require ConnectivityData.connectivity.")
    conn_frame = np.asarray(conn_raw, dtype=int)
    bo_values = np.asarray(bo_frame, dtype=float)
    if conn_frame.ndim != 2 or bo_values.ndim != 2 or conn_frame.shape != bo_values.shape:
        raise ValueError("Ragged connectivity and bond_orders frames must have matching 2D shapes.")

    dense = np.zeros((n_atoms, n_atoms), dtype=float)
    n_rows = min(n_atoms, conn_frame.shape[0])
    for ai in range(n_rows):
        for partner_id, order in zip(conn_frame[ai], bo_values[ai]):
            if not np.isfinite(order) or order <= 0.0:
                continue
            pj = int(partner_id) - 1
            if 0 <= pj < n_atoms:
                dense[ai, pj] = max(dense[ai, pj], float(order))
    return dense


def _bond_order_frame(data: ConnectivityTrajectoryData, frame_index: int) -> np.ndarray:
    bo = data.connectivity.bond_orders
    if bo is None:
        raise ValueError("Active-site structural analysis requires ConnectivityData.bond_orders.")
    if isinstance(bo, np.ndarray):
        if bo.ndim != 3:
            raise ValueError("bond_orders ndarray must have shape (n_frames, n_atoms, n_atoms).")
        frame = _as_dense_frame(bo[frame_index])
        n_atoms = len(data.trajectory.atom_ids)
        if frame.shape == (n_atoms, n_atoms):
            return frame
        return _ragged_bond_order_frame(data, frame_index, frame)
    if isinstance(bo, (list, tuple)):
        frame = _as_dense_frame(bo[frame_index])
        n_atoms = len(data.trajectory.atom_ids)
        if frame.shape == (n_atoms, n_atoms):
            return frame
        return _ragged_bond_order_frame(data, frame_index, frame)
    raise TypeError("Unsupported bond_orders type.")


_COV_RAD: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
}


def _pair_cutoff(ei: str, ej: str, *, scale: float, extra: float = 0.0) -> float:
    ri = _COV_RAD.get(str(ei), 0.75)
    rj = _COV_RAD.get(str(ej), 0.75)
    return float(scale) * (ri + rj) + float(extra)


def _build_geometric_bond_graph(
    xyz: np.ndarray,
    elements: list[str],
    cell: np.ndarray | None,
    *,
    bond_scale: float,
    bond_extra: float = 0.0,
) -> tuple[nx.Graph, list[list[int]]]:
    """Build TRACT-style geometric bond graph using ASE neighbor_list."""
    try:
        from ase import Atoms
        from ase.neighborlist import neighbor_list as ase_neighbor_list
    except Exception as exc:
        raise ValueError("Distance bond mode requires ASE to be installed.") from exc

    n_atoms = len(elements)
    is_periodic = bool(cell is not None and np.isfinite(cell).all() and np.linalg.norm(cell) > 0.0)
    atoms = Atoms(
        symbols=elements,
        positions=np.asarray(xyz, dtype=float),
        cell=cell if is_periodic else None,
        pbc=is_periodic,
    )

    unique_elems = sorted(set(elements))
    cutoff_max = max(_pair_cutoff(ei, ej, scale=bond_scale, extra=bond_extra) for ei in unique_elems for ej in unique_elems)
    i_arr, j_arr, d_arr = ase_neighbor_list("ijd", atoms, cutoff_max)

    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    neighbors: list[list[int]] = [[] for _ in range(n_atoms)]
    seen: set[tuple[int, int]] = set()

    for i, j, d in zip(i_arr.tolist(), j_arr.tolist(), d_arr.tolist()):
        a, b = (i, j) if i < j else (j, i)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        cutoff_ij = _pair_cutoff(elements[a], elements[b], scale=bond_scale, extra=bond_extra)
        if float(d) <= cutoff_ij:
            graph.add_edge(a, b, d=float(d), bo=float("nan"))
            neighbors[a].append(b)
            neighbors[b].append(a)

    for i in range(n_atoms):
        neighbors[i].sort()
    return graph, neighbors


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 0.0 or n2 <= 0.0:
        return float("nan")
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _pyramidalization_for_atom(
    atom_index: int,
    bonded_neighbors: list[int],
    xyz: np.ndarray,
    cell: np.ndarray | None,
) -> tuple[float, float]:
    """Return TRACT-parity (d_pyr, d_ang_deg) for one atom."""
    n = len(bonded_neighbors)
    if n < 2:
        return 0.0, float("nan")

    if n == 2:
        vectors = np.asarray(
            [minimum_image_vector(xyz[j] - xyz[atom_index], cell) for j in bonded_neighbors],
            dtype=float,
        )
        theta = _angle_deg(vectors[0], vectors[1])
        d_ang = float(180.0 - theta) if np.isfinite(theta) else float("nan")
        return 0.0, d_ang

    center = xyz[atom_index]
    if n == 3:
        j1, j2, j3 = bonded_neighbors[0], bonded_neighbors[1], bonded_neighbors[2]
        d_i1 = minimum_image_vector(xyz[j1] - center, cell)
        d_i2 = minimum_image_vector(xyz[j2] - center, cell)
        d_i3 = minimum_image_vector(xyz[j3] - center, cell)
        v12 = d_i2 - d_i1
        v13 = d_i3 - d_i1
        normal = np.cross(v12, v13)
        nlen = float(np.linalg.norm(normal))
        if nlen < 1.0e-10:
            return 0.0, float("nan")
        normal /= nlen
        d_pyr = float(np.dot(-d_i1, normal))
        return d_pyr, float("nan")

    vectors = np.asarray([minimum_image_vector(xyz[j] - center, cell) for j in bonded_neighbors], dtype=float)
    centroid = vectors.mean(axis=0)
    centered = vectors - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0, float("nan")
    normal = vh[-1]
    d_pyr = float(np.dot(-centroid, normal))
    return d_pyr, float("nan")


def _compute_has_hetero_bond(
    elements: list[str],
    neighbors: list[list[int]],
    _carbon_element: str,
) -> np.ndarray:
    out = np.zeros(len(elements), dtype=bool)
    for i, elem in enumerate(elements):
        out[i] = any(elements[j] != elem for j in neighbors[i])
    return out


def _compute_strain_roughness(
    graph: nx.Graph,
    neighbors: list[list[int]],
    xyz: np.ndarray,
    cell: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TRACT-parity bond strain, angle strain, and local roughness."""
    n_atoms = len(neighbors)
    all_bonds = [float(graph.edges[e]["d"]) for e in graph.edges()]
    d_median = float(np.median(all_bonds)) if all_bonds else 1.42

    bond_strain = np.zeros(n_atoms, dtype=float)
    angle_strain = np.zeros(n_atoms, dtype=float)
    local_roughness = np.zeros(n_atoms, dtype=float)

    for i in range(n_atoms):
        nn = neighbors[i]
        if nn:
            bond_lengths = [float(graph.edges[(min(i, j), max(i, j))]["d"]) for j in nn if graph.has_edge(min(i, j), max(i, j))]
            if bond_lengths:
                bond_strain[i] = float(np.mean([abs(d - d_median) / d_median for d in bond_lengths]))

        angle_devs: list[float] = []
        for a in range(len(nn)):
            for b in range(a + 1, len(nn)):
                j, k = nn[a], nn[b]
                v1 = xyz[j] - xyz[i]
                v2 = xyz[k] - xyz[i]
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 > 0.0 and n2 > 0.0:
                    cos_th = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    th = float(np.degrees(np.arccos(cos_th)))
                    angle_devs.append(abs(th - 120.0))
        if angle_devs:
            angle_strain[i] = float(np.mean(angle_devs))

        # Plane fit over local neighborhood (i + neighbors) using minimum-image unwrapped points.
        if len(nn) + 1 >= 3:
            pts_arr = np.asarray([xyz[i]] + [xyz[j] for j in nn], dtype=float)
            a_mat = np.c_[pts_arr[:, 0], pts_arr[:, 1], np.ones(len(pts_arr))]
            z_vec = pts_arr[:, 2]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(a_mat, z_vec, rcond=None)
                z_fit = a_mat @ coeffs
                local_roughness[i] = float(np.sqrt(np.mean((z_vec - z_fit) ** 2)))
            except np.linalg.LinAlgError:
                pass

    return bond_strain, angle_strain, local_roughness


def _compute_psi6(
    xyz: np.ndarray,
    neighbors: list[list[int]],
    cell: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TRACT-parity bond-orientational order psi_6 in XY plane."""
    n_atoms = len(neighbors)
    q = np.zeros(n_atoms, dtype=complex)
    for i in range(n_atoms):
        angles: list[float] = []
        for j in neighbors[i]:
            vec = (xyz[j] - xyz[i])[:2]
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                angles.append(math.atan2(float(vec[1]), float(vec[0])))
        if angles:
            q[i] = np.mean(np.exp(1j * 6.0 * np.asarray(angles, dtype=float)))
    psi6_mag = np.abs(q)
    psi6_ang = np.mod(np.degrees(np.angle(q) / 6.0), 180.0)
    return q.real.astype(float), q.imag.astype(float), psi6_mag.astype(float), psi6_ang.astype(float)


def _detect_grains(
    neighbors: list[list[int]],
    psi6_mag: np.ndarray,
    psi6_ang: np.ndarray,
    *,
    mag_min: float = 0.45,
    misorient_max: float = 10.0,
) -> np.ndarray:
    """TRACT-parity region-growing grain labels."""
    n_atoms = len(neighbors)
    grain_id = np.full(n_atoms, -1, dtype=int)
    gid = 0
    for seed in range(n_atoms):
        if grain_id[seed] != -1 or float(psi6_mag[seed]) < mag_min:
            continue
        stack = [seed]
        grain_id[seed] = gid
        while stack:
            u = stack.pop()
            for v in neighbors[u]:
                if grain_id[v] != -1 or float(psi6_mag[v]) < mag_min:
                    continue
                d = abs(float(psi6_ang[u]) - float(psi6_ang[v]))
                d = min(d, 180.0 - d)
                if d <= misorient_max:
                    grain_id[v] = gid
                    stack.append(v)
        gid += 1
    return grain_id


def _compute_soap(
    xyz: np.ndarray,
    elements: list[str],
    c_idx: np.ndarray,
    cell: np.ndarray | None,
    *,
    r_cut: float,
    n_max: int,
    l_max: int,
    soap_ref: np.ndarray | None,
    zeta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """TRACT-parity SOAP descriptors for carbon atoms only."""
    from ase import Atoms

    try:
        from dscribe.descriptors import SOAP
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize as sk_normalize
    except Exception as exc:
        raise ValueError(
            "SOAP requested but dependencies are unavailable. Install dscribe and scikit-learn."
        ) from exc

    if len(c_idx) == 0:
        return np.empty((0, 0), dtype=float), np.empty((0, 3), dtype=float), None

    is_periodic = bool(cell is not None and np.isfinite(cell).all() and np.linalg.norm(cell) > 0.0)
    c_only_positions = np.asarray(xyz[c_idx], dtype=float)
    c_only_atoms = Atoms(
        symbols=["C"] * len(c_idx),
        positions=c_only_positions,
        cell=cell if is_periodic else None,
        pbc=is_periodic,
    )

    soap_desc = SOAP(
        species=["C"],
        r_cut=float(r_cut),
        n_max=int(n_max),
        l_max=int(l_max),
        periodic=is_periodic,
        sparse=False,
    )
    centers = list(range(len(c_idx)))
    raw = soap_desc.create(c_only_atoms, centers=centers)
    descriptors = sk_normalize(np.asarray(raw, dtype=np.float64))

    n_c = len(c_idx)
    n_components = min(3, n_c, descriptors.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    pca_raw = pca.fit_transform(descriptors)
    if pca_raw.shape[1] < 3:
        pad = np.zeros((n_c, 3 - pca_raw.shape[1]), dtype=float)
        pca_raw = np.hstack([pca_raw, pad])

    soap_score: np.ndarray | None = None
    if soap_ref is not None:
        ref = sk_normalize(np.asarray(soap_ref, dtype=np.float64))
        k = (descriptors @ ref.T) ** int(zeta)
        soap_score = np.asarray(k.mean(axis=1), dtype=float)

    return descriptors, np.asarray(pca_raw, dtype=float), soap_score


def _alpha_boundary_nodes(points2d: np.ndarray, alpha_radius: float) -> set[int]:
    """Find 2D alpha-shape boundary nodes (local indexing)."""
    if len(points2d) < 4:
        return set(range(len(points2d)))
    if not _HAS_SCIPY:
        warnings.warn("scipy not installed; alpha-shape boundary disabled.", stacklevel=2)
        return set()

    tri = Delaunay(points2d)
    edge_count: dict[tuple[int, int], int] = {}

    def add_edge(a: int, b: int) -> None:
        if a > b:
            a, b = b, a
        edge_count[(a, b)] = edge_count.get((a, b), 0) + 1

    for simplex in tri.simplices:
        ia, ib, ic = [int(v) for v in simplex]
        pa, pb, pc = points2d[ia], points2d[ib], points2d[ic]
        side_a = float(np.linalg.norm(pb - pa))
        side_b = float(np.linalg.norm(pc - pb))
        side_c = float(np.linalg.norm(pa - pc))
        s = 0.5 * (side_a + side_b + side_c)
        area2 = s * (s - side_a) * (s - side_b) * (s - side_c)
        if area2 <= 0.0:
            continue
        radius = (side_a * side_b * side_c) / (4.0 * np.sqrt(area2))
        if radius > float(alpha_radius):
            continue
        add_edge(ia, ib)
        add_edge(ib, ic)
        add_edge(ia, ic)

    out: set[int] = set()
    for (i, j), count in edge_count.items():
        if count == 1:
            out.add(i)
            out.add(j)
    return out


def _largest_angular_gap(i: int, neighbors: list[list[int]], xy: np.ndarray) -> float:
    nbrs_i = neighbors[i]
    if len(nbrs_i) < 2:
        return 360.0
    c = xy[i]
    dirs = []
    for j in nbrs_i:
        vec = xy[j] - c
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            dirs.append(vec / norm)
    if len(dirs) < 2:
        return 360.0
    angs = np.sort(np.degrees(np.arctan2([v[1] for v in dirs], [v[0] for v in dirs])) % 360.0)
    diffs = np.diff(np.concatenate([angs, [angs[0] + 360.0]]))
    return float(np.max(diffs))


def _find_boundary_nodes(
    is_periodic: bool,
    coords: np.ndarray,
    neighbors: list[list[int]],
    undercoord: np.ndarray,
    alpha_radius: float,
    gap_deg: float,
) -> set[int]:
    """Find boundary-node set in local indexing."""
    if len(coords) == 0:
        return set()

    if is_periodic:
        return {int(i) for i, flag in enumerate(undercoord) if bool(flag)}

    bnodes = _alpha_boundary_nodes(coords[:, :2], alpha_radius=float(alpha_radius)) if float(alpha_radius) > 0.0 else set()
    xy = coords[:, :2]
    for i in range(len(coords)):
        if len(neighbors[i]) <= 2 or _largest_angular_gap(i, neighbors, xy) >= float(gap_deg):
            bnodes.add(i)
    return bnodes


def _singh_type(i: int, neighbors: list[list[int]]) -> str | None:
    nbr = neighbors[i]
    if len(nbr) != 2:
        return None
    a, b = nbr
    d1, d2 = len(neighbors[a]), len(neighbors[b])
    if d1 == 3 and d2 == 3:
        return "edge_zigzag"
    if {d1, d2} == {2, 3}:
        return "edge_armchair"
    return None


def _sublattice_segment_label(graph: nx.Graph, seg_nodes: list[int]) -> str:
    inner = set()
    for u in seg_nodes:
        for v in graph.neighbors(u):
            for w in graph.neighbors(v):
                inner.add(w)
    h = graph.subgraph(inner | set(seg_nodes))
    try:
        color = nx.algorithms.bipartite.color(h)
    except Exception:
        return "edge_zigzag"
    col_seq = [color[u] for u in seg_nodes if u in color]
    if len(col_seq) < 2:
        return "edge_zigzag"
    flips = sum(1 for a, b in zip(col_seq, col_seq[1:]) if a != b)
    same = len(col_seq) - 1 - flips
    return "edge_armchair" if flips > same else "edge_zigzag"


def _label_edge_atoms(
    graph: nx.Graph,
    neighbors: list[list[int]],
    boundary_nodes_local: set[int],
    n_nodes: int,
) -> tuple[dict[int, str], np.ndarray]:
    bgraph = graph.subgraph(boundary_nodes_local).copy()
    edge_label: dict[int, str] = {}
    seg_id = np.full(n_nodes, -1, dtype=int)
    sid = 0
    for comp in nx.connected_components(bgraph):
        seg = list(comp)
        provisional: dict[int, str] = {}
        for i in seg:
            lab = _singh_type(i, neighbors)
            if lab is not None:
                provisional[i] = lab
        seg_lab = _sublattice_segment_label(graph, seg)
        for i in seg:
            if i not in provisional:
                provisional[i] = seg_lab
        for i in seg:
            edge_label[int(i)] = provisional[i]
            seg_id[int(i)] = sid
        sid += 1
    return edge_label, seg_id


@register_task("active_site_structural", label="Active Site Structural")
class ActiveSiteStructuralTask(AnalysisTask):
    """Compute per-atom active-site structural descriptors on one frame."""

    required_data = ConnectivityTrajectoryData

    def required_data_for(self, request: object, args: dict | None = None):
        """Resolve required input type for structural task execution mode.

        Works on
        -----
        Active-site structural request objects and optional executor arguments

        Parameters
        -----
        request : object
            Request object that may specify `bond_mode`.
        args : dict | None, optional
            Optional execution-time argument map used by workflow dispatch.

        Returns
        -----
        Any
            Required input type (single type or tuple) for current mode.

        Examples
        -----
        ```python
        required = task.required_data_for(request, args)
        ```
        Sample output:
        `ConnectivityTrajectoryData` or `TrajectoryData` (or tuple fallback).
        Meaning:
        Data requirements adapt to bond graph source mode.
        """
        bond_mode = str(getattr(request, "bond_mode", "bo")).strip().lower()
        if bond_mode == "distance":
            # Validation wrappers call without executor args; accept both direct and executor-fed inputs there.
            if args is None:
                return (TrajectoryData, ConnectivityTrajectoryData)
            return TrajectoryData
        return ConnectivityTrajectoryData

    @staticmethod
    def recommended_presentations(
        _result: ActiveSiteStructuralResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Build default table/plot presentations for structural task output.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : ActiveSiteStructuralResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for table and `d_pyr` profile views.

        Examples
        -----
        ```python
        specs = ActiveSiteStructuralTask.recommended_presentations(result, payload)
        ```
        Sample output:
        A list with table and `d_pyr vs atom_id` plot views.
        Meaning:
        Structural outputs can be rendered without custom plotting metadata.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "atom_id" not in sample or "d_pyr" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="d_pyr vs atom_id",
                mapping={"x_col": "atom_id", "y_col": "d_pyr", "group_by_col": "label"},
                options={"title": "Pyramidalization by Atom", "xlabel": "atom_id", "ylabel": "d_pyr", "legend": True},
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ConnectivityTrajectoryData | TrajectoryData,
        request: ActiveSiteStructuralRequest,
        reporter=None,
    ) -> ActiveSiteStructuralResult:
        """Compute frame-level active-site structural descriptors and labels.

        Builds a bond graph (BO- or distance-based), derives geometric/ring
        descriptors, assigns TRACT-parity labels, and returns detailed plus
        TRACT-compatible structural tables.

        Works on
        -----
        `ConnectivityTrajectoryData` or `TrajectoryData` plus structural request

        Parameters
        -----
        data : ConnectivityTrajectoryData | TrajectoryData
            Input trajectory (and optional connectivity) data for selected frame.
        request : ActiveSiteStructuralRequest
            Structural analysis configuration for frame and descriptor options.
        reporter : Any, optional
            Optional progress callback invoked near completion.

        Returns
        -----
        ActiveSiteStructuralResult
            Structural descriptor table, TRACT projection, summary metrics, and
            optional SOAP descriptors.

        Examples
        -----
        ```python
        req = ActiveSiteStructuralRequest(frame=0, bond_mode="bo")
        result = ActiveSiteStructuralTask().run(bundle, req)
        ```
        Sample output:
        `result.table` with per-atom descriptors and labels.
        Meaning:
        One analyzed frame is transformed into rich structural site features.
        """
        if isinstance(data, ConnectivityTrajectoryData):
            trajectory = data.trajectory
            connectivity_data = data
        elif isinstance(data, TrajectoryData):
            trajectory = data
            connectivity_data = None
        else:
            raise TypeError(
                "ActiveSiteStructuralTask expects ConnectivityTrajectoryData or TrajectoryData."
            )

        positions = np.asarray(trajectory.positions, dtype=float)
        if positions.ndim != 3:
            raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")
        n_frames, n_atoms, _ = positions.shape
        frame_index = int(request.frame)
        if frame_index < 0 or frame_index >= n_frames:
            raise ValueError(f"frame must be in [0, {n_frames - 1}]")

        if len(trajectory.atom_ids) != n_atoms:
            raise ValueError("TrajectoryData.atom_ids length must match trajectory atom count.")
        if len(trajectory.elements) != n_atoms:
            raise ValueError("TrajectoryData.elements length must match trajectory atom count.")

        xyz = positions[frame_index]
        valid = np.isfinite(xyz).all(axis=1)
        atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
        elements = [str(e) for e in trajectory.elements]
        carbon_element = str(request.carbon_element)
        bond_mode = str(getattr(request, "bond_mode", "bo")).strip().lower()
        bond_scale = float(getattr(request, "bond_scale", 1.20))
        bo_threshold = float(request.bo_threshold)
        cell = frame_cell_matrix(data, frame_index)

        if bond_mode == "bo":
            if connectivity_data is None:
                raise ValueError(
                    "bond_mode='bo' requires ConnectivityTrajectoryData with ConnectivityData.bond_orders."
                )
            bo = _bond_order_frame(connectivity_data, frame_index)
            if bo.shape != (n_atoms, n_atoms):
                raise ValueError("bond-order frame shape must match trajectory atom dimension.")
            bo = np.maximum(bo, bo.T)
            bo[~np.isfinite(bo)] = 0.0
            bo[~valid, :] = 0.0
            bo[:, ~valid] = 0.0
            np.fill_diagonal(bo, 0.0)

            bonded = bo > bo_threshold
            graph = nx.Graph()
            graph.add_nodes_from(range(n_atoms))
            ii, jj = np.where(np.triu(bonded, 1))
            for i, j in zip(ii.tolist(), jj.tolist()):
                dist = float(np.linalg.norm(minimum_image_vector(xyz[j] - xyz[i], cell)))
                graph.add_edge(i, j, bo=float(bo[i, j]), d=dist)
            neighbors = [sorted(list(graph.neighbors(i))) for i in range(n_atoms)]
        elif bond_mode == "distance":
            graph, neighbors = _build_geometric_bond_graph(
                xyz=xyz,
                elements=elements,
                cell=cell,
                bond_scale=bond_scale,
            )
        else:
            raise ValueError("bond_mode must be one of: bo, distance")

        n_bonds = np.asarray([len(v) for v in neighbors], dtype=int)
        c_idx = np.where(np.asarray(elements, dtype=object) == carbon_element)[0]
        n_c = len(c_idx)

        n_bonds_c = np.zeros(n_atoms, dtype=int)
        for global_i in c_idx.tolist():
            n_bonds_c[global_i] = int(sum(1 for j in neighbors[global_i] if elements[j] == carbon_element))

        neighbors_same_or_all: list[list[int]] = []
        for i in range(n_atoms):
            same = [j for j in neighbors[i] if elements[j] == elements[i]]
            neighbors_same_or_all.append(same if same else list(neighbors[i]))

        has_hetero_bond = _compute_has_hetero_bond(elements, neighbors, carbon_element)

        is_undercoord = np.zeros(n_atoms, dtype=bool)
        for i in range(n_atoms):
            is_undercoord[i] = bool(len(neighbors_same_or_all[i]) < 3)

        is_periodic = bool(cell is not None and np.isfinite(cell).all() and np.linalg.norm(cell) > 0.0)

        bond_strain, angle_strain, local_roughness = _compute_strain_roughness(
            graph=graph,
            neighbors=neighbors,
            xyz=xyz,
            cell=cell,
        )
        psi6_re, psi6_im, psi6_mag, psi6_ang = _compute_psi6(
            xyz=xyz,
            neighbors=neighbors,
            cell=cell,
        )
        grain_id = _detect_grains(neighbors=neighbors, psi6_mag=psi6_mag, psi6_ang=psi6_ang)

        d_pyr = np.zeros(n_atoms, dtype=float)
        d_ang_deg = np.full(n_atoms, np.nan, dtype=float)
        for atom_i in range(n_atoms):
            if not valid[atom_i]:
                continue
            geom_neighbors = [int(j) for j in neighbors_same_or_all[atom_i] if valid[int(j)]]
            dp, da = _pyramidalization_for_atom(atom_i, geom_neighbors, xyz, cell)
            d_pyr[atom_i] = dp
            d_ang_deg[atom_i] = da

        soap_pc = np.full((n_atoms, 3), np.nan, dtype=float)
        soap_score_col = np.full(n_atoms, np.nan, dtype=float)
        soap_descriptors: np.ndarray | None = None
        if bool(getattr(request, "soap", False)) and n_c > 0:
            soap_ref = None
            soap_ref_path = getattr(request, "soap_ref_path", None)
            if soap_ref_path:
                soap_ref = np.load(Path(str(soap_ref_path)))
            soap_descriptors, pca_scores, soap_score = _compute_soap(
                xyz=xyz,
                elements=elements,
                c_idx=c_idx,
                cell=cell,
                r_cut=float(getattr(request, "soap_r_cut", 5.0)),
                n_max=int(getattr(request, "soap_n_max", 9)),
                l_max=int(getattr(request, "soap_l_max", 9)),
                soap_ref=soap_ref,
                zeta=int(getattr(request, "soap_zeta", 2)),
            )
            soap_pc[c_idx, :] = pca_scores
            if soap_score is not None:
                soap_score_col[c_idx] = soap_score

        boundary_nodes = _find_boundary_nodes(
            is_periodic=is_periodic,
            coords=xyz,
            neighbors=neighbors,
            undercoord=is_undercoord,
            alpha_radius=float(request.alpha_radius),
            gap_deg=float(request.gap_deg),
        )
        edge_label_global, seg_id = _label_edge_atoms(
            graph=graph,
            neighbors=neighbors,
            boundary_nodes_local=boundary_nodes,
            n_nodes=n_atoms,
        )
        boundary_global = set(boundary_nodes)

        rings = find_primitive_rings(graph)
        min_ring, max_ring, in_non6_ring = ring_membership(n_atoms, rings)
        defect_type = per_atom_defect_types(n_atoms, rings, boundary_nodes=boundary_nodes)
        ring_size_min = np.where(min_ring < 0, 0, min_ring).astype(int)
        ring_size_max = np.where(max_ring < 0, 0, max_ring).astype(int)

        labels = site_labels_tract_parity(
            elements=elements,
            carbon_element=carbon_element,
            is_undercoord=is_undercoord,
            in_non6_ring=in_non6_ring,
            defect_type=defect_type,
            edge_label_by_global=edge_label_global,
        )
        edge_label_col = np.asarray([edge_label_global.get(i, "") for i in range(n_atoms)], dtype=object)
        boundary_col = np.asarray([i in boundary_global for i in range(n_atoms)], dtype=bool)

        table = pd.DataFrame(
            {
                "frame_idx": np.full(n_atoms, frame_index, dtype=int),
                "atom_id": atom_ids,
                "element": elements,
                "x": xyz[:, 0],
                "y": xyz[:, 1],
                "z": xyz[:, 2],
                "valid": valid,
                "n_bonds": n_bonds,
                "n_bonds_c": n_bonds_c,
                "is_undercoord": is_undercoord,
                "has_hetero_bond": has_hetero_bond,
                "d_pyr": d_pyr,
                "d_ang_deg": d_ang_deg,
                "bond_strain": bond_strain,
                "angle_strain": angle_strain,
                "local_roughness": local_roughness,
                "psi6_re": psi6_re,
                "psi6_im": psi6_im,
                "psi6_mag": psi6_mag,
                "psi6_ang": psi6_ang,
                "grain_id": grain_id,
                "ring_size_min": ring_size_min,
                "ring_size_max": ring_size_max,
                "in_non6_ring": in_non6_ring,
                "defect_type": defect_type,
                "boundary": boundary_col,
                "edge_label": edge_label_col,
                "seg_id": seg_id,
                "label": labels,
                "soap_pc1": soap_pc[:, 0],
                "soap_pc2": soap_pc[:, 1],
                "soap_pc3": soap_pc[:, 2],
                "soap_score": soap_score_col,
            }
        )

        if not bool(request.include_noncarbon):
            table = table[table["element"] == carbon_element].reset_index(drop=True)

        ring_histogram: dict[str, int] = {}
        for cyc in rings:
            key = str(len(cyc))
            ring_histogram[key] = int(ring_histogram.get(key, 0)) + 1

        carbon_table = table[table["element"] == carbon_element]
        defect_cluster_counts = {
            str(k): int(v)
            for k, v in carbon_table["defect_type"].value_counts().to_dict().items()
            if str(k) != "none"
        }
        regular_carbon = carbon_table[~carbon_table["is_undercoord"]]
        if len(regular_carbon) > 0:
            abs_dp = regular_carbon["d_pyr"].abs().to_numpy(dtype=float)
            dpyr_stats = {
                "mean_abs": float(np.mean(abs_dp)),
                "median_abs": float(np.median(abs_dp)),
                "frac_above_tau": float((abs_dp > 0.229).sum() / max(len(abs_dp), 1)),
            }
        else:
            dpyr_stats = {"mean_abs": 0.0, "median_abs": 0.0, "frac_above_tau": 0.0}

        summary = {
            "frame_idx": frame_index,
            "bond_mode": bond_mode,
            "bond_scale": float(bond_scale),
            "bo_threshold": bo_threshold,
            "n_atoms_total": int(n_atoms),
            "n_atoms_valid": int(valid.sum()),
            "n_atoms_carbon": int(sum(1 for e in elements if e == carbon_element)),
            "n_undercoord_carbon": int(((table["element"] == carbon_element) & table["is_undercoord"]).sum()),
            "n_boundary_carbon": int(((table["element"] == carbon_element) & table["boundary"]).sum()),
            "n_edge_zigzag": int((table["edge_label"] == "edge_zigzag").sum()),
            "n_edge_armchair": int((table["edge_label"] == "edge_armchair").sum()),
            "n_bonds_total": int(graph.number_of_edges()),
            "n_rings_primitive": int(len(rings)),
            "ring_histogram": ring_histogram,
            "defect_cluster_counts": defect_cluster_counts,
            "n_grains": int(np.max(grain_id) + 1) if np.any(grain_id >= 0) else 0,
            "dpyr_stats": dpyr_stats,
            "is_periodic": bool(is_periodic),
            "label_counts": {str(k): int(v) for k, v in table["label"].value_counts().to_dict().items()},
        }

        if reporter:
            reporter("analyze", 1, 1, "Active-site structural descriptors assembled")
        tract_table = to_tract_structural_table(table, strict=bool(request.strict_tract))
        return ActiveSiteStructuralResult(
            table=table,
            tract_table=tract_table,
            summary=summary,
            request=request,
            soap_descriptors=soap_descriptors,
        )


__all__ = [
    "ActiveSiteStructuralRequest",
    "ActiveSiteStructuralResult",
    "ActiveSiteStructuralTask",
]
