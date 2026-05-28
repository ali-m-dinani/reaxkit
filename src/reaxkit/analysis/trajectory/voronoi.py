"""Compute Voronoi-based geometric analyzers from trajectory snapshots.

This module performs Voronoi tessellation over selected trajectory frames and
exports geometric metrics and optional cell geometry summaries. It is limited
to Voronoi-derived descriptors and does not compute bond-order connectivity.

**Usage context**

- Spatial partitioning: Quantify local free volume and neighborhood geometry.
- Frame diagnostics: Track Voronoi metrics over selected simulation steps.
- Structural reporting: Export tessellation-derived tables for analysis UIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Voronoi

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec


@dataclass
class VoronoiRequest(BaseRequest):
    """Request for Voronoi tessellation analysis.

    This request is shared by metric-only and geometry-producing Voronoi tasks.

    Fields
    -----
    atom_ids : Optional[Sequence[int]]
        Atom IDs to include. When set, takes precedence over `atom_types`.
    atom_types : Optional[Sequence[str]]
        Atom types/elements used for selection when `atom_ids` is not set.
    frames : Optional[Sequence[int]]
        Frame indices to evaluate. `None` means all frames.
    every : int
        Stride over selected frames. Must be `>= 1`.
    backend : str
        Voronoi backend identifier, `"scipy"` or `"pyvoro"`.

    Examples
    -----
    ```python
    req = VoronoiRequest(atom_types=["O"], frames=[0, 50, 100], backend="scipy")
    ```
    Sample output:
    `VoronoiRequest(...)`
    Meaning:
    The request selects atoms/frames and backend for Voronoi evaluation.
    """

    atom_ids: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Atom IDs", "help": "Atom IDs to include in output. Empty means all atoms.", "units": "index"},
    )
    atom_types: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={"label": "Atom types", "help": "Element symbols to include when atom_ids is empty."},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Frame indices to evaluate. Empty means all frames.", "units": "frame_index"},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Stride", "help": "Stride over selected frames.", "min": 1, "units": "frames"},
    )
    backend: str = dc_field(
        default="scipy",
        metadata={"label": "Backend", "help": "Voronoi backend.", "choices": ["scipy", "pyvoro"]},
    )


@dataclass
class VoronoiResult(BaseResult):
    """Result of Voronoi metric analysis.

    Fields
    -----
    table : pd.DataFrame
        Output table with columns
        `["frame_index", "iter", "atom_id", "atom_type", "voronoi_volume", "num_faces", "is_bounded", "backend"]`.
        `voronoi_volume` can be `NaN` for unbounded/degenerate cells.
    request : VoronoiRequest
        Request object used for this analysis execution.

    Examples
    -----
    ```python
    result = VoronoiScipyTask().run(data, req)
    result.table[["atom_id", "voronoi_volume"]].head()
    ```
    Sample output:
    DataFrame rows with one Voronoi metric record per selected atom/frame.
    Meaning:
    The table summarizes per-atom Voronoi cell properties.
    """

    table: pd.DataFrame
    request: VoronoiRequest


@dataclass
class VoronoiGeometryResult(BaseResult):
    """Result of Voronoi geometry analysis for 2D/3D visualization.

    Fields
    -----
    table : pd.DataFrame
        Output geometry table with columns
        `["frame_index", "iter", "atom_id", "atom_type", "site_position", "vertices", "faces", "neighbor_atom_ids", "voronoi_volume", "num_faces", "is_bounded", "backend"]`.
        Geometry fields contain JSON-serializable list/dict structures.
    request : VoronoiRequest
        Request object used for this analysis execution.

    Notes
    -----
    `faces` entries include local vertex indices and neighbor references.

    Examples
    -----
    ```python
    result = VoronoiGeometryScipyTask().run(data, req)
    result.table[["atom_id", "vertices", "faces"]].head(1)
    ```
    Sample output:
    One-row DataFrame containing full cell geometry for a selected atom.
    Meaning:
    The result can drive 2D/3D Voronoi visualization pipelines.
    """

    table: pd.DataFrame
    request: VoronoiRequest


def _recommended_presentations(_result: VoronoiResult, payload: dict[str, Any]) -> list[PresentationSpec]:
    rows = payload.get("table")
    if not isinstance(rows, list) or not rows:
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]
    sample = rows[0] if isinstance(rows[0], dict) else {}
    if "iter" not in sample or "voronoi_volume" not in sample:
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]
    return [
        PresentationSpec(renderer="table", label="Table", view_type="table"),
        PresentationSpec(
            renderer="single_plot",
            label="Voronoi Volume vs iter",
            mapping={"x_col": "iter", "y_col": "voronoi_volume", "group_by_col": "atom_id"},
            options={"title": "Voronoi Volume", "xlabel": "iter", "ylabel": "voronoi_volume", "legend": True},
            view_type="plot2d",
        ),
    ]


def _select_atom_indices(data: TrajectoryData, request: VoronoiRequest) -> list[int]:
    if request.atom_ids is not None:
        wanted = set(int(v) for v in request.atom_ids)
        return [j for j, aid in enumerate(data.atom_ids) if int(aid) in wanted]
    if request.atom_types:
        wanted_types = {str(v) for v in request.atom_types}
        return [j for j, element in enumerate(data.elements) if str(element) in wanted_types]
    return list(range(data.positions.shape[1]))


def _select_frames(data: TrajectoryData, request: VoronoiRequest) -> list[int]:
    n_frames = data.positions.shape[0]
    frame_idx = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames]
    return frame_idx[:: max(1, int(request.every))]


def _rows_for_frame_scipy(
    data: TrajectoryData,
    frame_index: int,
    selected_indices: Sequence[int],
    *,
    iter_val: int,
) -> list[dict[str, Any]]:
    frame = np.asarray(data.positions[int(frame_index)], dtype=float)
    valid = np.isfinite(frame).all(axis=1)
    valid_selected = [idx for idx in selected_indices if idx < frame.shape[0] and bool(valid[idx])]
    if not valid_selected:
        return []

    points = frame[valid]
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(np.where(valid)[0].tolist())}

    if points.shape[0] < 5:
        rows: list[dict[str, Any]] = []
        for global_idx in valid_selected:
            atom_id = int(data.atom_ids[global_idx])
            rows.append(
                {
                    "frame_index": int(frame_index),
                    "iter": int(iter_val),
                    "atom_id": atom_id,
                    "atom_type": str(data.elements[global_idx]),
                    "voronoi_volume": np.nan,
                    "num_faces": 0,
                    "is_bounded": False,
                    "backend": "scipy",
                }
            )
        return rows

    vor = Voronoi(points, qhull_options="Qbb Qc Qx")

    neighbors = {i: set() for i in range(points.shape[0])}
    for p1, p2 in vor.ridge_points:
        neighbors[int(p1)].add(int(p2))
        neighbors[int(p2)].add(int(p1))

    rows = []
    for global_idx in valid_selected:
        local_idx = global_to_local[global_idx]
        region_idx = int(vor.point_region[local_idx])
        region = vor.regions[region_idx] if region_idx >= 0 else []

        bounded = bool(region) and (-1 not in region)
        volume = np.nan
        if bounded:
            verts = vor.vertices[np.asarray(region, dtype=int)]
            if verts.shape[0] >= 4:
                try:
                    volume = float(ConvexHull(verts).volume)
                except Exception:
                    volume = np.nan

        rows.append(
            {
                "frame_index": int(frame_index),
                "iter": int(iter_val),
                "atom_id": int(data.atom_ids[global_idx]),
                "atom_type": str(data.elements[global_idx]),
                "voronoi_volume": float(volume) if np.isfinite(volume) else np.nan,
                "num_faces": int(len(neighbors.get(local_idx, ()))),
                "is_bounded": bool(bounded),
                "backend": "scipy",
            }
        )
    return rows


def _pyvoro_limits_and_periodic(
    data: TrajectoryData,
    frame_index: int,
    points: np.ndarray,
) -> tuple[list[list[float]], bool]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad = 1e-6

    sim = data.simulation
    if sim is not None and sim.cell_lengths is not None:
        lengths = np.asarray(sim.cell_lengths[int(frame_index)], dtype=float)
        if lengths.shape[0] == 3 and np.all(np.isfinite(lengths)) and np.all(lengths > 0):
            span = maxs - mins
            if np.all(span <= lengths + 1e-6):
                limits = [[float(mins[k]), float(mins[k] + lengths[k])] for k in range(3)]
                return limits, True

    limits = [[float(mins[k] - pad), float(maxs[k] + pad)] for k in range(3)]
    return limits, False


def _rows_for_frame_pyvoro(
    data: TrajectoryData,
    frame_index: int,
    selected_indices: Sequence[int],
    *,
    iter_val: int,
) -> list[dict[str, Any]]:
    try:
        import pyvoro
    except Exception as exc:
        raise ImportError("pyvoro backend is not available. Install pyvoro or use --backend scipy.") from exc

    frame = np.asarray(data.positions[int(frame_index)], dtype=float)
    valid = np.isfinite(frame).all(axis=1)
    valid_selected = [idx for idx in selected_indices if idx < frame.shape[0] and bool(valid[idx])]
    if not valid_selected:
        return []

    points = frame[valid]
    global_valid = np.where(valid)[0].tolist()
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(global_valid)}

    if points.shape[0] < 2:
        rows: list[dict[str, Any]] = []
        for global_idx in valid_selected:
            rows.append(
                {
                    "frame_index": int(frame_index),
                    "iter": int(iter_val),
                    "atom_id": int(data.atom_ids[global_idx]),
                    "atom_type": str(data.elements[global_idx]),
                    "voronoi_volume": np.nan,
                    "num_faces": 0,
                    "is_bounded": False,
                    "backend": "pyvoro",
                }
            )
        return rows

    limits, periodic = _pyvoro_limits_and_periodic(data, int(frame_index), points)
    extent = np.asarray([hi - lo for lo, hi in limits], dtype=float)
    block_size = float(max(1e-3, np.min(extent) / 5.0))

    cells = pyvoro.compute_voronoi(
        points=np.asarray(points, dtype=float).tolist(),
        limits=limits,
        block_size=block_size,
        periodic=bool(periodic),
    )

    rows = []
    for global_idx in valid_selected:
        local_idx = global_to_local[global_idx]
        cell = cells[local_idx]
        faces = cell.get("faces", [])
        volume = cell.get("volume", np.nan)
        bounded = all(int(face.get("adjacent_cell", -1)) >= 0 for face in faces)
        rows.append(
            {
                "frame_index": int(frame_index),
                "iter": int(iter_val),
                "atom_id": int(data.atom_ids[global_idx]),
                "atom_type": str(data.elements[global_idx]),
                "voronoi_volume": float(volume) if np.isfinite(float(volume)) else np.nan,
                "num_faces": int(len(faces)),
                "is_bounded": bool(bounded),
                "backend": "pyvoro",
            }
        )
    return rows


def _run_voronoi(data: TrajectoryData, request: VoronoiRequest, *, backend: str, reporter=None) -> VoronoiResult:
    cols = ["frame_index", "iter", "atom_id", "atom_type", "voronoi_volume", "num_faces", "is_bounded", "backend"]
    n_frames = data.positions.shape[0]
    if n_frames == 0:
        return VoronoiResult(table=pd.DataFrame(columns=cols), request=request)

    frame_idx = _select_frames(data, request)
    if not frame_idx:
        return VoronoiResult(table=pd.DataFrame(columns=cols), request=request)

    selected_indices = _select_atom_indices(data, request)
    if not selected_indices:
        return VoronoiResult(table=pd.DataFrame(columns=cols), request=request)

    backend_l = str(backend).strip().lower()
    rows: list[dict[str, Any]] = []
    total = len(frame_idx)

    for step_i, frame_index in enumerate(frame_idx, start=1):
        iter_val = int(data.iterations[frame_index]) if data.iterations is not None else int(frame_index)
        if backend_l == "scipy":
            rows.extend(_rows_for_frame_scipy(data, int(frame_index), selected_indices, iter_val=iter_val))
        elif backend_l == "pyvoro":
            rows.extend(_rows_for_frame_pyvoro(data, int(frame_index), selected_indices, iter_val=iter_val))
        else:
            raise ValueError("backend must be 'scipy' or 'pyvoro'")
        if reporter:
            reporter("analyze", step_i, total, "Computing Voronoi metrics")

    table = pd.DataFrame(rows, columns=cols)
    if not table.empty:
        table = table.sort_values(["frame_index", "atom_id"], kind="stable").reset_index(drop=True)
    return VoronoiResult(table=table, request=request)


def _geometry_rows_for_frame_scipy(
    data: TrajectoryData,
    frame_index: int,
    selected_indices: Sequence[int],
    *,
    iter_val: int,
) -> list[dict[str, Any]]:
    frame = np.asarray(data.positions[int(frame_index)], dtype=float)
    valid = np.isfinite(frame).all(axis=1)
    valid_selected = [idx for idx in selected_indices if idx < frame.shape[0] and bool(valid[idx])]
    if not valid_selected:
        return []

    points = frame[valid]
    global_valid = np.where(valid)[0].tolist()
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(global_valid)}

    if points.shape[0] < 5:
        out: list[dict[str, Any]] = []
        for global_idx in valid_selected:
            out.append(
                {
                    "frame_index": int(frame_index),
                    "iter": int(iter_val),
                    "atom_id": int(data.atom_ids[global_idx]),
                    "atom_type": str(data.elements[global_idx]),
                    "site_position": frame[global_idx].astype(float).tolist(),
                    "vertices": [],
                    "faces": [],
                    "neighbor_atom_ids": [],
                    "voronoi_volume": np.nan,
                    "num_faces": 0,
                    "is_bounded": False,
                    "backend": "scipy",
                }
            )
        return out

    vor = Voronoi(points, qhull_options="Qbb Qc Qx")
    # Build per-point ridge membership explicitly from zipped ridge arrays.
    ridge_by_point = {i: [] for i in range(points.shape[0])}
    for (p1, p2), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        pair = (int(p1), int(p2), [int(v) for v in ridge_vertices])
        ridge_by_point[int(p1)].append(pair)
        ridge_by_point[int(p2)].append(pair)

    rows: list[dict[str, Any]] = []
    for global_idx in valid_selected:
        local_idx = global_to_local[global_idx]
        region_idx = int(vor.point_region[local_idx])
        region = vor.regions[region_idx] if region_idx >= 0 else []
        bounded = bool(region) and (-1 not in region)

        ridge_items = ridge_by_point.get(local_idx, [])
        finite_vertex_ids = {
            int(v)
            for _, _, rv in ridge_items
            for v in rv
            if int(v) >= 0
        }
        finite_vertex_ids.update(int(v) for v in region if int(v) >= 0)
        sorted_vids = sorted(finite_vertex_ids)
        vertex_map = {vid: i for i, vid in enumerate(sorted_vids)}
        cell_vertices = [vor.vertices[int(vid)].astype(float).tolist() for vid in sorted_vids]

        faces: list[dict[str, Any]] = []
        neighbor_atom_ids: list[int] = []
        for p1, p2, ridge_vertices in ridge_items:
            neigh_local = int(p2) if int(p1) == int(local_idx) else int(p1)
            neigh_global = global_valid[neigh_local]
            neigh_atom_id = int(data.atom_ids[neigh_global])
            neighbor_atom_ids.append(neigh_atom_id)
            faces.append(
                {
                    "vertex_indices": [int(vertex_map[v]) for v in ridge_vertices if int(v) >= 0 and int(v) in vertex_map],
                    "neighbor_atom_id": neigh_atom_id,
                    "is_open": bool(any(int(v) < 0 for v in ridge_vertices)),
                }
            )

        volume = np.nan
        if bounded:
            verts = vor.vertices[np.asarray(region, dtype=int)]
            if verts.shape[0] >= 4:
                try:
                    volume = float(ConvexHull(verts).volume)
                except Exception:
                    volume = np.nan

        rows.append(
            {
                "frame_index": int(frame_index),
                "iter": int(iter_val),
                "atom_id": int(data.atom_ids[global_idx]),
                "atom_type": str(data.elements[global_idx]),
                "site_position": frame[global_idx].astype(float).tolist(),
                "vertices": cell_vertices,
                "faces": faces,
                "neighbor_atom_ids": sorted(set(int(v) for v in neighbor_atom_ids)),
                "voronoi_volume": float(volume) if np.isfinite(volume) else np.nan,
                "num_faces": int(len(ridge_items)),
                "is_bounded": bool(bounded),
                "backend": "scipy",
            }
        )
    return rows


def _geometry_rows_for_frame_pyvoro(
    data: TrajectoryData,
    frame_index: int,
    selected_indices: Sequence[int],
    *,
    iter_val: int,
) -> list[dict[str, Any]]:
    try:
        import pyvoro
    except Exception as exc:
        raise ImportError("pyvoro backend is not available. Install pyvoro or use --backend scipy.") from exc

    frame = np.asarray(data.positions[int(frame_index)], dtype=float)
    valid = np.isfinite(frame).all(axis=1)
    valid_selected = [idx for idx in selected_indices if idx < frame.shape[0] and bool(valid[idx])]
    if not valid_selected:
        return []

    points = frame[valid]
    global_valid = np.where(valid)[0].tolist()
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(global_valid)}

    if points.shape[0] < 2:
        rows: list[dict[str, Any]] = []
        for global_idx in valid_selected:
            rows.append(
                {
                    "frame_index": int(frame_index),
                    "iter": int(iter_val),
                    "atom_id": int(data.atom_ids[global_idx]),
                    "atom_type": str(data.elements[global_idx]),
                    "site_position": frame[global_idx].astype(float).tolist(),
                    "vertices": [],
                    "faces": [],
                    "neighbor_atom_ids": [],
                    "voronoi_volume": np.nan,
                    "num_faces": 0,
                    "is_bounded": False,
                    "backend": "pyvoro",
                }
            )
        return rows

    limits, periodic = _pyvoro_limits_and_periodic(data, int(frame_index), points)
    extent = np.asarray([hi - lo for lo, hi in limits], dtype=float)
    block_size = float(max(1e-3, np.min(extent) / 5.0))
    cells = pyvoro.compute_voronoi(
        points=np.asarray(points, dtype=float).tolist(),
        limits=limits,
        block_size=block_size,
        periodic=bool(periodic),
    )

    rows: list[dict[str, Any]] = []
    for global_idx in valid_selected:
        local_idx = global_to_local[global_idx]
        cell = cells[local_idx]
        vertices = [[float(x), float(y), float(z)] for x, y, z in cell.get("vertices", [])]
        faces_raw = cell.get("faces", [])

        faces: list[dict[str, Any]] = []
        neighbor_atom_ids: list[int] = []
        for face in faces_raw:
            adj = int(face.get("adjacent_cell", -1))
            neigh_atom_id = None
            if adj >= 0 and adj < len(global_valid):
                neigh_atom_id = int(data.atom_ids[global_valid[adj]])
                neighbor_atom_ids.append(neigh_atom_id)
            faces.append(
                {
                    "vertex_indices": [int(v) for v in face.get("vertices", [])],
                    "neighbor_atom_id": neigh_atom_id,
                    "is_open": bool(adj < 0),
                }
            )

        volume = cell.get("volume", np.nan)
        bounded = all(int(face.get("adjacent_cell", -1)) >= 0 for face in faces_raw)
        rows.append(
            {
                "frame_index": int(frame_index),
                "iter": int(iter_val),
                "atom_id": int(data.atom_ids[global_idx]),
                "atom_type": str(data.elements[global_idx]),
                "site_position": frame[global_idx].astype(float).tolist(),
                "vertices": vertices,
                "faces": faces,
                "neighbor_atom_ids": sorted(set(int(v) for v in neighbor_atom_ids)),
                "voronoi_volume": float(volume) if np.isfinite(float(volume)) else np.nan,
                "num_faces": int(len(faces_raw)),
                "is_bounded": bool(bounded),
                "backend": "pyvoro",
            }
        )
    return rows


def _run_voronoi_geometry(data: TrajectoryData, request: VoronoiRequest, *, backend: str, reporter=None) -> VoronoiGeometryResult:
    cols = [
        "frame_index",
        "iter",
        "atom_id",
        "atom_type",
        "site_position",
        "vertices",
        "faces",
        "neighbor_atom_ids",
        "voronoi_volume",
        "num_faces",
        "is_bounded",
        "backend",
    ]
    n_frames = data.positions.shape[0]
    if n_frames == 0:
        return VoronoiGeometryResult(table=pd.DataFrame(columns=cols), request=request)

    frame_idx = _select_frames(data, request)
    if not frame_idx:
        return VoronoiGeometryResult(table=pd.DataFrame(columns=cols), request=request)

    selected_indices = _select_atom_indices(data, request)
    if not selected_indices:
        return VoronoiGeometryResult(table=pd.DataFrame(columns=cols), request=request)

    backend_l = str(backend).strip().lower()
    rows: list[dict[str, Any]] = []
    total = len(frame_idx)
    for step_i, frame_index in enumerate(frame_idx, start=1):
        iter_val = int(data.iterations[frame_index]) if data.iterations is not None else int(frame_index)
        if backend_l == "scipy":
            rows.extend(_geometry_rows_for_frame_scipy(data, int(frame_index), selected_indices, iter_val=iter_val))
        elif backend_l == "pyvoro":
            rows.extend(_geometry_rows_for_frame_pyvoro(data, int(frame_index), selected_indices, iter_val=iter_val))
        else:
            raise ValueError("backend must be 'scipy' or 'pyvoro'")
        if reporter:
            reporter("analyze", step_i, total, "Computing Voronoi geometry")

    table = pd.DataFrame(rows, columns=cols)
    if not table.empty:
        table = table.sort_values(["frame_index", "atom_id"], kind="stable").reset_index(drop=True)
    return VoronoiGeometryResult(table=table, request=request)


@register_task("get_voronoi_scipy", label="Voronoi (SciPy)")
class VoronoiScipyTask(AnalysisTask):
    """Compute per-atom Voronoi metrics using SciPy.

    Notes
    -----
    SciPy Voronoi is non-periodic and can yield unbounded cells near boundaries.
    """

    required_data = TrajectoryData
    recommended_presentations = staticmethod(_recommended_presentations)

    def run(self, data: TrajectoryData, request: VoronoiRequest, reporter=None) -> VoronoiResult:
        """Compute Voronoi metric table using the SciPy backend.

        Works on
        -----
        `TrajectoryData` plus `VoronoiRequest` analyzer inputs

        Parameters
        -----
        data : TrajectoryData
            Trajectory coordinates, atom metadata, and optional iterations.
        request : VoronoiRequest
            Atom/frame selection and backend configuration.
        reporter : Any, optional
            Optional progress callback for frame-wise processing.

        Returns
        -----
        VoronoiResult
            Metric-only Voronoi table and request echo.

        Examples
        -----
        ```python
        result = VoronoiScipyTask().run(data, req)
        ```
        Sample output:
        `result.table` with `voronoi_volume`, `num_faces`, `is_bounded`.
        Meaning:
        One row is produced per selected atom and frame.
        """
        return _run_voronoi(data, request, backend="scipy", reporter=reporter)


@register_task("get_voronoi_pyvoro", label="Voronoi (pyvoro)")
class VoronoiPyvoroTask(AnalysisTask):
    """Compute per-atom Voronoi metrics using pyvoro.

    Notes
    -----
    Uses pyvoro native cell outputs and enables periodic tessellation when box
    lengths are available and consistent with frame coordinates.
    """

    required_data = TrajectoryData
    recommended_presentations = staticmethod(_recommended_presentations)

    def run(self, data: TrajectoryData, request: VoronoiRequest, reporter=None) -> VoronoiResult:
        """Compute Voronoi metric table using the pyvoro backend.

        Works on
        -----
        `TrajectoryData` plus `VoronoiRequest` analyzer inputs

        Parameters
        -----
        data : TrajectoryData
            Trajectory coordinates, atom metadata, and optional iterations.
        request : VoronoiRequest
            Atom/frame selection and backend configuration.
        reporter : Any, optional
            Optional progress callback for frame-wise processing.

        Returns
        -----
        VoronoiResult
            Metric-only Voronoi table and request echo.

        Examples
        -----
        ```python
        result = VoronoiPyvoroTask().run(data, req)
        ```
        Sample output:
        Table rows with `backend="pyvoro"` and per-cell metrics.
        Meaning:
        The task returns pyvoro-derived Voronoi metrics per selected atom/frame.
        """
        return _run_voronoi(data, request, backend="pyvoro", reporter=reporter)


@register_task("get_voronoi_geometry_scipy", label="Voronoi Geometry (SciPy)")
class VoronoiGeometryScipyTask(AnalysisTask):
    """Compute per-atom Voronoi geometry using SciPy.

    Face connectivity and neighbor mapping are reconstructed from SciPy ridge
    structures to produce per-cell geometry records.
    """

    required_data = TrajectoryData
    recommended_presentations = staticmethod(_recommended_presentations)

    def run(self, data: TrajectoryData, request: VoronoiRequest, reporter=None) -> VoronoiGeometryResult:
        """Compute Voronoi geometry table using the SciPy backend.

        Works on
        -----
        `TrajectoryData` plus `VoronoiRequest` analyzer inputs

        Parameters
        -----
        data : TrajectoryData
            Trajectory coordinates, atom metadata, and optional iterations.
        request : VoronoiRequest
            Atom/frame selection and backend configuration.
        reporter : Any, optional
            Optional progress callback for frame-wise processing.

        Returns
        -----
        VoronoiGeometryResult
            Geometry-rich Voronoi table and request echo.

        Examples
        -----
        ```python
        result = VoronoiGeometryScipyTask().run(data, req)
        ```
        Sample output:
        Rows containing `vertices`, `faces`, and neighbor atom IDs per cell.
        Meaning:
        Each row captures full geometric context for one Voronoi cell.
        """
        return _run_voronoi_geometry(data, request, backend="scipy", reporter=reporter)


@register_task("get_voronoi_geometry_pyvoro", label="Voronoi Geometry (pyvoro)")
class VoronoiGeometryPyvoroTask(AnalysisTask):
    """Compute per-atom Voronoi geometry using pyvoro native cell outputs."""

    required_data = TrajectoryData
    recommended_presentations = staticmethod(_recommended_presentations)

    def run(self, data: TrajectoryData, request: VoronoiRequest, reporter=None) -> VoronoiGeometryResult:
        """Compute Voronoi geometry table using the pyvoro backend.

        Works on
        -----
        `TrajectoryData` plus `VoronoiRequest` analyzer inputs

        Parameters
        -----
        data : TrajectoryData
            Trajectory coordinates, atom metadata, and optional iterations.
        request : VoronoiRequest
            Atom/frame selection and backend configuration.
        reporter : Any, optional
            Optional progress callback for frame-wise processing.

        Returns
        -----
        VoronoiGeometryResult
            Geometry-rich Voronoi table and request echo.

        Examples
        -----
        ```python
        result = VoronoiGeometryPyvoroTask().run(data, req)
        ```
        Sample output:
        Rows containing pyvoro-native cell geometry fields.
        Meaning:
        The output is suitable for geometry-aware Voronoi visualization.
        """
        return _run_voronoi_geometry(data, request, backend="pyvoro", reporter=reporter)


__all__ = [
    "VoronoiRequest",
    "VoronoiResult",
    "VoronoiGeometryResult",
    "VoronoiScipyTask",
    "VoronoiPyvoroTask",
    "VoronoiGeometryScipyTask",
    "VoronoiGeometryPyvoroTask",
]
