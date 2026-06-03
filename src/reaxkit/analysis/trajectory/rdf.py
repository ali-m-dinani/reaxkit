"""Compute radial-distribution-function analyzers from trajectory data.

This module generates RDF curves and derived RDF properties for selected atom
groups across chosen frame ranges. It is scoped to pair-distribution analysis
and does not perform broader structural classification.

**Usage context**

- Local-structure analysis: Quantify pair correlations via `g(r)` curves.
- Peak/area metrics: Extract first-peak and integral-style RDF descriptors.
- Comparative studies: Compare RDF behavior across atom-group selections.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import TrajectoryData
from reaxkit.presentation.specs import PresentationSpec

_RDF_PROPERTY_CHOICES = ("first_peak", "dominant_peak", "area", "excess_area")


@dataclass
class RDFRequest(BaseRequest):
    """Request payload for RDF curve analysis.

    Captures atom-group selection, frame sampling, radial discretization, and
    backend options for radial distribution function evaluation.

    Fields
    -----
    atom_ids_a : Optional[Sequence[int]]
        Atom IDs defining group A. If `None`, group A may be selected by types
        or default to all atoms.
    atom_ids_b : Optional[Sequence[int]]
        Atom IDs defining group B. If `None`, group B may be selected by types
        or default to all atoms.
    atom_types_a : Optional[Sequence[str]]
        Atom types for group A when `atom_ids_a` is not provided.
    atom_types_b : Optional[Sequence[str]]
        Atom types for group B when `atom_ids_b` is not provided.
    frames : Optional[Sequence[int]]
        Frame indices to include. `None` means all frames.
    every : int
        Stride over selected frames. Must be `>= 1`. Default is `1`.
    bins : int
        Number of radial bins. Must be positive. Default is `200`.
    r_max : Optional[float]
        Maximum RDF radius. `None` uses backend-specific default logic.
    backend : str
        RDF backend name, `"freud"` or `"ovito"`.

    Examples
    -----
    ```python
    req = RDFRequest(atom_types_a=["O"], atom_types_b=["H"], bins=300, backend="freud")
    ```
    Sample output:
    `RDFRequest(...)`
    Meaning:
    The request configures an O-H partial RDF with selected discretization.
    """

    atom_ids_a: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Group A atom IDs", "help": "Atom IDs for group A. Empty means all atoms.", "units": "index"},
    )
    atom_ids_b: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Group B atom IDs", "help": "Atom IDs for group B. Empty means all atoms.", "units": "index"},
    )
    atom_types_a: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={"label": "Group A atom types", "help": "Element symbols for group A when atom_ids_a is empty."},
    )
    atom_types_b: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={"label": "Group B atom types", "help": "Element symbols for group B when atom_ids_b is empty."},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Frame indices to include. Empty means all frames.", "units": "frame_index"},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Stride", "help": "Stride over selected frames.", "min": 1, "units": "frames"},
    )
    bins: int = dc_field(
        default=200,
        metadata={"label": "Number of bins", "help": "Number of radial bins.", "min": 1, "max": 5000, "units": "bins"},
    )
    r_max: Optional[float] = dc_field(
        default=None,
        metadata={
            "label": "Maximum radius",
            "help": "Maximum radius. Empty uses half of the shortest box length.",
            "min": 0.0,
            "max": 20.0,
            "units": "distance",
        },
    )
    backend: str = dc_field(
        default="freud",
        metadata={"label": "RDF backend", "help": "RDF computation backend.", "choices": ["freud", "ovito"]},
    )


@dataclass
class RDFResult(BaseResult):
    """Result payload for RDF curve analysis.

    Stores per-frame RDF samples and the originating request configuration.

    Fields
    -----
    table : pd.DataFrame
        Output table with columns `["frame_index", "iter", "r", "g"]` where
        `r` is radial position and `g` is RDF value.
    request : RDFRequest
        Request object used for this RDF analysis run.

    Examples
    -----
    ```python
    result = RDFTask().run(data, req)
    print(result.table.head())
    ```
    Sample output:
    DataFrame rows with one `g(r)` sample per radial bin and frame.
    Meaning:
    The table contains the discretized RDF curve(s) for selected frames.
    """

    table: pd.DataFrame
    request: RDFRequest


@dataclass
class RDFPropertyRequest(BaseRequest):
    """Request payload for RDF-derived property extraction.

    Configures which single RDF property to extract and reuses RDF selection/
    discretization controls for per-frame property computation.

    Fields
    -----
    property : str
        RDF property name: `"first_peak"`, `"dominant_peak"`, `"area"`, or
        `"excess_area"`.
    atom_ids_a : Optional[Sequence[int]]
        Atom IDs for RDF group A.
    atom_ids_b : Optional[Sequence[int]]
        Atom IDs for RDF group B.
    atom_types_a : Optional[Sequence[str]]
        Atom types for group A when `atom_ids_a` is not provided.
    atom_types_b : Optional[Sequence[str]]
        Atom types for group B when `atom_ids_b` is not provided.
    frames : Optional[Sequence[int]]
        Frame indices to include. `None` means all frames.
    every : int
        Stride over selected frames. Must be `>= 1`. Default is `1`.
    bins : int
        Number of radial bins used for RDF calculation.
    r_max : Optional[float]
        Maximum RDF radius. `None` uses backend-specific default logic.
    backend : str
        RDF backend name, `"freud"` or `"ovito"`.

    Examples
    -----
    ```python
    req = RDFPropertyRequest(property="first_peak", atom_types_a=["O"], atom_types_b=["O"])
    ```
    Sample output:
    `RDFPropertyRequest(...)`
    Meaning:
    The request asks for frame-resolved first-peak properties for O-O RDF.
    """

    property: str = dc_field(
        default="first_peak",
        metadata={
            "label": "Property",
            "help": "RDF-derived property to compute from g(r).",
            "choices": ["first_peak", "dominant_peak", "area", "excess_area"],
        },
    )
    atom_ids_a: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Group A atom IDs", "help": "Atom IDs for group A. Empty means all atoms.", "units": "index"},
    )
    atom_ids_b: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Group B atom IDs", "help": "Atom IDs for group B. Empty means all atoms.", "units": "index"},
    )
    atom_types_a: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={"label": "Group A atom types", "help": "Element symbols for group A when atom_ids_a is empty."},
    )
    atom_types_b: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={"label": "Group B atom types", "help": "Element symbols for group B when atom_ids_b is empty."},
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={"label": "Frames", "help": "Frame indices to include. Empty means all frames.", "units": "frame_index"},
    )
    every: int = dc_field(
        default=1,
        metadata={"label": "Stride", "help": "Stride over selected frames.", "min": 1, "units": "frames"},
    )
    bins: int = dc_field(
        default=200,
        metadata={"label": "Number of bins", "help": "Number of radial bins.", "min": 1, "max": 5000, "units": "bins"},
    )
    r_max: Optional[float] = dc_field(
        default=None,
        metadata={
            "label": "Maximum radius",
            "help": "Maximum radius. Empty uses half of the shortest box length.",
            "min": 0.0,
            "max": 20.0,
            "units": "distance",
        },
    )
    backend: str = dc_field(
        default="freud",
        metadata={"label": "RDF backend", "help": "RDF computation backend.", "choices": ["freud", "ovito"]},
    )


@dataclass
class RDFPropertyResult(BaseResult):
    """Result payload for RDF-derived property analysis.

    Stores frame-wise derived RDF properties and the request used to compute
    them.

    Fields
    -----
    table : pd.DataFrame
        Output table always includes `["frame_index", "iter"]` plus
        property-specific columns:
        - `first_peak` -> `["r_first_peak", "g_first_peak"]`
        - `dominant_peak` -> `["r_peak", "g_peak"]`
        - `area` -> `["area"]`
        - `excess_area` -> `["excess_area"]`
    request : RDFPropertyRequest
        Request object used for this property analysis run.

    Examples
    -----
    ```python
    result = RDFPropertyTask().run(data, req)
    result.table.columns
    ```
    Sample output:
    Index with `frame_index`, `iter`, and selected property columns.
    Meaning:
    Rows map each analyzed frame to its derived RDF property value(s).
    """

    table: pd.DataFrame
    request: RDFPropertyRequest


def _normalize_property_selection(properties: Optional[Sequence[str]]) -> list[str]:
    if properties is None:
        selected = list(_RDF_PROPERTY_CHOICES)
    elif isinstance(properties, str):
        selected = [properties]
    else:
        selected = [str(p) for p in properties]

    normalized: list[str] = []
    for item in selected:
        key = str(item).strip().lower()
        if key == "all":
            return list(_RDF_PROPERTY_CHOICES)
        if key and key not in normalized:
            normalized.append(key)
    if not normalized:
        return list(_RDF_PROPERTY_CHOICES)

    invalid = [p for p in normalized if p not in _RDF_PROPERTY_CHOICES]
    if invalid:
        raise ValueError(f"Unknown RDF property selection: {invalid}. Allowed: {list(_RDF_PROPERTY_CHOICES)}")
    return normalized


def _rdf_properties_for_curve(r: np.ndarray, g: np.ndarray, selected: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    selected_set = set(selected)
    if "first_peak" in selected_set:
        rp, gp = _first_local_max(r, g)
        out["r_first_peak"] = rp
        out["g_first_peak"] = gp
    if "dominant_peak" in selected_set:
        rp, gp = _dominant_peak(r, g)
        out["r_peak"] = rp
        out["g_peak"] = gp
    if "area" in selected_set:
        out["area"] = float(np.trapezoid(g, r)) if len(r) else np.nan
    if "excess_area" in selected_set:
        out["excess_area"] = float(np.trapezoid(g - 1.0, r)) if len(r) else np.nan
    return out


def _build_properties_table(
    data: TrajectoryData,
    *,
    r_ref: np.ndarray,
    stack: list[np.ndarray],
    frame_idx: list[int],
    selected_properties: Sequence[str],
) -> pd.DataFrame:
    if len(r_ref) == 0 or not stack:
        cols = ["frame_index", "iter"]
        sample = _rdf_properties_for_curve(np.array([], dtype=float), np.array([], dtype=float), selected_properties)
        cols.extend(list(sample.keys()))
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, float | int]] = []
    for j, i in enumerate(frame_idx):
        r = r_ref
        g = stack[j]
        out = _rdf_properties_for_curve(r, g, selected_properties)
        iter_val = int(data.iterations[i]) if data.iterations is not None else int(i)
        rows.append({"frame_index": int(i), "iter": iter_val, **out})
    return pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)


def _dominant_peak(r: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    if len(g) == 0:
        return float("nan"), float("nan")
    k = int(np.argmax(g))
    return float(r[k]), float(g[k])


def _first_local_max(r: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    n = len(g)
    for i in range(1, n - 1):
        if g[i] > g[i - 1] and g[i] > g[i + 1]:
            return float(r[i]), float(g[i])
    return _dominant_peak(r, g)


def _mask_from_request(
    data: TrajectoryData,
    ids: Optional[Sequence[int]],
    types: Optional[Sequence[str]],
) -> np.ndarray:
    n_atoms = data.positions.shape[1]
    if ids is not None:
        selected = set(int(i) for i in ids)
        return np.array([aid in selected for aid in data.atom_ids], dtype=bool)
    if types:
        tset = {str(t) for t in types}
        return np.array([str(t) in tset for t in data.elements], dtype=bool)
    return np.ones(n_atoms, dtype=bool)


def _frame_grid_and_rdf_freud(
    data: TrajectoryData,
    frame_index: int,
    *,
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    bins: int,
    r_max: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import freud
    except Exception as e:
        raise ImportError(
            "freud backend is not available. Install it or set backend='ovito'."
        ) from e

    sim = data.simulation
    if sim is None or sim.cell_lengths is None:
        raise ValueError("TrajectoryData.simulation.cell_lengths is required for RDF analysis.")

    cell_l = np.asarray(sim.cell_lengths[int(frame_index)], dtype=float)
    if cell_l.shape[0] != 3:
        raise ValueError("cell_lengths must have shape (n_frames, 3).")

    if sim.cell_angles is not None:
        cell_a = np.asarray(sim.cell_angles[int(frame_index)], dtype=float)
        if cell_a.shape[0] != 3:
            raise ValueError("cell_angles must have shape (n_frames, 3).")
        alpha, beta, gamma = cell_a
    else:
        alpha = beta = gamma = 90.0

    a, b, c = float(cell_l[0]), float(cell_l[1]), float(cell_l[2])
    half_box = 0.5 * float(min(a, b, c))
    # Freud AABBQuery can fail when r_max is at/above half-box; keep a small safety margin.
    max_safe_r = max(0.0, half_box - 1e-6)
    if max_safe_r <= 0.0:
        raise ValueError("Cell is too small for RDF cutoff selection.")
    if r_max is None:
        r_eff = max_safe_r
    else:
        try:
            r_eff = float(r_max)
        except Exception:
            r_eff = max_safe_r
        if r_eff > max_safe_r:
            r_eff = max_safe_r
        if r_eff <= 0.0:
            raise ValueError("r_max must be > 0.")

    box = freud.box.Box.from_box_lengths_and_angles(
        a, b, c, np.radians(float(alpha)), np.radians(float(beta)), np.radians(float(gamma))
    )

    coords = data.positions[int(frame_index)].astype(float)
    a_coords = coords[a_mask]
    b_coords = coords[b_mask]

    rdf = freud.density.RDF(bins=int(bins), r_max=float(r_eff))
    rdf.compute((box, b_coords), query_points=a_coords)

    r = rdf.bin_centers.copy()
    g = rdf.rdf.copy()
    if len(r) > 1:
        r, g = r[1:], g[1:]
    return r, g


def _write_xyz_temp(coords: np.ndarray, types: Sequence[str]) -> str:
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".xyz", prefix="reaxkit_rdf_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{coords.shape[0]}\n")
        f.write("generated by reaxkit RDF\n")
        for t, (x, y, z) in zip(types, coords):
            f.write(f"{str(t)} {x:.9f} {y:.9f} {z:.9f}\n")
    return path


def _frame_grid_and_rdf_ovito(
    data: TrajectoryData,
    frame_index: int,
    *,
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    bins: int,
    r_max: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    import os

    try:
        from ovito.io import import_file
        from ovito.modifiers import CoordinationAnalysisModifier
    except Exception as e:
        raise ImportError(
            "OVITO is required for RDF analysis. Install reaxkit[viz] and ensure OVITO Python modules are available."
        ) from e

    coords = data.positions[int(frame_index)].astype(float)
    types = np.asarray(data.elements, dtype=str)
    if types.shape[0] != coords.shape[0]:
        raise ValueError("TrajectoryData.elements length must match atom dimension in positions.")

    has_custom_selection = (not np.all(a_mask)) or (not np.all(b_mask))
    if has_custom_selection:
        labels = np.full(coords.shape[0], "X", dtype=object)
        labels[b_mask] = "B"
        labels[a_mask] = "A"
        xyz_types = labels.tolist()
        want_partial = True
        pair_name = "A-B"
    else:
        xyz_types = types.tolist()
        want_partial = False
        pair_name = ""

    xyz = _write_xyz_temp(coords, xyz_types)
    cutoff = float(r_max) if r_max is not None else 4.0

    try:
        pipe = import_file(xyz)
        pipe.modifiers.append(
            CoordinationAnalysisModifier(
                cutoff=cutoff,
                number_of_bins=int(bins),
                partial=want_partial,
            )
        )
        out = pipe.compute()
        table = out.tables["coordination-rdf"]
        arr = table.xy()
        r = np.asarray(arr[:, 0], dtype=float)

        if not want_partial:
            g = np.asarray(arr[:, 1], dtype=float)
            return r, g

        try:
            names = list(table.y.component_names)
        except Exception:
            names = list(getattr(table.y, "components", []))

        if pair_name not in names and "B-A" in names:
            pair_name = "B-A"
        idx = names.index(pair_name)
        g = np.asarray(arr[:, 1 + idx], dtype=float)
        return r, g
    finally:
        try:
            os.remove(xyz)
        except OSError:
            pass


def _compute_rdfs(
    data: TrajectoryData,
    *,
    atom_ids_a: Optional[Sequence[int]],
    atom_ids_b: Optional[Sequence[int]],
    atom_types_a: Optional[Sequence[str]],
    atom_types_b: Optional[Sequence[str]],
    frames: Optional[Sequence[int]],
    every: int,
    bins: int,
    r_max: Optional[float],
    backend: str,
    reporter=None,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    backend_l = str(backend).lower()
    if backend_l not in {"freud", "ovito"}:
        raise ValueError("backend must be 'freud' or 'ovito'")

    n_frames = data.positions.shape[0]
    frame_idx = list(range(n_frames)) if frames is None else [int(i) for i in frames]
    frame_idx = frame_idx[:: max(1, int(every))]
    if not frame_idx:
        return np.array([]), [], []

    a_mask = _mask_from_request(data, atom_ids_a, atom_types_a)
    b_mask = _mask_from_request(data, atom_ids_b, atom_types_b)
    if not np.any(a_mask) or not np.any(b_mask):
        return np.array([]), [], frame_idx

    r_ref: np.ndarray | None = None
    stack: list[np.ndarray] = []

    total = len(frame_idx)
    for step_i, i in enumerate(frame_idx, start=1):
        if backend_l == "freud":
            r, g = _frame_grid_and_rdf_freud(
                data,
                i,
                a_mask=a_mask,
                b_mask=b_mask,
                bins=int(bins),
                r_max=r_max,
            )
        else:
            r, g = _frame_grid_and_rdf_ovito(
                data,
                i,
                a_mask=a_mask,
                b_mask=b_mask,
                bins=int(bins),
                r_max=r_max,
            )

        if r_ref is None:
            r_ref = r
        elif len(r) != len(r_ref) or np.max(np.abs(r - r_ref)) > 1e-10:
            raise ValueError("R grids differ between frames; fix bins/r_max.")
        stack.append(g)
        if reporter:
            reporter("analyze", step_i, total, "Computing RDF")

    if r_ref is None:
        return np.array([]), [], frame_idx
    return r_ref, stack, frame_idx


@register_task("get_rdf", label="RDF")
class RDFTask(AnalysisTask):
    """RDF curve task (total/partial)."""

    required_data = TrajectoryData

    @staticmethod
    def recommended_presentations(_result: RDFResult, payload: dict[str, object]) -> list[PresentationSpec]:
        """Build default table/plot presentations for RDF curve outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : RDFResult
            Analysis result object for the executed task.
        payload : dict[str, object]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for table and `g(r)` plotting.

        Examples
        -----
        ```python
        specs = RDFTask.recommended_presentations(result, payload)
        ```
        Sample output:
        A table view and an RDF line-plot view.
        Meaning:
        The payload can be rendered as numeric table and radial profile plot.
        """
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        if "r" not in sample or "g" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "frame_index" if "frame_index" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="RDF g(r)",
                mapping={"x_col": "r", "y_col": "g", "group_by_col": group_by},
                options={"title": "RDF g(r)", "xlabel": "r", "ylabel": "g", "legend": bool(group_by)},
                view_type="plot2d",
            ),
        ]

    def run(self, data: TrajectoryData, request: RDFRequest, reporter=None) -> RDFResult:
        """Compute RDF curves for selected atom groups across sampled frames.

        Works on
        -----
        `TrajectoryData` plus `RDFRequest` analyzer inputs

        Parameters
        -----
        data : TrajectoryData
            Trajectory coordinates, atom metadata, and periodic-cell metadata.
        request : RDFRequest
            RDF configuration including groups, bins, radius, and backend.
        reporter : Any, optional
            Optional progress callback invoked during frame processing.

        Returns
        -----
        RDFResult
            Result object containing per-frame RDF sample rows.

        Examples
        -----
        ```python
        req = RDFRequest(atom_types_a=["O"], atom_types_b=["H"], backend="freud")
        result = RDFTask().run(data, req)
        ```
        Sample output:
        `result.table` with columns `frame_index`, `iter`, `r`, `g`.
        Meaning:
        Each frame contributes one discretized `g(r)` curve to the table.
        """
        r_ref, stack, frame_idx = _compute_rdfs(
            data,
            atom_ids_a=request.atom_ids_a,
            atom_ids_b=request.atom_ids_b,
            atom_types_a=request.atom_types_a,
            atom_types_b=request.atom_types_b,
            frames=request.frames,
            every=request.every,
            bins=request.bins,
            r_max=request.r_max,
            backend=request.backend,
            reporter=reporter,
        )
        if len(r_ref) == 0 or not stack:
            return RDFResult(table=pd.DataFrame(columns=["frame_index", "iter", "r", "g"]), request=request)
        rows: list[dict[str, float | int]] = []
        for j, i in enumerate(frame_idx):
            iter_val = int(data.iterations[i]) if data.iterations is not None else int(i)
            for rr, gg in zip(r_ref, stack[j]):
                rows.append(
                    {
                        "frame_index": int(i),
                        "iter": iter_val,
                        "r": float(rr),
                        "g": float(gg),
                    }
                )
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["frame_index", "r"], kind="stable").reset_index(drop=True)
        return RDFResult(table=table, request=request)


@register_task("get_rdf_property", label="RDF Property")
class RDFPropertyTask(AnalysisTask):
    """RDF-derived property task."""

    required_data = TrajectoryData

    @staticmethod
    def recommended_presentations(_result: RDFPropertyResult, payload: dict[str, object]) -> list[PresentationSpec]:
        """Build default table/plot presentations for RDF property outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : RDFPropertyResult
            Analysis result object for the executed task.
        payload : dict[str, object]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for property tables and line plots.

        Examples
        -----
        ```python
        specs = RDFPropertyTask.recommended_presentations(result, payload)
        ```
        Sample output:
        A list containing table view and one property-vs-time plot view.
        Meaning:
        The selected property can be visualized directly over frames/iterations.
        """
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        y_candidates = ("r_first_peak", "g_first_peak", "r_peak", "g_peak", "area", "excess_area")
        y_axis = next((name for name in y_candidates if name in sample), "")
        if not y_axis:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_axis} vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": y_axis, "group_by_col": ""},
                options={"title": f"{y_axis} vs {x_axis}", "xlabel": x_axis, "ylabel": y_axis, "legend": False},
                view_type="plot2d",
            ),
        ]

    def run(self, data: TrajectoryData, request: RDFPropertyRequest, reporter=None) -> RDFPropertyResult:
        """Compute one selected RDF-derived property per sampled frame.

        Works on
        -----
        `TrajectoryData` plus `RDFPropertyRequest` analyzer inputs

        Parameters
        -----
        data : TrajectoryData
            Trajectory coordinates and metadata used to generate RDF curves.
        request : RDFPropertyRequest
            Property selection and RDF computation configuration.
        reporter : Any, optional
            Optional progress callback invoked during RDF evaluation.

        Returns
        -----
        RDFPropertyResult
            Result object containing frame-wise derived RDF property rows.

        Examples
        -----
        ```python
        req = RDFPropertyRequest(property="dominant_peak")
        result = RDFPropertyTask().run(data, req)
        ```
        Sample output:
        `result.table` with columns such as `r_peak` and `g_peak`.
        Meaning:
        Each row summarizes the selected RDF property for one analyzed frame.
        """
        prop = _normalize_property_selection([request.property])[0]
        r_ref, stack, frame_idx = _compute_rdfs(
            data,
            atom_ids_a=request.atom_ids_a,
            atom_ids_b=request.atom_ids_b,
            atom_types_a=request.atom_types_a,
            atom_types_b=request.atom_types_b,
            frames=request.frames,
            every=request.every,
            bins=request.bins,
            r_max=request.r_max,
            backend=request.backend,
            reporter=reporter,
        )
        properties_table = _build_properties_table(
            data,
            r_ref=r_ref,
            stack=stack,
            frame_idx=frame_idx,
            selected_properties=[prop],
        )
        if properties_table.empty:
            return RDFPropertyResult(table=pd.DataFrame(), request=request)
        return RDFPropertyResult(table=properties_table.copy(), request=request)


__all__ = [
    "RDFRequest",
    "RDFResult",
    "RDFTask",
    "RDFPropertyRequest",
    "RDFPropertyResult",
    "RDFPropertyTask",
]
