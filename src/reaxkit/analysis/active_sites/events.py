"""Extract active-site event descriptors from trajectory and connectivity snapshots.

This module builds frame-resolved active-site event tables from geometry and
bond-order inputs used by the active-sites pipeline. It focuses on event
assembly and export, while structural site classification is handled by
`reaxkit.analysis.active_sites.structural`.

**Usage context**

- Active-site screening: Track likely reactive events over selected frames.
- TrACT interoperability: Emit tables mapped to TrACT-compatible event schema.
- Reporting workflows: Provide normalized event rows for plotting and summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reaxkit.analysis.active_sites.models import ActiveSiteEventsRequest, ActiveSiteEventsResult
from reaxkit.analysis.active_sites.pbc import frame_cell_matrix, pairwise_min_image_distances
from reaxkit.analysis.active_sites.tract_compat import to_tract_events_table
from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.data_models import ConnectivityTrajectoryData, TrajectoryData
from reaxkit.presentation.specs import PresentationSpec


def _as_dense_frame(frame_obj: Any) -> np.ndarray:
    if isinstance(frame_obj, np.ndarray):
        return frame_obj.astype(float)
    if hasattr(frame_obj, "toarray"):
        return np.asarray(frame_obj.toarray(), dtype=float)
    if hasattr(frame_obj, "todense"):
        return np.asarray(frame_obj.todense(), dtype=float)
    return np.asarray(frame_obj, dtype=float)


def _bond_order_frame(data: ConnectivityTrajectoryData, frame_index: int) -> np.ndarray:
    bo = data.connectivity.bond_orders
    if bo is None:
        raise ValueError("ConnectivityData.bond_orders is required for bo mode.")
    if isinstance(bo, np.ndarray):
        if bo.ndim != 3:
            raise ValueError("bond_orders ndarray must have shape (n_frames, n_atoms, n_atoms).")
        return _as_dense_frame(bo[frame_index])
    if isinstance(bo, (list, tuple)):
        return _as_dense_frame(bo[frame_index])
    raise TypeError("Unsupported bond_orders type.")


def merge_active_site_tables(
    structural_table: pd.DataFrame,
    events_table: pd.DataFrame,
) -> pd.DataFrame:
    """Merge structural and event active-site tables on `atom_id`.

    Applies left-join semantics from structural rows and normalizes key event
    columns to deterministic defaults and dtypes.

    Parameters
    -----
    structural_table : pd.DataFrame
        Structural active-site table keyed by `atom_id`.
    events_table : pd.DataFrame
        Events active-site table keyed by `atom_id`.

    Returns
    -----
    pd.DataFrame
        Merged table with filled event counts/flags/frame markers.

    Examples
    -----
    ```python
    merged = merge_active_site_tables(structural_df, events_df)
    ```
    Sample output:
    DataFrame with structural columns plus normalized event columns.
    Meaning:
    Each structural atom row gains event metrics when available.
    """
    merged = structural_table.merge(events_table, on="atom_id", how="left")
    for col in ["n_events_O", "n_events_Si", "total_bound_frames_O", "total_bound_frames_Si"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    for col in ["is_reactive_O", "is_reactive_Si", "is_reactive_any"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False)
    for col in ["first_event_frame_O", "first_event_frame_Si"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(-1).astype(int)
    return merged


def _fort7_available(args: dict) -> bool:
    candidates: list[Path] = []
    raw_fort7 = args.get("fort7")
    if raw_fort7:
        p = Path(str(raw_fort7))
        candidates.append(p / "fort.7" if p.is_dir() else p)

    for key in ("run_dir", "input"):
        raw = args.get(key)
        if not raw:
            continue
        p = Path(str(raw))
        if p.is_dir():
            candidates.append(p / "fort.7")

    candidates.append(Path("fort.7"))
    return any(path.exists() for path in candidates)


def _rkf_like_input(args: dict) -> bool:
    def _is_kf(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in {".rkf", ".kf"}

    for key in ("rkf", "kf", "ams_kf", "ams_rkf"):
        raw = args.get(key)
        if raw:
            p = Path(str(raw))
            if _is_kf(p):
                return True

    raw_input = args.get("input")
    if raw_input:
        p = Path(str(raw_input))
        if _is_kf(p):
            return True
        if p.is_dir() and (list(p.glob("*.rkf")) or list(p.glob("*.kf"))):
            return True
    return False


@register_task("active_site_events", label="Active Site Events")
class ActiveSiteEventsTask(AnalysisTask):
    """Extract persistent C-O and C-Si active-site events over trajectory frames."""

    required_data = ConnectivityTrajectoryData

    def required_data_for(self, request: object, args: dict | None = None):
        """Resolve required input type for event task execution mode.

        Works on
        -----
        Active-site events request objects and optional executor arguments

        Parameters
        -----
        request : object
            Request object that may specify detection `mode`.
        args : dict | None, optional
            Optional execution argument map used for auto-mode inference.

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
        `ConnectivityTrajectoryData`, `TrajectoryData`, or tuple fallback.
        Meaning:
        Required data adapts to BO/distance availability and execution context.
        """
        mode = str(getattr(request, "mode", "auto")).strip().lower()
        if mode == "dist":
            if args is None:
                return (TrajectoryData, ConnectivityTrajectoryData)
            return TrajectoryData
        if mode == "bo":
            return ConnectivityTrajectoryData

        # auto mode
        if args is None:
            return (TrajectoryData, ConnectivityTrajectoryData)
        engine = str(args.get("engine") or "").strip().lower()
        if engine == "ams" or _rkf_like_input(args):
            return ConnectivityTrajectoryData
        if engine == "reaxff":
            return ConnectivityTrajectoryData if _fort7_available(args) else TrajectoryData
        if engine == "lammps":
            return TrajectoryData
        return ConnectivityTrajectoryData if _fort7_available(args) else TrajectoryData

    @staticmethod
    def recommended_presentations(
        _result: ActiveSiteEventsResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Build default table/plot presentations for event task outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : ActiveSiteEventsResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for table and event-count plot views.

        Examples
        -----
        ```python
        specs = ActiveSiteEventsTask.recommended_presentations(result, payload)
        ```
        Sample output:
        A list with table and `n_events_O vs atom_id` plot specs.
        Meaning:
        Event outputs can be rendered with default plotting metadata.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "atom_id" not in sample or "n_events_O" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="n_events_O vs atom_id",
                mapping={"x_col": "atom_id", "y_col": "n_events_O", "group_by_col": "is_reactive_O"},
                options={"title": "C-O Event Counts", "xlabel": "atom_id", "ylabel": "n_events_O", "legend": True},
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ConnectivityTrajectoryData | TrajectoryData,
        request: ActiveSiteEventsRequest,
        reporter=None,
    ) -> ActiveSiteEventsResult:
        """Extract persistent C-O and C-Si event descriptors over sampled frames.

        Selects BO- or distance-based contact mode, applies persistence filtering
        to suppress flicker, and returns per-carbon event aggregates with
        TRACT-compatible projection.

        Works on
        -----
        `ConnectivityTrajectoryData` or `TrajectoryData` plus events request

        Parameters
        -----
        data : ConnectivityTrajectoryData | TrajectoryData
            Input trajectory (and optional connectivity) for event extraction.
        request : ActiveSiteEventsRequest
            Event extraction configuration including mode, thresholds, and frame
            sampling controls.
        reporter : Any, optional
            Optional progress callback invoked during frame iteration.

        Returns
        -----
        ActiveSiteEventsResult
            Per-carbon events table, TRACT projection, and run summary.

        Examples
        -----
        ```python
        req = ActiveSiteEventsRequest(mode="auto", every=10, persist=50)
        result = ActiveSiteEventsTask().run(bundle, req)
        ```
        Sample output:
        `result.table` with reactive flags and event counts per carbon atom.
        Meaning:
        Time-resolved contacts are condensed into persistent event descriptors.
        """
        if isinstance(data, ConnectivityTrajectoryData):
            trajectory = data.trajectory
            connectivity_data = data
        elif isinstance(data, TrajectoryData):
            trajectory = data
            connectivity_data = None
        else:
            raise TypeError(
                "ActiveSiteEventsTask expects ConnectivityTrajectoryData or TrajectoryData."
            )

        positions = np.asarray(trajectory.positions, dtype=float)
        if positions.ndim != 3:
            raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")
        n_frames, n_atoms, _ = positions.shape
        if len(trajectory.atom_ids) != n_atoms:
            raise ValueError("TrajectoryData.atom_ids length must match trajectory atom count.")
        if len(trajectory.elements) != n_atoms:
            raise ValueError("TrajectoryData.elements length must match trajectory atom count.")

        frames = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames if 0 <= int(i) < n_frames]
        frames = frames[:: max(1, int(request.every))]
        if not frames:
            raise ValueError("No frames selected for event extraction.")
        frame_markers = np.arange(n_frames, dtype=int)
        if trajectory.iterations is not None:
            iters = np.asarray(trajectory.iterations, dtype=int)
            if iters.ndim == 1 and iters.shape[0] == n_frames:
                frame_markers = iters
        elif trajectory.simulation is not None and trajectory.simulation.iterations is not None:
            iters = np.asarray(trajectory.simulation.iterations, dtype=int)
            if iters.ndim == 1 and iters.shape[0] == n_frames:
                frame_markers = iters

        elements = np.asarray([str(e) for e in trajectory.elements], dtype=object)
        atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
        carbon_element = str(request.carbon_element)
        oxygen_element = str(request.oxygen_element)
        silicon_element = str(request.silicon_element)

        c_idx = np.where(elements == carbon_element)[0]
        o_idx = np.where(elements == oxygen_element)[0]
        si_idx = np.where(elements == silicon_element)[0]
        n_c = len(c_idx)
        if n_c == 0:
            empty = pd.DataFrame(
                {
                    "atom_id": np.asarray([], dtype=int),
                    "element": np.asarray([], dtype=object),
                    "n_events_O": np.asarray([], dtype=int),
                    "n_events_Si": np.asarray([], dtype=int),
                    "first_event_frame_O": np.asarray([], dtype=int),
                    "first_event_frame_Si": np.asarray([], dtype=int),
                    "is_reactive_O": np.asarray([], dtype=bool),
                    "is_reactive_Si": np.asarray([], dtype=bool),
                    "is_reactive_any": np.asarray([], dtype=bool),
                    "total_bound_frames_O": np.asarray([], dtype=int),
                    "total_bound_frames_Si": np.asarray([], dtype=int),
                    "mean_contact_O_when_bound": np.asarray([], dtype=float),
                    "mean_contact_Si_when_bound": np.asarray([], dtype=float),
                    "contact_metric": np.asarray([], dtype=object),
                }
            )
            summary = {
                "mode": str(request.mode).lower(),
                "frames_analyzed": int(len(frames)),
                "frame_first": int(frame_markers[frames[0]]),
                "frame_last": int(frame_markers[frames[-1]]),
                "every": int(request.every),
                "persist": int(request.persist),
                "n_carbon": 0,
                "n_reactive_O": 0,
                "n_reactive_Si": 0,
                "n_reactive_any": 0,
                "note": f"No atoms matched carbon_element='{carbon_element}'.",
            }
            tract_table = to_tract_events_table(empty, strict=bool(request.strict_tract))
            return ActiveSiteEventsResult(table=empty, tract_table=tract_table, summary=summary, request=request)

        mode = str(request.mode).lower()
        has_bo = bool(connectivity_data is not None and connectivity_data.connectivity.bond_orders is not None)
        if mode == "auto":
            use_bo = has_bo
        elif mode == "bo":
            if not has_bo:
                raise ValueError("mode='bo' requested but ConnectivityData.bond_orders is not available.")
            use_bo = True
        elif mode == "dist":
            use_bo = False
        else:
            raise ValueError("mode must be one of: auto, bo, dist")

        persist = max(1, int(request.persist))
        bo_th = float(request.bo_threshold)
        r_co = float(request.r_CO)
        r_csi = float(request.r_CSi)

        consec_o = np.zeros(n_c, dtype=int)
        consec_si = np.zeros(n_c, dtype=int)
        in_event_o = np.zeros(n_c, dtype=bool)
        in_event_si = np.zeros(n_c, dtype=bool)
        n_events_o = np.zeros(n_c, dtype=int)
        n_events_si = np.zeros(n_c, dtype=int)
        first_o = np.full(n_c, -1, dtype=int)
        first_si = np.full(n_c, -1, dtype=int)
        total_bound_o = np.zeros(n_c, dtype=int)
        total_bound_si = np.zeros(n_c, dtype=int)
        sum_contact_o = np.zeros(n_c, dtype=float)
        sum_contact_si = np.zeros(n_c, dtype=float)
        n_contact_o = np.zeros(n_c, dtype=int)
        n_contact_si = np.zeros(n_c, dtype=int)

        total_steps = len(frames)
        for step_i, fi in enumerate(frames, start=1):
            xyz = positions[fi]
            valid = np.isfinite(xyz).all(axis=1)
            valid_c = valid[c_idx]
            cell = frame_cell_matrix(data, fi)

            if use_bo:
                if connectivity_data is None:
                    raise ValueError("mode='bo' requires ConnectivityTrajectoryData with bond_orders.")
                bo = _bond_order_frame(connectivity_data, fi)
                if bo.shape != (n_atoms, n_atoms):
                    raise ValueError("bond-order frame shape must match trajectory atom dimension.")
                bo = np.maximum(bo, bo.T)
                bo[~np.isfinite(bo)] = 0.0
                bo[~valid, :] = 0.0
                bo[:, ~valid] = 0.0
                np.fill_diagonal(bo, 0.0)

                if len(o_idx) > 0:
                    bo_co = bo[np.ix_(c_idx, o_idx)]
                    q = bo_co > bo_th
                    bound_o = q.any(axis=1) & valid_c
                    metric_o = np.full(n_c, np.nan, dtype=float)
                    if q.any():
                        metric_o[bound_o] = np.nanmean(np.where(q[bound_o], bo_co[bound_o], np.nan), axis=1)
                else:
                    bound_o = np.zeros(n_c, dtype=bool)
                    metric_o = np.full(n_c, np.nan, dtype=float)

                if len(si_idx) > 0:
                    bo_csi = bo[np.ix_(c_idx, si_idx)]
                    q = bo_csi > bo_th
                    bound_si = q.any(axis=1) & valid_c
                    metric_si = np.full(n_c, np.nan, dtype=float)
                    if q.any():
                        metric_si[bound_si] = np.nanmean(np.where(q[bound_si], bo_csi[bound_si], np.nan), axis=1)
                else:
                    bound_si = np.zeros(n_c, dtype=bool)
                    metric_si = np.full(n_c, np.nan, dtype=float)
            else:
                c_pos = xyz[c_idx]
                if len(o_idx) > 0:
                    d = pairwise_min_image_distances(c_pos, xyz[o_idx], cell)
                    metric_o = np.nanmin(d, axis=1)
                    bound_o = (metric_o < r_co) & valid_c
                else:
                    metric_o = np.full(n_c, np.nan, dtype=float)
                    bound_o = np.zeros(n_c, dtype=bool)

                if len(si_idx) > 0:
                    d = pairwise_min_image_distances(c_pos, xyz[si_idx], cell)
                    metric_si = np.nanmin(d, axis=1)
                    bound_si = (metric_si < r_csi) & valid_c
                else:
                    metric_si = np.full(n_c, np.nan, dtype=float)
                    bound_si = np.zeros(n_c, dtype=bool)

            consec_o[bound_o] += 1
            consec_o[~bound_o] = 0
            consec_si[bound_si] += 1
            consec_si[~bound_si] = 0

            new_o = (consec_o == persist) & (~in_event_o)
            new_si = (consec_si == persist) & (~in_event_si)
            n_events_o[new_o] += 1
            n_events_si[new_si] += 1
            frame_value = int(frame_markers[fi])
            first_o[(new_o) & (first_o < 0)] = frame_value
            first_si[(new_si) & (first_si < 0)] = frame_value

            in_event_o |= new_o
            in_event_si |= new_si
            in_event_o &= bound_o
            in_event_si &= bound_si

            total_bound_o[bound_o] += 1
            total_bound_si[bound_si] += 1
            finite_o = bound_o & np.isfinite(metric_o)
            finite_si = bound_si & np.isfinite(metric_si)
            sum_contact_o[finite_o] += metric_o[finite_o]
            sum_contact_si[finite_si] += metric_si[finite_si]
            n_contact_o[finite_o] += 1
            n_contact_si[finite_si] += 1

            if reporter:
                reporter("analyze", step_i, total_steps, "Extracting active-site events")

        mean_contact_o = np.full(n_c, np.nan, dtype=float)
        mean_contact_si = np.full(n_c, np.nan, dtype=float)
        mask_o = n_contact_o > 0
        mask_si = n_contact_si > 0
        mean_contact_o[mask_o] = sum_contact_o[mask_o] / n_contact_o[mask_o]
        mean_contact_si[mask_si] = sum_contact_si[mask_si] / n_contact_si[mask_si]

        table = pd.DataFrame(
            {
                "atom_id": atom_ids[c_idx],
                "element": elements[c_idx],
                "n_events_O": n_events_o,
                "n_events_Si": n_events_si,
                "first_event_frame_O": first_o,
                "first_event_frame_Si": first_si,
                "is_reactive_O": n_events_o > 0,
                "is_reactive_Si": n_events_si > 0,
                "is_reactive_any": (n_events_o + n_events_si) > 0,
                "total_bound_frames_O": total_bound_o,
                "total_bound_frames_Si": total_bound_si,
                "mean_contact_O_when_bound": mean_contact_o,
                "mean_contact_Si_when_bound": mean_contact_si,
                "contact_metric": "bo" if use_bo else "distance_ang",
            }
        )

        summary = {
            "mode": "bo" if use_bo else "dist",
            "frames_analyzed": int(len(frames)),
            "frame_first": int(frame_markers[frames[0]]),
            "frame_last": int(frame_markers[frames[-1]]),
            "every": int(request.every),
            "persist": persist,
            "n_carbon": int(n_c),
            "n_reactive_O": int((table["is_reactive_O"]).sum()),
            "n_reactive_Si": int((table["is_reactive_Si"]).sum()),
            "n_reactive_any": int((table["is_reactive_any"]).sum()),
        }
        if use_bo:
            summary["bo_threshold"] = bo_th
        else:
            summary["r_CO"] = r_co
            summary["r_CSi"] = r_csi

        tract_table = to_tract_events_table(table, strict=bool(request.strict_tract))
        return ActiveSiteEventsResult(table=table, tract_table=tract_table, summary=summary, request=request)


__all__ = [
    "ActiveSiteEventsRequest",
    "ActiveSiteEventsResult",
    "ActiveSiteEventsTask",
    "merge_active_site_tables",
]
