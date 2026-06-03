"""Diagnose active-site event distance cutoffs from trajectory samples.

This module implements the TRACT-style diagnostic step for event extraction:
sample trajectory frames, collect nearest C-O/C-Si distances, record
close-approach episode lengths, and suggest distance/persistence thresholds.
It is scoped to analysis data generation and does not render figures directly.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from reaxkit.analysis.active_sites.models import (
    ActiveSiteEventDiagnosticsRequest,
    ActiveSiteEventDiagnosticsResult,
)
from reaxkit.analysis.active_sites.pbc import frame_cell_matrix, pairwise_min_image_distances
from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.data_models import ConnectivityTrajectoryData, TrajectoryData
from reaxkit.presentation.specs import PresentationSpec


def _selected_frames(n_frames: int, request: ActiveSiteEventDiagnosticsRequest) -> list[int]:
    frames = list(range(n_frames)) if request.frames is None else [int(i) for i in request.frames if 0 <= int(i) < n_frames]
    frames = frames[:: max(1, int(request.every))]
    return frames[: max(1, int(request.max_diag_frames))]


def _frame_markers(trajectory: TrajectoryData, n_frames: int) -> np.ndarray:
    markers = np.arange(n_frames, dtype=int)
    if trajectory.iterations is not None:
        iters = np.asarray(trajectory.iterations, dtype=int)
        if iters.ndim == 1 and iters.shape[0] == n_frames:
            return iters
    if trajectory.simulation is not None and trajectory.simulation.iterations is not None:
        iters = np.asarray(trajectory.simulation.iterations, dtype=int)
        if iters.ndim == 1 and iters.shape[0] == n_frames:
            return iters
    return markers


def _valley_cutoff(values: list[float], *, r_max: float = 3.5) -> Optional[float]:
    arr = np.asarray([v for v in values if np.isfinite(v) and v < r_max], dtype=float)
    if arr.size < 10:
        return None
    counts, edges = np.histogram(arr, bins=100)
    mids = 0.5 * (edges[:-1] + edges[1:])
    if counts.sum() <= 0:
        return None
    kernel = np.ones(5, dtype=float) / 5.0
    smooth = np.convolve(counts.astype(float), kernel, mode="same")
    first_peak = int(np.argmax(smooth))
    if first_peak >= smooth.size - 5:
        return None
    nonzero_after = np.where(smooth[first_peak:] > 0)[0]
    if nonzero_after.size < 3:
        return None
    stop = first_peak + int(nonzero_after[-1]) + 1
    valley_idx = first_peak + int(np.argmin(smooth[first_peak:stop]))
    return float(mids[valley_idx])


def _persist_suggestion(durations: list[int]) -> Optional[int]:
    if len(durations) < 5:
        return None
    arr = np.asarray(durations, dtype=int)
    short = arr[arr <= 10]
    long_ = arr[arr > 10]
    if long_.size == 0:
        return None
    return int(np.percentile(short, 95)) + 1 if short.size else 10


def _summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for species in ("C-O", "C-Si"):
        data = summary.get("species", {}).get(species, {})
        if not data:
            continue
        rows.append(
            {
                "species": species,
                "n_distance_samples": data.get("n_distance_samples", 0),
                "n_episodes": data.get("n_episodes", 0),
                "suggested_r_cut": data.get("suggested_r_cut"),
                "suggested_persist_frames": data.get("suggested_persist_frames"),
                "suggested_persist_ps": data.get("suggested_persist_ps"),
            }
        )
    return pd.DataFrame(rows)


@register_task("active_site_event_diagnostics", label="Active Site Event Diagnostics")
class ActiveSiteEventDiagnosticsTask(AnalysisTask):
    """Sample C-X distances to diagnose event-extraction thresholds."""

    VERSION = "3"
    required_data = TrajectoryData

    def required_data_for(self, request: object, args: dict | None = None):
        return TrajectoryData

    @staticmethod
    def recommended_presentations(
        _result: ActiveSiteEventDiagnosticsResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]

    def run(
        self,
        data: ConnectivityTrajectoryData | TrajectoryData,
        request: ActiveSiteEventDiagnosticsRequest,
        reporter=None,
    ) -> ActiveSiteEventDiagnosticsResult:
        if isinstance(data, ConnectivityTrajectoryData):
            trajectory = data.trajectory
            cell_source: ConnectivityTrajectoryData | TrajectoryData = data
        elif isinstance(data, TrajectoryData):
            trajectory = data
            cell_source = data
        else:
            raise TypeError(
                "ActiveSiteEventDiagnosticsTask expects TrajectoryData or ConnectivityTrajectoryData."
            )

        positions = np.asarray(trajectory.positions, dtype=float)
        if positions.ndim != 3:
            raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")
        n_frames, n_atoms, _ = positions.shape
        if len(trajectory.atom_ids) != n_atoms:
            raise ValueError("TrajectoryData.atom_ids length must match trajectory atom count.")
        if len(trajectory.elements) != n_atoms:
            raise ValueError("TrajectoryData.elements length must match trajectory atom count.")

        frames = _selected_frames(n_frames, request)
        if not frames:
            raise ValueError("No frames selected for event diagnostics.")

        elements = np.asarray([str(e) for e in trajectory.elements], dtype=object)
        atom_ids = np.asarray(trajectory.atom_ids, dtype=int)
        c_idx = np.where(elements == str(request.carbon_element))[0]
        o_idx = np.where(elements == str(request.oxygen_element))[0]
        si_idx = np.where(elements == str(request.silicon_element))[0]
        markers = _frame_markers(trajectory, n_frames)

        distance_rows: list[dict[str, Any]] = []
        episode_rows: list[dict[str, Any]] = []
        min_distances: dict[str, list[float]] = {"C-O": [], "C-Si": []}
        episode_lengths: dict[str, list[int]] = {"C-O": [], "C-Si": []}
        counters: dict[str, np.ndarray] = {
            "C-O": np.zeros(len(c_idx), dtype=int),
            "C-Si": np.zeros(len(c_idx), dtype=int),
        }

        def _record_ended(species: str, ending: np.ndarray) -> None:
            ended = counters[species][ending]
            if ended.size:
                episode_lengths[species].extend([int(v) for v in ended if int(v) > 0])

        total_steps = len(frames)
        for step_i, frame_index in enumerate(frames, start=1):
            xyz = positions[frame_index]
            valid = np.isfinite(xyz).all(axis=1)
            valid_c = valid[c_idx]
            c_pos = xyz[c_idx]
            cell = frame_cell_matrix(cell_source, frame_index)
            frame_marker = int(markers[frame_index])

            for species, target_idx in (("C-O", o_idx), ("C-Si", si_idx)):
                valid_target = valid[target_idx] if target_idx.size else np.asarray([], dtype=bool)
                if c_idx.size == 0 or target_idx.size == 0 or not valid_target.any():
                    bound = np.zeros(len(c_idx), dtype=bool)
                    ending = (~bound) & (counters[species] > 0)
                    _record_ended(species, ending)
                    counters[species][bound] += 1
                    counters[species][~bound] = 0
                    continue

                d = pairwise_min_image_distances(c_pos, xyz[target_idx][valid_target], cell)
                min_d = np.nanmin(d, axis=1)
                finite = np.isfinite(min_d) & valid_c
                min_distances[species].extend(min_d[finite].astype(float).tolist())
                for local_i, dist in enumerate(min_d):
                    if not finite[local_i]:
                        continue
                    distance_rows.append(
                        {
                            "species": species,
                            "frame_index": int(frame_index),
                            "frame_marker": frame_marker,
                            "atom_id": int(atom_ids[c_idx[local_i]]),
                            "min_distance": float(dist),
                        }
                    )

                bound = (min_d < float(request.r_probe)) & finite
                ending = (~bound) & (counters[species] > 0)
                _record_ended(species, ending)
                counters[species][bound] += 1
                counters[species][~bound] = 0

            if reporter:
                reporter("analyze", step_i, total_steps, "Diagnosing active-site event cutoffs")

        duration_ps_factor = max(1, int(request.every)) * float(request.timestep_fs) * 1.0e-3
        for species, lengths in episode_lengths.items():
            for length in lengths:
                episode_rows.append(
                    {
                        "species": species,
                        "duration_frames": int(length),
                        "duration_ps": float(length) * duration_ps_factor,
                        "r_probe": float(request.r_probe),
                    }
                )

        species_summary: dict[str, dict[str, Any]] = {}
        for species in ("C-O", "C-Si"):
            suggested_r = _valley_cutoff(min_distances[species])
            suggested_persist = _persist_suggestion(episode_lengths[species])
            duration_ps = np.asarray(episode_lengths[species], dtype=float) * duration_ps_factor
            species_summary[species] = {
                "n_distance_samples": int(len(min_distances[species])),
                "n_episodes": int(len(episode_lengths[species])),
                "n_thermal_episodes_lt_1ps": int(np.sum(duration_ps < 1.0)) if duration_ps.size else 0,
                "n_stable_episodes_ge_1ps": int(np.sum(duration_ps >= 1.0)) if duration_ps.size else 0,
                "suggested_r_cut": suggested_r,
                "suggested_persist_frames": suggested_persist,
                "suggested_persist_ps": (
                    float(suggested_persist) * duration_ps_factor
                    if suggested_persist is not None
                    else None
                ),
            }

        summary = {
            "diagnostic": True,
            "frames_analyzed": int(len(frames)),
            "frame_first": int(markers[frames[0]]),
            "frame_last": int(markers[frames[-1]]),
            "every": int(request.every),
            "r_probe": float(request.r_probe),
            "max_diag_frames": int(request.max_diag_frames),
            "timestep_fs": float(request.timestep_fs),
            "duration_scale": "physical",
            "duration_ps_factor": float(duration_ps_factor),
            "n_carbon": int(len(c_idx)),
            "n_oxygen": int(len(o_idx)),
            "n_silicon": int(len(si_idx)),
            "species": species_summary,
        }

        return ActiveSiteEventDiagnosticsResult(
            table=_summary_table(summary),
            distance_table=pd.DataFrame(distance_rows),
            episode_table=pd.DataFrame(episode_rows),
            summary=summary,
            request=request,
        )


__all__ = [
    "ActiveSiteEventDiagnosticsRequest",
    "ActiveSiteEventDiagnosticsResult",
    "ActiveSiteEventDiagnosticsTask",
]
