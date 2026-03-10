"""Engine-agnostic molecular analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import re
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import MolecularAnalysisData
from reaxkit.presentation.specs import PresentationSpec


def _selected_iterations(data: MolecularAnalysisData, frames: Optional[Sequence[int]], every: int) -> tuple[np.ndarray, np.ndarray]:
    iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
    idx = list(range(len(iterations))) if frames is None else [int(i) for i in frames]
    idx = [i for i in idx if 0 <= i < len(iterations)][:: max(1, int(every))]
    return iterations, np.asarray(idx, dtype=int)


@dataclass
class DominantSpeciesRequest(BaseRequest):
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Frames",
            "help": "Optional frame indices to evaluate. Empty means all frames.",
            "units": "frame_index",
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            "label": "Every",
            "help": "Stride over selected frames.",
            "min": 1,
            "units": "frames",
            "choices": [1, 2, 5, 10],
        },
    )
    top_n: int = dc_field(
        default=1,
        metadata={
            "label": "Top N",
            "help": "Number of dominant species retained per sampled frame/iteration.",
            "min": 1,
            "choices": [1, 3, 5, 10],
        },
    )
    min_freq: float = dc_field(
        default=0.0,
        metadata={
            "label": "Min Freq",
            "help": "Minimum species frequency threshold used before ranking.",
            "min": 0.0,
            "choices": [0.0, 1.0, 2.0],
        },
    )


@dataclass
class DominantSpeciesResult(BaseResult):
    """Dominant-species analysis result.

    Output structure:
    - request: DominantSpeciesRequest used to generate this result
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'rank', 'molecular_formula', 'freq', 'molecular_mass']
      - frame_index: sampled frame index
      - iter: simulation iteration for that frame
      - rank: per-iteration rank after sorting by frequency/mass
      - molecular_formula: species formula label
      - freq: species frequency at that iteration
      - molecular_mass: species mass value when available
    """

    table: pd.DataFrame
    request: DominantSpeciesRequest


@dataclass
class MoleculeLifetimeRequest(BaseRequest):
    molecules: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={
            "label": "Molecules",
            "help": "Optional molecular formulas to track (for example ['H2O', 'OH']). Empty means all formulas.",
            "choices": ["H2O", "OH", "CO2"],
        },
    )
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Frames",
            "help": "Optional frame indices to evaluate. Empty means all frames.",
            "units": "frame_index",
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            "label": "Every",
            "help": "Stride over selected frames.",
            "min": 1,
            "units": "frames",
            "choices": [1, 2, 5, 10],
        },
    )
    min_freq: float = dc_field(
        default=1.0,
        metadata={
            "label": "Min Freq",
            "help": "Minimum frequency threshold for a species to be considered active.",
            "min": 0.0,
            "choices": [0.0, 1.0, 2.0],
        },
    )


@dataclass
class MoleculeLifetimeResult(BaseResult):
    """Molecule-lifetime analysis result.

    Output structure:
    - request: MoleculeLifetimeRequest used to generate this result
    - table: pandas.DataFrame with columns
      ['molecular_formula', 'lifetime_segment_id', 'start_frame_index', 'end_frame_index',
       'start_iter', 'end_iter', 'lifetime_segment_sampled_step_count', 'peak_freq', 'mean_freq']
      - molecular_formula: tracked species label
      - lifetime_segment_id: contiguous active-segment index per species
        (1 for first active interval, 2 for second, ...)
      - start_frame_index/end_frame_index: active-run frame bounds
      - start_iter/end_iter: active-run iteration bounds
      - lifetime_segment_sampled_step_count: number of sampled points in the active segment
        Example:
        If sampled iterations are [0, 10, 20, 30, 40] and the molecule is
        active for iter 10, 20, 30, then lifetime_segment_sampled_step_count = 3.
      - peak_freq: maximum frequency inside the run
      - mean_freq: mean frequency inside the run
        Example:
        For frequencies [2.0, 5.0, 3.0] within one segment,
        peak_freq = 5.0 and mean_freq = 10.0 / 3.
    """

    table: pd.DataFrame
    request: MoleculeLifetimeRequest


@dataclass
class LargestMoleculeByMassRequest(BaseRequest):
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Frames",
            "help": "Optional frame indices to evaluate. Empty means all frames.",
            "units": "frame_index",
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            "label": "Every",
            "help": "Stride over selected frames. Example: every=5 evaluates frames 0,5,10,...",
            "min": 1,
            "units": "frames",
            "choices": [1, 2, 5, 10],
        },
    )


@dataclass
class LargestMoleculeByMassResult(BaseResult):
    """Largest-molecule-by-mass analysis result.

    Output structure:
    - request: LargestMoleculeByMassRequest used to generate this result
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'molecular_formula', 'freq', 'molecular_mass']
      - frame_index: sampled frame index in the trajectory subset
      - iter: simulation iteration corresponding to frame_index
      - molecular_formula: formula of the heaviest molecule at that iteration
      - freq: frequency/count of that molecule at that iteration
      - molecular_mass: mass value used for selecting the dominant-by-mass row

    Example:
    If at iter=100 the species in data are H2O (18.0) and Al2O3 (102.0),
    this table keeps Al2O3 for that iteration.
    """

    table: pd.DataFrame
    request: LargestMoleculeByMassRequest


@dataclass
class LargestMoleculeCompositionRequest(BaseRequest):
    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Frames",
            "help": "Optional frame indices to evaluate. Empty means all frames.",
            "units": "frame_index",
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            "label": "Every",
            "help": "Stride over selected frames. Example: every=5 evaluates frames 0,5,10,...",
            "min": 1,
            "units": "frames",
            "choices": [1, 2, 5, 10],
        },
    )


@dataclass
class LargestMoleculeCompositionResult(BaseResult):
    """Largest-molecule composition analysis result.

    Output structure:
    - request: LargestMoleculeCompositionRequest used to generate this result
    - table: pandas.DataFrame with columns
      ['frame_index', 'iter', 'element', 'count']
      - frame_index: sampled frame index
      - iter: simulation iteration corresponding to frame_index
      - element: element symbol parsed from the selected largest molecule formula
      - count: number of atoms of that element in the formula

    Example:
    If the largest molecule formula at one iteration is Al2O3,
    the table has two rows for that iteration:
    - element='Al', count=2
    - element='O', count=3
    """

    table: pd.DataFrame
    request: LargestMoleculeCompositionRequest


@register_task("dominant_species")
class DominantSpeciesTask(AnalysisTask):
    """Return the dominant molecular species per selected iteration."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(
        _result: DominantSpeciesResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "freq" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "molecular_formula" if "molecular_formula" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Dominant Species Frequency",
                mapping={"x_col": x_axis, "y_col": "freq", "group_by_col": group_by},
                options={
                    "title": "Dominant Species Frequency",
                    "xlabel": x_axis,
                    "ylabel": "freq",
                    "legend": bool(group_by),
                },
                view_type="plot2d",
            ),
        ]

    def run(self, data: MolecularAnalysisData, request: DominantSpeciesRequest, reporter=None) -> DominantSpeciesResult:
        df = data.molecular_species.copy()
        if df.empty:
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"]),
                request=request,
            )

        iterations, frame_idx = _selected_iterations(data, request.frames, request.every)
        selected_iters = iterations[frame_idx]
        df = df[df["iter"].isin(set(int(it) for it in selected_iters))].copy()
        if df.empty:
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"]),
                request=request,
            )

        df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0.0)
        df["molecular_mass"] = pd.to_numeric(df["molecular_mass"], errors="coerce")
        df = df[df["freq"] >= float(request.min_freq)].copy()
        if df.empty:
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"]),
                request=request,
            )

        rows = []
        frame_lookup = {int(iterations[i]): int(i) for i in frame_idx.tolist()}
        top_n = max(1, int(request.top_n))
        for iter_val, sub in df.groupby("iter", sort=True):
            ranked = sub.sort_values(["freq", "molecular_mass", "molecular_formula"], ascending=[False, False, True]).head(top_n)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "frame_index": frame_lookup[int(iter_val)],
                        "iter": int(iter_val),
                        "rank": int(rank),
                        "molecular_formula": str(row["molecular_formula"]),
                        "freq": float(row["freq"]),
                        "molecular_mass": float(row["molecular_mass"]) if pd.notna(row["molecular_mass"]) else np.nan,
                    }
                )

        out = pd.DataFrame(rows)
        if out.empty:
            out = pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"])
        else:
            out = out.sort_values(["frame_index", "rank"], kind="stable").reset_index(drop=True)
        return DominantSpeciesResult(table=out, request=request)


@register_task("largest_molecule_by_mass")
class LargestMoleculeByMassTask(AnalysisTask):
    """Return the heaviest individual molecular species per selected iteration."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(
        _result: LargestMoleculeByMassResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "molecular_mass" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "molecular_formula" if "molecular_formula" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Largest Molecule Mass",
                mapping={"x_col": x_axis, "y_col": "molecular_mass", "group_by_col": group_by},
                options={
                    "title": "Largest Molecule By Mass",
                    "xlabel": x_axis,
                    "ylabel": "molecular_mass",
                    "legend": bool(group_by),
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: MolecularAnalysisData,
        request: LargestMoleculeByMassRequest,
        reporter=None,
    ) -> LargestMoleculeByMassResult:
        columns = ["frame_index", "iter", "molecular_formula", "freq", "molecular_mass"]
        df = data.molecular_species.copy()
        if df.empty:
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns), request=request)

        iterations, frame_idx = _selected_iterations(data, request.frames, request.every)
        selected_iters = iterations[frame_idx]
        if selected_iters.size == 0:
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns), request=request)

        df = df[df["iter"].isin(set(int(it) for it in selected_iters))].copy()
        if df.empty:
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns), request=request)

        df["molecular_mass"] = pd.to_numeric(df["molecular_mass"], errors="coerce")
        df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0.0)
        df = df[df["molecular_mass"].notna()].copy()
        if df.empty:
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns), request=request)

        idx = df.groupby("iter")["molecular_mass"].idxmax()
        out = df.loc[idx, ["iter", "molecular_formula", "freq", "molecular_mass"]].copy()
        frame_lookup = {int(iterations[i]): int(i) for i in frame_idx.tolist()}
        out.insert(0, "frame_index", out["iter"].map(lambda it: frame_lookup.get(int(it), -1)).astype(int))
        out = out.sort_values(["frame_index", "iter"], kind="stable").reset_index(drop=True)
        return LargestMoleculeByMassResult(table=out[columns], request=request)


@register_task("largest_molecule_composition")
class LargestMoleculeCompositionTask(AnalysisTask):
    """Return per-element composition of the heaviest molecule per selected iteration."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(
        _result: LargestMoleculeCompositionResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "iter" if "iter" in sample else ("frame_index" if "frame_index" in sample else "")
        if not x_axis or "count" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "element" if "element" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Largest Molecule Composition",
                mapping={"x_col": x_axis, "y_col": "count", "group_by_col": group_by},
                options={
                    "title": "Largest Molecule Composition",
                    "xlabel": x_axis,
                    "ylabel": "count",
                    "legend": bool(group_by),
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: MolecularAnalysisData,
        request: LargestMoleculeCompositionRequest,
        reporter=None,
    ) -> LargestMoleculeCompositionResult:
        largest = LargestMoleculeByMassTask().run(
            data,
            LargestMoleculeByMassRequest(frames=request.frames, every=request.every),
            reporter=reporter,
        ).table
        if largest.empty:
            return LargestMoleculeCompositionResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "element", "count"]),
                request=request,
            )

        rows = []
        for _, row in largest.iterrows():
            pairs = re.findall(r"([A-Z][a-z]*)(\d+)", str(row["molecular_formula"]))
            for element, count in pairs:
                rows.append(
                    {
                        "frame_index": int(row["frame_index"]),
                        "iter": int(row["iter"]),
                        "element": str(element),
                        "count": int(count),
                    }
                )

        long = pd.DataFrame(rows)
        if long.empty:
            long = pd.DataFrame(columns=["frame_index", "iter", "element", "count"])
        else:
            long = long.sort_values(["frame_index", "element"], kind="stable").reset_index(drop=True)
        return LargestMoleculeCompositionResult(table=long, request=request)


@register_task("molecule_lifetime")
class MoleculeLifetimeTask(AnalysisTask):
    """Compute active lifetimes and birth/death events for molecular species."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(
        _result: MoleculeLifetimeResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        table_rows = payload.get("table")
        if not isinstance(table_rows, list) or not table_rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
        x_axis = "start_iter" if "start_iter" in sample else ("start_frame_index" if "start_frame_index" in sample else "")
        if not x_axis or "lifetime_segment_sampled_step_count" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_by = "molecular_formula" if "molecular_formula" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Molecule Lifetimes",
                mapping={"x_col": x_axis, "y_col": "lifetime_segment_sampled_step_count", "group_by_col": group_by},
                options={
                    "title": "Molecule Lifetimes",
                    "xlabel": x_axis,
                    "ylabel": "lifetime_segment_sampled_step_count",
                    "legend": bool(group_by),
                },
                view_type="plot2d",
            ),
        ]

    def run(self, data: MolecularAnalysisData, request: MoleculeLifetimeRequest, reporter=None) -> MoleculeLifetimeResult:
        df = data.molecular_species.copy()
        iterations, frame_idx = _selected_iterations(data, request.frames, request.every)
        selected_iters = iterations[frame_idx]

        lifetime_cols = [
            "molecular_formula",
            "lifetime_segment_id",
            "start_frame_index",
            "end_frame_index",
            "start_iter",
            "end_iter",
            "lifetime_segment_sampled_step_count",
            "peak_freq",
            "mean_freq",
        ]
        if selected_iters.size == 0:
            return MoleculeLifetimeResult(table=pd.DataFrame(columns=lifetime_cols), request=request)

        if request.molecules is not None:
            df = df[df["molecular_formula"].isin(set(str(m) for m in request.molecules))].copy()
        formulas = sorted(df["molecular_formula"].astype(str).unique().tolist()) if not df.empty else []
        if request.molecules is not None:
            formulas = [str(m) for m in request.molecules]

        lifetimes_rows = []
        for formula in formulas:
            sub = df[df["molecular_formula"] == str(formula)][["iter", "freq"]].copy()
            sub["iter"] = pd.to_numeric(sub["iter"], errors="coerce").astype(int)
            sub["freq"] = pd.to_numeric(sub["freq"], errors="coerce").fillna(0.0)
            freq_by_iter = pd.Series(sub["freq"].to_numpy(dtype=float), index=sub["iter"].to_numpy(dtype=int))
            freq = freq_by_iter.reindex(selected_iters, fill_value=0.0).to_numpy(dtype=float)
            active = freq >= float(request.min_freq)
            if not active.any():
                continue

            starts = np.where(active & ~np.r_[False, active[:-1]])[0]
            ends = np.where(active & ~np.r_[active[1:], False])[0]
            for segment_id, (start_idx, end_idx) in enumerate(zip(starts, ends), start=1):
                run_freq = freq[start_idx : end_idx + 1]
                lifetimes_rows.append(
                    {
                        "molecular_formula": str(formula),
                        "lifetime_segment_id": int(segment_id),
                        "start_frame_index": int(frame_idx[start_idx]),
                        "end_frame_index": int(frame_idx[end_idx]),
                        "start_iter": int(selected_iters[start_idx]),
                        "end_iter": int(selected_iters[end_idx]),
                        "lifetime_segment_sampled_step_count": int(end_idx - start_idx + 1),
                        "peak_freq": float(np.max(run_freq)),
                        "mean_freq": float(np.mean(run_freq)),
                    }
                )

        table = pd.DataFrame(lifetimes_rows)
        if table.empty:
            table = pd.DataFrame(columns=lifetime_cols)
        else:
            table = table.sort_values(["molecular_formula", "start_iter"], kind="stable").reset_index(drop=True)

        return MoleculeLifetimeResult(table=table, request=request)


__all__ = [
    "DominantSpeciesRequest",
    "DominantSpeciesResult",
    "DominantSpeciesTask",
    "LargestMoleculeByMassRequest",
    "LargestMoleculeByMassResult",
    "LargestMoleculeByMassTask",
    "LargestMoleculeCompositionRequest",
    "LargestMoleculeCompositionResult",
    "LargestMoleculeCompositionTask",
    "MoleculeLifetimeRequest",
    "MoleculeLifetimeResult",
    "MoleculeLifetimeTask",
]
