"""Engine-agnostic molecular analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence
import re

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import MolecularAnalysisData


def _selected_iterations(data: MolecularAnalysisData, frames: Optional[Sequence[int]], every: int) -> tuple[np.ndarray, np.ndarray]:
    iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
    idx = list(range(len(iterations))) if frames is None else [int(i) for i in frames]
    idx = [i for i in idx if 0 <= i < len(iterations)][:: max(1, int(every))]
    return iterations, np.asarray(idx, dtype=int)


@dataclass
class DominantSpeciesRequest(BaseRequest):
    frames: Optional[Sequence[int]] = None
    every: int = 1
    top_n: int = 1
    min_freq: float = 0.0


@dataclass
class DominantSpeciesResult(BaseResult):
    table: pd.DataFrame


@dataclass
class MoleculeLifetimeRequest(BaseRequest):
    molecules: Optional[Sequence[str]] = None
    frames: Optional[Sequence[int]] = None
    every: int = 1
    min_freq: float = 1.0


@dataclass
class MoleculeLifetimeResult(BaseResult):
    lifetimes: pd.DataFrame
    events: pd.DataFrame


@dataclass
class LargestMoleculeByMassRequest(BaseRequest):
    frames: Optional[Sequence[int]] = None
    every: int = 1


@dataclass
class LargestMoleculeByMassResult(BaseResult):
    table: pd.DataFrame


@dataclass
class LargestMoleculeCompositionRequest(BaseRequest):
    frames: Optional[Sequence[int]] = None
    every: int = 1
    format: Literal["wide", "long"] = "wide"


@dataclass
class LargestMoleculeCompositionResult(BaseResult):
    table: pd.DataFrame


@register_task("dominant_species")
class DominantSpeciesTask(AnalysisTask):
    """Return the dominant molecular species per selected iteration."""

    required_data = MolecularAnalysisData

    def run(self, data: MolecularAnalysisData, request: DominantSpeciesRequest, reporter=None) -> DominantSpeciesResult:
        df = data.molecular_species.copy()
        if df.empty:
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"])
            )

        iterations, frame_idx = _selected_iterations(data, request.frames, request.every)
        selected_iters = iterations[frame_idx]
        df = df[df["iter"].isin(set(int(it) for it in selected_iters))].copy()
        if df.empty:
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"])
            )

        df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0.0)
        df["molecular_mass"] = pd.to_numeric(df["molecular_mass"], errors="coerce")
        df = df[df["freq"] >= float(request.min_freq)].copy()
        if df.empty:
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"])
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
        return DominantSpeciesResult(table=out)


@register_task("largest_molecule_by_mass")
class LargestMoleculeByMassTask(AnalysisTask):
    """Return the heaviest individual molecular species per selected iteration."""

    required_data = MolecularAnalysisData

    def run(
        self,
        data: MolecularAnalysisData,
        request: LargestMoleculeByMassRequest,
        reporter=None,
    ) -> LargestMoleculeByMassResult:
        columns = ["frame_index", "iter", "molecular_formula", "freq", "molecular_mass"]
        df = data.molecular_species.copy()
        if df.empty:
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns))

        iterations, frame_idx = _selected_iterations(data, request.frames, request.every)
        selected_iters = iterations[frame_idx]
        if selected_iters.size == 0:
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns))

        df = df[df["iter"].isin(set(int(it) for it in selected_iters))].copy()
        if df.empty:
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns))

        df["molecular_mass"] = pd.to_numeric(df["molecular_mass"], errors="coerce")
        df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0.0)
        df = df[df["molecular_mass"].notna()].copy()
        if df.empty:
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns))

        idx = df.groupby("iter")["molecular_mass"].idxmax()
        out = df.loc[idx, ["iter", "molecular_formula", "freq", "molecular_mass"]].copy()
        frame_lookup = {int(iterations[i]): int(i) for i in frame_idx.tolist()}
        out.insert(0, "frame_index", out["iter"].map(lambda it: frame_lookup.get(int(it), -1)).astype(int))
        out = out.sort_values(["frame_index", "iter"], kind="stable").reset_index(drop=True)
        return LargestMoleculeByMassResult(table=out[columns])


@register_task("largest_molecule_composition")
class LargestMoleculeCompositionTask(AnalysisTask):
    """Return per-element composition of the heaviest molecule per selected iteration."""

    required_data = MolecularAnalysisData

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
            cols = ["frame_index", "iter"] if request.format == "wide" else ["frame_index", "iter", "element", "count"]
            return LargestMoleculeCompositionResult(table=pd.DataFrame(columns=cols))

        rows = []
        all_elements = set()
        for _, row in largest.iterrows():
            counts = {
                "frame_index": int(row["frame_index"]),
                "iter": int(row["iter"]),
            }
            pairs = re.findall(r"([A-Z][a-z]*)(\d+)", str(row["molecular_formula"]))
            for element, count in pairs:
                count_i = int(count)
                counts[element] = counts.get(element, 0) + count_i
                all_elements.add(element)
            rows.append(counts)

        wide = pd.DataFrame(rows).sort_values(["frame_index", "iter"], kind="stable").reset_index(drop=True)
        for element in sorted(all_elements):
            if element not in wide.columns:
                wide[element] = 0
        wide_cols = ["frame_index", "iter"] + sorted(col for col in wide.columns if col not in {"frame_index", "iter"})
        wide = wide[wide_cols]

        if request.format == "wide":
            return LargestMoleculeCompositionResult(table=wide)

        long = wide.melt(
            id_vars=["frame_index", "iter"],
            var_name="element",
            value_name="count",
        ).sort_values(["frame_index", "element"], kind="stable").reset_index(drop=True)
        return LargestMoleculeCompositionResult(table=long)


@register_task("molecule_lifetime")
class MoleculeLifetimeTask(AnalysisTask):
    """Compute active lifetimes and birth/death events for molecular species."""

    required_data = MolecularAnalysisData

    def run(self, data: MolecularAnalysisData, request: MoleculeLifetimeRequest, reporter=None) -> MoleculeLifetimeResult:
        df = data.molecular_species.copy()
        iterations, frame_idx = _selected_iterations(data, request.frames, request.every)
        selected_iters = iterations[frame_idx]

        lifetime_cols = [
            "molecular_formula",
            "run_id",
            "start_frame_index",
            "end_frame_index",
            "start_iter",
            "end_iter",
            "n_samples",
            "peak_freq",
            "mean_freq",
        ]
        event_cols = ["molecular_formula", "event", "frame_index", "iter", "run_id", "freq"]
        if selected_iters.size == 0:
            return MoleculeLifetimeResult(
                lifetimes=pd.DataFrame(columns=lifetime_cols),
                events=pd.DataFrame(columns=event_cols),
            )

        if request.molecules is not None:
            df = df[df["molecular_formula"].isin(set(str(m) for m in request.molecules))].copy()
        formulas = sorted(df["molecular_formula"].astype(str).unique().tolist()) if not df.empty else []
        if request.molecules is not None:
            formulas = [str(m) for m in request.molecules]

        lifetimes_rows = []
        event_rows = []
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
            for run_id, (start_idx, end_idx) in enumerate(zip(starts, ends), start=1):
                run_freq = freq[start_idx : end_idx + 1]
                lifetimes_rows.append(
                    {
                        "molecular_formula": str(formula),
                        "run_id": int(run_id),
                        "start_frame_index": int(frame_idx[start_idx]),
                        "end_frame_index": int(frame_idx[end_idx]),
                        "start_iter": int(selected_iters[start_idx]),
                        "end_iter": int(selected_iters[end_idx]),
                        "n_samples": int(end_idx - start_idx + 1),
                        "peak_freq": float(np.max(run_freq)),
                        "mean_freq": float(np.mean(run_freq)),
                    }
                )
                event_rows.append(
                    {
                        "molecular_formula": str(formula),
                        "event": "birth",
                        "frame_index": int(frame_idx[start_idx]),
                        "iter": int(selected_iters[start_idx]),
                        "run_id": int(run_id),
                        "freq": float(freq[start_idx]),
                    }
                )
                event_rows.append(
                    {
                        "molecular_formula": str(formula),
                        "event": "death",
                        "frame_index": int(frame_idx[end_idx]),
                        "iter": int(selected_iters[end_idx]),
                        "run_id": int(run_id),
                        "freq": float(freq[end_idx]),
                    }
                )

        lifetimes = pd.DataFrame(lifetimes_rows)
        if lifetimes.empty:
            lifetimes = pd.DataFrame(columns=lifetime_cols)
        else:
            lifetimes = lifetimes.sort_values(["molecular_formula", "start_iter"], kind="stable").reset_index(drop=True)

        events = pd.DataFrame(event_rows)
        if events.empty:
            events = pd.DataFrame(columns=event_cols)
        else:
            events = events.sort_values(["molecular_formula", "iter", "event"], kind="stable").reset_index(drop=True)

        return MoleculeLifetimeResult(lifetimes=lifetimes, events=events)


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
