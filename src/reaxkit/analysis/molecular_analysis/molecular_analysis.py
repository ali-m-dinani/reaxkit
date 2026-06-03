"""Provide engine-agnostic molecular-population analyzer tasks.

This module derives dominant species, molecule lifetimes, and largest-molecule
summaries from molecular analysis datasets. It is scoped to population-level
molecular metrics and not to per-bond event detection.

**Usage context**

- Species tracking: Identify dominant molecular formulas over selected frames.
- Lifetime studies: Measure persistence of molecular species through time.
- Composition summaries: Extract largest-molecule mass/composition diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import re
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import MolecularAnalysisData
from reaxkit.presentation.specs import PresentationSpec


def _selected_iterations(data: MolecularAnalysisData, frames: Optional[Sequence[int]], every: int) -> tuple[np.ndarray, np.ndarray]:
    """Return full iterations and sampled frame indices after frame/stride filtering."""
    iterations = np.asarray(data.iterations, dtype=int).reshape(-1)
    idx = list(range(len(iterations))) if frames is None else [int(i) for i in frames]
    idx = [i for i in idx if 0 <= i < len(iterations)][:: max(1, int(every))]
    return iterations, np.asarray(idx, dtype=int)


@dataclass
class DominantSpeciesRequest(BaseRequest):
    """Request payload for dominant-species extraction.

    This request configures frame sampling and ranking criteria used to keep
    the most frequent species per sampled iteration.

    Fields
    -----
    frames : Optional[Sequence[int]]
        Optional frame indices to evaluate. If omitted, all frames are used.
    every : int
        Sampling stride applied after frame selection.
    top_n : int
        Number of top-ranked species to retain per sampled iteration.
    min_freq : float
        Minimum frequency threshold applied before ranking species.

    Examples
    -----
    ```python
    request = DominantSpeciesRequest(frames=[0, 10, 20], every=1, top_n=3, min_freq=1.0)
    ```
    The request keeps up to 3 dominant species for each selected frame.
    """
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
    """Result payload for dominant-species analysis.

    The analyzer returns ranked species rows per sampled iteration, including
    frequency and molecular mass values used in ranking.

    Fields
    -----
    table : pandas.DataFrame
        Table with columns ``frame_index``, ``iter``, ``rank``,
        ``molecular_formula``, ``freq``, and ``molecular_mass``.
    request : DominantSpeciesRequest
        Request object used to generate this result.

    Examples
    -----
    ```python
    row = {
        "frame_index": 10,
        "iter": 1000,
        "rank": 1,
        "molecular_formula": "H2O",
        "freq": 24.0,
        "molecular_mass": 18.015,
    }
    ```
    The row represents the top-ranked species at one sampled iteration.
    """

    table: pd.DataFrame
    request: DominantSpeciesRequest


@dataclass
class MoleculeLifetimeRequest(BaseRequest):
    """Request payload for molecule lifetime segmentation.

    This request selects formulas, frame sampling, and the activity threshold
    used to detect contiguous lifetime segments.

    Fields
    -----
    molecules : Optional[Sequence[str]]
        Optional set of molecular formulas to track. If omitted, all formulas
        in sampled data are considered.
    frames : Optional[Sequence[int]]
        Optional frame indices to evaluate. If omitted, all frames are used.
    every : int
        Sampling stride applied after frame selection.
    min_freq : float
        Minimum per-iteration frequency for a species to count as active.

    Examples
    -----
    ```python
    request = MoleculeLifetimeRequest(molecules=["H2O", "OH"], every=5, min_freq=1.0)
    ```
    The request tracks lifetime segments for selected formulas at stride 5.
    """
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
    """Result payload for molecule lifetime analysis.

    The analyzer returns contiguous active segments for each tracked molecular
    formula, including segment bounds and summary frequency statistics.

    Fields
    -----
    table : pandas.DataFrame
        Table with lifetime segment columns such as ``molecular_formula``,
        ``start_iter``, ``end_iter``, and
        ``lifetime_segment_sampled_step_count``.
    request : MoleculeLifetimeRequest
        Request object used to generate this result.

    Examples
    -----
    ```python
    row = {
        "molecular_formula": "OH",
        "lifetime_segment_id": 1,
        "start_iter": 10,
        "end_iter": 30,
        "lifetime_segment_sampled_step_count": 3,
        "peak_freq": 5.0,
        "mean_freq": 3.3333333333,
    }
    ```
    The sample row captures one contiguous active interval for ``OH``.
    """

    table: pd.DataFrame
    request: MoleculeLifetimeRequest


@dataclass
class LargestMoleculeByMassRequest(BaseRequest):
    """Request payload for largest-molecule-by-mass extraction.

    This request configures frame sampling used to pick the heaviest species
    per sampled iteration.

    Fields
    -----
    frames : Optional[Sequence[int]]
        Optional frame indices to evaluate. If omitted, all frames are used.
    every : int
        Sampling stride applied after frame selection.

    Examples
    -----
    ```python
    request = LargestMoleculeByMassRequest(frames=[0, 5, 10], every=1)
    ```
    The request evaluates the heaviest species on selected frames.
    """
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
    """Result payload for largest-molecule-by-mass analysis.

    The analyzer returns one selected heaviest species row per sampled
    iteration with associated frequency and molecular mass.

    Fields
    -----
    table : pandas.DataFrame
        Table with columns ``frame_index``, ``iter``, ``molecular_formula``,
        ``freq``, and ``molecular_mass``.
    request : LargestMoleculeByMassRequest
        Request object used to generate this result.

    Examples
    -----
    ```python
    row = {
        "frame_index": 20,
        "iter": 100,
        "molecular_formula": "Al2O3",
        "freq": 3.0,
        "molecular_mass": 101.96,
    }
    ```
    The row indicates the heaviest species at one sampled iteration.
    """

    table: pd.DataFrame
    request: LargestMoleculeByMassRequest


@dataclass
class LargestMoleculeCompositionRequest(BaseRequest):
    """Request payload for largest-molecule composition expansion.

    This request controls frame sampling used before decomposing each selected
    heaviest formula into per-element counts.

    Fields
    -----
    frames : Optional[Sequence[int]]
        Optional frame indices to evaluate. If omitted, all frames are used.
    every : int
        Sampling stride applied after frame selection.

    Examples
    -----
    ```python
    request = LargestMoleculeCompositionRequest(every=5)
    ```
    The request expands composition on every fifth sampled frame.
    """
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
    """Result payload for largest-molecule composition analysis.

    The analyzer expands each selected largest molecular formula into
    element-count rows per sampled iteration.

    Fields
    -----
    table : pandas.DataFrame
        Table with columns ``frame_index``, ``iter``, ``element``, and
        ``count``.
    request : LargestMoleculeCompositionRequest
        Request object used to generate this result.

    Examples
    -----
    ```python
    rows = [
        {"frame_index": 20, "iter": 100, "element": "Al", "count": 2},
        {"frame_index": 20, "iter": 100, "element": "O", "count": 3},
    ]
    ```
    The sample rows represent composition for ``Al2O3`` at one iteration.
    """

    table: pd.DataFrame
    request: LargestMoleculeCompositionRequest


@register_task("get_dominant_species", label="Dominant Species")
class DominantSpeciesTask(AnalysisTask):
    """Return the dominant molecular species per selected iteration."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(
        _result: DominantSpeciesResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Recommend table and species-frequency plot views.

        Returns a table presentation by default and adds frequency-vs-iteration
        plotting when required fields exist in serialized rows.

        Works on
        Analyzer task output for ``get_dominant_species``.

        Parameters
        -----
        _result : DominantSpeciesResult
            Typed analyzer result instance (unused by current logic).
        payload : dict[str, Any]
            Serialized payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specifications.

        Examples
        -----
        ```python
        specs = DominantSpeciesTask.recommended_presentations(
            _result,
            {"table": [{"iter": 10, "molecular_formula": "H2O", "freq": 4.0}]},
        )
        ```
        The returned list includes a table and a grouped frequency plot.
        """
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
        """Run dominant-species ranking on sampled molecular-analysis frames.

        Filters molecular species to sampled iterations, applies frequency
        thresholds, ranks species per iteration, and returns top-N rows.

        Works on
        ``MolecularAnalysisData``.

        Parameters
        -----
        data : MolecularAnalysisData
            Parsed molecular analysis dataset containing species frequencies.
        request : DominantSpeciesRequest
            Sampling and ranking configuration.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks.

        Returns
        -----
        DominantSpeciesResult
            Result containing ranked dominant species rows.

        Examples
        -----
        ```python
        result = DominantSpeciesTask().run(
            data,
            DominantSpeciesRequest(top_n=3, min_freq=1.0),
        )
        ```
        ``result.table`` contains up to 3 ranked species per sampled iteration.
        """
        df = data.molecular_species.copy()
        if df.empty:
            if reporter:
                reporter("analyze", 1, 1, "Finished dominant species analysis")
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"]),
                request=request,
            )

        iterations, frame_idx = _selected_iterations(data, request.frames, request.every)
        selected_iters = iterations[frame_idx]
        df = df[df["iter"].isin(set(int(it) for it in selected_iters))].copy()
        if df.empty:
            if reporter:
                reporter("analyze", 1, 1, "Finished dominant species analysis")
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"]),
                request=request,
            )

        df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0.0)
        df["molecular_mass"] = pd.to_numeric(df["molecular_mass"], errors="coerce")
        df = df[df["freq"] >= float(request.min_freq)].copy()
        if df.empty:
            if reporter:
                reporter("analyze", 1, 1, "Finished dominant species analysis")
            return DominantSpeciesResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"]),
                request=request,
            )

        rows = []
        frame_lookup = {int(iterations[i]): int(i) for i in frame_idx.tolist()}
        top_n = max(1, int(request.top_n))
        grouped = list(df.groupby("iter", sort=True))
        total_groups = max(1, len(grouped))
        if reporter:
            reporter("analyze", 0, total_groups, "Preparing dominant species analysis")
        for step_i, (iter_val, sub) in enumerate(grouped, start=1):
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
            if reporter:
                reporter("analyze", step_i, total_groups, "Computing dominant species")

        out = pd.DataFrame(rows)
        if out.empty:
            out = pd.DataFrame(columns=["frame_index", "iter", "rank", "molecular_formula", "freq", "molecular_mass"])
        else:
            out = out.sort_values(["frame_index", "rank"], kind="stable").reset_index(drop=True)
        if reporter:
            reporter("analyze", total_groups, total_groups, "Finished dominant species analysis")
        return DominantSpeciesResult(table=out, request=request)


@register_task("get_largest_molecule_by_mass", label="Largest Molecule by Mass")
class LargestMoleculeByMassTask(AnalysisTask):
    """Return the heaviest individual molecular species per selected iteration."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(
        _result: LargestMoleculeByMassResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Recommend table and mass-vs-iteration plot views.

        Returns a table presentation and adds a molecular-mass trend plot when
        required fields are present in serialized rows.

        Works on
        Analyzer task output for ``get_largest_molecule_by_mass``.

        Parameters
        -----
        _result : LargestMoleculeByMassResult
            Typed analyzer result instance (unused by current logic).
        payload : dict[str, Any]
            Serialized payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Recommended presentation specifications.

        Examples
        -----
        ```python
        specs = LargestMoleculeByMassTask.recommended_presentations(
            _result,
            {"table": [{"iter": 100, "molecular_formula": "Al2O3", "molecular_mass": 101.96}]},
        )
        ```
        The output includes a table and a mass trend plot.
        """
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
        """Run largest-molecule-by-mass selection on sampled iterations.

        Samples iterations, filters molecular rows, and selects the heaviest
        species at each sampled iteration.

        Works on
        ``MolecularAnalysisData``.

        Parameters
        -----
        data : MolecularAnalysisData
            Parsed molecular analysis dataset.
        request : LargestMoleculeByMassRequest
            Sampling configuration for iteration selection.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks.

        Returns
        -----
        LargestMoleculeByMassResult
            Result containing one heaviest-species row per sampled iteration.

        Examples
        -----
        ```python
        result = LargestMoleculeByMassTask().run(
            data,
            LargestMoleculeByMassRequest(every=2),
        )
        ```
        ``result.table`` contains heaviest species rows for sampled iterations.
        """
        if reporter:
            reporter("analyze", 0, 3, "Preparing largest molecule-by-mass analysis")
        columns = ["frame_index", "iter", "molecular_formula", "freq", "molecular_mass"]
        df = data.molecular_species.copy()
        if df.empty:
            if reporter:
                reporter("analyze", 3, 3, "Finished largest molecule-by-mass analysis")
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns), request=request)

        iterations, frame_idx = _selected_iterations(data, request.frames, request.every)
        selected_iters = iterations[frame_idx]
        if selected_iters.size == 0:
            if reporter:
                reporter("analyze", 3, 3, "Finished largest molecule-by-mass analysis")
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns), request=request)

        df = df[df["iter"].isin(set(int(it) for it in selected_iters))].copy()
        if df.empty:
            if reporter:
                reporter("analyze", 3, 3, "Finished largest molecule-by-mass analysis")
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns), request=request)
        if reporter:
            reporter("analyze", 1, 3, "Filtering selected iterations")

        df["molecular_mass"] = pd.to_numeric(df["molecular_mass"], errors="coerce")
        df["freq"] = pd.to_numeric(df["freq"], errors="coerce").fillna(0.0)
        df = df[df["molecular_mass"].notna()].copy()
        if df.empty:
            if reporter:
                reporter("analyze", 3, 3, "Finished largest molecule-by-mass analysis")
            return LargestMoleculeByMassResult(table=pd.DataFrame(columns=columns), request=request)
        if reporter:
            reporter("analyze", 2, 3, "Selecting heaviest species per iteration")

        idx = df.groupby("iter")["molecular_mass"].idxmax()
        out = df.loc[idx, ["iter", "molecular_formula", "freq", "molecular_mass"]].copy()
        frame_lookup = {int(iterations[i]): int(i) for i in frame_idx.tolist()}
        out.insert(0, "frame_index", out["iter"].map(lambda it: frame_lookup.get(int(it), -1)).astype(int))
        out = out.sort_values(["frame_index", "iter"], kind="stable").reset_index(drop=True)
        if reporter:
            reporter("analyze", 3, 3, "Finished largest molecule-by-mass analysis")
        return LargestMoleculeByMassResult(table=out[columns], request=request)


@register_task("get_largest_molecule_composition", label="Largest Molecule Composition")
class LargestMoleculeCompositionTask(AnalysisTask):
    """Return per-element composition of the heaviest molecule per selected iteration."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(
        _result: LargestMoleculeCompositionResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Recommend table and element-count plot views.

        Returns a table by default and adds an element-count trend plot when
        required fields are present in serialized rows.

        Works on
        Analyzer task output for ``get_largest_molecule_composition``.

        Parameters
        -----
        _result : LargestMoleculeCompositionResult
            Typed analyzer result instance (unused by current logic).
        payload : dict[str, Any]
            Serialized payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Recommended presentation specifications.

        Examples
        -----
        ```python
        specs = LargestMoleculeCompositionTask.recommended_presentations(
            _result,
            {"table": [{"iter": 100, "element": "O", "count": 3}]},
        )
        ```
        The returned list includes a table and grouped element-count plot.
        """
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
        """Run largest-molecule composition expansion from sampled iterations.

        Reuses largest-molecule-by-mass selection, parses selected formulas into
        element/count pairs, and emits long-form composition rows.

        Works on
        ``MolecularAnalysisData``.

        Parameters
        -----
        data : MolecularAnalysisData
            Parsed molecular analysis dataset.
        request : LargestMoleculeCompositionRequest
            Sampling configuration for iteration selection.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks.

        Returns
        -----
        LargestMoleculeCompositionResult
            Result containing long-form element composition rows.

        Examples
        -----
        ```python
        result = LargestMoleculeCompositionTask().run(
            data,
            LargestMoleculeCompositionRequest(frames=[0, 1, 2]),
        )
        ```
        ``result.table`` contains one row per element per sampled iteration.
        """
        if reporter:
            reporter("analyze", 0, 1, "Preparing largest molecule composition analysis")
        largest = LargestMoleculeByMassTask().run(
            data,
            LargestMoleculeByMassRequest(frames=request.frames, every=request.every),
            reporter=reporter,
        ).table
        if largest.empty:
            if reporter:
                reporter("analyze", 1, 1, "Finished largest molecule composition analysis")
            return LargestMoleculeCompositionResult(
                table=pd.DataFrame(columns=["frame_index", "iter", "element", "count"]),
                request=request,
            )

        rows = []
        total = max(1, len(largest))
        for step_i, (_, row) in enumerate(largest.iterrows(), start=1):
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
            if reporter:
                reporter("analyze", step_i, total, "Expanding molecular composition")

        long = pd.DataFrame(rows)
        if long.empty:
            long = pd.DataFrame(columns=["frame_index", "iter", "element", "count"])
        else:
            long = long.sort_values(["frame_index", "element"], kind="stable").reset_index(drop=True)
        if reporter:
            reporter("analyze", total, total, "Finished largest molecule composition analysis")
        return LargestMoleculeCompositionResult(table=long, request=request)


@register_task("get_molecule_lifetime", label="Molecule Lifetime")
class MoleculeLifetimeTask(AnalysisTask):
    """Compute active lifetimes and birth/death events for molecular species."""

    required_data = MolecularAnalysisData

    @staticmethod
    def recommended_presentations(
        _result: MoleculeLifetimeResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Recommend table and lifetime-segment plot views.

        Returns a table presentation and adds a lifetime length plot when
        required segment fields are present in serialized rows.

        Works on
        Analyzer task output for ``get_molecule_lifetime``.

        Parameters
        -----
        _result : MoleculeLifetimeResult
            Typed analyzer result instance (unused by current logic).
        payload : dict[str, Any]
            Serialized payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Recommended presentation specifications.

        Examples
        -----
        ```python
        specs = MoleculeLifetimeTask.recommended_presentations(
            _result,
            {"table": [{"start_iter": 10, "molecular_formula": "OH", "lifetime_segment_sampled_step_count": 3}]},
        )
        ```
        The returned list includes a table and a lifetime-length plot.
        """
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
        """Run molecule lifetime segmentation on sampled species frequencies.

        Samples iterations, tracks per-formula active periods above threshold,
        and returns contiguous segment statistics for each tracked formula.

        Works on
        ``MolecularAnalysisData``.

        Parameters
        -----
        data : MolecularAnalysisData
            Parsed molecular analysis dataset containing frequency time series.
        request : MoleculeLifetimeRequest
            Formula selection, sampling, and activity-threshold configuration.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks.

        Returns
        -----
        MoleculeLifetimeResult
            Result containing lifetime segment rows per molecular formula.

        Examples
        -----
        ```python
        result = MoleculeLifetimeTask().run(
            data,
            MoleculeLifetimeRequest(min_freq=1.0),
        )
        ```
        ``result.table`` lists active segments with start/end iteration bounds.
        """
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
            if reporter:
                reporter("analyze", 1, 1, "Finished molecule lifetime analysis")
            return MoleculeLifetimeResult(table=pd.DataFrame(columns=lifetime_cols), request=request)

        if request.molecules is not None:
            df = df[df["molecular_formula"].isin(set(str(m) for m in request.molecules))].copy()
        formulas = sorted(df["molecular_formula"].astype(str).unique().tolist()) if not df.empty else []
        if request.molecules is not None:
            formulas = [str(m) for m in request.molecules]

        lifetimes_rows = []
        total_formulas = max(1, len(formulas))
        if reporter:
            reporter("analyze", 0, total_formulas, "Preparing molecule lifetime analysis")
        for step_i, formula in enumerate(formulas, start=1):
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
            if reporter:
                reporter("analyze", step_i, total_formulas, "Computing molecule lifetimes")

        table = pd.DataFrame(lifetimes_rows)
        if table.empty:
            table = pd.DataFrame(columns=lifetime_cols)
        else:
            table = table.sort_values(["molecular_formula", "start_iter"], kind="stable").reset_index(drop=True)

        if reporter:
            reporter("analyze", total_formulas, total_formulas, "Finished molecule lifetime analysis")
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
