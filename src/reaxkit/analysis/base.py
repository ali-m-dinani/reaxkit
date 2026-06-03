"""Define base interfaces and presentation defaults for analysis tasks.

This module provides the abstract analysis-task contract and a default
presentation-spec builder used across analyzer outputs. It is scoped to shared
analysis-layer infrastructure and does not implement domain-specific analysis.

**Usage context**

- Task foundation: Subclass `AnalysisTask` to implement analyzer execution.
- Validation wiring: Auto-wrap `run` with shared input validation.
- UI defaults: Build fallback table/plot presentation specs from payloads.

Notes
-----
- `AnalysisTask.__init_subclass__` injects validation into subclass `run`
  implementations unless already wrapped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any

from reaxkit.presentation.specs import PresentationSpec

_DEFAULT_X_CANDIDATES = ("iter", "frame_index", "frame_idx", "frame", "time", "x")
_DEFAULT_GROUP_CANDIDATES = ("atom_id", "atom_type", "species", "src", "dst", "molecule")
_TASK_VIEW_HINTS: dict[str, dict[str, Any]] = {
    "get_kinematics": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("value", "speed", "kinetic_energy", "vx", "vy", "vz"),
        "group_candidates": ("atom_id",),
    },
    "get_bond_events": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("bo_at_event",),
        "group_candidates": ("event", "source", "destination"),
    },
    "charge_table": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("charge",),
        "group_candidates": ("atom_id", "atom_type"),
    },
    "get_connection_list": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("bo",),
        "group_candidates": ("src", "dst"),
    },
    "get_connection_stats": {
        "x_candidates": ("src", "dst"),
        "y_candidates": ("value",),
    },
    "get_connection_table": {"disable_plot": True},
    "get_control_data": {"disable_plot": True},
    "get_coordination": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("coordination", "status"),
        "group_candidates": ("atom_id", "atom_type"),
    },
    "dipole": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("dipole_magnitude", "dipole_z", "dipole_y", "dipole_x"),
    },
    "get_dominant_species": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("frequency", "count"),
        "group_candidates": ("species", "formula"),
    },
    "force_field_optimization": {
        "x_candidates": ("epoch", "iter", "step"),
        "y_candidates": ("loss", "objective", "value"),
    },
    "force_field_optimization_parameters": {
        "x_candidates": ("epoch", "iter", "step"),
        "y_candidates": ("value",),
        "group_candidates": ("parameter", "name"),
    },
    "force_field_data": {"disable_plot": True},
    "force_field_optimization_report": {"disable_plot": True},
    "force_field_optimization_report_bulk_modulus": {
        "x_candidates": ("volume", "strain", "x"),
        "y_candidates": ("energy", "bulk_modulus", "y"),
    },
    "force_field_optimization_report_eos": {
        "x_candidates": ("volume", "strain", "x"),
        "y_candidates": ("energy", "y"),
    },
    "get_hybridization": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("hybridization", "status"),
        "group_candidates": ("atom_id", "atom_type"),
    },
    "get_largest_molecule_by_mass": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("mass",),
    },
    "get_largest_molecule_composition": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("count", "mass_fraction"),
        "group_candidates": ("element", "species"),
    },
    "get_molecule_lifetime": {
        "x_candidates": ("molecule", "species"),
        "y_candidates": ("lifetime", "mean_lifetime", "count"),
    },
    "parameter_optimization_diagnostic": {
        "x_candidates": ("epoch", "iter", "step"),
        "y_candidates": ("loss", "objective", "value"),
    },
    "polarization": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("P_z (uC/cm^2)", "P_z", "polarization_z", "polarization"),
    },
    "polarization_field": {
        "x_candidates": ("field_z", "field", "E"),
        "y_candidates": ("P_z (uC/cm^2)", "P_z", "polarization_z"),
    },
    "get_msd": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("get_msd",),
        "group_candidates": ("atom_id",),
    },
    "get_diffusivity": {
        "x_candidates": ("atom_id",),
        "y_candidates": ("get_diffusivity",),
        "group_candidates": ("atom_type",),
    },
    "get_rdf": {
        "x_candidates": ("r", "radius", "distance"),
        "y_candidates": ("g_r", "g(r)", "get_rdf"),
    },
    "get_rdf_property": {
        "x_candidates": ("frame_index", "frame_idx", "iter"),
        "y_candidates": ("value", "peak_height", "area"),
    },
    "structure_summary_data": {"disable_plot": True},
    "trainset_data": {"disable_plot": True},
    "trainset_group_comments": {"disable_plot": True},
}


def _pick_first(candidates: tuple[str, ...], columns: list[str]) -> str:
    colset = set(columns)
    for name in candidates:
        if name in colset:
            return name
    return ""


def default_recommended_presentations(
    payload: dict[str, Any],
    *,
    task_name: str = "",
) -> list[PresentationSpec]:
    """Build default presentation specs for serialized analysis payload tables.

    Infers candidate x/y/group columns from task-specific hints and sample row
    content, always returning at least a table presentation.

    Works on
    -----
    Analyzer task payload dictionaries containing a serialized `table` field

    Parameters
    -----
    payload : dict[str, Any]
        Serialized analysis payload, usually produced from a `BaseResult`.
    task_name : str, optional
        Registered task name used to apply task-specific plot-column hints.

    Returns
    -----
    list[PresentationSpec]
        Ordered presentation specs containing a table view and, when possible,
        a single-plot view inferred from available columns.

    Examples
    -----
    ```python
    from reaxkit.analysis.base import default_recommended_presentations

    payload = {"table": [{"iter": 0, "atom_id": 1, "msd": 0.0}]}
    views = default_recommended_presentations(payload, task_name="get_msd")
    ```
    Sample output:
    `[PresentationSpec(renderer='table', ...), PresentationSpec(renderer='single_plot', ...)]`
    Meaning:
    The payload supports both tabular and inferred 2D-plot representations.
    """
    table_rows = payload.get("table")
    if not isinstance(table_rows, list) or not table_rows:
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]

    sample = table_rows[0] if isinstance(table_rows[0], dict) else {}
    if not isinstance(sample, dict):
        return [PresentationSpec(renderer="table", label="Table", view_type="table")]

    cols = [str(k) for k in sample.keys()]
    hints = _TASK_VIEW_HINTS.get(str(task_name or "").strip().lower(), {})

    x_candidates = tuple(hints.get("x_candidates") or _DEFAULT_X_CANDIDATES)
    y_candidates = tuple(hints.get("y_candidates") or ())
    group_candidates = tuple(hints.get("group_candidates") or _DEFAULT_GROUP_CANDIDATES)

    x_axis = _pick_first(x_candidates, cols)
    numeric_cols = [k for k, v in sample.items() if isinstance(v, (int, float)) and k != x_axis]
    y_axis = _pick_first(y_candidates, cols)
    if not y_axis:
        y_axis = numeric_cols[0] if numeric_cols else ""
    group_by = _pick_first(group_candidates, cols)

    views: list[PresentationSpec] = [PresentationSpec(renderer="table", label="Table", view_type="table")]
    if bool(hints.get("disable_plot")):
        return views
    if x_axis and y_axis:
        views.append(
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_axis} vs {x_axis}",
                mapping={"x_col": x_axis, "y_col": y_axis, "group_by_col": group_by},
                options={"title": f"{y_axis} vs {x_axis}", "xlabel": x_axis, "ylabel": y_axis, "legend": bool(group_by)},
                view_type="plot2d",
            )
        )
    return views


class AnalysisTask(ABC):
    """Abstract base class for all analysis tasks.

    Defines the `run` execution contract, required-data resolution, and default
    presentation recommendation behavior for analyzer tasks.
    """

    required_data = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        run_fn = cls.__dict__.get("run")
        if run_fn is None or getattr(run_fn, "_reaxkit_validated_run", False):
            return

        @wraps(run_fn)
        def _validated_run(self, data, request, *args, **kwargs):
            from reaxkit.analysis.validation import validate_task_inputs

            validate_task_inputs(self, data, request)
            return run_fn(self, data, request, *args, **kwargs)

        _validated_run._reaxkit_validated_run = True
        cls.run = _validated_run

    @abstractmethod
    def run(self, data, request, reporter=None):
        """Run analyzer logic on normalized domain data and request configuration.

        Subclasses must implement this method and return a task-specific result
        object derived from `BaseResult`.

        Works on
        -----
        Analyzer task domain data objects and their corresponding request objects

        Parameters
        -----
        data : Any
            Domain data object required by the concrete task implementation.
        request : Any
            Request/configuration object controlling analysis behavior.
        reporter : Any, optional
            Optional progress callback accepted by task implementations.

        Returns
        -----
        Any
            Task-specific analysis result object (typically a `BaseResult`
            subclass).

        Examples
        -----
        ```python
        class MyTask(AnalysisTask):
            required_data = MyData
            def run(self, data, request, reporter=None):
                return MyResult(...)
        ```
        Sample output:
        `MyResult(...)`
        Meaning:
        A concrete analyzer returns a structured result payload for downstream
        serialization and presentation.
        """

    def required_data_for(self, request: object, args: dict | None = None):
        """Resolve the required input data type for a specific run request.

        Default behavior preserves static `required_data` semantics. Subclasses
        can override this to select data requirements dynamically.

        Works on
        -----
        Analyzer task request objects and optional invocation argument maps

        Parameters
        -----
        request : object
            Request object used to determine data requirements.
        args : dict | None, optional
            Optional invocation arguments that may influence type resolution.

        Returns
        -----
        Any
            Required input type (or tuple of types) expected by the task.

        Examples
        -----
        ```python
        required = task.required_data_for(request)
        ```
        Sample output:
        `<class 'reaxkit.domain.data_models.TrajectoryData'>`
        Meaning:
        The task expects trajectory-domain input for the given request.
        """
        _ = (request, args)
        return self.required_data

    @classmethod
    def recommended_presentations(cls, _result: object, payload: dict[str, Any]) -> list[PresentationSpec]:
        """Return default presentation recommendations for a task result payload.

        Delegates to the shared fallback presenter using the registered task
        name hint when available.

        Works on
        -----
        Analyzer result objects and serialized analyzer payload dictionaries

        Parameters
        -----
        _result : object
            Result object instance; accepted for API consistency.
        payload : dict[str, Any]
            Serialized payload, typically containing a `table` list.

        Returns
        -----
        list[PresentationSpec]
            Presentation specs suitable for rendering the task output.

        Examples
        -----
        ```python
        views = MyTask.recommended_presentations(result, {"table": [{"x": 1, "y": 2}]})
        ```
        Sample output:
        `[PresentationSpec(...)]`
        Meaning:
        The task exposes renderer-ready view metadata for the provided payload.
        """
        return default_recommended_presentations(
            payload,
            task_name=str(getattr(cls, "_reaxkit_task_name", "")).strip().lower(),
        )
