"""Base interfaces for analysis layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any

from reaxkit.presentation.specs import PresentationSpec

_DEFAULT_X_CANDIDATES = ("iter", "frame_index", "frame_idx", "frame", "time", "x")
_DEFAULT_GROUP_CANDIDATES = ("atom_id", "atom_type", "species", "src", "dst", "molecule")
_TASK_VIEW_HINTS: dict[str, dict[str, Any]] = {
    "atomic_kinematics": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("value", "speed", "kinetic_energy", "vx", "vy", "vz"),
        "group_candidates": ("atom_id",),
    },
    "bond_events": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("bo_at_event",),
        "group_candidates": ("event", "source", "destination"),
    },
    "charge_table": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("charge",),
        "group_candidates": ("atom_id", "atom_type"),
    },
    "connection_list": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("bo",),
        "group_candidates": ("src", "dst"),
    },
    "connection_stats": {
        "x_candidates": ("src", "dst"),
        "y_candidates": ("value",),
    },
    "connection_table": {"disable_plot": True},
    "control_value": {"disable_plot": True},
    "coordination": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("coordination", "status"),
        "group_candidates": ("atom_id", "atom_type"),
    },
    "dipole": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("dipole_magnitude", "dipole_z", "dipole_y", "dipole_x"),
    },
    "dominant_species": {
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
    "hybridization": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("hybridization", "status"),
        "group_candidates": ("atom_id", "atom_type"),
    },
    "largest_molecule_by_mass": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("mass",),
    },
    "largest_molecule_composition": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("count", "mass_fraction"),
        "group_candidates": ("element", "species"),
    },
    "molecule_lifetime": {
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
    "msd": {
        "x_candidates": ("iter", "frame_index", "frame_idx"),
        "y_candidates": ("msd",),
        "group_candidates": ("atom_id",),
    },
    "diffusivity": {
        "x_candidates": ("atom_id",),
        "y_candidates": ("diffusivity",),
        "group_candidates": ("atom_type",),
    },
    "rdf": {
        "x_candidates": ("r", "radius", "distance"),
        "y_candidates": ("g_r", "g(r)", "rdf"),
    },
    "rdf_property": {
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
    """Build default typed presentation specs for an analysis payload."""
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
    """Abstract analysis task with declarative data requirement."""

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
        """Run scientific analysis on normalized domain data."""

    def required_data_for(self, request: object, args: dict | None = None):
        """Resolve required input dataclass for this run.

        Default behavior keeps existing static ``required_data`` semantics.
        """
        _ = (request, args)
        return self.required_data

    @classmethod
    def recommended_presentations(cls, _result: object, payload: dict[str, Any]) -> list[PresentationSpec]:
        """Default typed presentation specs for analysis results."""
        return default_recommended_presentations(
            payload,
            task_name=str(getattr(cls, "_reaxkit_task_name", "")).strip().lower(),
        )
