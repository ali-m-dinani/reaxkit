"""Repository-wide execution and artifact policy for registered analyses."""

from __future__ import annotations

from typing import Any


# These sets are intentionally exhaustive for the tasks imported by
# ``reaxkit.analysis``.  A test compares them with TASK_REGISTRY so adding a
# command requires an explicit decision here instead of silently inheriting a
# parallel strategy.
INDEPENDENT_FRAME_TASKS = frozenset({
    "active_site_structural",
    "cell_dimensions",
    "charge_series",
    "charge_table",
    "electric_field_series",
    "eregime_series",
    "get-dipole",
    "get-polarization",
    "get_connection_list",
    "get_connection_table",
    "get_coordination",
    "get_dihedral",
    "get_hybridization",
    "get_voronoi_geometry_pyvoro",
    "get_voronoi_geometry_scipy",
    "get_voronoi_pyvoro",
    "get_voronoi_scipy",
    "get-wurtzite-neighbors",
    "molecular_frequency_series",
    "molecular_totals_series",
    "partial_energy_series",
    "restraint_series",
    "simulation_series",
    "trajectory_coordinate_series",
})

REFERENCE_FRAME_TASKS = frozenset({
    "get-potential-and-electric-field",
    "get_binned_dynamic_charges",
    "get_dynamic_charge_changes",
    "get-wurtzite-polarity",
    "trajectory_displacement_series",
})

STREAMING_REDUCTION_TASKS = frozenset({
    "active_site_event_diagnostics",
    "get_connection_stats",
    "get_charge_vs_electric_field",
    "get_polarization_field",
    "get_rdf",
    "get_rdf_property",
    "isomer_representative_detection",
})

ORDERED_STATEFUL_TASKS = frozenset({
    "active_site_events",
    "get_bond_events",
    "get_dominant_species",
    "get_largest_molecule_by_mass",
    "get_largest_molecule_composition",
    "get_molecule_lifetime",
    "molecule_isomer_detection",
    "trajectory_relabel_by_coordination",
    "write-trajectory-with-polarity",
    "write-trajectory-with-potential-and-electric-field",
    "write_trajectory_with_charges",
})

GLOBAL_TASKS = frozenset({
    "force_field_data",
    "force_field_optimization",
    "force_field_optimization_parameters",
    "force_field_optimization_report",
    "force_field_optimization_report_bulk_modulus",
    "force_field_optimization_report_eos",
    "force_field_optimization_report_restraints",
    "geometry_optimization_data",
    "get-dielectric-constant",
    "get_control_data",
    "get_diffusivity",
    "get_frames_count",
    "get_kinematics",
    "get_msd",
    "get_z_binned_deformation_gradient_strain",
    "get_z_binned_top_bottom_strain",
    "msd",
    "parameter_optimization_diagnostic",
    "parameter_optimization_diagnostic_beeswarm",
    "structure_summary_data",
    "trainset_data",
    "trainset_group_comments",
})

ALL_GENERAL_TASKS = frozenset().union(
    INDEPENDENT_FRAME_TASKS,
    REFERENCE_FRAME_TASKS,
    STREAMING_REDUCTION_TASKS,
    ORDERED_STATEFUL_TASKS,
    GLOBAL_TASKS,
)

_SHAPE_BY_TASK = {
    **{name: "independent_frame_map" for name in INDEPENDENT_FRAME_TASKS},
    **{name: "reference_frame_map" for name in REFERENCE_FRAME_TASKS},
    **{name: "streaming_reduction" for name in STREAMING_REDUCTION_TASKS},
    **{name: "ordered_stateful_stream" for name in ORDERED_STATEFUL_TASKS},
    **{name: "global" for name in GLOBAL_TASKS},
}

# Only these implementations currently execute their numerical frame kernel
# through BoundedFramePipeline. Other tasks retain an explicit serial policy
# until their command-specific result combiner is migrated.
SHARED_PIPELINE_TASKS = frozenset({
    "charge_table",
    "get-dipole",
    "get-polarization",
})


def classification_for_registered_task(task: Any) -> dict[str, Any] | None:
    name = str(getattr(task.__class__, "_reaxkit_task_name", "")).strip().lower()
    shape = _SHAPE_BY_TASK.get(name)
    if shape is None:
        return None
    streaming = callable(getattr(task, "run_stream", None))
    return {
        "command": name,
        "shape": shape,
        "thread_safe": name in SHARED_PIPELINE_TASKS,
        "pipeline_aware": name in SHARED_PIPELINE_TASKS,
        "streaming": streaming,
        "artifact_profile": "standard_csv_compatibility",
        "status": (
            "migrated_shared_pipeline"
            if name in SHARED_PIPELINE_TASKS
            else "classified_existing_stream"
            if streaming
            else "classified_materialized"
            if shape != "global"
            else "classified_global"
        ),
    }


__all__ = [
    "ALL_GENERAL_TASKS",
    "GLOBAL_TASKS",
    "INDEPENDENT_FRAME_TASKS",
    "ORDERED_STATEFUL_TASKS",
    "REFERENCE_FRAME_TASKS",
    "SHARED_PIPELINE_TASKS",
    "STREAMING_REDUCTION_TASKS",
    "classification_for_registered_task",
]
