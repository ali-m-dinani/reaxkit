"""Capability inventory for electrostatics and ferroelectrics analyses."""

from __future__ import annotations

from typing import Any


PRIORITY_TASK_MANIFEST: tuple[dict[str, Any], ...] = (
    {"command": "charge_table", "module": "reaxkit.analysis.electrostatics.charges", "class": "ChargeTableTask", "shape": "independent_frame_map", "thread_safe": True, "status": "migrated_shared_pipeline"},
    {"command": "get-dielectric-constant", "module": "reaxkit.analysis.electrostatics.dielectric_constant", "class": "DielectricConstantTask", "shape": "global", "thread_safe": False, "status": "classified_global"},
    {"command": "get-dipole", "module": "reaxkit.analysis.electrostatics.electrostatics", "class": "DipoleTask", "shape": "independent_frame_map", "thread_safe": True, "status": "migrated_shared_pipeline"},
    {"command": "get-polarization", "module": "reaxkit.analysis.electrostatics.electrostatics", "class": "PolarizationTask", "shape": "independent_frame_map", "thread_safe": True, "status": "migrated_shared_pipeline"},
    {"command": "get_polarization_field", "module": "reaxkit.analysis.electrostatics.electrostatics", "class": "PolarizationFieldTask", "shape": "streaming_reduction", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "get-potential-and-electric-field", "module": "reaxkit.analysis.electrostatics.potential_and_electric_field.analysis", "class": "PotentialElectricFieldTask", "shape": "reference_frame_map", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "write-trajectory-with-potential-and-electric-field", "module": "reaxkit.analysis.electrostatics.potential_and_electric_field.trajectory", "class": "PotentialElectricFieldTrajectoryTask", "shape": "ordered_stateful_stream", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "get_binned_dynamic_charges", "module": "reaxkit.analysis.ferroelectrics.binned_dynamic_charge", "class": "BinnedDynamicChargeTask", "shape": "reference_frame_map", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "get_charge_vs_electric_field", "module": "reaxkit.analysis.ferroelectrics.charge_field", "class": "ChargeFieldTask", "shape": "streaming_reduction", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "write_trajectory_with_charges", "module": "reaxkit.analysis.ferroelectrics.charge_extxyz", "class": "ChargeExtendedXYZTask", "shape": "ordered_stateful_stream", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "get_dynamic_charge_changes", "module": "reaxkit.analysis.ferroelectrics.dynamic_charge", "class": "DynamicChargeChangeTask", "shape": "reference_frame_map", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "get-basal-plane-displacement-dipole", "module": "reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.dipole", "class": "BasalPlaneDipoleTask", "shape": "reference_frame_map", "thread_safe": False, "status": "classified_materialized"},
    {"command": "get-basal-plane-displacement-local-polarization", "module": "reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.local_polarization", "class": "BasalPlaneLocalPolarizationTask", "shape": "reference_frame_map", "thread_safe": False, "status": "classified_materialized"},
    {"command": "get-basal-plane-displacement-polarization", "module": "reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.polarization", "class": "BasalPlanePolarizationTask", "shape": "reference_frame_map", "thread_safe": False, "status": "classified_materialized"},
    {"command": "get-basal-plane-displacement-projected-polarity", "module": "reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.projected_polarity", "class": "BasalPlaneProjectedPolarityTask", "shape": "reference_frame_map", "thread_safe": True, "status": "migrated_shared_pipeline"},
    {"command": "get-wurtzite-neighbors", "module": "reaxkit.analysis.ferroelectrics.four_folded_wurtzite.neighbors", "class": "WurtziteNeighborTask", "shape": "independent_frame_map", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "get-wurtzite-polarity", "module": "reaxkit.analysis.ferroelectrics.four_folded_wurtzite.polarity", "class": "WurtzitePolarityTask", "shape": "independent_frame_map", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "write-trajectory-with-polarity", "module": "reaxkit.analysis.ferroelectrics.four_folded_wurtzite.trajectory", "class": "PolarityExtendedXYZTask", "shape": "ordered_stateful_stream", "thread_safe": False, "status": "classified_existing_stream"},
    {"command": "get-hbn-reference-local-polarization", "module": "reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization", "class": "HBNReferenceLocalPolarizationTask", "shape": "reference_frame_map", "thread_safe": False, "status": "classified_materialized"},
    {"command": "get-hbn-reference-polarization", "module": "reaxkit.analysis.ferroelectrics.hbn_reference.polarization", "class": "HBNReferencePolarizationTask", "shape": "reference_frame_map", "thread_safe": False, "status": "classified_materialized"},
    {"command": "get-hbn-reference-projected-polarity", "module": "reaxkit.analysis.ferroelectrics.hbn_reference.projected_polarity", "class": "HBNReferenceProjectedPolarityTask", "shape": "reference_frame_map", "thread_safe": True, "status": "migrated_shared_pipeline"},
    {"command": "get-three-folded-wurtzite-neighbors", "module": "reaxkit.analysis.ferroelectrics.three_folded_wurtzite.neighbors", "class": "WurtziteNeighborTask", "shape": "independent_frame_map", "thread_safe": False, "status": "classified_materialized"},
    {"command": "get-three-folded-wurtzite-polarity", "module": "reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarity", "class": "WurtzitePolarityTask", "shape": "independent_frame_map", "thread_safe": False, "status": "classified_materialized"},
    {"command": "get-three-folded-wurtzite-polarization", "module": "reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization", "class": "BinnedPolarizationTask", "shape": "streaming_reduction", "thread_safe": False, "status": "classified_materialized"},
    {"command": "write-three-folded-trajectory-with-polarity", "module": "reaxkit.analysis.ferroelectrics.three_folded_wurtzite.trajectory", "class": "PolarityExtendedXYZTask", "shape": "ordered_stateful_stream", "thread_safe": False, "status": "classified_materialized"},
)

_BY_CLASS = {
    f"{record['module']}.{record['class']}": record
    for record in PRIORITY_TASK_MANIFEST
}


def classification_for_task(task: Any) -> dict[str, Any] | None:
    key = f"{task.__class__.__module__}.{task.__class__.__name__}"
    record = _BY_CLASS.get(key)
    return dict(record) if record is not None else None


__all__ = ["PRIORITY_TASK_MANIFEST", "classification_for_task"]
