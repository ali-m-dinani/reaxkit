"""Registry for routing top-level analysis commands to workflow modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisCommandSpec:
    """Metadata for a direct analysis command."""

    name: str
    module_path: str


ANALYSIS_COMMAND_REGISTRY: dict[str, AnalysisCommandSpec] = {}


def register_analysis_command(name: str, *, module_path: str) -> AnalysisCommandSpec:
    """Register a direct analysis command route."""
    spec = AnalysisCommandSpec(name=name, module_path=module_path)
    ANALYSIS_COMMAND_REGISTRY[name] = spec
    return spec


def get_registered_analysis_commands() -> dict[str, AnalysisCommandSpec]:
    """Return all registered direct analysis command routes."""
    return dict(ANALYSIS_COMMAND_REGISTRY)


register_analysis_command("dipole", module_path="reaxkit.workflows.electrostatics_workflow")
register_analysis_command("polarization", module_path="reaxkit.workflows.electrostatics_workflow")
register_analysis_command("charge-table", module_path="reaxkit.workflows.electrostatics_workflow")
register_analysis_command("polarization_field", module_path="reaxkit.workflows.electrostatics_workflow")
register_analysis_command("kinematics", module_path="reaxkit.workflows.kinematics_workflow")
register_analysis_command("kinematics_plot3d", module_path="reaxkit.workflows.kinematics_workflow")
register_analysis_command("kinematics_heatmap2d", module_path="reaxkit.workflows.kinematics_workflow")
register_analysis_command("dominant_species", module_path="reaxkit.workflows.molecular_analysis_workflow")
register_analysis_command("largest_molecule_by_mass", module_path="reaxkit.workflows.molecular_analysis_workflow")
register_analysis_command("largest_molecule_composition", module_path="reaxkit.workflows.molecular_analysis_workflow")
register_analysis_command("molecule_lifetime", module_path="reaxkit.workflows.molecular_analysis_workflow")
register_analysis_command("get_ffield_data", module_path="reaxkit.workflows.file_tools.ffield_workflow")
register_analysis_command("get_ffield_opt_progress_data", module_path="reaxkit.workflows.file_tools.ffield_workflow")
register_analysis_command("get_energy_min_summary_data", module_path="reaxkit.workflows.file_tools.ffield_workflow")
register_analysis_command(
    "get_ffield_diagnostic_data",
    module_path="reaxkit.workflows.file_tools.ffield_workflow",
)
register_analysis_command(
    "parameter_optimization_most_sensitive",
    module_path="reaxkit.workflows.file_tools.ffield_workflow",
)
register_analysis_command("parameter_optimization_tornado", module_path="reaxkit.workflows.file_tools.ffield_workflow")
register_analysis_command("get_ffield_opt_results", module_path="reaxkit.workflows.file_tools.ffield_workflow")
register_analysis_command(
    "get_ffield_opt_eos",
    module_path="reaxkit.workflows.file_tools.ffield_workflow",
)
register_analysis_command(
    "ffield_opt_bulk_modulus",
    module_path="reaxkit.workflows.file_tools.ffield_workflow",
)
register_analysis_command("get_trainset_data", module_path="reaxkit.workflows.file_tools.trainset_workflow")
register_analysis_command("get_trainset_group_comments", module_path="reaxkit.workflows.file_tools.trainset_workflow")
register_analysis_command("get-params", module_path="reaxkit.workflows.params_workflow")
register_analysis_command("msd", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command("diffusivity", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command("rdf", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command("rdf_property", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command("voronoi", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command("connection_list", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("connection_table", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("connection_stats", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("bond_events", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("coordination", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("coordination_relabel", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("hybridization", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("atom_property_plot3d", module_path="reaxkit.workflows.spatial_property_workflow")
register_analysis_command("atom_property_heatmap2d", module_path="reaxkit.workflows.spatial_property_workflow")
register_analysis_command("active_site_structural", module_path="reaxkit.workflows.active_site_workflow")
register_analysis_command("active_site_events", module_path="reaxkit.workflows.active_site_workflow")
