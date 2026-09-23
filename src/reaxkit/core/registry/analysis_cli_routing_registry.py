"""
Registry for routing top-level analysis commands to workflow modules.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AnalysisCommandSpec:
    """
    Metadata for a direct analysis command.
    
    
    Fields
    -----
    name : str
        Field value used by this structured record.
    module_path : str
        Field value used by this structured record.
    aliases : tuple[str, ...], optional
        Backward-compatible command names resolved to ``name``.
    """

    name: str
    module_path: str
    aliases: tuple[str, ...] = ()


ANALYSIS_COMMAND_REGISTRY: dict[str, AnalysisCommandSpec] = {}


def register_analysis_command(
        name: str,
        *,
        module_path: str,
        aliases: Iterable[str] = (),
) -> AnalysisCommandSpec:
    """
    Register a direct analysis command route.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    name : str
        Input parameter used by this function.
    module_path : str
        Input parameter used by this function.
    aliases : Iterable[str], optional
        Backward-compatible command names that resolve to ``name``.
    
    Returns
    -----
    AnalysisCommandSpec
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.registry.analysis_cli_routing_registry import register_analysis_command
    # Configure required arguments for your case.
    result = register_analysis_command(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    spec = AnalysisCommandSpec(
        name=name,
        module_path=module_path,
        aliases=tuple(str(alias) for alias in aliases),
    )
    ANALYSIS_COMMAND_REGISTRY[name] = spec
    return spec


def get_registered_analysis_commands() -> dict[str, AnalysisCommandSpec]:
    """
    Return all registered direct analysis command routes.
    
    This function is part of the ReaxKit core API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    None
    
    Returns
    -----
    dict[str, AnalysisCommandSpec]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.registry.analysis_cli_routing_registry import get_registered_analysis_commands
    # Configure required arguments for your case.
    result = get_registered_analysis_commands(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return dict(ANALYSIS_COMMAND_REGISTRY)


register_analysis_command(
    "get-dipole",
    module_path="reaxkit.workflows.electrostatics.electrostatics_workflow",
    aliases=("get_dipole", "dipole"),
)
register_analysis_command(
    "get-polarization",
    module_path="reaxkit.workflows.electrostatics.electrostatics_workflow",
    aliases=("polarization",),
)
register_analysis_command(
    "charge-table",
    module_path="reaxkit.workflows.electrostatics.electrostatics_workflow",
)
register_analysis_command(
    "charge_table",
    module_path="reaxkit.workflows.electrostatics.electrostatics_workflow",
)
register_analysis_command(
    "get_polarization_field",
    module_path="reaxkit.workflows.electrostatics.electrostatics_workflow",
    aliases=("polarization_field",),
)
register_analysis_command(
    "get_dynamic_charge_changes",
    module_path="reaxkit.workflows.ferroelectrics.dynamic_charge_workflow",
    aliases=("dynamic_charge_changes", "dynamic-charge-changes"),
)
register_analysis_command(
    "get_binned_dynamic_charges",
    module_path="reaxkit.workflows.ferroelectrics.binned_dynamic_charge_workflow",
    aliases=("binned_dynamic_charges", "binned-dynamic-charges"),
)
register_analysis_command(
    "get_charge_vs_electric_field",
    module_path="reaxkit.workflows.ferroelectrics.charge_field_workflow",
    aliases=("charge_vs_electric_field", "charge-field"),
)
register_analysis_command(
    "write_trajectory_with_charges",
    module_path="reaxkit.workflows.ferroelectrics.charge_extxyz_workflow",
    aliases=("generate_charge_extxyz", "charge_extxyz", "charge-extended-xyz"),
)
register_analysis_command(
    "get-wurtzite-neighbors",
    module_path=(
        "reaxkit.workflows.ferroelectrics.four_folded_wurtzite.neighbors_workflow"
    ),
    aliases=("get_wurtzite_neighbors", "wurtzite-neighbors"),
)
register_analysis_command(
    "get-wurtzite-polarity",
    module_path=(
        "reaxkit.workflows.ferroelectrics.four_folded_wurtzite.polarity_workflow"
    ),
    aliases=("get_wurtzite_polarity", "wurtzite-polarity"),
)
register_analysis_command(
    "write-trajectory-with-polarity",
    module_path=(
        "reaxkit.workflows.ferroelectrics.four_folded_wurtzite."
        "polarity_trajectory_workflow"
    ),
    aliases=("write_trajectory_with_polarity", "polarity-extxyz"),
)
register_analysis_command(
    "get-three-folded-wurtzite-neighbors",
    module_path=(
        "reaxkit.workflows.ferroelectrics.three_folded_wurtzite.neighbors_workflow"
    ),
    aliases=(
        "get_three_folded_wurtzite_neighbors",
        "three-folded-wurtzite-neighbors",
    ),
)
register_analysis_command(
    "get-three-folded-wurtzite-polarity",
    module_path=(
        "reaxkit.workflows.ferroelectrics.three_folded_wurtzite.polarity_workflow"
    ),
    aliases=(
        "get_three_folded_wurtzite_polarity",
        "three-folded-wurtzite-polarity",
    ),
)
register_analysis_command(
    "get-three-folded-wurtzite-polarization",
    module_path=(
        "reaxkit.workflows.ferroelectrics.three_folded_wurtzite."
        "polarization_workflow"
    ),
    aliases=(
        "get_three_folded_wurtzite_polarization",
        "three-folded-wurtzite-polarization",
    ),
)
register_analysis_command(
    "write-three-folded-trajectory-with-polarity",
    module_path=(
        "reaxkit.workflows.ferroelectrics.three_folded_wurtzite."
        "polarity_trajectory_workflow"
    ),
    aliases=(
        "write_three_folded_trajectory_with_polarity",
        "three-folded-polarity-extxyz",
    ),
)
register_analysis_command(
    "get-basal-plane-displacement-dipole",
    module_path=(
        "reaxkit.workflows.ferroelectrics."
        "basal_plane_displacement_for_dipole_moment.dipole_workflow"
    ),
    aliases=("get_basal_plane_displacement_dipole", "basal-plane-dipole"),
)
register_analysis_command(
    "get-basal-plane-displacement-polarization",
    module_path=(
        "reaxkit.workflows.ferroelectrics."
        "basal_plane_displacement_for_dipole_moment.polarization_workflow"
    ),
    aliases=(
        "get_basal_plane_displacement_polarization",
        "basal-plane-polarization",
    ),
)
register_analysis_command(
    "get-basal-plane-displacement-local-polarization",
    module_path=(
        "reaxkit.workflows.ferroelectrics."
        "basal_plane_displacement_for_dipole_moment.local_polarization_workflow"
    ),
    aliases=(
        "get_basal_plane_displacement_local_polarization",
        "basal-plane-local-polarization",
    ),
)
register_analysis_command(
    "get-basal-plane-displacement-projected-polarity",
    module_path=(
        "reaxkit.workflows.ferroelectrics."
        "basal_plane_displacement_for_dipole_moment.projected_polarity_workflow"
    ),
    aliases=(
        "get_basal_plane_displacement_projected_polarity",
        "basal-plane-projected-polarity",
    ),
)
register_analysis_command(
    "get-hbn-reference-polarization",
    module_path=(
        "reaxkit.workflows.ferroelectrics.hbn_reference.polarization_workflow"
    ),
    aliases=("get_hbn_reference_polarization", "hbn-reference-polarization"),
)
register_analysis_command(
    "get-hbn-reference-local-polarization",
    module_path=(
        "reaxkit.workflows.ferroelectrics.hbn_reference."
        "local_polarization_workflow"
    ),
    aliases=(
        "get_hbn_reference_local_polarization",
        "hbn-reference-local-polarization",
    ),
)
register_analysis_command(
    "get-hbn-reference-projected-polarity",
    module_path=(
        "reaxkit.workflows.ferroelectrics.hbn_reference."
        "projected_polarity_workflow"
    ),
    aliases=(
        "get_hbn_reference_projected_polarity",
        "hbn-reference-projected-polarity",
    ),
)
register_analysis_command(
    "get-potential-and-electric-field",
    module_path=(
        "reaxkit.workflows.electrostatics.potential_and_electric_field."
        "potential_and_electric_field_workflow"
    ),
    aliases=("get_potential_and_electric_field", "reaxff-local-field"),
)
register_analysis_command(
    "write-trajectory-with-potential-and-electric-field",
    module_path=(
        "reaxkit.workflows.electrostatics.potential_and_electric_field.trajectory_workflow"
    ),
    aliases=("write_trajectory_with_potential_and_electric_field", "local-field-extxyz"),
)
register_analysis_command(
    "get-dielectric-constant",
    module_path="reaxkit.workflows.electrostatics.dielectric_constant_workflow",
    aliases=("get_dielectric_constant", "dielectric-constant"),
)
register_analysis_command(
    "fit-switching-kinetics",
    module_path=(
        "reaxkit.workflows.ferroelectrics.switching_kinetics.comparison_workflow"
    ),
    aliases=("fit_switching_kinetics",),
)
register_analysis_command(
    "fit-kai-switching",
    module_path="reaxkit.workflows.ferroelectrics.switching_kinetics.kai_workflow",
    aliases=("fit_kai_switching",),
)
register_analysis_command(
    "fit-nls-switching",
    module_path="reaxkit.workflows.ferroelectrics.switching_kinetics.nls_workflow",
    aliases=("fit_nls_switching",),
)
register_analysis_command(
    "fit-snng-switching",
    module_path="reaxkit.workflows.ferroelectrics.switching_kinetics.snng_workflow",
    aliases=("fit_snng_switching",),
)
register_analysis_command("kinematics", module_path="reaxkit.workflows.kinematics_workflow")
register_analysis_command("get_kinematics", module_path="reaxkit.workflows.kinematics_workflow")
register_analysis_command("kinematics_plot3d", module_path="reaxkit.workflows.kinematics_workflow")
register_analysis_command("kinematics_heatmap2d", module_path="reaxkit.workflows.kinematics_workflow")
register_analysis_command("get_dominant_species", module_path="reaxkit.workflows.molecular_analysis_workflow")
register_analysis_command("get_largest_molecule_by_mass", module_path="reaxkit.workflows.molecular_analysis_workflow")
register_analysis_command("get_largest_molecule_composition",
                          module_path="reaxkit.workflows.molecular_analysis_workflow")
register_analysis_command("get_molecule_lifetime", module_path="reaxkit.workflows.molecular_analysis_workflow")
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
    "get_ffield_diagnostics_sensitivity",
    module_path="reaxkit.workflows.file_tools.ffield_workflow",
)
register_analysis_command(
    "get_ffield_diagnostics_evolution",
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
    "get_ffield_opt_bulk_modulus",
    module_path="reaxkit.workflows.file_tools.ffield_workflow",
    aliases=("ffield_opt_bulk_modulus",),
)
register_analysis_command(
    "get_ffield_opt_plots",
    module_path="reaxkit.workflows.force_field_opt.get_ffield_opt_plots",
)
register_analysis_command(
    "get_force_field_opt_geo_files",
    module_path="reaxkit.workflows.force_field_opt.get_force_field_opt_geo_files",
    aliases=("get-force-field-opt-geo-files",),
)
register_analysis_command(
    "get-ffield-opt-report",
    module_path="reaxkit.workflows.force_field_opt.get_ffield_opt_report",
    aliases=("get_ffield_opt_report",),
)
register_analysis_command("get_trainset_data", module_path="reaxkit.workflows.file_tools.trainset_workflow")
register_analysis_command("get_trainset_group_comments", module_path="reaxkit.workflows.file_tools.trainset_workflow")
register_analysis_command("get-params", module_path="reaxkit.workflows.params_workflow")
register_analysis_command(
    "msd",
    module_path="reaxkit.workflows.trajectory_workflow",
    aliases=("mean-square-displacement", "mean_square_displacement"),
)
register_analysis_command(
    "get_msd",
    module_path="reaxkit.workflows.trajectory_workflow",
    aliases=("msd", "get-msd", "mean-square-displacement", "mean_square_displacement"),
)
register_analysis_command(
    "diffusivity",
    module_path="reaxkit.workflows.trajectory_workflow",
    aliases=("diffusion-coefficient", "diffusion_coefficient"),
)
register_analysis_command(
    "get_diffusivity",
    module_path="reaxkit.workflows.trajectory_workflow",
    aliases=("diffusivity", "get-diffusivity", "diffusion-coefficient", "diffusion_coefficient"),
)
register_analysis_command("rdf", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command(
    "get_rdf",
    module_path="reaxkit.workflows.trajectory_workflow",
    aliases=("rdf", "get-rdf"),
)
register_analysis_command("rdf_property", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command(
    "get_rdf_property",
    module_path="reaxkit.workflows.trajectory_workflow",
    aliases=("rdf_property", "rdf-property", "get-rdf-property"),
)
register_analysis_command("voronoi", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command("get_dihedral", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command("get_voronoi", module_path="reaxkit.workflows.trajectory_workflow")
register_analysis_command(
    "get_z_binned_top_bottom_strain",
    module_path="reaxkit.workflows.stress_strain.z_binned_strain_workflow",
    aliases=("get-z-binned-top-bottom-strain", "z_binned_strain_using_top_bottom_atoms",
             "z-binned-strain-using-top-bottom-atoms"),
)
register_analysis_command(
    "get_z_binned_deformation_gradient_strain",
    module_path="reaxkit.workflows.stress_strain.z_binned_strain_workflow",
    aliases=("get-z-binned-deformation-gradient-strain", "z_binned_deformation_gradient_strain",
             "z-binned-deformation-gradient-strain"),
)
register_analysis_command("connection_list", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("get_connection_list", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("connection_table", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("get_connection_table", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("connection_stats", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("get_connection_stats", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("bond_events", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("get_bond_events", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("coordination", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("get_coordination", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("coordination_relabel", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("relabel_traj_using_coordination", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("hybridization", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("get_hybridization", module_path="reaxkit.workflows.connectivity_workflow")
register_analysis_command("plot_atom_property", module_path="reaxkit.workflows.meta.plot_atom_property_workflow")
register_analysis_command(
    "get_active_site_structural",
    module_path="reaxkit.workflows.active_site_workflow",
    aliases=("active_site_structural", "active-site-structural", "get-active-site-structural"),
)
register_analysis_command(
    "get_active_site_events",
    module_path="reaxkit.workflows.active_site_workflow",
    aliases=("active_site_events", "active-site-events", "get-active-site-events"),
)

_TIMESERIES_WORKFLOW_COMMANDS = (
    "get_potential_energy",
    "get_num_of_atoms",
    "get_volume",
    "get_temperature",
    "get_pressure",
    "get_density",
    "get_elapsed_time",
    "get_a",
    "get_b",
    "get_c",
    "get_alpha",
    "get_beta",
    "get_gamma",
    "get_frames_count",
    "get_trajectory",
    "get_displacement",
    "get_charge",
    "get_cell_dimensions",
    "get_electric_field",
    "get_eregime",
    "get_partial_energy",
    "get_restraint",
    "get_molecular_frequency",
    "get_molecular_totals",
    "get_total_molecules",
    "get_total_atoms",
    "get_total_molecular_mass",
    "get_geometry_optimization",
)
for _command in _TIMESERIES_WORKFLOW_COMMANDS:
    register_analysis_command(
        _command,
        module_path=f"reaxkit.workflows.timeseries.{_command}",
    )
