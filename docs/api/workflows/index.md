# Workflows API

This section documents ReaxKit workflow modules (CLI-facing orchestration layer).
Workflows parse user arguments, load data/adapters, execute tasks, and route outputs.

## Structure

Workflow docs are organized into:

- top-level domain workflows
- `file_tools` workflows
- `meta` workflows
- `presentation` workflows
- `study_design` workflows

## Module Index

### Top-Level Workflows

- [alaki_workflow](alaki_workflow.md)
- [connectivity_workflow](connectivity_workflow.md)
- [electrostatics_workflow](electrostatics_workflow.md)
- [kinematics_workflow](kinematics_workflow.md)
- [molecular_analysis_workflow](molecular_analysis_workflow.md)
- [timeseries_workflow](timeseries_workflow.md)
- [trajectory_workflow](trajectory_workflow.md)

### File Tools

- [addmol_workflow](file_tools/addmol_workflow.md)
- [charges_workflow](file_tools/charges_workflow.md)
- [control_workflow](file_tools/control_workflow.md)
- [eregime_workflow](file_tools/eregime_workflow.md)
- [ffield_workflow](file_tools/ffield_workflow.md)
- [fort7_workflow](file_tools/fort7_workflow.md)
- [fort83_workflow](file_tools/fort83_workflow.md)
- [geo_workflow](file_tools/geo_workflow.md)
- [kopple2_workflow](file_tools/kopple2_workflow.md)
- [params_workflow](file_tools/params_workflow.md)
- [trainset_workflow](file_tools/trainset_workflow.md)
- [tregime_workflow](file_tools/tregime_workflow.md)
- [vregime_workflow](file_tools/vregime_workflow.md)
- [xmolout_workflow](file_tools/xmolout_workflow.md)

### Meta

- [command_alias_workflow](meta/command_alias_workflow.md)
- [gui_workflow](meta/gui_workflow.md)
- [help_workflow](meta/help_workflow.md)
- [introspection_workflow](meta/introspection_workflow.md)
- [manage_workspace_workflow](meta/manage_workspace_workflow.md)

### Presentation

- [gen_plot_workflow](presentation/gen_plot_workflow.md)
- [gen_video_workflow](presentation/gen_video_workflow.md)
- [plot_atom_property_workflow](presentation/plot_atom_property_workflow.md)

### Study Design

- [runtime](study_design/runtime.md)
- [study_workflow](study_design/study_workflow.md)
