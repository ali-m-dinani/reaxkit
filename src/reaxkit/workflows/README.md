# workflows

## Purpose
Defines command/workflow orchestration that connects CLI inputs to core runtime, analyzers, and presentation outputs.

## What Belongs Here
- Top-level workflow modules and command entry wrappers.
- Workflow-level metadata maps and doc rules.
- File-tool and meta/presentation workflow utilities.

## What Does Not Belong Here
- Low-level engine parsing internals.
- Core runtime infrastructure.
- Heavy scientific analysis logic (keep in `analysis/`).

## Structure
- `file_tools/`: generator/file transformation workflows.
- `meta/`: help/introspection/study/workspace workflows.
- `presentation/`: plot/video/plot-atom-property workflows.
- `data/`: workflow dataclass mapping metadata.
- Top-level workflows: `active_site_workflow.py`, `connectivity_workflow.py`, `electrostatics_workflow.py`, `kinematics_workflow.py`, `molecular_analysis_workflow.py`, `timeseries_workflow.py`, `trajectory_workflow.py`.

## Flow
Workflow functions normalize command args, prepare request objects, invoke core runtime execution, then pass results to presentation/persist layers.

## Extension Points
- Add new command workflows and register routes in core routing registries.
- Extend workflow metadata in `workflows/data/workflow_dataclass_map.yaml`.
