# workflows/presentation

## Purpose
Workflow-level commands that generate visual/presentation artifacts.

## What Belongs Here
- Plot/video workflow entry modules.
- Presentation command wrappers that bridge command args to presentation utilities.

## What Does Not Belong Here
- Low-level renderer implementations (belong in `reaxkit/presentation`).

## Structure
- `gen_plot_workflow.py`
- `gen_video_workflow.py`
- `plot_atom_property_workflow.py`

## Flow
These workflows call core/runtime + presentation modules to emit figures/videos and related metadata.

## Extension Points
- Add new visualization-oriented workflow commands and register routes.
