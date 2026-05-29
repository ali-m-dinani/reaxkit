# analysis

## Purpose
Implements analyzers and analysis task modules that transform parsed engine/domain data into computed scientific outputs.

## What Belongs Here
- Analyzer implementations and helper modules by domain.
- Analysis task request/result mapping metadata.
- Shared analysis base classes and validation helpers.

## What Does Not Belong Here
- CLI parsing and command routing.
- Engine file parsing adapters.
- Presentation-only rendering/report export concerns.

## Structure
- `active_sites/`: structural/event active-site analysis modules.
- `connectivity/`, `trajectory/`, `timeseries/`, `kinematics/`, `electrostatics/`, `force_field/`, `molecular_analysis/`, `params/`, `control/`: domain analyzers.
- `data/`: analyzer dataclass mapping metadata (`analysis_task_dataclass_map.yaml`).
- `base.py`, `validation.py`: shared analyzer/task scaffolding.

## Flow
Inputs come from typed domain objects prepared by engine/core runtime. Analyzers compute tables/results, then workflows/core/presentation layers persist or render them.

## Extension Points
- Register new tasks via `reaxkit.core.registry.analysis_task_registry.register_task`.
- Add analyzer dataclass mapping entries under `analysis/data/`.
