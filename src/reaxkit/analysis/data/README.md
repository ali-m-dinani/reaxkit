# analysis/data

## Purpose
Stores analysis-layer metadata maps used to associate task names with dataclass/request contracts.

## What Belongs Here
- Analyzer task dataclass mapping YAMLs.

## What Does Not Belong Here
- Analyzer code implementations.
- Workflow routing maps.

## Structure
- `analysis_task_dataclass_map.yaml`

## Flow
Loaded by analysis/workflow plumbing to build request objects from command/task names.

## Extension Points
- Update this map whenever a new analysis task dataclass contract is introduced.
