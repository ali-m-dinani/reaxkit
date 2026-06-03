# workflows/data

## Purpose
Stores workflow-layer metadata used for command-to-workflow dataclass/request mapping.

## What Belongs Here
- Workflow dataclass mapping YAMLs.

## What Does Not Belong Here
- Workflow execution code.
- Global package constants.

## Structure
- `workflow_dataclass_map.yaml`

## Flow
Used by workflow orchestration to resolve which request schema a command/workflow should instantiate.

## Extension Points
- Add/modify entries when workflow command contracts change.
