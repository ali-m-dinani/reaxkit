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
- `workflow_command_metadata.py` regenerates dedicated command entries from
  `command_to_workflow_module` while enriching analyzer-backed commands from
  the analysis task metadata map.

## Flow
Used by workflow orchestration to resolve which request schema a command/workflow should instantiate.

## Extension Points
- Add/modify entries when workflow command contracts change.
- Run `python -m reaxkit.workflows.data.workflow_command_metadata` after adding
  or removing a routed workflow command.
