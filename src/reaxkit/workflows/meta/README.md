# workflows/meta

## Purpose
Meta workflows for help, introspection, workspace management, command aliases, and study orchestration.

## What Belongs Here
- Operational workflows that manage or introspect the tool itself.
- Study and workspace utility workflows.

## What Does Not Belong Here
- Domain analysis implementations.

## Structure
- `help_workflow.py`, `introspection_workflow.py`, `gui_workflow.py`, `manage_workspace_workflow.py`, `command_alias_workflow.py`, `study_workflow.py`.

## Flow
Invoked as meta commands; they orchestrate core/help/study modules rather than running scientific analyzers directly.

## Extension Points
- Add new operational workflows for developer or user productivity tasks.
