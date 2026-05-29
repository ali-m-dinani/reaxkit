# core/registry

## Purpose
Central registries for tasks, workflows, commands, generators, and analysis routing.

## What Belongs Here
- Route/task registration data structures.
- Lightweight registration decorators/helpers.

## What Does Not Belong Here
- Heavy execution logic.
- Engine parsing or analysis algorithms.

## Structure
- `analysis_task_registry.py`
- `analysis_cli_routing_registry.py`
- `workflow_cli_routing_registry.py`
- `generator_cli_routing_registry.py`
- `command_catalog.py`

## Flow
CLI/workflows query these registries to resolve what command name maps to which runtime target.

## Extension Points
- Register new command/workflow/generator/task names here.
