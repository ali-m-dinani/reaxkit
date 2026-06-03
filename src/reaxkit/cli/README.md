# cli

## Purpose
Provides command-line entrypoints and bootstrap path setup for invoking ReaxKit workflows and commands.

## What Belongs Here
- CLI startup/entrypoint modules.
- Path/bootstrap helpers required before command dispatch.

## What Does Not Belong Here
- Analysis algorithm implementations.
- Workflow business logic.

## Structure
- `main.py`: main CLI entrypoint.
- `path.py`: path/environment bootstrapping.
- `__init__.py`: package exports.

## Flow
CLI parses and normalizes user command input, then dispatches into workflow/core command resolution.

## Extension Points
- Add new top-level CLI behavior only when it cannot live cleanly in workflows/core registries.
