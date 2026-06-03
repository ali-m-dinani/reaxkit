# core/resolve

## Purpose
Resolves canonical names and aliases for commands/variables, including user-defined alias overlays.

## What Belongs Here
- Command alias normalization and tolerant matching.
- User alias persistence/load logic.
- Variable alias resolution utilities.

## What Does Not Belong Here
- Task execution.
- Registry ownership/state mutation outside resolution concerns.

## Structure
- `command_alias_resolver.py`
- `user_command_aliases.py`
- `alias.py`

## Flow
Used before dispatch to map user-provided tokens into canonical command/data names.

## Extension Points
- Add alias domains by extending normalization and alias map loaders.
