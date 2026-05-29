# utils

## Purpose
Hosts shared utility modules that are not tightly coupled to one package layer.

## What Belongs Here
- Reusable numerical/media/data helper modules with low coupling.
- Utility metadata used by helper tooling.

## What Does Not Belong Here
- Domain-specific analyzer implementations.
- Core runtime orchestration code.

## Structure
- `numerical/`: signal/statistics/numerical helper functions.
- `media/`: media helpers (currently sparse).
- `data/`: utility metadata YAML.
- `equation_of_states.py`

## Flow
Imported by analysis/workflow/presentation modules when generic utility logic is needed.

## Extension Points
- Add utility modules only when they are genuinely cross-package and not owned by a single domain folder.
