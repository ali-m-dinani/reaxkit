# engine/reaxff

## Purpose
ReaxFF-specific adapter and parsing integration.

## What Belongs Here
- ReaxFF adapter detection/load behavior.
- ReaxFF format-specific parsing entrypoints owned by this engine layer.

## What Does Not Belong Here
- Analysis task logic.
- Cross-engine runtime orchestration.

## Structure
- `adapter.py`

## Flow
Used when ReaxFF outputs are detected; provides domain model objects to runtime/analyzers.

## Extension Points
- Expand supported ReaxFF input markers/load paths in adapter logic.
