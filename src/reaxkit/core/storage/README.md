# core/storage

## Purpose
Provides storage layout, cache management, and parsed artifact persistence for run-scoped reproducible execution.

## What Belongs Here
- Cache key/value persistence for analysis results.
- Run/index directory layout conventions.
- Parsed artifact serialization utilities.

## What Does Not Belong Here
- UI-level export formatting.
- Domain-specific analysis computation.

## Structure
- `cache_manager.py`
- `storage_layout.py`
- `parsed_store.py`

## Flow
Runtime writes/reads parsed and analyzed artifacts via this subpackage to support cache hits and reproducibility.

## Extension Points
- Extend storage schema/index versioning in `storage_layout.py`.
