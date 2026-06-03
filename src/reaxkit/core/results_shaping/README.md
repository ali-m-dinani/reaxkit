# core/results_shaping

## Purpose
Applies core-level result shaping utilities before downstream persistence/presentation.

## What Belongs Here
- Multi-table result packaging helpers.
- Time-axis enrichment for analysis tables.

## What Does Not Belong Here
- Renderer-specific plotting/report view logic.

## Structure
- `result_bundle.py`
- `result_time_enrichment.py`

## Flow
Called by runtime/workflows to normalize result object structure and enrich time information.

## Extension Points
- Add result-shaping utilities that are presentation-agnostic and runtime-shared.
