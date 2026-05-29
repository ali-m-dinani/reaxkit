# core/runtime

## Purpose
Implements runtime execution orchestration for analysis and generator command flows.

## What Belongs Here
- Analysis executor lifecycle.
- Generator output/runtime normalization helpers.
- Progress reporter resolution.

## What Does Not Belong Here
- Static registration metadata.
- Engine-specific parsing code.

## Structure
- `analysis_executor.py`
- `generator_runtime.py`
- `progress.py`

## Flow
Runtime receives normalized args + request/task, resolves inputs/load/cache/execute/persist timing, and returns result objects.

## Extension Points
- Add execution policies (timing/cache/reporting hooks) in `analysis_executor.py`.
