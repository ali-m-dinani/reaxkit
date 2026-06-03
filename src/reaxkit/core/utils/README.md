# core/utils

## Purpose
Holds core-scoped utility helpers used by runtime/registry/resolve modules.

## What Belongs Here
- Frame/atom selection parsing helpers used by core/workflow flows.

## What Does Not Belong Here
- General-purpose helpers with broader package ownership (use `reaxkit/utils`).

## Structure
- `frame_utils.py`

## Flow
Used during request normalization and frame index resolution before runtime execution.

## Extension Points
- Add utilities here only when tightly coupled to core orchestration behavior.
