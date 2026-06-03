# engine/ams

## Purpose
AMS-specific adapter and RKF parsing handlers.

## What Belongs Here
- AMS adapter detection/load logic.
- RKF/KF handler implementations.

## What Does Not Belong Here
- Generic cross-engine registry/runtime code.

## Structure
- `adapter.py`
- `rkf_handler.py`

## Flow
Selected by engine resolver when AMS inputs are detected, then produces domain data for analysis tasks.

## Extension Points
- Add AMS-file coverage by extending handler parsing methods.
