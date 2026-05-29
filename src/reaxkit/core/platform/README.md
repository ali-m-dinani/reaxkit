# core/platform

## Purpose
Platform-level primitives shared across core runtime: logging, exceptions, constants, and engine resolution.

## What Belongs Here
- `exceptions.py`, `log.py`, `constants.py`.
- Engine resolution/detection entrypoint (`engine_resolver.py`).

## What Does Not Belong Here
- Command registries.
- Workflow/business orchestration.

## Structure
- `engine_resolver.py`: choose engine adapter from hints/detection.
- `log.py`: logging setup helpers.
- `exceptions.py`: core exception classes.
- `constants.py`: packaged constant loading helpers.

## Flow
Used by runtime/storage/resolve modules to enforce consistent platform behavior.

## Extension Points
- Add shared platform primitives only when broadly reused by core subpackages.
