# webui/presentation

## Purpose
Web-UI-specific presentation registry and performance configuration.

## What Belongs Here
- UI presentation registry wiring.
- Performance config JSON/settings for UI rendering behavior.

## What Does Not Belong Here
- General package presentation logic (belongs in `reaxkit/presentation`).

## Structure
- `registry.py`
- `perf_config.py`
- `ui_performance.json`

## Flow
Used by web UI layer to determine available presentation handlers and runtime rendering constraints.

## Extension Points
- Add new web presentation handlers and tune performance settings.
