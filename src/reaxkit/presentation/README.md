# presentation

## Purpose
Formats analysis outputs for user-facing consumption: report payloads, persistence/export outputs, and plot/video rendering integration.

## What Belongs Here
- Report payload builders and registry wiring.
- Presentation specs and conversion/dispatch helpers.
- Plot and movie rendering integration code.

## What Does Not Belong Here
- Core analysis algorithms.
- Engine parsing logic.

## Structure
- `active_sites/`: active-site-specific report and plot exports.
- `plot/`: plot registry and renderer integration.
- `movie/`: video generation helpers.
- `report_registry.py`, `reporting.py`, `persist.py`, `dispatcher.py`, `convert.py`, `specs.py`, `units.py`, `export_utils.py`.

## Flow
Workflows/core pass result payloads into presentation utilities to create report structures, figures, and persisted output artifacts.

## Extension Points
- Register report payload builders in `report_registry.py`.
- Add new renderer behavior in `plot/` and update dispatch mappings.
