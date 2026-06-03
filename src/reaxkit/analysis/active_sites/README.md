# analysis/active_sites

## Purpose
Active-site analysis package implementing structural and event tasks plus local helper modules.

## What Belongs Here
- Active-site task implementations (`structural.py`, `events.py`).
- Active-site-local helper modules (classification, defects, PBC, schemas, TRACT projection).

## What Does Not Belong Here
- Generic plotting/reporting not specific to active-sites.
- Cross-domain analysis utilities shared broadly across analysis packages.

## Structure
- Task modules: `structural.py`, `events.py`.
- Request/result models: `models.py`.
- Helpers: `classification.py`, `defects.py`, `pbc.py`, `tract_compat.py`.
- Package exports/registration: `__init__.py`.

## Flow
Workflows invoke active-site tasks via core runtime; task outputs are returned as full table + TRACT-compatible table + summary.

## Extension Points
- Register new active-site task variants via task registry.
- Keep helper logic local here unless reused across multiple analysis domains.
