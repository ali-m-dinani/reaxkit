# webui

## Purpose
Implements the web application layer, including API/backend logic, UI composition, and web-specific presentation registry/performance settings.

## What Belongs Here
- UI layout/callback modules and Dash app wiring.
- Web backend payload/schema/serialization helpers.
- Web presentation registry/perf config.

## What Does Not Belong Here
- Engine parsing and analysis task implementations.
- Core runtime registry infrastructure.

## Structure
- `backend/`: API schemas, serializers, backend registries.
- `ui/`: page/layout and callback-facing UI composition.
- `presentation/`: web presentation registry and perf config.
- `app.py`, `dash_app.py`, `callbacks.py`, `components.py`, `layouts.py`, `utils.py`.

## Flow
Web UI issues command/workflow requests, receives analyzed/persisted payloads, and renders interactive views/export actions.

## Extension Points
- Add UI feature surfaces under `ui/`.
- Add backend payload transformations under `backend/`.
