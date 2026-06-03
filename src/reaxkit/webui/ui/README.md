# webui/ui

## Purpose
Owns web UI layout composition and view structure.

## What Belongs Here
- UI layout module(s) and component composition wrappers.

## What Does Not Belong Here
- Backend payload serialization.
- Command/workflow execution logic.

## Structure
- `layout.py`

## Flow
Receives prepared backend/presentation payloads and renders interactive UI surfaces.

## Extension Points
- Add new UI screens/panels and keep layout contracts aligned with backend payload schemas.
