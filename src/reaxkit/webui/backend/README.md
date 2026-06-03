# webui/backend

## Purpose
Backend utilities for web UI request handling, payload shaping, serialization, and API schema support.

## What Belongs Here
- API request/response helpers.
- Payload schemas/serializers and backend registries.

## What Does Not Belong Here
- UI layout components.
- Core runtime orchestration internals.

## Structure
- `api.py`, `schemas.py`, `serializer.py`, `tabular_payload.py`, `pipeline_store.py`, `utility_registry.py`, `node_runtime.py`.

## Flow
Transforms workflow/runtime outputs into web-friendly payloads consumed by UI callbacks/components.

## Extension Points
- Add new payload serializers and schema validators for new UI views/endpoints.
