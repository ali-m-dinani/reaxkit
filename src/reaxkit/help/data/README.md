# help/data

## Purpose
Contains YAML sources for help content, intent routing, and help information provenance.

## What Belongs Here
- Help information source metadata.
- Help intent definitions and mappings.

## What Does Not Belong Here
- Executable help logic (belongs in `reaxkit/help/*.py`).

## Structure
- `help_information_sources.yaml`
- `help_intents.yaml`

## Flow
Loaded by help index/introspection utilities to answer help discovery queries.

## Extension Points
- Add new intents/sources with stable keys expected by help loaders.
