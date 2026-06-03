# help

## Purpose
Provides packaged help/intent metadata and utilities for command help/introspection experiences.

## What Belongs Here
- Help index loaders and intent utilities.
- YAML metadata powering help search/intents.

## What Does Not Belong Here
- Runtime command execution logic.
- Analyzer implementations.

## Structure
- `data/`: help information sources and intent YAML files.
- `help_index_loader.py`
- `introspection_utils.py`

## Flow
Help workflows/modules load this metadata to answer command discovery and intent mapping requests.

## Extension Points
- Add new help metadata fields in `help/data/` and extend loader/parser logic accordingly.
