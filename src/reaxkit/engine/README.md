# engine

## Purpose
Implements engine adapters and handlers that parse raw simulation outputs into typed domain data for analysis tasks.

## What Belongs Here
- Adapter classes for each supported engine.
- Handler modules for engine-specific file parsing.
- Shared engine abstraction/base classes.

## What Does Not Belong Here
- Analyzer math/business logic.
- CLI command definitions.

## Structure
- `ams/`: AMS adapter and RKF handlers.
- `lammps/`: LAMMPS adapter and dump/log handlers.
- `reaxff/`: ReaxFF adapter and handlers.
- `common/`: cross-engine shared helpers.
- `base.py`: adapter base abstractions.

## Flow
Core runtime resolves an engine adapter, then calls adapter load functions to produce domain model objects required by tasks.

## Extension Points
- Register new engine adapters via core platform engine resolver/registry integration.
