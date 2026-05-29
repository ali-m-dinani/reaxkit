# core

## Purpose
Owns command/task registries, runtime execution orchestration, storage/caching, and command/alias resolution used by workflows and CLI.

## What Belongs Here
- Runtime orchestration and execution infrastructure.
- Registry/resolve/storage/platform internals shared across package layers.
- Study orchestration internals.

## What Does Not Belong Here
- Domain-specific scientific analysis algorithms.
- Engine-format parsing implementations.
- UI-specific rendering logic.

## Structure
- [platform](platform/README.md): constants, exceptions, logging, engine resolver.
- [registry](registry/README.md): command/task/workflow route registries.
- [resolve](resolve/README.md): alias and command resolution.
- [runtime](runtime/README.md): executor/generator runtime and progress.
- [storage](storage/README.md): cache, parsed artifacts, run layout.
- [results_shaping](results_shaping/README.md): result packaging/time enrichment.
- [study](study/README.md): study-run engines and study schema/io.
- [utils](utils/README.md): core-scoped helper utilities.

## Flow
Workflows call core runtime, which resolves engine/commands, loads data, executes tasks, applies caching/storage, then returns enriched results.

## Extension Points
- Register tasks/commands/workflows/generators via registry subpackage.
- Extend runtime behavior in `runtime/analysis_executor.py` and storage policy in `storage/`.
