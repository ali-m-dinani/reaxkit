# ReaxKit Package Map

## Purpose
This directory is the import root for the ReaxKit Python package. It contains analysis logic, workflow orchestration, execution/runtime infrastructure, and UI-facing layers.

## What Belongs Here
- ReaxKit package code and package-scoped metadata.
- Cross-layer registries and contracts used by CLI/workflow execution.
- Domain-specific analysis and presentation modules.

## What Does Not Belong Here
- One-off experiments not intended for importable package use.
- Generated runtime artifacts (except explicitly tracked data/config assets).

## Structure
- [analysis](analysis/README.md): analyzers and analysis task implementations.
- [workflows](workflows/README.md): command/workflow orchestration.
- [core](core/README.md): execution runtime, registries, storage, and resolution.
- [engine](engine/README.md): engine adapters and parsing entrypoints.
- [presentation](presentation/README.md): report/plot conversion and persistence.
- [domain](domain/README.md): typed request/result/data contracts.
- [cli](cli/README.md): CLI entrypoints and path setup.
- [help](help/README.md): help-intent metadata and loaders.
- [data](data/README.md): packaged YAML constants/aliases/units.
- [utils](utils/README.md): general shared utilities.
- [webui](webui/README.md): web UI backend and presentation registry.
- [under_dev](../../../other/under_dev/README.md): in-progress modules not yet stabilized.
- [trash](../../../other/trash/README.md): deprecated or archival modules.

## Flow
Typical path: `cli -> workflows -> core runtime/registries -> engine/domain/analysis -> presentation -> webui/outputs`.

## Extension Points
- Register analysis tasks via core task registry.
- Register command/workflow routes via core routing registries.
- Add new workflow modules under `workflows/`.
