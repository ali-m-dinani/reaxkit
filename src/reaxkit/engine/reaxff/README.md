# engine/reaxff

## Purpose
ReaxFF-specific adapter, file I/O handlers, generators, and normalization/load integration.

## What Belongs Here
- ReaxFF adapter detection and engine-facing load/write entrypoints.
- ReaxFF format handlers and parser logic under `io/`.
- ReaxFF output/input generators under `generators/`.
- ReaxFF reference data files under `data/`.
- Internal adapter implementation split by responsibility under `adapter_parts/`.

## What Does Not Belong Here
- Analysis task logic.
- Cross-engine runtime orchestration.
- Web UI presentation logic.
- Domain model definitions (they belong in `reaxkit/domain`).

## Structure
- `adapter.py`: Public ReaxFF adapter facade registered via engine resolver.
- `adapter_parts/`:
  - `io_paths.py`: Path resolution and fast frame-count probes.
  - `timing.py`: Shared load timing and handler construction helpers.
  - `normalizers.py`: Handler-to-domain-model conversion functions.
  - `loaders_dynamics.py`: Trajectory/geometry/simulation/connectivity loaders.
  - `loaders_forcefield.py`: Force-field and optimization loaders/bundles.
  - `loaders_properties.py`: Energies, charges, electrostatics, control, and related loaders.
  - `writers.py`: Write-side helpers for control and xmolout outputs.
- `io/`: ReaxFF file handlers (e.g., `fort.*`, `xmolout`, `geo`, `summary`, etc.).
- `generators/`: File writers and training-set/control generation helpers.
- `data/`: Engine-specific static/reference data.

## Flow
Used when ReaxFF outputs are detected:
1. `adapter.py` receives runtime load/write requests.
2. Adapter delegates to `adapter_parts/loaders_*` and `writers`.
3. Loaders build handlers from `io/`, normalize through `normalizers.py`, and return domain models.
4. Runtime/analyzers consume the returned canonical data objects.

## Extension Points
- Add/adjust file-path resolution and probe logic in `adapter_parts/io_paths.py`.
- Add new handler-to-model conversions in `adapter_parts/normalizers.py`.
- Add loader families in new `adapter_parts/loaders_*.py` modules when new domains are introduced.
- Keep `adapter.py` thin; prefer expanding submodules instead of adding large in-class logic blocks.
