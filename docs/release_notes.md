# Release notes

## 3.1.0 (2026-09-24)

- Added polar surface generation for signed Miller indices, stable XYZ-to-GEO
  descriptions from input filenames, and dielectric spectrum PNG plots.
- Matplotlib is now a core dependency because the dielectric workflow writes a
  spectrum plot by default.

- CLI `-h` now highlights required inputs, common scientific choices, and
  outputs. `--help-all` (also `--all-flags`) shows every accepted option in
  purpose-based tables, including workflow tasks and advanced file overrides.
  Parsing defaults and existing flags are unchanged. Cache overrides such as
  `--no-input-cache` remain available for reproducibility and benchmarks.
- Repaired legacy kinematics, molecular-analysis, connectivity, Voronoi, and
  control-file command aliases. Restored parser support for kinematics 3D
  plots and 2D heatmaps and normalized control-generation command spellings.
- Added shared execution overrides, affinity/Slurm-aware worker selection,
  bounded scheduling, scoped native-thread limits, and execution-plan metadata.
  Lightweight kernels stay serial automatically when thread overhead dominates.
- Added bounded coordinate, charge, displacement, dihedral, connectivity,
  neighbor and SciPy Voronoi paths; streaming bond events and connection
  statistics; and disk-backed blocked time-origin MSD and per-atom diffusivity.
- Completed strain history scans, molecule-lifetime/population streams,
  reference-dependent polarity/polarization, and incremental polarity extxyz.
  Local trajectory results preserve labels and time in temporary disk storage.
- Added `standard`, `minimal`, `full`, and `legacy` artifact profiles. Declared
  optional details are omitted by default and stream to Parquet when requested.
  Core filenames remain stable. Use `--output-profile legacy` for historical
  detail CSV output, or `--detail-format csv` for a specific format override.
  Legacy compatibility is retained throughout 3.x, including at least the next
  minor release. Dynamic-charge details now require `--write-detailed-charges`
  or `full`/`legacy`; polarity neighbor diagnostics are optional, and requested
  h-BN displacement details default to Parquet. `minimal` suppresses details.
- Routed workflow table writers (including study and snapshot row writers) and
  generator metadata through shared publication; fixed three stale CLI routes.
  Inventory checks now reject undeclared tasks and missing workflow modules.
- Added staged artifact publication, rollback on write failure, empty-table
  schemas, lazy table producers, and schema/provenance manifests. See the
  [execution and output guide](for_developers/unified_execution.md) for coverage.

- Added a persistent, dependency-free source-frame cache for overlapping
  xmolout and fort.7 selections. Finite selections now reuse individual parsed
  frames across commands and processes and seek directly to uncached frames.
- Added `--input-cache` / `--no-input-cache`, a 10 GiB default
  `--frame-cache-max-gb` limit, cache hit/miss diagnostics, and frame-cache
  inspection and clearing through `manage-workspace --folder cache/frames`.
- Kept analysis-result caching independent, so changes such as dynamic versus
  formal charge still produce separate results while sharing compatible input
  frames.
