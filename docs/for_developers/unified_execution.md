# Execution and artifact contracts

Every registered analysis has an explicit dependency classification. The
generated [command inventory](../command_execution_inventory.json) records
capabilities, result tables, public names, default tiers, and streaming/blocked
entry points. Regenerate it with:

```bash
python -m reaxkit.core.runtime.command_inventory
```

`AnalysisExecutor` resolves one policy after checking the engine's streaming
support. CPU selection uses Slurm, process affinity, then host CPU count;
queue size also accounts for selected frames and memory estimates. Multiple
frame workers temporarily limit native BLAS/OpenMP pools to one thread.
The timing log and saved settings include the policy and fallback reason.
Generators declare `single` and always use one worker.

Advanced options are accepted by the shared CLI:

```text
--execution auto|serial|threads|processes
--workers auto|N --chunk-size auto|N
--output-profile standard|minimal|full|legacy
--detail-format parquet|csv
```

An explicit worker count is capped by the allocation and memory budget.
`processes` currently falls back to serial with a reason: no kernel has a
validated process transport. Thread safety alone does not justify enabling
automatic parallelism. Set `automatic_parallel=False` until equivalence and
performance measurements justify the default. Small coordinate, charge,
displacement, dihedral, connectivity, and neighbor kernels currently use
bounded serial streaming automatically; explicit threads remain available.
The SciPy Voronoi kernels passed both gates.

## Implementing a task

Declare `TaskCapabilities` with dependency shape, thread safety, selective-frame
support, reference fields, and conservative per-frame/result memory estimates.
Keep the materialized `run` path as a scientific reference during migration.
Use `BoundedFramePipeline.map_ordered`, `map_reference`, `scan_ordered`, or
`reduce_ordered` rather than creating another executor. Results carry original
source-frame indices; collection order is deterministic. Close the iterator
on cancellation so readers, native thread limits, and temporary files are
released. Worker errors report the source frame.

`reference_frames` reads a dependency once, spooling a forward-only prefix to
temporary storage if necessary. Preparation marks shared NumPy arrays read-only.
Dependency-only frames must be filtered from public outputs. Coordinate and
charge frame maps, displacement, dihedral, connectivity categories, both
neighbor analyses, both SciPy Voronoi analyses, electrostatics totals, and the
two projected-polarity prototypes use shared bounded execution. The remaining
polarity, basal-plane/h-BN polarization, molecular ranking/series, relabeling,
and RDF paths use the same runtime. Both polarity trajectory exporters publish
incremental extxyz through the shared artifact transaction.

Connection statistics keep per-pair count/sum/max state. Bond events maintain
bounded moving-average/EMA, hysteresis, and flicker state per observed pair.
Time-origin averaged MSD uses temporary coordinate storage and bounded lag
blocks, preserving the serial unwrapping convention. Its temporary disk use
scales with selected coordinates; its RAM excludes the full trajectory.
Result tables can still scale with the size of the requested primary result.

Diffusivity uses one source pass and temporary coordinate storage, fitting one
atom trace at a time with the original periodic unwrapping and polynomial fit.
Its working memory is O(frames + atoms), excluding its compact per-atom result.
Exact dynamic-charge medians read bounded atom-trace blocks from scratch files,
avoiding whole-matrix memory-map residency. Local-polarization and relabeling
results retain their public trajectory through
owned temporary disk mappings; labels and physical times are preserved. Close
`result._trajectory_spool` when finished with a long-lived API result, or release
the result. These temporary results bypass the persistent analysis-result cache.

The global/single-command audit is explicit in each inventory execution note:

- Dielectric autocorrelation/FFT consumes a compact dipole/time series and stays
  global; frame splitting would change the correlation and spectrum.
- Isomer assignment compares a supplied geometry collection and shared
  equivalence classes, so it retains its global comparison algorithm.
- Force-field, optimization, trainset, geometry reports and kinematics operate
  on one supplied file/report; connectivity matrices and structural-site queries
  select one frame. They do not use generic frame workers.
- Simulation, cell, external-field, energy-regime, partial-energy and restraint
  series consume engine-provided scalar tables. Those engines do not expose a
  canonical frame iterator for these tables; their primary outputs already scale
  with the scalar series, without a coordinate trajectory.
- External pyvoro and RDF backends stream bounded frames serially until their
  thread safety is validated. RDF returns a curve for each requested frame;
  averaging histograms across frames would change the public scientific result.
  Mergeable fixed-edge histogram and count/sum reducers are available, and fixed
  charge bins use the shared count/sum accumulator.

Both strain analyses scan contiguous periodic history with fixed reference
geometry. Molecule lifetimes retain active segment state per formula; active-site
scans preserve chronological transitions and diagnostic frame limits. Molecular
population parsing reads one iteration at a time, directly from the original
source instead of copying or materializing the entire file.

## Output profiles and compatibility

| Profile | Artifacts |
| --- | --- |
| `standard` | Declared core/summary outputs and explicit detail opt-ins |
| `minimal` | Core outputs |
| `full` | All declared artifacts; tabular details prefer Parquet |
| `legacy` | All declared artifacts with historical detail CSV formats |

Existing core filenames remain stable. `legacy` will remain available throughout
the 3.x release series, including at least the next minor release after this
transition. An explicit `--detail-format` overrides the profile's detail format.
Plots retain their command-specific opt-in behavior.

Declare `ArtifactSpec` by scientific role; a large primary result remains core.
Use `TableChunks` for lazy details and `TableAccumulator` for incremental output.
Disabled producers must never be consumed. `ArtifactWriter` supports CSV,
Parquet row groups, JSON, and extxyz producer callbacks. It stages outputs,
publishes the manifest last, and rolls back caught publication failures.
The manifest records schemas, row counts, units, source-frame bounds, omitted
artifacts, and run metadata. Files are individually atomic; the manifest is the
completion marker. This is not a filesystem-wide atomic transaction across
power loss or termination during publication.

All workflow DataFrame CSV/Parquet and legacy row-writer sites now use the shared
writer, including study aggregation and force-field snapshots. CI rejects new
direct table writers. Generator settings and primary-output metadata use the
same manifest policy; the original generator owns its primary file creation.
Arbitrary plots, text reports and generator-specific primary files are not a
single atomic filesystem transaction.

Dynamic-charge details require `--write-detailed-charges` (or `full`/`legacy`).
Polarity neighbor diagnostics and h-BN atom displacements are optional; h-BN also
accepts `--write-displacements`. Requested details spool bounded frame batches
and publish Parquet row groups (`legacy` selects CSV). `minimal` overrides detail
opt-ins. Neighbor geometry needed to calculate a frame is temporary scientific
working state, and is discarded after its derived results are collected.
Per-site dipoles, unique-ion contributions, coordination lists and complete
Voronoi geometry are core when they are the requested scientific result; their
size is not hidden by silently omitting primary rows.

## Validation

`tests/core/test_unified_rollout.py` compares frozen serial results, selected
frames, noncontiguous IDs, periodic images, categories, undefined values, and
floating-point results (up to `1e-12` where order changes). It also tests
subprocess peak RSS, failure rollback, disabled producers, reference ordering,
native-thread restoration, and policy overrides. Existing engine tests cover
coordinate/charge alignment and selected-frame cache reuse. The completion
suite adds strain history, molecular segments/rankings, late polarity references,
all output profiles, incremental charge/h-BN details, trajectory metadata, and
blocked diffusivity comparisons. `benchmarks/rollout_completion.py` adds worker
comparisons and isolated short/long scientific-memory gates.

See [benchmark instructions](https://github.com/ali-m-dinani/reaxkit/blob/master/benchmarks/README.md) for the deterministic
medium corpus and private 28,800-atom, 4,000-frame Slurm recipe. Workstation
measurements are checked in; production Slurm acceptance remains external.

The final local regression coverage is 899 passed and 12 skipped across 911
collected tests. `benchmark_results/rollout_validation.json` records the two
complementary suite runs and the final 561-test affected-area rerun. In particular,
streamed connectivity bundles preserve optional force-field valences loaded once;
the real relabeling integration test passes with this dependency available.
