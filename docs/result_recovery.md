# Incremental results and recovery

Audited streaming CLI tasks now save bounded, durable result batches by default.
Input-frame caches and result checkpoints are separate: cache eviction and the
input-cache size limit do not delete checkpoints. Direct Python calls keep their
existing in-memory result API unless a result store is explicitly supplied.

## Supported work

The live `result_checkpoint` field in `docs/command_execution_inventory.json`
lists supported tasks. These include total/local/projected h-BN polarization,
potential/electric-field tables and trajectories, charge Extended XYZ, binned
dynamic charges, charge versus field, the shared independent frame-table tasks,
connection-statistics reducers, and ordered bond-event scans.

Other tasks reject explicit checkpoint/resume requests. Basal-plane reference
variants, other ordered molecular/strain scans, global MSD/FFT/correlation tasks,
and materialized scalar energy/temperature/partial-energy loaders remain explicit
exceptions. Their current scientific algorithms/input loaders cannot acquire
correct restart behavior just by saving output rows. They are not advertised as
recoverable or constant-memory result paths.

## Running and resuming

The shared options are:

```text
--checkpoint / --no-checkpoint
--resume CHECKPOINT
--checkpoint-buffer-mb 32
--checkpoint-interval-seconds 30
```

Repeat the original scientific and output-profile options when using `--resume`.
The failure message prints the checkpoint directory, last committed selection
position, and an exact resume command when launched through the CLI. Source-frame
numbers are distinct from selection positions. Kernel errors identify their
source frame when available. Source-reading errors may have no known frame.

Each new invocation gets a separate checkpoint under its workspace analysis
directory. An explicit export destination does not move recovery metadata.
Resume validates full SHA-256 source identities, normalized scientific settings,
task/checkpoint versions, and every committed batch before numerical work.
Changed inputs, appended trajectories, changed selections or enabled artifacts
are rejected. A fresh invocation never silently reuses a partial result.

Reference matching and bins are reconstructed deterministically from validated
source/reference inputs on an interrupted resume. Invariant reference geometry,
atom mapping, bin edges, baseline metadata exposed by the result, and result
descriptors are retained in portable JSON/Parquet data. Fixed potential midpoint
and charge baselines are recomputed from the same first selected/reference frame.
Connection reducers persist compensated sums/counts/maxima. Bond-event snapshots
include smoothing windows, EMA, hysteresis and delayed event confirmation state.

`analysis_complete` means numerical work is saved but export/plotting is not yet
confirmed. Resuming this state rebuilds exports without running numerical kernels
or reloading trajectory frames. Input identity and batch validation still run.
Only successful completion of the entire CLI workflow marks `complete`.

## Storage and failure semantics

A single parent-process writer stages all enabled tables for each batch, closes
and fsyncs them, publishes the batch directory, then atomically replaces the
manifest. Checksums cover tables, state and a chain of batch metadata. Only
manifest-referenced batches count as committed. A crash after publishing a
directory but before replacing the manifest leaves an ignored orphan.

Extended XYZ is saved as complete-frame text segments in batch tables alongside
associated numerical tables. Final XYZ and CSV outputs stream those committed
parts into temporary files before publication. A failure leaves the recoverable
parts intact rather than publishing a truncated trajectory as successful.

Parquet is preferred; CSV is the fallback if PyArrow is unavailable. Batch flushes
occur at 32 MiB, 128 completed frames, or the first completed frame after 30
seconds, whichever happens first. One oversized frame is allowed. Serialization,
table copies and workers require additional memory above the buffer setting.
No timer can commit a frame while its numerical kernel is still running.

`writer.lock` records host, PID and token. Concurrent writers are rejected.
After process/node death, first confirm that the recorded writer/job is no longer
running on the recorded host. Then explicitly remove **only that checkpoint's
`writer.lock`** and resume. Never remove a live writer's lock. Tests exercise
abrupt subprocess exit without cleanup handlers; physical power loss and remote
filesystem durability still depend on storage semantics.

Checkpoints are retained after success and failure. After checking all final
outputs, users may explicitly delete a completed checkpoint to reclaim space.
Keep failed/interrupted checkpoints until recovery is no longer needed. Export
temporarily duplicates data; budget disk for batches plus final artifacts.

## Memory boundaries

Primary table exports iterate batch references without concatenating all rows.
Per-frame local-polarization and potential/field plots use saved tables and
streaming extrema passes. Coordinates for local XYZ are retained only when
requested, then rebuilt into a temporary disk-backed trajectory from committed
data. Ordered bond events use an external SQLite sort to preserve public order.

Frame/iteration/time index arrays and lists of artifact paths still grow with
frame count. Sparse local trajectory export also retains source-indexed metadata.
These are metadata costs, not constant-memory claims. Unique connectivity pairs
and ordered per-pair state can grow with the scientific system.

Global kymographs explicitly materialize their binned matrices. Generic global
plot/report adapters can materialize selected result tables, and charge/field
plots retain one atom's exact time trace at a time. These optional global views
are outside the bounded primary-export claim; no scientific downsampling is used.

## Reproducing validation

Run from the repository using its configured Python environment:

```text
python -m pytest tests/core/test_result_store.py tests/core/test_checkpoint_commands.py
python tools/validate_result_recovery.py --input-dir /path/to/field_iout2_npt_inpt3 --workspace /fresh/validation --engine ams --charge-source reaxff
python tools/validate_result_recovery.py --input-dir /path/to/field_iout2_npt_inpt3 --workspace /fresh/local --engine ams --charge-source reaxff --command get-hbn-reference-local-polarization --frames 0:30
python tools/benchmark_result_recovery.py --directory /fresh/store-benchmark --frames 10000 --rows 1000 --buffer-mb 4
```

Use a fresh benchmark directory for each measurement. The sample comparison
checks baseline, checkpointed and resumed CSV schemas/order/values **within each
engine**. AMS and ReaxFF frames in this sample are different trajectories.
Reports include wall time, sampled process-tree RSS and checkpoint disk bytes.
The storage-only benchmark deliberately excludes source loading and scientific
kernels; it cannot establish full-CLI memory behavior by itself.

Local measurements and outstanding production gates are recorded at the bottom
of `improvement plans and files/INCREMENTAL_RESULTS_AND_RECOVERY_PLAN.md`.
Production acceptance still needs an actual Slurm allocation and the intended
shared filesystem. Set a site-specific throughput budget before that deployment;
full input hashing can dominate short selected-frame runs and restarts.
