# Selected-Frame Pipeline Plan

Status: Phases 1-3 are implemented. ReaxKit now has shared execution contracts, automatic resource policy, stage/resource instrumentation, and a bounded ordered frame pipeline. Later command migration and shared incremental-output work remain in progress.

This document is a focused companion to [UNIFIED_EXECUTION_AND_OUTPUT_PLAN.md](UNIFIED_EXECUTION_AND_OUTPUT_PLAN.md). It describes how ReaxKit should overlap input parsing, frame-level analysis, reduction, and output while keeping memory bounded.

## 1. Purpose

For analyses whose selected frames can be evaluated independently, ReaxKit should not load all selected frames before beginning the calculation. It should instead:

1. Read only frames selected by `--frames`.
2. Align the data required for each frame, such as `xmolout` coordinates and `fort.7` charges.
3. Submit that frame for analysis immediately.
4. Retain only a bounded number of in-flight frames and results.
5. Reduce or append completed results in deterministic frame order.
6. Release the frame data as soon as no downstream stage needs it.

This overlaps parsing with calculation and prevents the full selected trajectory from becoming one large in-memory object. It also preserves the existing user experience: a computational chemist runs a scientific command with its normal flags, while ReaxKit selects the execution policy automatically.

The pipeline will improve utilization when parsing and calculation have comparable costs. If parsing alone is slower than all available computation, the reader remains the limiting stage; the reader optimizations and pipeline therefore complement each other.

## 2. Scope

The shared runtime will support several analysis shapes rather than forcing every command into independent frame processing:

| Analysis shape | Execution behavior | Typical dependency |
|---|---|---|
| Independent frame map | Pipeline selected frames through parallel workers | Current frame only |
| Reference-frame map | Prepare the reference once, then pipeline selected frames | Immutable reference plus current frame |
| Streaming reduction | Update a bounded accumulator for each selected frame | Running sums, histograms, or moments |
| Ordered stateful stream | Process frames in source order with bounded memory | Previous state or event history |
| Global analysis | Use a command-specific blocked or global algorithm | Entire trajectory or nonlocal frame relationships |

The first implementation target is independent and reference-frame map analysis. Streaming reductions follow once their reducer contract is stable. Ordered and global analyses will share the selected-frame reader and instrumentation but will not claim unsafe frame-level parallelism.

## 3. Target architecture

```text
selected-frame reader
        |
        v
input aligner and minimal preprocessing
        |
        v
bounded work queue  -->  frame workers
                              |
                              v
                     ordered result collector
                              |
                              v
                 reducer / incremental artifact sink
                              |
                              v
                    final result and plot generation
```

Each stage has backpressure. When workers or output cannot keep up, the reader pauses instead of growing memory without limit.

### 3.1 Selected-frame reader

The reader must:

- Interpret `--frames` once and parse only selected frame payloads.
- Skip unselected atom records by the cheapest safe path.
- Stop after the highest requested source frame when the selection is finite.
- Traverse each required input forward once on a cold read.
- Reuse a valid frame index or compact input cache when that is faster.
- Request only the fields needed by the command.
- Preserve original source-frame indices and simulation time metadata.

For paired ReaxFF inputs, the reader must align `xmolout` and `fort.7` records without separately materializing either selected trajectory. Alignment errors must identify the source frame and input file.

For AMS input, selected-frame reads should access the required `History` variables directly from `reaxout.kf` or RKF input. A trajectory-only request must not load atomic-charge arrays, while a native-charge request should load coordinates, cells, and charges only for the selected frames.

### 3.2 Frame envelope

The reader and analysis runtime should exchange a small, explicit frame envelope containing:

- Source-frame index and selected-frame position.
- Required coordinates, cell, elements, charges, bonds, or metadata.
- References to immutable prepared state where needed.
- An estimate of payload memory for queue control.

The envelope must not carry unused trajectory fields or retain parser-owned buffers longer than necessary.

### 3.3 Prepared state

Reference-dependent commands should have a preparation phase that runs once. It may load a reference frame, construct neighbor mappings, prepare replicated reference data, or build immutable lookup arrays. Workers receive this shared read-only state and the current frame envelope.

Preparation must be separated from per-frame calculation so that reference data is not rebuilt or copied for every selected frame.

### 3.4 Frame kernel

Each eligible command should expose a small per-frame kernel. The kernel receives prepared state and one frame envelope and returns a compact frame result. It must not decide worker counts, create its own executor, read future frames, or write shared output files.

This boundary lets the common runtime control scheduling while the command retains responsibility for scientific calculations.

### 3.5 Ordered collector and reducer

Workers may finish out of order, but externally visible results must remain deterministic. The collector should hold only the small number of results needed to emit the next selected frame.

The reducer should retain the smallest sufficient state:

- Append projected rows or time-series rows incrementally.
- Maintain NumPy sums, counts, histograms, or moments for aggregate results.
- Keep compact plot matrices only when the requested plot requires them.
- Avoid retaining atom-level frame tables unless the user requested detailed output.

## 4. Output strategy

“Save the result immediately” should mean incremental reduction or batched append, not one small file per frame. Thousands of per-frame files would add metadata overhead and make later use difficult.

The shared output layer should provide:

- Bounded row batches for tabular output.
- Parquet row groups for large optional detail tables.
- Compact in-memory accumulators for small final summaries.
- Temporary output followed by atomic finalization on success.
- Stable selected-frame ordering regardless of worker completion order.
- A manifest recording the selection, execution policy, schema, and completed frame count.

CSV detail tables should remain opt-in when they are large. Parquet should be the preferred format for optional atom-level or center-level detail because it supports typed, compressed, incremental row groups.

## 5. Automatic scheduling and memory control

The top-level runtime should choose safe defaults from the command capability, available CPUs, available memory, estimated frame size, and execution environment.

The policy should:

- Detect CPUs assigned by Slurm and avoid using CPUs outside the allocation.
- Keep BLAS and OpenMP libraries single-threaded when frame workers provide parallelism.
- Prefer threads for NumPy-heavy kernels that release the GIL and avoid copying large frames between processes.
- Permit a process backend only for measured Python-bound kernels with a safe data-transfer strategy.
- Start with a small read-ahead queue, normally around one to two work items per worker.
- Reduce queue depth when frame payloads or results are large.
- Reserve memory for the parser, prepared reference state, reducers, plots, and output buffers.
- fall back to one worker when a command is stateful, not thread-safe, or memory estimates make parallel execution unsafe.

`--workers` and `--chunk-size` may remain expert overrides during migration, but normal users should not need them. Once the automatic policy is validated, these controls can be hidden from routine command help or moved under an advanced execution group.

## 6. Execution behavior by command capability

### Independent frame map

The runtime reads selected frames into the bounded queue and begins calculation as soon as the first complete frame is available. Completed compact results flow through the ordered collector to the reducer or output sink. Frame payloads are released after their workers finish.

### Reference-frame map

The runtime reads and prepares the required reference frame first, even when it lies outside the requested output selection. It then streams the selected frames through workers using the immutable prepared reference. Reference arrays remain resident once; current-frame arrays remain bounded by queue depth.

### Streaming reduction

The reader sends one selected frame at a time to a frame contribution function. Contributions update a bounded reducer. Parallel contribution calculation is allowed only when merging is associative, numerically controlled, and deterministic under the selected merge order.

### Ordered stateful stream

Frames remain in source order and update one state machine. The same selected-frame reader and incremental writer apply, but frame-level workers are disabled unless a safe inner-frame operation can be parallelized.

### Global analysis

Commands that truly require all frames must declare that requirement. They should use blocked algorithms, disk-backed arrays, or explicit global materialization with a preflight memory estimate. The runtime must never silently label these commands as independent maps.

## 7. Reader and computation overlap

The initial design should use one forward reader feeding a bounded queue. This provides overlap without concurrent seeks against the same large files. A separate reader thread is appropriate when parsing and NumPy work release the GIL; a process-based reader should be considered only after profiling serialization and memory costs.

Benchmarks must determine whether an allocation should reserve a logical CPU for reading. On small allocations, the reader may share time with workers. On larger allocations or slow parsing, one reader plus `N-1` workers may deliver better throughput than `N` workers competing with the reader.

The runtime should not read the entire trajectory merely to count frames before starting. Progress can use an index or cached count when available and otherwise report processed selected frames without requiring a preliminary full scan.

## 8. Failure, cancellation, and recovery

The pipeline should fail as one coordinated operation:

- A reader, worker, reducer, or writer exception stops new submissions.
- Pending work is cancelled where possible and queues are drained safely.
- The reported error includes the command, stage, and source-frame index.
- Final output paths are published only after successful finalization.
- Interrupted temporary artifacts are clearly marked and can be removed safely.

Checkpoint and resume support can be added after deterministic incremental output is established. Its checkpoint must record the exact command configuration, input identity, completed source frames, and reducer state.

## 9. Instrumentation

`--timing` should report stage-level measurements that reveal the actual bottleneck:

- Reference/preparation time.
- Frames scanned, skipped, parsed, and selected.
- Reader throughput and reader active time.
- Queue occupancy and time workers waited for input.
- Per-frame calculation time and worker utilization.
- Reducer and output-writing time.
- Peak in-flight payload and result memory.
- End-to-end elapsed time.

This distinguishes a parser-bound run from a compute-bound or output-bound run. A progress line that says a final task completed quickly must not conceal the preceding input and per-frame work.

## 10. Command migration contract

Each analysis command must be audited and assigned one capability. Migration requires the command to declare:

- Required input fields and data sources.
- How normalized scientific settings alter those requirements after engine resolution. For example, projected polarity with `--charge-source formal` requires trajectory data, while `--charge-source auto` requires the selected or detected engine's trajectory and atomic charges.
- Whether it needs a reference frame or other prepared state.
- Whether frames are independent, reducible, ordered, or global.
- Its per-frame result schema and estimated size.
- Requested compact and detailed artifacts.
- Whether its kernel is safe for threads or processes.

Commands should no longer create private worker pools, accumulate full-frame tables by default, or repeat input traversal. Existing scientific flags and result meanings must remain compatible unless a documented correction is required.

## 11. Implementation phases

### Phase 1: Measure the current path

- Add or consolidate stage timing around reading, alignment, preparation, frame calculation, reduction, and writing.
- Record baseline wall time, CPU utilization, peak RSS, bytes read, and output sizes on representative ReaxFF trajectories.
- Include contiguous, strided, sparse, and single-frame selections.

### Phase 2: Define shared contracts

- Introduce capability metadata and the frame envelope, prepared-state, frame-kernel, reducer, and artifact-sink contracts.
- Add a common automatic execution policy.
- Keep legacy execution available internally while commands migrate.

### Phase 3: Build the bounded pipeline

- Implement the selected-frame producer, aligned input stage, bounded queue, worker scheduler, ordered collector, and coordinated cancellation.
- Enforce deterministic ordering and bounded memory.
- Add queue and stage instrumentation.

### Phase 4: Build incremental outputs

- Extend the artifact writer with buffered table append and Parquet row groups.
- Add compact reducers for common time series, bins, histograms, and plot matrices.
- Add atomic finalization and incomplete-output handling.

### Phase 5: Migrate the two optimized prototypes

- Move `get-hbn-reference-projected-polarity` and `get-basal-plane-displacement-projected-polarity` from command-owned worker pools to the shared runtime.
- Confirm numerical equivalence, output compatibility, lower duplicated code, and equal or better performance.

### Phase 6: Migrate electrostatics and ferroelectrics

- Classify every analysis command in the two priority folders.
- Migrate independent and reference-frame commands first.
- Convert suitable aggregate commands to streaming reducers.
- Preserve ordered execution for history-dependent calculations.

### Phase 7: Migrate remaining analysis commands

- Use the command inventory and priority tracker to migrate the rest of the analysis surface.
- Add specialized blocked execution for global commands.
- Remove obsolete private pools and full-trajectory accumulation paths after compatibility coverage passes.

### Phase 8: Validate on workstation and Slurm

- Compare serial and automatic execution across worker counts and queue depths.
- Test slow shared storage, warm cache, cold cache, and local scratch.
- Verify bounded RSS as trajectory length increases.
- Document scheduler guidance only where the automatic policy cannot infer a safe value.

## 12. Verification and acceptance criteria

The pipeline is ready for broad use when:

- Independent commands start calculation after the first selected frame is ready.
- Unselected frames are not fully parsed or retained.
- Peak frame memory is bounded by prepared state, reducer state, and configured in-flight work rather than selected-frame count.
- Selected-frame results match the current implementation within defined numerical tolerances.
- Output ordering and filenames are deterministic for one or many workers.
- Reference-dependent calculations prepare the reference once.
- Worker failures cancel reading and leave no apparently complete partial output.
- `--timing` identifies reader, calculation, and writer bottlenecks separately.
- Default execution respects Slurm CPU and memory allocations without requiring multiprocessing flags from the user.
- Representative compute-heavy runs show useful reader/compute overlap, while parser-bound runs show measurable gains from the selective-reader improvements.

### Real-data validation (2026-09-23)

Both projected-polarity commands passed two-frame streamed runs on the 28,880-atom ReaxFF text files and the downloaded `reaxout.kf`, using two workers and a two-frame queue. Automatic engine detection and noncontiguous `0:3:2` formal-charge runs also passed and loaded trajectory data only; compact outputs omitted detail tables. The shared physical frame produced identical polarity counts and means across engines. Testing found and fixed standalone-KF handling for step-indexed atom names and angstrom-valued coordinates; differing later iterations reflect the files' 5-versus-50 frame-to-iteration ratios.

## 13. Phase status

| Phase | Status | Completed work | Work left to complete the phase |
|---|---|---|---|
| **1. Measure the current path** | **Implemented** | Timing records now separate execution-policy selection, reference preparation, reader activity, cumulative worker calculation, collector waiting, total pipeline time, loading, analysis, and end-to-end execution. End-to-end records include process CPU time, resident memory when available, and unique source bytes. Existing Slurm observations and selective-reader tests provide initial baselines. | No implementation work remains. The larger controlled workstation/Slurm benchmark matrix is part of Phase 8 validation. |
| **2. Define shared contracts** | **Implemented** | The runtime defines execution shapes, task capabilities, frame producers, input aligners, frame envelopes, immutable prepared state, frame kernels, reducers, artifact sinks, execution policies, and compact frame results. Policy selection respects task safety, explicit overrides, Slurm CPU allocation, Slurm memory allocation, and per-frame memory estimates. Engine resolution and scientific settings determine required canonical data before execution. | No Phase 2 implementation work remains. New commands must declare capabilities as they migrate. |
| **3. Build the bounded pipeline** | **Implemented** | The engine-independent pipeline overlaps forward input iteration with frame workers, applies bounded read-ahead and memory-aware queue limits, preserves deterministic input order, tracks in-flight items and bytes, propagates frame context, cancels pending work after failure, and emits stage metrics. ReaxFF and AMS/KF selective streams act as producers. The two projected-polarity commands now use the shared scheduler instead of private thread pools. | No Phase 3 implementation work remains. Additional commands will adopt the pipeline during Phases 5-7. |
| **4. Build incremental outputs** | **Implemented** | The runtime now provides declared artifact specifications, output profiles, buffered CSV append, Parquet row groups, compact table/count-sum/histogram/plot-matrix reducers, atomic publication, a machine-readable artifact manifest, and temporary-output cleanup after failure. | No Phase 4 implementation work remains. Additional workflows can adopt these services during later command migrations. |
| **5. Migrate the two optimized prototypes** | **Implemented** | `get-hbn-reference-projected-polarity` and `get-basal-plane-displacement-projected-polarity` use the shared scheduler, automatic resource policy, shared accumulators, and the atomic artifact writer. Detailed center rows stream to CSV or Parquet without being retained in the result, while compact projected and kymograph tables remain compatible. Focused numerical, failure, and workflow tests pass. | No code migration remains. Controlled workstation and Slurm performance comparison belongs to Phase 8. |
| **6. Migrate electrostatics and ferroelectrics** | **Implemented for the safe common-runtime scope** | All 25 registered analysis tasks in the two priority folders have an explicit execution-shape and thread-safety classification. `get-dipole`, `get-polarization`, and `charge-table` now join the two projected-polarity commands on the shared bounded frame scheduler. Existing streaming reductions and ordered writers remain serial by contract; global and still-materialized reference analyses cannot receive unsafe frame parallelism. The tracker records each distinction. | Further command-specific conversion of classified materialized analyses to selected-frame producers is tracked for Phase 7. Performance validation remains Phase 8. |
| **7. Migrate remaining analysis commands** | **Implemented for the shared-policy migration** | Every automatically registered analysis task is covered by an exhaustive repository manifest, while workflow-loaded electrostatics and ferroelectrics tasks remain covered by the priority manifest. The executor records a policy and serial/parallel reason for streaming, materialized, ordered, and global tasks. Generic result persistence now uses atomic artifact writing and supports `minimal`, `standard`, `full`, and `legacy` profiles. A bounded block iterator is available for global algorithms; unsupported global mathematics stays explicitly serial. | Individual global algorithms such as MSD and diffusivity may adopt specialized block kernels later when numerical-equivalence and performance evidence justify them. Their conservative serial status is an intentional completed policy decision, not an implicit fallback. |
| **8. Validate on workstation and Slurm** | **Implemented; production Slurm execution pending** | Determinism, cancellation, atomic-failure cleanup, ordered collection, queue bounds, payload-memory bounds, output profiles, and serial/automatic numerical equivalence have automated coverage. The portable benchmark compares worker counts and first/repeated file reads and emits machine-readable platform, timing, speedup, digest, and peak-in-flight data. The committed workstation report passed nine cases with identical digests and bounded payload retention. A Roar Collab Slurm job runs the same matrix and focused tests under `/usr/bin/time -v`. | Submit the supplied job on Roar Collab against its shared filesystem and local scratch, then retain its JSON, timing output, and `seff` report. Real Slurm results cannot be produced from this workstation. Production scientific-command benchmarks remain advisable before changing task-specific defaults. |

### Completed foundations

- Phases 1 through 7 are implemented for the shared-runtime scope defined above.
- Selective frame loading is implemented for the relevant ReaxFF text and AMS/KF paths.
- Engine detection now occurs before task data requirements are chosen.
- For the two projected-polarity commands, `--charge-source auto` requests native AMS or ReaxFF electrostatics and `--charge-source formal` requests trajectory data only.
- The two prototype commands avoid large detail tables by default, stream optional details through the shared artifact writer, and use the shared bounded scheduler.
- Dipole, polarization, and charge-table analyses also use the shared bounded scheduler.
- All 25 registered electrostatics and ferroelectrics analysis tasks have conservative execution classifications; unsafe ordered/global paths remain serial.
- All automatically registered analyses participate in the repository-wide execution and artifact policy.
- Generic persisted tables use atomic artifact publication and shared output profiles.
- Phase 8 has a reproducible workstation/Slurm benchmark harness; the workstation matrix is complete.
- The unified execution plan and analysis-command priority tracker exist.

### Remaining critical path

1. Submit `benchmarks/slurm_selected_frame_pipeline.sh` on Roar Collab and archive the JSON, `/usr/bin/time -v`, and `seff` results.
2. Benchmark the two production projected-polarity commands with representative ReaxFF and AMS/KF trajectories.
3. Add specialized blocked kernels to global analyses only when their scientific equivalence and measured benefit are established.
