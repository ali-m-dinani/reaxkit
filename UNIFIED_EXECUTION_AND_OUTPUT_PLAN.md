# Unified Execution and Output Plan

Status: the shared-policy implementation is complete. Registered analyses have explicit conservative capability decisions, shared-pipeline tasks use bounded scheduling, generic persisted tables use atomic artifact profiles, and a reproducible workstation/Slurm validation harness exists. Real-data tests passed for both projected-polarity commands with 28,880-atom ReaxFF text input and standalone `reaxout.kf`; the tests also verified automatic engine detection, native/formal charge routing, noncontiguous selected frames, and compact default output. `SELECTED_FRAME_PIPELINE_PLAN.md` remains the detailed implementation record and phase-status source of truth; production Slurm validation is still pending.

## Objective

Give every ReaxKit command a shared execution and artifact policy so a computational chemist continues to choose only the scientific command and its scientific flags. ReaxKit should automatically:

- select bounded-memory streaming when the engine and analysis support it;
- use the CPUs assigned to the process without nested-thread oversubscription;
- parallelize independent frame work while preserving deterministic frame order;
- retain compact scientific results by default;
- avoid constructing detailed intermediate tables when they are not requested;
- write large optional tables incrementally in Parquet;
- keep serial execution for analyses whose mathematics or external libraries require it.

The optimized `get-hbn-reference-projected-polarity` and `get-basal-plane-displacement-projected-polarity` commands are prototypes for this architecture. Their worker scheduling, bounded queues, incremental optional details, and atomic table publication now use shared runtime infrastructure.

## Important constraint

A single unconditional multiprocessing wrapper is not safe for all commands. ReaxKit contains several kinds of work:

1. **Independent frame maps**: each frame can be analyzed independently and results can be concatenated in source-frame order. These are the easiest commands to stream and parallelize.
2. **Reference-dependent frame maps**: a reference frame or small preparation state must be computed first; later frames are independent. The two optimized projected-polarity commands belong here.
3. **Streaming reductions**: frames can be processed incrementally, but partial accumulators must be merged rather than concatenated. Histograms, binned statistics, and some correlation calculations belong here.
4. **Stateful trajectory scans**: frame order affects the next step, as in event detection, molecule lifetimes, and some connectivity tracking. These can stream but usually cannot map frames independently.
5. **Global trajectory algorithms**: the mathematical result depends on relationships among many or all frames, as in some MSD, diffusivity, fitting, and time-correlation workflows. These need specialized blocked algorithms rather than generic frame workers.
6. **One-shot file/report commands**: parsing a force field, generating a control file, or producing a small report gains nothing from frame multiprocessing.

All commands should participate in the same top-level policy, but the policy must select the correct execution strategy from declared task capabilities.

## User experience

The normal interface remains:

```text
reaxkit COMMAND [scientific options]
```

No new performance flag should be required. In automatic mode ReaxKit should:

- read `SLURM_CPUS_PER_TASK` when present;
- otherwise respect the process CPU affinity and then fall back to `os.cpu_count()`;
- choose a conservative worker count from available CPUs, selected-frame count, task limits, and estimated per-frame memory;
- choose a bounded chunk size from the same information;
- restrict BLAS/OpenMP libraries while frame workers are active;
- report the selected strategy, worker count, chunk size, cache choice, and output profile in the timing log.

Advanced global overrides may be offered for benchmarking and troubleshooting:

```text
--execution auto|serial|threads|processes
--workers auto|N
--chunk-size auto|N
--output-profile standard|minimal|full|legacy
--detail-format parquet|csv
```

Their defaults must be automatic. Existing command-specific `--workers`, `--chunk-size`, `--write-*`, and format flags remain accepted during migration and become aliases or overrides of the shared policy.

## 1. Add an execution-capability contract

Add a small immutable contract under `reaxkit.core.runtime`, for example `execution_policy.py`:

```python
@dataclass(frozen=True)
class ExecutionCapabilities:
    strategy: Literal[
        "single",
        "frame_map",
        "reference_frame_map",
        "stream_reduce",
        "ordered_stream",
        "global",
    ]
    supported_backends: tuple[Literal["serial", "threads", "processes"], ...]
    preserves_frame_order: bool = True
    supports_selective_frames: bool = False
    reference_fields: tuple[str, ...] = ()
    estimated_bytes_per_atom: int | None = None
    max_workers: int | None = None
```

Extend `AnalysisTask` with conservative defaults equivalent to the current behavior. Tasks then override capabilities instead of scattering booleans such as `supports_selective_streaming` through the codebase.

The contract should answer four separate questions:

- Can the engine deliver the required data as bounded frame objects?
- Can the task process one frame independently?
- Does the task require preparation or ordered state?
- How are partial results merged?

Keep compatibility shims for `supports_selective_streaming` and existing `run_stream` methods until every registered task has migrated.

## 2. Introduce one execution policy resolver

Create an `ExecutionPolicy` value object resolved once by `AnalysisExecutor` from:

- task capabilities;
- engine streaming capabilities;
- selected and dependency frames;
- CPU affinity and Slurm variables;
- configured or detected memory limits;
- input-cache settings;
- explicit advanced overrides.

The resolver should produce a concrete decision such as:

```text
strategy=reference_frame_map
backend=threads
workers=4
chunk_size=16
ordered=true
blas_threads_per_worker=1
```

The policy decision must be recorded in human timing output and machine-readable run metadata. Falling back to serial execution must include a reason, such as `task_requires_ordered_state` or `engine_cannot_stream_required_data`.

## 3. Add shared bounded map/reduce primitives

Move the duplicated chunked executor from the two projected-polarity commands into reusable runtime code.

Required primitives:

- ordered bounded map over frame payloads;
- reference-state preparation followed by ordered bounded map;
- ordered streaming scan for stateful tasks;
- mergeable reduction for counts, sums, histograms, and compact tables;
- cancellation and exception propagation that stops input parsing and other workers promptly;
- progress reporting based on completed frames rather than submitted futures.

Only `chunk_size` payloads should be retained beyond active workers. Avoid submitting an entire trajectory to `Executor.map` at once.

Threads should be the first backend for NumPy/SciPy-heavy tasks because they avoid copying large coordinate arrays. Processes should be enabled only for tasks that declare their payload and prepared state safely picklable and demonstrate a benchmark benefit. Serial remains a valid automatic choice.

Use `threadpoolctl` or an equivalent scoped mechanism to budget native-library threads. Do not rely solely on users exporting `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, or `OPENBLAS_NUM_THREADS` in job scripts.

## 4. Standardize reference-frame dependencies

Replace ad hoc `stream_reference_first` handling with explicit dependency declarations in the capability contract.

The executor should:

1. request reference frames before primary selected frames;
2. prepare an immutable, read-only reference state once;
3. exclude dependency-only frames from public results unless the user selected them;
4. pass the prepared state to frame workers without rebuilding it;
5. preserve original source-frame indices in every result and artifact.

This mechanism should cover displacement references, aligned structures, fixed spatial bins, initial connectivity, and other small reusable preparation states.

## 5. Centralize artifact policy and writing

Current workflows write tables directly with many independent `to_csv` calls. Introduce an artifact contract, for example:

```python
@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    tier: Literal["core", "summary", "detail", "debug"]
    default_enabled: bool
    preferred_format: Literal["csv", "parquet", "extxyz", "json"]
    incremental: bool = False
```

Each task or result declares its artifacts. A shared `ArtifactWriter` should then:

- always write small core and summary results in stable public formats;
- omit detail and debug artifacts under the default `standard` profile;
- write large tabular details as Parquet by default;
- support incremental Parquet row groups so details need not exist as one DataFrame;
- write a manifest containing schema, row count, units, source frames, format, and whether an artifact was omitted by policy;
- use atomic temporary-file replacement so failed jobs do not leave apparently complete outputs;
- reject accidental overwrite conflicts when multiple commands share an output directory.

The default policy must be based on declared scientific meaning, not only a row-count threshold. A large table can be the primary result of one command and optional detail for another.

Provide `--output-profile legacy` during migration to reproduce the former CSV set. Announce output-default changes in release notes because removing a formerly automatic CSV is externally visible behavior.

## 6. Make results incrementally consumable

Do not require every result field to be a fully materialized pandas DataFrame. Add one or both of:

- a `TableChunks`/`RecordBatch` abstraction for bounded producers;
- reducer state objects whose `finalize()` method returns only compact results.

The presentation layer should consume compact result tables. Detailed artifacts should be streamed directly to `ArtifactWriter` when requested rather than retained solely so a workflow can call `to_csv` later.

Pandas remains suitable for compact public tables. Use NumPy arrays and vectorized reductions inside frame kernels, and PyArrow record batches for large optional tabular output.

## 7. Inventory and classify every registered command

Generate a checked-in command capability manifest from the task and CLI registries. Each registered analysis command must be assigned one category and an output inventory.

Initial migration groups:

### Group A: frame-map candidates

Start with commands that already have `run_stream`, selective-frame support, or obvious independent frame loops, including charge extraction, coordinate series, electrostatics totals, neighbor finding, coordination, hybridization, Voronoi, per-frame strain, and per-frame polarization analyses.

For each command:

- isolate a pure per-frame kernel;
- retain only fields needed by requested outputs;
- implement deterministic chunk combination;
- add serial-versus-parallel equivalence tests;
- classify detailed per-atom/per-neighbor tables as optional when they are not the command's primary result.

### Group B: reference-dependent frame maps

Migrate displacement, fixed-reference polarization, polarity, and fixed-bin commands to the shared reference-state primitive. Replace the custom worker loops in the two prototype commands after the shared implementation is proven.

### Group C: streaming reductions

Add mergeable accumulators for RDF histograms, binned charges, density/profile summaries, aggregate connection statistics, and similar commands. Validate that different chunk boundaries produce identical results within documented floating-point tolerance.

### Group D: ordered scans

Stream bond events, molecule lifetimes, active-site events, and other stateful analyses through a single ordered state machine. Optimize memory first. Parallelize only independent subproblems inside a frame or independent species/groups when scientifically equivalent.

### Group E: global algorithms

Design command-specific blocked algorithms for MSD, diffusivity, correlations, fitting, and other cross-frame mathematics. Do not label these commands parallel merely because their input loader streams. Their capability records may remain `global` or `single` until an equivalent blocked implementation exists.

### Group F: one-shot tools and generators

Register them with `strategy="single"`. They still use the shared artifact policy and timing metadata, but automatic workers remain 1.

## 8. Preserve scientific and numerical behavior

Every migrated command needs a frozen serial reference path during transition. Tests must compare:

- selected source frames and output order;
- atom/site identity mapping;
- periodic-image choices;
- reference-frame behavior;
- counts and categorical assignments exactly;
- floating-point values within command-specific tolerances;
- missing/undefined value behavior;
- output schemas, units, and artifact names for core outputs.

Parallel execution must be deterministic for a fixed input and command. Reductions should use stable ordering or numerically stable merge strategies where ordinary summation order would materially change results.

## 9. Add framework-level validation

Add tests that fail when a registered command lacks an explicit capability and artifact declaration after its migration phase.

Required test layers:

1. Policy-unit tests for Slurm CPU detection, affinity, memory limits, overrides, and serial fallbacks.
2. Executor tests for bounded submission, reference-first dependencies, ordered output, cancellation, and worker exceptions.
3. Artifact tests for profiles, Parquet row groups, atomic writes, manifests, and legacy output compatibility.
4. Engine tests proving selected frames are physically read once and aligned across coordinate, charge, and connectivity sources.
5. Command equivalence tests comparing legacy serial and new automatic execution.
6. Peak-memory tests using subprocess resource measurements rather than assertions about Python object counts.
7. Performance benchmarks using representative small fixtures in CI and larger opt-in HPC fixtures.

Create a benchmark corpus with at least:

- a small correctness trajectory;
- a medium local benchmark;
- a documented 28,800-atom, 4,000-frame HPC benchmark recipe that can use private/local data without committing it.

Record wall time, CPU time, CPU efficiency, peak RSS, bytes read, bytes written, artifact sizes, and frame throughput.

## 10. Rollout phases

### Phase 0: baseline and inventory

- Generate the command/capability/output inventory.
- Capture baseline tests and performance for representative commands.
- Identify which current CSV files are core results versus optional detail.
- Document current public output names before changing defaults.

### Phase 1: runtime contracts

- Add `ExecutionCapabilities`, `ExecutionPolicy`, CPU/memory detection, and policy logging.
- Keep all commands serial unless explicitly classified.
- Add compatibility adapters for current `run` and `run_stream` tasks.

### Phase 2: shared bounded execution

- Add ordered frame map, reference-frame map, ordered scan, and reducer primitives.
- Migrate the two projected-polarity prototypes onto the shared primitives.
- Prove no regression against their current optimized implementations.

### Phase 3: artifact system

- Add `ArtifactSpec`, `ArtifactWriter`, output profiles, Parquet chunk writing, and manifests.
- Route existing `csv_tables` through a compatibility adapter.
- Migrate direct workflow `to_csv` calls in small coherent groups.

### Phase 4: frame-local commands

- Migrate Group A commands by domain.
- Enable automatic workers only after equivalence and benchmark gates pass.
- Remove duplicated command-level worker/chunk implementations as each domain migrates.

### Phase 5: reducers and ordered scans

- Implement Group C accumulators and Group D state machines.
- Focus first on bounded memory; enable safe parallel substructure only where benchmarks justify it.

### Phase 6: global algorithms

- Implement blocked or domain-specific parallel versions command by command.
- Leave unsupported commands explicitly serial with a logged reason.

### Phase 7: default-output transition

- Release `standard`, `minimal`, `full`, and `legacy` output profiles.
- Keep `legacy` available for at least one documented compatibility cycle.
- Change large detail artifacts to opt-in only after documentation and migration notices are complete.

## 11. Acceptance criteria

The framework is complete when:

- every registered command has an explicit execution capability and artifact inventory;
- ordinary users need no performance flags on a laptop or Slurm allocation;
- frame-map tasks automatically use bounded workers when more than one CPU is available;
- stateful and global tasks remain correct and never receive unsafe generic multiprocessing;
- selected trajectories can run with memory proportional to one bounded chunk rather than total selected frames whenever the algorithm permits;
- large optional tables are not constructed or written under the standard profile;
- requested detailed tables stream to Parquet without full materialization;
- core scientific outputs remain stable and documented;
- serial and automatic modes pass numerical-equivalence tests;
- timing output explains the chosen execution plan and its fallback reasons;
- representative HPC benchmarks show reduced wall time or memory without increased source-file rereads.

## 12. Recorded implementation decisions

The implementation resolved the original Phase 0 questions as follows:

1. Artifact declarations and the command tracker identify core, summary, detail, and debug outputs; `standard` omits declared optional details.
2. `legacy` remains available as an explicit compatibility profile during the output transition.
3. Automatic scheduling uses the allocated CPUs conservatively and reserves no CPU on allocations of four or fewer CPUs.
4. Queue depth is bounded by detected memory when available and otherwise uses conservative per-frame estimates.
5. The checked-in manifests record thread safety and execution shape; unsafe ordered and global tasks remain serial with a recorded reason.
6. One forward reader supplies canonical frame payloads to bounded workers instead of letting workers reread shared source files.
7. Results are collected and reduced in deterministic source-frame order; command-specific compensated reductions remain appropriate where numerical evidence requires them.
8. Compact scientific tables follow the selected output profile, while plots and large detail artifacts remain explicit opt-ins unless a command declares otherwise.

Standalone `reaxout.kf` is treated as an AMS/KF input while preserving its step-indexed History layout and angstrom-valued coordinates. `--charge-source auto` reads the detected engine's native charges; `--charge-source formal` requests trajectory data only.
