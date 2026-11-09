# Optimization time budgets and durable output

These changes require updated optimizer checkouts, not only updated ReaxKit.
They have local regression coverage; new timed engine runs on Slurm are still
required before long campaigns.

## Settings

Add at the top of a campaign YAML:

```yaml
optimization_time_limit_seconds: 3600
collection_interval_seconds: 30
```

Each optimizer's existing runtime mapping can override that budget:

```yaml
runtime:
  optimization_time_limit_seconds: 3600
  shutdown_grace_seconds: 600
  timeout_seconds: 4800
```

Retain the existing command, source and environment fields. A runtime budget of
null disables the inherited optimization budget. Iteration/evaluation/convergence
criteria still apply: the first satisfied criterion wins.

The optimization budget requests a cooperative stop. An evaluation, parameter
row, generation or native dispatch batch already running may finish first.
The grace period begins when the worker requests stopping. Expiry forcibly
interrupts the process; it is not a successful optimizer convergence.
The existing runtime timeout includes preparation and initialization and is a
hard interruption. Slurm wall time must additionally leave room for worker
shutdown and final collection. For example, 60 minutes optimization, a measured
initialization allowance, and 10 minutes shutdown require more than 70 minutes
of allocation if initialization is substantial. No fixed margin guarantees
completion of arbitrarily slow engine calls.

## Supported paths and boundaries

| Integrated path | Cooperative boundary |
| --- | --- |
| JAX-SOPPI | Complete parameter row, including retries |
| JAX-ReaxFF | True evaluations, SciPy loss calls and random-search samples |
| INDEEDopt short sampling | Candidate evaluation |
| PIBO cobalt driver | Warm-start/BO candidate evaluation |
| EZFF cobalt driver | Objective call; interrupted epoch recommendations may be absent |
| GloMPO AMS/ParAMS | Existing MaxSeconds manager criterion; live received evaluations |
| Flocky serial | Population/generation boundary; rebuild the executable |
| OGolem | Existing STOP file checked by the native dispatcher; timed runs set MaxTasksToSubmit to the worker count |
| AMS MCFF | Native istop: annotated AMS 2023 Key uses 2 to stop; StopKey layouts use 1; checks can lag by multiple iterations |
| Standalone SOPPI | Local source stop hook after a saved parameter row; rebuild required |
| AMS CMA-ES | No verified cooperative wall-time hook; configuration rejects this budget |

PIBO and INDEEDopt entries do not promise time limits for unrelated native
workflows. Proprietary AMS binaries are not modified. Do not describe a hard
timeout of AMS CMA-ES as graceful stopping.

OGolem's default dispatch batch is 1000 tasks. For timed runs, ReaxKit limits
that batch to the configured worker count in the generated fit.ogo. This avoids
waiting for a large queued batch before checking STOP. An active optimization
task can still outlast the shutdown allowance; this is not cancellation inside
an engine call or local optimization. Prepare a fresh campaign to apply the change.

For standalone SOPPI, only set runtime.cooperative_stop_file to true after
rebuilding and testing your locally patched executable. The local hook reads
REAXKIT_STOP_FILE and optionally REAXKIT_OPTIMIZATION_TIME_LIMIT_SECONDS, then
uses the existing final-report path after saving a parameter row. Original
Fortran sources and compiled binaries must not be included in ReaxKit or
published with these changes. The local patch still needs a compiler/engine
smoke test. Its local syntax check passes; pre-existing ffext array-bound warnings
remain and were not changed as part of this work.

## Timing and output semantics

Instrumented Python paths write optimization-timing.json and append
optimization-events.jsonl. Initialization is measured from the optimizer
entrypoint to its explicit start boundary, excluding interpreter startup/imports.
JAX-SOPPI, PIBO and EZFF include their initial baseline evaluation in
initialization; JAX-ReaxFF currently includes its first trial/JIT evaluation
in optimization. Do not compare these as identical compute budgets.

The worker records total elapsed time, initialization_seconds,
optimization_elapsed_seconds, termination_mode and termination_reason.
Without native timing instrumentation (AMS, OGolem and legacy SOPPI), separate
phase times remain null rather than fabricated, and the cooperative budget
uses process elapsed time including initialization.

Finalization is included in elapsed time up to the final timing publication;
it is not claimed as isolated objective-evaluation time. Hard interruptions
can leave stale optimizer timing. worker.json is authoritative for process
termination; scheduler status must also be checked after allocation expiry.

Best-field updates occur at safe improvement checkpoints. Periodic collection
copies recognized restart artifacts into content-addressed normalized
checkpoints, so later overwrites do not destroy previously collected versions.
An artifact changing while copied is retried at a later collection.
Uncollected improvements between polls are not promised to be preserved.
Live uniform events are a fallback when native progress is unavailable, not
extra evaluations to add to native histories. Their unit is a completed
optimizer work unit; candidate identity and cross-method equivalence remain
unverified. No fabricated parity data is produced.

## Before long runs

Run a tiny time-budget test for each updated engine/runner, verify a finite
saved field and timing/termination records, then test one deliberate hard
interruption. Keep datasets, logs, licensed source and binaries private.
