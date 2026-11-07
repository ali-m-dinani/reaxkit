# Force-field optimization on Roar

ReaxKit owns YAML configuration, independent directories, Slurm submission,
provenance and converters. Separately installed optimizers own their engines,
algorithms and objectives. GUI development remains deferred.

**Status:** native mappings are source/contract-tested, not Roar-validated.
Linux process tests and CI are provided. Windows checks do not establish
Linux/MPI/engine correctness. Complete the acceptance procedure before production.

## Supported launch paths

All nine adapters have output converters. Native launching has explicit limits:

| Adapter | Native entrypoint | Requirements |
| --- | --- | --- |
| ams_mcff | SCM standalone reaxff | User control; iopt=4; min/max params |
| ams_cmaes | SCM standalone reaxff | User control; iopt=7; optional explicit restart |
| standalone_soppi | User-built legacy ffopt | Audited unit layout; control and random_file; unlinked parameters only; near-zero bounds rejected because the engine treats them as unbounded |
| jax_soppi | python -m soppi.cli | Explicit JAX-ReaxFF/JAX-MD source directories |
| jax_reaxff | python -m jaxreaxff.driver | No control-file interpretation; machine_report only with num_trials=0 |
| ogolem | Compatible Java/OGolem jar | runtime.source and runtime.bridge_python; explicit AMS/JAX/custom evaluator |
| pibo cobalt | python -m pibo_calculation.cobalt_pibo | Explicit acknowledgement of the native cobalt subset/LAMMPS objective |
| pibo MoS2, indeedopt, ezff | External runner accepting run-config.json | Not universal native common-input interfaces |

Unsupported native workflows fail before submission. INDEEDopt has dataset-specific
constants and rewrites bounds using gpf=5. MoS2 PIBO uses its own VASP data, fixed
parameter catalog and checkout-relative files. EZFF needs an objective callback.
Configurable isolated runners belong in those projects under their own licenses.
See src/reaxkit/optimization/AUDIT.md for inspected revisions and output formats.

## Environment isolation

Use separate Linux virtual/Conda environments for ReaxKit and each optimizer;
never merge incompatible NumPy/JAX/LAMMPS requirements. Set absolute executable,
interpreter, script and jar paths in runtime.command. Install each optimizer into
its own environment or explicitly configure its process's PYTHONPATH.
runtime.source records provenance; it does not install packages or change imports.
OGolem's bridge_python and engine_python paths retain interpreter symlinks so
Linux virtual environments are not bypassed when their Python links to a system binary.

Prepare on Linux using the ReaxKit environment that workers will use. Its absolute
interpreter is captured in job/worker.sh; activating another environment in the
job does not change it. Native subprocesses use their separately configured
interpreters. Keep modules/library paths/MPI variables in the user's job.
Never submit a copied Windows-prepared campaign: prepare a new one on Linux.

## Configuration

```bash
reaxkit optimize-ffield --generate-template settings.yaml
reaxkit optimize-ffield --generate-job-template roar_job.sh
reaxkit optimize-ffield --list-optimizers
```

Templates refuse overwrite. Fill paths, selected optimizers and job settings.
The [two-method smoke example](examples/ffield_optimization/smoke.yaml) is a
smaller alternative to the full catalog.

- replicas counts independent jobs per method, not native population size.
- seed derives requested per-method/replica seeds; null leaves native defaults.
- inputs requires ffield, params, trainset and geo; control is optional.
- inputs.params_bounds_order explicitly selects min_max or max_min.
- job is the batch-script filename, not a job name; each optimizer may override it.
- output must be a new directory.
- collection_interval_seconds does not force native programs to flush reports.

Input/setting paths are YAML-relative. CLI overrides use the current directory.
Command argv is literal and should use absolute paths, not YAML-relative ones.

```yaml
optimizer_settings:
  jax_reaxff:
    runtime:
      mode: native
      command: [/path/jax-env/bin/python, -u, -m, jaxreaxff.driver]
      version: your-installed-revision
      engine_version: your-installed-jax-md-revision
      timeout_seconds: null
      environment: {}
    settings:
      num_trials: 1
      num_steps: 2
```

Native mode appends mapped flags. Supply only entrypoint/launcher options, not
conflicting optimizer flags. Optional runtime.timeout_seconds limits the whole
worker, including bridge preparation; timeout stops its child group, collects
output and records exit 124. Slurm walltime remains authoritative.

Only the first argv element supports a leading environment variable, such as
`${AMSBIN}/reaxff`, expanded after module setup. Other arguments
are literal. There is no shell evaluation or automatic srun. Configure any
necessary native MPI launcher explicitly; never launch the orchestrator worker
once per rank.

## Scientific input fidelity

Original files stay in input/. Native working copies convert only declared
parameter bound ordering. The fourth column is not rescaled: delta, sigma and
sensitivity mean different things. Duplicate/nonfinite/ill-ordered rows fail.

Control precedence is **explicit non-null YAML > supplied control > catalog
default when absent**. Null preserves native/control defaults. The full template
lists defaults explicitly: remove a field or use null to preserve control values.
Duplicate control keywords fail. Fixed-width records keep keywords at column
nine; values that cannot fit without losing precision fail instead of rounding.
JAX-SOPPI forwards control and overrides; JAX-ReaxFF rejects supplied control
rather than ignoring it.

JAX preflight checks supported energy/charge/geometry/per-atom-force sections,
record shapes, finite values, positive accuracy divisors and geometry references.
It rejects unsupported/RMSG-only entries and structures the native reader might
silently filter. This is not full topology or engine compatibility validation.

Native PIBO cobalt requires settings.driver=cobalt, budget, and
acknowledge_cobalt_subset=true. Remove MoS2-only keys from that block. Its native
Co_clusters/Co2_cluster energy subset and distinct loss are recorded explicitly;
it is not a complete-trainset objective or a fair cross-method ranking.

OGolem calls its existing input bridge inside the allocation, then Java once in
native/bundle/. Custom evaluators require evaluator_command argv. An AMS candidate
script runs inside the allocation, not through sbatch per candidate. Coordinate
OGolem worker counts and candidate engine ranks to avoid oversubscription.
ReaxKit never submits example Slurm files generated by the external bridge.

## Job handoff and commands

Preserve appropriate resource directives, modules and parallel variables from
your job. Add the marker and replace the final engine invocation:

```bash
# reaxkit-worker-contract: 1
exec bash "$1"
```

Do not retain another engine invocation earlier. ReaxKit passes the worker as
argument one and sets unique job names/working/log directories. No site account,
partition, module version, hardware constraint, NSCM or rank count is hardcoded.
Arrays are unsupported. For graceful pre-timeout collection, consider the
user-selected directive `#SBATCH --signal=B:TERM@60`. See the official
[Roar batch-job guide](https://docs.icds.psu.edu/running-jobs/batch-jobs/) for script
arguments and scheduler commands.

```bash
reaxkit optimize-ffield --settings settings.yaml --validate
reaxkit optimize-ffield --settings settings.yaml --dry-run
reaxkit optimize-ffield --submit ./optimization_runs
reaxkit optimize-ffield --status ./optimization_runs
reaxkit optimize-ffield --collect ./optimization_runs
```

Inspect run-config.json launch plans, staged inputs/control and job scripts.
Dry-run executes neither engines nor bridge commands. Installation/module/Java
compatibility is checked in the allocation, not inferred from YAML validation.
Without --validate/--dry-run, --settings prepares and submits on Linux.

Worker reports retain commands, exits, allocation variables, elapsed time and
seed forwarding. Requested, forwarded and scientifically effective seeds differ:
seed_applied remains unverified. Deterministic SOPPI replicas may be identical;
AMS/OGolem have no documented orchestrator RNG-seed mapping. Version strings are
supplied provenance, not automatic engine detection.

## Outputs and recovery

Runs retain input/, native/, job/, run.json, run-config.json and worker.json.
Staged inputs/settings are hashed before submission/execution. External setting
files are hashed without relocating interpreters or external configuration trees.
normalized/conversion.json atomically points to a snapshot containing:

- progress.jsonl: native steps, scalar/vector scores, candidate IDs, measured
  counts/time, objective and dataset scope.
- predictions.jsonl: available references/predictions, units, identity and weight
  semantics; never guessed positional matches.
- artifacts.json: native paths, hashes and restart capability.
- conversion.json and optional source_changes.json: coverage and diagnostics.

Read one snapshot at a time. Repeated collection is idempotent, partial lines
wait, missing values remain null, and failed/nonfinite evaluations retain status.
Best-only outputs never become fabricated histories. Different objectives and
native steps are not automatically comparable. Converters never run engines.

Snapshots reference rather than copy force fields: preserve the whole campaign.
OGolem artifacts are relative to native_results_relative (native/bundle under
the run directory). Collection also works on a copied campaign; execution still
requires original staged paths or a newly prepared campaign.
Recognized AMS 2023.103 MCFF headers and CMA-ES scalar report blocks are parsed
automatically. Explicit conversion options retain the generic mapped-table reader;
unknown layouts still require version-specific mappings. MCFF records distinguish
initial/current errors and the running minimum of observed current errors per native
replica. Its native elapsed column is not assumed cumulative. CMA-ES uses reported
candidate evaluation counts (not invented generation numbers), generation-best
scores, candidate-best scores and CPU time (not walltime). A fort.13 initial baseline
is associated only when its count and candidate minima agree with the report blocks;
it does not replace the optimizer's candidate-best score. Missing predictions remain
missing; fort.99 parity is not guessed. These reports alone do not certify convergence
or scientific equivalence between methods.
See the backend README for mapping examples.

Submission intent is durable before sbatch. Ambiguous jobs are never retried
automatically. --status gives read-only reconciliation candidates; absent
accounting is UNKNOWN, not success. Repeated --submit skips attempted jobs but
can submit remaining staged jobs. Never edit a ledger to retry an uncertain job.
Native resume is not automatic: use a new campaign with an explicit supported
checkpoint or recovered field. After forced kill/node loss, verify the writer
has gone before removing a stale lock, then collect offline. Zero exit with no
recognized results is not scientific success.

## Acceptance gates

1. Run `python -m pytest tests/optimization -q` in the Linux ReaxKit
   environment. Four Linux-only cases exercise real Bash/process groups/signals
   using synthetic runners, never engines or sbatch. Dedicated Linux CI is provided.
2. Use a tiny scientifically valid user-owned dataset, two installed methods and
   two replicas. The example has small budgets, not a dataset-validity or
   compilation-time guarantee. Review allocations and dry-run before submission.
3. Check actual allocations/job IDs, directory isolation, seed forwarding,
   native-to-normalized numbers/weights/units/identities and complete best fields.
4. Run `reaxkit optimize-ffield --smoke-report ./optimization_runs`.
   This read-only report checks evidence and current artifact hashes. Exit 1 means
   automated evidence is incomplete/failed. Manual scientific/site review is
   always required; the report never certifies cluster validation.
5. In a separately authorized tiny run, wait for a report, terminate that exact
   job, then verify final/offline collection and no duplicate submission. Never
   cancel unrelated jobs. Expand to available methods; keep others unvalidated.

No usable WSL distribution, Docker runtime or configured Roar SSH connection was
found here. Actual Linux execution and Roar engine jobs remain acceptance gates,
not accomplishments inferred from Windows tests. The installation-specific
walkthrough needs cluster paths and a permitted job file; none are guessed.

## External runners and licenses

Use runtime.mode=external_runner with command
`[/optimizer-env/bin/python, /optimizer-project/runner.py, "{run_config}"]`.
The runner consumes reaxkit-external-runner-v1 settings, applies supported seeds,
keeps mutable files in native_directory, stages dataset-specific dependencies,
preserves output and propagates failures. prepared_inputs is copied only in this
mode: provide clean inputs, not stale reports/checkpoints. External settings
retain original paths so relative dependencies do not change. Keep files stable
while jobs are pending. Never put credentials in saved YAML/environment maps.

ReaxKit neither vendors optimizer code/binaries nor imports their libraries.
Keep each component's license/notices in its own distribution. Legacy Fortran
headers specifically request no redistribution; absence of a LICENSE file is
not permission. The process boundary is not a blanket legal guarantee for future
tighter integrations.
