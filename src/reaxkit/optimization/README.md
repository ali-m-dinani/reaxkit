# Force-field optimization backend

ReaxKit stages YAML campaigns, submits isolated Slurm jobs, records scheduler and
worker state, and converts native results. Engines/objectives remain external.
Native command mappings and external-runner escape hatches are available.
No GUI or real HPC validation is claimed.
See [the permanent user guide](../../../docs/ffield_optimization.md) for native
launch capabilities, environment isolation and the Roar smoke-test procedure.
See [validation and limitations](../../../docs/optimization_validation.md) for evidence levels and adapter contracts.

## Commands

```bash
reaxkit optimize-ffield --generate-template settings.yaml
reaxkit optimize-ffield --generate-job-template roar_job.sh
reaxkit optimize-ffield --list-optimizers
reaxkit optimize-ffield --settings settings.yaml --validate
reaxkit optimize-ffield --settings settings.yaml --dry-run
reaxkit optimize-ffield --submit ./optimization_runs
reaxkit optimize-ffield --status ./optimization_runs
reaxkit optimize-ffield --collect ./optimization_runs
reaxkit optimize-ffield --smoke-report ./optimization_runs
reaxkit optimize-ffield --convert /path/native --optimizer jax_soppi --output /path/normalized
```

Normal `--settings` invocation prepares and submits on Linux. Template generation,
conversion, and dry-run work without installed engines or Slurm. Prepare again
on Linux before submission if inspecting a configuration on Windows. YAML-relative
paths use the settings directory; CLI input overrides use the current directory.
An existing campaign output directory is never overwritten. Use `--submit` after
a dry-run; attempted jobs are not submitted again. No jobs are submitted during
ReaxKit tests.

## Job and runner contracts

A supplied batch script owns resource directives, site modules, and environment.
Add the exact line `# reaxkit-worker-contract: 1`. Replace its final engine
invocation with `exec bash "$1"`. ReaxKit passes the generated worker launcher as
the first script argument. Do not retain a previous engine invocation elsewhere
in the script. Arrays are not supported. Per-optimizer `job` overrides the shared
file. ReaxKit overrides job name, working directory, and log paths for uniqueness.

Native mode appends verified arguments/control settings to the configured
entrypoint; see per-optimizer launch.py files and the user guide. Workflows
without a suitable native CLI require an **external runner**, configured as argv:

```yaml
runtime:
  mode: external_runner
  command: [/path/to/runner-env/bin/python, /path/to/runner.py, "{run_config}"]
  version: null
  engine_version: null
  source: null
  environment: {}
```

This is not the optimizer's unmodified CLI. The runner receives one JSON file
with contract `reaxkit-external-runner-v1`: absolute staged input paths, input
hashes, explicit settings and resolved defaults, replica, requested seed/support,
native output directory, bounds order, and runtime metadata. It must consume
those settings, keep the native engine/objective, invoke its optimizer, and
propagate failure through a nonzero exit code. It runs in the run's `native/`
directory. `prepared_inputs` is copied there if supplied. Dataset conversion and
native mappings are deliberately bounded; ReaxKit does not claim an arbitrary
optimizer executable can consume this JSON.

The external runner is responsible for explicit-setting > native-control > native-default
precedence, objective/engine fidelity, input compatibility, and native seeding.
The run manifest records `seed_applied: null` because the infrastructure cannot
prove a runner applied it. Native workers additionally record documented
seed-argument forwarding; this is not proof of scientifically effective seeding.
Unsupported seeding is not called reproducible.
Version fields are supplied provenance, not automatic engine detection. Comparison
fingerprints include configured settings and runtime identity, but comparison
status stays unverified: equal fingerprints alone do not establish scientific
comparability or confirm engine versions.

An executable may begin with an environment-variable reference such as
`${RUNNER_HOME}/bin/runner`; expansion occurs only for the executable at worker
startup after the job's module setup. Arguments are passed literally without a
shell. The worker's own ReaxKit interpreter is captured when preparing on the
cluster. Keep separate environments for ReaxKit, each optimizer, and engines.
Do not store passwords/tokens in YAML: explicitly configured environment values
are saved with the resolved settings.

## Results and recovery

`campaign.json` indexes independent run directories. Each contains immutable
inputs/configuration, a submission ledger, worker state, native outputs, and
normalized results. `normalized/conversion.json` atomically points to a complete
snapshot under `normalized/snapshots/<id>/`, containing `progress.jsonl`,
`predictions.jsonl`, `artifacts.json`, and conversion diagnostics. Readers should
follow that pointer rather than combine files from different snapshots.

Unchanged conversion produces the same snapshot ID. Source changes/truncation
are recorded, previous snapshots remain, trailing partial lines wait for a later
collection, and unknown data remains null. Artifact checksums reference native
files at collection time; a normalized snapshot does not itself copy large
force-field files. Preserve native outputs/checkpoints for restart.

Progress retains native step kinds, scalar/vector objectives, available timing,
and measured counts. No inferred engine-call counts, cross-objective rankings,
or surrogate-to-engine relabeling. Missing prediction identities prevent automatic
cross-file matching. JAX text reports are rounded to two decimals; JAX-SOPPI
residual array identities are source-local. No engine runs merely to populate plots.

AMS progress requires a version-specific mapping until native samples establish
defaults. Example conversion options (indices must match the actual file):

```yaml
progress_file: MCFFOptimizer.log
skip_rows: 1
columns:
  step: 0
  score: 1
  elapsed_seconds: 2
```

Pass this file using `--conversion-config`, or put the mapping in the optimizer's
`conversion` section. The example is illustrative, not an AMS format claim.

The worker periodically collects output and collects again on exit. SIGTERM or
SIGINT stops its child process group where possible and preserves available
results. SIGKILL/node loss can leave stale writer locks; verify the writer is
gone before removing the named lock. Offline `--collect` needs no Slurm access.
Exact native resume is distinct from parameter-only restart and is not automatic.

Submission intent is saved before `sbatch`. Timeout or ambiguous responses leave
an unknown state, never an automatic retry. Status queries attempt read-only
reconciliation by the unique job name, including completed jobs. Reconciliation
candidates require inspection; no job is resubmitted or adopted implicitly.
Accounting absence is UNKNOWN, not success. Scheduler and scientific-result
states stay separate. A zero exit with no recognized results is reported as
completed with scientific results unavailable.

## Validation scope

Synthetic fixtures test every adapter's declared format, including explicit AMS
column mappings. Six existing local optimizer output sets also passed read-only
conversion checks. Mocked Slurm and subprocess tests do not validate cluster
resources, MPI launch behavior, a licensed engine, or native optimization.
Linux-specific real-process tests and a dedicated CI workflow are provided.
Actual Linux execution and authorized Roar smoke tests remain open gates:
no usable local Linux distribution or configured Roar connection was found.
The read-only smoke report checks evidence but never certifies scientific/site
validation automatically. The broader
CLI suite currently has two independently reproduced pre-existing failures:
stale generated active-site workflow documentation and missing help metadata for
the unrelated --export-per-atom-csv option. The optimization command's own help
inventory is checked separately.
