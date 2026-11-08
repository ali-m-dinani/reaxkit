# GloMPO integration

GloMPO is the tenth registered optimizer framework. Its separate GPL-3.0 fork
is https://github.com/ali-m-dinani/glompo (upstream mfgustavo/glompo).
ReaxKit contains only its own file converter and launch contract metadata.
No GloMPO code, SCM software, environments or run results are bundled.

## Evidence and limitations

A real Windows/Python 3.12 sphere/CMA benchmark completed 12 evaluations and
produced HDF5, YAML and a trajectory PNG. The fork exports numerical table data
to JSONL with source hashes. ReaxKit checks these hashes and preserves every
evaluation's call number, optimizer ID, parameter vector, score and running
best. HDF5 and YAML remain artifacts, not unsafe Python deserialization inputs.
An incomplete/stale export is rejected; it can be regenerated after termination.

The AMS/ParAMS runner has **engine smoke evidence, not full scientific validation**.
Job 57122446 reached nine finite recorded evaluations and exited
successfully at the manager time limit; full scientific acceptance is still
pending. Its first candidate was transformed by CMA and is not an unchanged
original-field baseline.
Its objective is native ParAMS SSE with dataset sigma/weights. Parameters in
its history are scaled to [0,1], not literal force-field values. Native summary
and best-field artifacts are retained, but no labeled parity predictions or
cross-optimizer scientific equivalence are claimed. A downloaded analytic
benchmark must never be presented as ReaxFF training.

Roar's AMS 2023.103 Python 3.8 completed a 10-evaluation sphere benchmark;
downloaded HDF5/JSONL hashes and scores agree. Its official SCM conversion
preserved 12 active parameter bounds, 144 energy expressions/references and
147 jobs. Weight/Sigma-squared matches inverse accuracy-squared. The converted
jobs include 130 geometry optimizations and 17 single points, not an entirely
single-point smoke workload. These checks are not a successful engine fit.

## Roar: AMS 2023.103 prerequisite gate

From the existing campaign parent directory:

```bash
git -C reaxkit pull --ff-only
git clone https://github.com/ali-m-dinani/glompo.git glompo
module load ams/2023.103
(
  cd glompo
  "$AMSBIN/amspython" -c "from glompo.interfaces.params import ReaxFFError; print('GloMPO/ParAMS import OK')"
)
```

Stop on an import error and preserve the full traceback. The old upstream
interface uses SCM internals that may differ in installed AMS versions.
The sphere benchmark does not validate that API. Do not submit a calculation
until this check passes. Keep optimizer dependencies out of ReaxKit's venv.
If the checkout already exists, use `git -C glompo pull --ff-only` instead of
cloning again. Use AMS's bundled interpreter for this profile, not the separate
modern `envs/glompo` interpreter. Do not install modern pinned dependencies into
AMS's shared environment. The latter environment remains useful for benchmarks.

The fork's ReaxKit runner explicitly requests SCM-style conversion, including
`use_weights=True` and the supplied control path. It stages params in min/max
order. Strict checking accepts either endpoint (cobalt p_ovun2=-2.5 is valid)
without recentering, while rejecting invalid ranges and missing training jobs.
`--check-only` now performs conversion on temporary copies, not merely imports.
Actual runs save converted YAML in `native/converted_to_params` for inspection.

The updated GloMPO runner preserves the exact requested scaled vector as the
first CMA population member (no extra evaluation outside the budget). Profiles
record `initial_candidate_policy` and `initial_parameters_scaled`. The converter
marks an `initial_parameter_evaluation` only if call 1 matches the recorded
vector; older/mismatching histories remain ordinary evaluations with a warning.
Existing runs are not retroactively treated as original-field baselines.
The fork also fixes active/full range selection and endpoint handling in its
parameter-setting warning. Run `examples/reaxkit_initial_check.py` with AMS's
`amspython` from the GloMPO checkout before preparing a fresh engine campaign.

## After the prerequisite gate passes

Generate a fresh settings template with
`reaxkit optimize-ffield --generate-template cobalt_glompo_smoke.yaml`. Select only `glompo`,
one replica and `seed: null`. Supply ffield, params, geo, trainset **and control**,
with the correct `params_bounds_order` and a new campaign output directory.
Configure the GloMPO section as follows (replace absolute paths):

```yaml
optimizer_settings:
  glompo:
    runtime:
      mode: external_runner
      command:
        - /storage/icds/RISE/sw8/ams2023.103/bin/amspython
        - /absolute/campaign-parent/glompo/reaxkit_runner.py
        - '{run_config}'
    settings:
      max_function_calls: 12
      max_seconds: 300
      population_size: 4
      max_iterations: 3
      sigma0: 0.1
```

Use a user-owned job file with one node, one task, one CPU, the account and
partition you already use, a short walltime, and **no genoa constraint**.
Load the same verified AMS/Python environment, set NSCM=1 and OMP_NUM_THREADS=1,
and end with the ReaxKit worker contract, not a direct engine command:

```bash
# reaxkit-worker-contract: 1
exec bash "$1"
```

Validate and stage using `--settings cobalt_glompo_smoke.yaml --validate` and
then `--dry-run`. Run the fork's `reaxkit_runner.py` with the generated
`runs/glompo/replica-0001/run-config.json --check-only`. Submit once with
`reaxkit optimize-ffield --submit ./cobalt_glompo_smoke` only after checks pass.
Inspect `--status`, then `--collect` once finished. Download the whole campaign
for scientific and converter review; do not commit it.

If interrupted, use the GloMPO interpreter to run
`glompo/reaxkit_runner.py --export-native <run>/native/manager` after the job
stops, then collect again. No automatic resubmission or exact restart is implied.
# Geometry-level parallel smoke test

The serial AMS 2023.103 job 57122639 completed 11 evaluations in 328.7 seconds.
Its exact initial candidate remained best (1451.4515029431068); termination was
the manager time limit, not scientific convergence. No parity data is available.

Set `optimizer_settings.glompo.settings.job_workers: 4` in a fresh campaign to
request four concurrent ParAMS geometry jobs per candidate. Keep one node, one
Slurm task, and request `--cpus-per-task=4`. Keep NSCM, OMP_NUM_THREADS and
SCM_USE_THREADS at 1: the four CPUs are for separate geometry jobs, not MPI ranks.
Do not add a hardware constraint. The default `job_workers: 1` remains serial.

This does not enable concurrent CMA candidates, multiple optimizers, or
multi-node execution. The runner rejects insufficient allocation/affinity and
records its resource plan in `manager/glompo-profile.json`. Check-only validates
the installed ParallelLevels constructor but defers allocation checks outside
Slurm. Run check-only with AMS's Python before submitting; parallel AMS execution
is not yet cluster-validated. Preserve previous campaign directories.
