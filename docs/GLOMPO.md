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

The AMS/ParAMS runner is implemented but **not yet engine- or Roar-validated**.
Its objective is native ParAMS SSE with dataset sigma/weights. Parameters in
its history are scaled to [0,1], not literal force-field values. Native summary
and best-field artifacts are retained, but no labeled parity predictions or
cross-optimizer scientific equivalence are claimed. A downloaded analytic
benchmark must never be presented as ReaxFF training.

## Roar: first prerequisite gate

From the existing campaign parent directory:

```bash
git -C reaxkit pull --ff-only
git clone https://github.com/ali-m-dinani/glompo.git glompo
reaxkit/.venv/bin/python -m venv envs/glompo
envs/glompo/bin/python -m pip install -r glompo/requirements-reaxkit.txt
envs/glompo/bin/python -m pip install -e ./glompo
envs/glompo/bin/python -m pip check
module load ams/2023.103
envs/glompo/bin/python -c "from glompo.interfaces.params import ReaxFFError; print('GloMPO/ParAMS import OK')"
```

Stop on an import error and preserve the full traceback. The old upstream
interface uses SCM internals that may differ in installed AMS versions.
The sphere benchmark does not validate that API. Do not submit a calculation
until this check passes. Keep optimizer dependencies out of ReaxKit's venv.

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
        - /absolute/campaign-parent/envs/glompo/bin/python
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
