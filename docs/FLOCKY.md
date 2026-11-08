# Flocky integration and acceptance gates

Fork: https://github.com/ali-m-dinani/flocky
Upstream: https://github.com/df398/flocky (a472e3c).

Flocky is a C++ RiPSOGM optimizer using a separate Tapered ReaxFF engine. The
ReaxKit adapter reads files only. Its first runner profile is serial and disables
regularization and local minimization; MPI and auxiliary force/frequency targets
are not yet exposed. Native objectives are not assumed comparable to AMS/JAX.

## Licensing

The C++ optimizer is GPL-3.0-or-later. Upstream bundled Fortran headers contain
redistribution restrictions and name Adri van Duin and David Furman. Obtain
authorization for a compatible engine. No optimizer/engine implementation or
engine binary is copied into ReaxKit. Do not use the GPL repository label as
permission to redistribute those engine files.

## Validation status

The converter is based on native output writers and upstream example_01 outputs:
iteration/global-best checkpoints, parameter-only fields and overwritten native
reports. Synthetic tests cover cycles, incomplete lines, duplicate steps,
nonfinite values and campaign preparation. This is not proof of an engine run.
Local native build/sample and Roar execution remain acceptance gates until
explicitly recorded as completed. Parity extraction and exact restart are absent.

Local inspection converted the upstream saved example into 20 checkpoints and
nine artifacts (first/last native best score 10928700.08 / 6208860.0073).
ReaxKit optimization tests: 143 passed, 12 skipped. These are parser/contract
checks, not a new optimization. Cygwin GCC/Fortran/Boost installation succeeded;
NLopt/build prerequisites remain unresolved, so no fresh local run is claimed.

## Roar prerequisites (first step)

From the existing campaign base directory:

```bash
git -C reaxkit pull --ff-only
git clone https://github.com/ali-m-dinani/flocky.git
module spider gcc
module spider boost
module spider nlopt
```

Inspect available compiler/library versions before constructing build commands;
do not assume library modules exist or mix incompatible compiler stacks. Use the
fork's Makefile.reaxkit for a serial optimizer build. The separate engine must be
authorized, compiled and evaluated before optimization is accepted. No genoa
constraint is required or recommended here.

## Settings skeleton (replace all placeholder paths)

```yaml
schema_version: 1
replicas: 1
optimizers: [flocky]
seed: null
output: ./cobalt_flocky_smoke
job: ./flocky_job.sh
inputs:
  ffield: ./ffield_lit
  params: ./params
  geo: ./geo
  trainset: ./trainset.in
  control: ./control_tapered
  params_bounds_order: min_max
scheduler: {type: slurm, profile: roar_collab}
optimizer_settings:
  flocky:
    runtime:
      mode: external_runner
      command: [/absolute/path/reaxkit/.venv/bin/python, /absolute/path/flocky/reaxkit_runner.py, '{run_config}']
    settings:
      executable: /absolute/path/flocky/flocky_serial
      engine_executable: /absolute/path/authorized/tapreaxff
      params_mod: ./params_flocky.mod
      swarm_size: 2
      iterations: 2
      cycles: 1
```

The job must load the matching runtime libraries, use one node/task/CPU, set
OMP_NUM_THREADS=1, and include the ReaxKit worker contract followed by
`exec bash "$1"`. Start with a 20-minute limit. Do not call flocky directly in
the job template; the worker launches its external runner.

params_flocky.mod requires seven columns: absolute ffield line, token column,
section, row, parameter, lower, upper. Manually verify its semantic mapping.
The runner validates matching common bounds and numeric locations; it never
silently reinterprets a six-column params file as native params.mod.

After configuring all prerequisites, validate/dry-run the YAML, then invoke the
runner with the generated run-config.json and --check-only before --submit.
Download the complete campaign after collection for native-output inspection.
