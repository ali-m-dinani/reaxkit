# Submit a multi-optimizer campaign

One invocation of optimize-ffield already prepares and submits every optimizer
in the YAML optimizers list, once per replica. Each receives separate input,
native, normalized and job directories. Jobs are independent Slurm submissions;
they run concurrently only if scheduler capacity/policy permits.

There is one job per optimizer per replica, not one job for all replicas.
For example, three optimizers with two replicas create six jobs. Each generated
script and submitted job uses a name such as ogo-r02-<campaign-id>, with
a three-letter optimizer abbreviation first, followed by the replica and a unique campaign suffix.
This replaces the template's job name; existing prepared campaigns are unchanged.
Abbreviations: ams_mcff=mcf, ams_cmaes=cma, standalone_soppi=sop,
jax_soppi=jsp, jax_reaxff=jrx, indeedopt=ind, ogolem=ogo, pibo=pib,
ezff=ezf, glompo=glo, flocky=flk. YAML optimizer identifiers and directory names
remain unchanged.

The --job option supplies a shared job template. A non-null job in an
optimizer's section overrides it. Each template must end with exec bash "$1"
and contain the reaxkit-worker-contract marker. Never put an optimizer command
in place of that worker invocation.

## Shared template and per-method allocations

Top-level resources override corresponding SBATCH directives in the template.
An optimizer's resources mapping overrides shared resources. Supported keys:
nodes, ntasks, ntasks_per_node, cpus_per_task, time (quoted HH:MM:SS or
D-HH:MM:SS), and mem (for example "64G"). Keep one option per SBATCH line.
Other directives and environment setup are retained. No hardware constraint
is added.

Standalone SOPPI and the integrated serial Flocky runner automatically receive
one node/task/CPU, overriding shared allocation defaults. Explicit conflicting
per-method requests are rejected. Allocating more CPUs never automatically
makes an optimizer parallel: its worker/population/engine settings must agree.
The renderer does not convert serial runners into MPI runners.

Generated job scripts export REAXKIT_OPTIMIZER before running environment
setup. A single shared template can therefore use a shell case statement to
load different module stacks. Alternatively, retain separate previously tested
job files in the per-method sections; submission is still a single command.

## Cobalt: 60-minute budget, 75-minute allocation

First synchronize the modified ReaxKit and optimizer repositories to the
cluster. Rebuild Flocky and, if used, your privately patched standalone SOPPI
executable. These local changes are not automatically available through git
pull until committed/pushed. Do not publish licensed Fortran sources/binaries.

Create cobalt_all_60min.yaml by retaining the runtime and settings mappings
from your successfully tested YAMLs under optimizer_settings. Use one shared
inputs mapping. Geometry is required in addition to ffield, params and trainset.
Use a new output directory; never reuse a completed smoke campaign.

Set these shared values:

```yaml
schema_version: 1
replicas: 1
optimizers: [ams_mcff, jax_soppi, jax_reaxff, pibo, indeedopt, ezff, ogolem, glompo, flocky]
output: ./cobalt_all_60min
job: ./campaign_job.sh
optimization_time_limit_seconds: 3600
collection_interval_seconds: 30
resources:
  nodes: 1
  ntasks: 1
  ntasks_per_node: 1
  cpus_per_task: 1
  time: "01:15:00"
  mem: "64G"
inputs:
  ffield: ./ffield_lit
  params: ./params
  trainset: ./trainset.in
  geo: ./geo
  control: ./control
  params_bounds_order: min_max
```

This is the shared header, not a complete configuration: append your tested
optimizer_settings sections. Within each section, retain runtime.command,
environment, source and engine-specific paths. Set job to null to use the
shared template, or retain that method's tested job file.

Set runtime.shutdown_grace_seconds to 600 and runtime.timeout_seconds to 4380
in each selected section. Remove stale per-method
runtime.optimization_time_limit_seconds overrides, or set them to 3600.
The 73-minute worker hard timeout leaves about two minutes before Slurm's
75-minute kill for cleanup/collection. Preparation, initialization, a slow
in-flight evaluation, and final writing still consume allocation time:
75 minutes cannot guarantee a full 60 optimization minutes plus arbitrary
initialization and shutdown. Hard interruptions remain labeled as such.

Suggested allocations for the previously tested integration modes:

| Method | nodes | ntasks | ntasks_per_node | cpus_per_task | Additional setting |
| --- | --- | --- | --- | --- | --- |
| AMS MCFF | 2 | 16 | 8 | 1 | Native AMS MPI environment |
| OGolem + AMS | 2 | 16 | 8 | 1 | workers: 1; one candidate at a time |
| JAX-SOPPI / JAX-ReaxFF / INDEEDopt short JAX | 1 | 1 | 1 | 8 | CPU backend; independent Python environments |
| GloMPO | 1 | 1 | 1 | 4 | job_workers: 4; single-process engines |
| PIBO / EZFF cobalt serial | 1 | 1 | 1 | 1 | Existing serial LAMMPS evaluator |
| Flocky / standalone SOPPI | 1 | 1 | 1 | 1 | Rebuilt serial executable |

For example, insert resources: {cpus_per_task: 8} alongside runtime/settings
in the jax_reaxff section. For AMS/OGolem insert all four topology values from
the table; shared time/memory remain inherited.

Raise the smoke-test iteration/evaluation ceilings; otherwise methods will
finish after their tiny original budgets. Example ceilings (not a recommendation
for scientific convergence or equal work):

| Method | Settings to review |
| --- | --- |
| AMS MCFF | mcffit: 100000 |
| JAX-SOPPI | max_rows: null (one complete parameter sweep; may finish early) |
| JAX-ReaxFF | num_steps: 1000; retain num_trials and initial-field policy |
| PIBO cobalt | budget: 1000 |
| INDEEDopt | short_samples: 1000 (sampling, not full learned optimization) |
| EZFF | num_epochs: 1000; retain tested population/algorithm |
| OGolem | iterations: 10000; retain tested population |
| GloMPO | max_seconds: 3600, max_function_calls: 10000, max_iterations: 1000 |
| Flocky | iterations: 10000; retain tested swarm_size/cycles |

Convergence or completing a finite sweep can still end a run early. Do not
artificially repeat completed optimizations just to fill the hour. PIBO/EZFF
still use the cobalt subset and objectives are not assumed interchangeable.

AMS CMA-ES is deliberately absent: no verified graceful time-limit hook exists.
Standalone SOPPI is absent until its private rebuilt stop hook passes testing;
then add it with runtime.cooperative_stop_file: true. Neither should be silently
presented as an equivalent validated 60-minute optimization.

## Shared job template

Start with:

```bash
reaxkit optimize-ffield --generate-job-template campaign_job.sh
```

The template uses generic SBATCH values --account=ACCOUNT and
--partition=PARTITION. Replace these placeholders locally before submission;
they are literal placeholders, not shell variables. Keep your actual account
details and site-specific job files private. Add environment setup before the final exec.
Use the following structure for setup; replace each branch with the module
loads and exports from that method's tested job. Do not launch engines here:

```bash
module purge
case "$REAXKIT_OPTIMIZER" in
  ams_mcff|ogolem)
    # Copy tested AMS/Intel/MPI setup, including NSCM and thread variables.
    ;;
  glompo)
    # Copy tested AMS/ParAMS setup; one process/thread per geometry worker.
    ;;
  jax_soppi|jax_reaxff|indeedopt)
    # Copy tested Python/JAX setup; use each runtime's absolute interpreter.
    ;;
  pibo|ezff)
    # Copy tested GCC/OpenMPI/LAMMPS module setup.
    ;;
  flocky)
    # Copy GCC/Boost setup and the local NLopt library path.
    ;;
  standalone_soppi)
    # Copy tested GCC setup; OMP_NUM_THREADS=1.
    ;;
  *) echo "Unsupported optimizer: $REAXKIT_OPTIMIZER" >&2; exit 2 ;;
esac
exec bash "$1"
```

The branch comments are placeholders, not a functioning module setup. To avoid
rebuilding environment setup now, simply keep the existing per-method job
paths in YAML; the new resource renderer overrides their wall time and
allocation. Remove any unwanted hardware constraints from those templates.

## Run once, monitor all methods

From your cluster working directory, with the updated ReaxKit environment active:

```bash
reaxkit optimize-ffield --settings cobalt_all_60min.yaml \
  --ffield ./ffield_lit --params ./params --trainset ./trainset.in \
  --geo ./geo --control ./control --job ./campaign_job.sh --validate

reaxkit optimize-ffield --settings cobalt_all_60min.yaml \
  --ffield ./ffield_lit --params ./params --trainset ./trainset.in \
  --geo ./geo --control ./control --job ./campaign_job.sh --dry-run

reaxkit optimize-ffield --submit ./cobalt_all_60min
reaxkit optimize-ffield --status ./cobalt_all_60min
reaxkit optimize-ffield --collect ./cobalt_all_60min
```

Inspect runs/*/replica-*/job/submit.sh after dry-run. Submission sends all
selected methods/replicas, not just one. For a fresh campaign you may omit
--dry-run to prepare and submit in one command instead; do not do both for
the same output directory. Download the whole campaign after collection.
