# ReaxKit

**ReaxKit** is a modular, extensible Python toolkit for **pre‑processing, post‑processing, and analysis of ReaxFF** molecular dynamics simulations.
It provides a clean separation between file parsing, analysis routines, and reproducible workflows, with **Python APIs**, a **CLI interface**, and a **Dash-based GUI**.

ReaxKit is designed for researchers who want a transparent, scriptable bridge between raw ReaxFF data and quantitative, publication‑ready results.

---

## Key Capabilities

### Engine
- Shared engine abstractions for consistent parsing and generating input/output files across 3 different ReaxFF simulation engines, namely `ams`, `lammps`, `reaxff`
- Example IO and generator modules under the `reaxff` engine:
  - ReaxFF input and output files such as `xmolout`, `fort.7`, `fort.13`, `molfra`, and more
  - Input generators for `control`, `geo`, `eregime`, `tregime`, and related files

### Analysis
- `get-potential-energy --relative --num-atoms N --export energy.csv` exports `potential_energy`, `relative_potential_energy`, `potential_energy_per_atom`, and `relative_potential_energy_per_atom` alongside the existing axes and plotting columns. Relative values use source frame 0 by default (or `--reference-frame`), before sampling. `--per-atom` can instead use each frame's atom count; relative per-atom energy is the difference between the normalized frame and normalized reference energies.
- Successful checkpointed CLI/workflow analyses consolidate their full recovery data into one verified `*.results.zip` file in the analysis cache, then remove their recovery checkpoint files after all requested outputs are saved. Repeating the same scientific options reuses this cache directly, without extracting batches or rerunning calculations. Interrupted analyses and failed output/cache writes keep their checkpoints for `--resume`. Explicit `--resume` takes precedence over cache reuse. With analysis caching disabled, checkpoints remain available; direct Python executor calls outside a workflow also retain them. Existing checkpoint folders are not swept automatically; successfully resuming one with caching enabled applies the same consolidation and cleanup.
- `get-hbn-reference-local-polarization` skips the large local, cell, and layer CSV tables by default; add `--export-local-csv` to request them. Summary/mapping output and requested plots or `--write-extxyz` remain available. This only changes exports, not calculations or checkpoint data, and does not delete older CSVs. Per-atom displacement exports remain separately controlled by `--write-displacements` and the output profile.
- AMS analyses can combine KF/RKF coordinates and charges with an external `fort.78` field schedule: use `--engine ams --input reaxout.kf --fort78 /path/to/fort.78`. Without an override, the adapter looks beside the KF/RKF file. The existing ReaxFF field reader preserves field components and MD iteration numbers; field-dependent analyses require matching iterations, not matching row numbers. Streaming parses the field schedule once. This supports charged-trajectory field export, local potential/field analyses, and charge-versus-field analysis. Missing fields are not replaced with zero.
- Engine-agnostic analyzers built on ReaxKit's domain data models
- Analyzer tasks organized around separated request, result, and task objects
- Analysis components separated from engine IO so the same computations can be reused across scripts, workflows, web UI, and presentation modules

### Workflows
- Reproducible workflows organized by data handling, file tools, meta orchestration, presentation, and study design
- Automation paths for common ReaxFF pre-processing, analysis, post-processing, and presentation tasks

### Web UI
- Dash-based interface components for interactive ReaxFF data inspection
- Backend, UI, and presentation layers for browser-driven workflows

### Presentation
- Publication-ready plotting utilities:
  - 2D plots, dual-axis plots, tornado plots, 3D scatter, and heatmaps
- Video generators

### Utilities
- Shared data, media, and numerical utilities
- Common infrastructure used by analysis, workflows, web UI, and presentation modules

See the full documentation (API reference, tutorials, examples) on [ReaxKit Site](https://ali-m-dinani.github.io/reaxkit/).

---

## Project Layout

```
src/reaxkit/
├── analysis/        # Engine-agnostic analysis tasks
├── cli/             # Command-line entry points
├── core/            # Registries and shared core infrastructure
├── data/            # Packaged reference data and resources
├── domain/          # Central data models, requests, and results
├── engine/          # IO handlers and input generators
├── help/            # Introspection and help system
├── presentation/    # Plotting, active-site views, and media presentation
├── utils/           # Shared data, media, and numerical utilities
├── webui/           # Dash/web interface backend, UI, and presentation layers
└── workflows/       # Workflow orchestration and automation
```

---

##  Testing

Run unit tests with:

```bash
pytest -s tests/
```

to test the package and get the timing for their execution.

---

## Command diagnostics

Diagnostics are disabled by default for every CLI command. Pass `--diagnostics`
after the command name to write a unique report folder under
`reaxkit_workspace/logs/diagnostics/<command>_<UTC timestamp>_<unique suffix>/`
(or beneath your `--project-root`). Its location is printed at command startup.
Help and argument-parsing errors do not create reports. Recording starts after
CLI imports and argument parsing, before command execution.

- `report.json`: command arguments, interpreter/module paths, host, Slurm allocation,
  timestamps, exit status, and the execution-step report.
- `console.log`: Python stdout/stderr, including progress, also displayed normally.
- `stacks.log`: Python stacks for all threads every 60 seconds, annotated with
  UTC timestamps and elapsed seconds. Sample headers contain collection times;
  observer annotations follow within roughly 0.25 seconds. The sampler's own
  stack is excluded, and each stack is limited to 100 frames.
- `stacks.raw.log`: original cooperative Python stack samples, retained
  independently of the observer.
- `stack_summary.txt` and `stack_summary.json`: live per-function tables grouped
  by thread, with self/inclusive sample counts and estimated seconds (count times
  sampling interval). Self means the top Python frame; inclusive means anywhere
  in that thread's stack, counting recursive occurrences once per sample.
  These are observations, not function-call counts or exact profiler timings.
  Waiting time is included; inclusive and cross-thread times overlap and must
  not be summed as command elapsed time. Short work between samples is missed.
- `resources.jsonl`: approximately five-second samples of process CPU time,
  resident memory, and thread count. CPU values are cumulative seconds, not percentages;
  measurements cover the command process, not its subprocesses.
- `steps.log`: finalized human-readable execution report; `exception.log` records
  Python exceptions when present. `observer.log` contains observer errors, if any.

Use `--diagnostics --diagnostic-interval-seconds 30` to enable diagnostics with a
different stack interval. Place these flags after the command
name. A Python thread collects referenced frames through `sys._current_frames()`.
Native code holding the GIL delays stack sampling; missed samples are not
backfilled, and estimated times can undercount that work. A separate lightweight
observer continues resource monitoring even while analysis holds the GIL.
Stack traces are written to files, not repeated on the console.

Periodic diagnostics no longer use `faulthandler.dump_traceback_later`: its native
watchdog can race with thread/frame teardown and crash the interpreter itself.
ReaxKit does not install or cancel process-wide faulthandler timers or change
fatal-signal handlers. `PYTHONFAULTHANDLER=1` can still record fatal errors in the
scheduler log. Remove legacy watchdog wrappers such as `formal_diagnostic.sh`
when using these built-in diagnostics.

On forced termination (including Slurm time limits), finalization may not run.
Existing console, stack, and resource samples remain; a `running` report with no
finish time means **unfinished reporting**, not proof the job is still active.
Native/subprocess writes that bypass Python streams stay in the scheduler logs.
Reports contain supplied command arguments; inspect them before sharing.

### Deterministic profiling (optional)

Install from your clone with `python -m pip install -e '.[profiling]'`, using the
same interpreter as the ReaxKit executable. Add `--diagnostics --profile` after the command name to
record actual function-call events and measured durations with Yappi, across the
main thread and Python worker threads. This is opt-in because instrumentation
adds overhead. `--profile-clock wall` (default) includes waiting; use
`--profile-clock cpu` for thread CPU time instead. Only one clock is recorded per run.

The same per-command **diagnostics** folder contains `profile_summary.txt`
(all functions sorted by cumulative time, with calls, primitive/nonrecursive
calls, self seconds, cumulative seconds, and thread timings), `profile.json`
(machine-readable equivalent), and `profile.pstat` (call graph for pstats-compatible
viewers). These files are separate from the estimated `stack_summary.*` tables.
Self time excludes callees; cumulative time includes them. Inclusive/thread wall
times overlap and must not be added as job elapsed time. Timings are measured,
not sampling estimates, but are not overhead-free or infinitely precise.

Profiling begins just before command execution and is finalized on normal exit
or a handled Python error. Startup imports/argument parsing, subprocesses, and
internal C/Fortran/BLAS stacks are not profiled; visible native call boundaries
are included. Forced termination can prevent profile files being saved; the
periodic stack diagnostics still help investigate hangs. `--profile` requires
diagnostics enabled and a fresh process without another active profiler or
retained Yappi session. No extra profiler dependency is needed without `--profile`.

## Citation

If you use ReaxKit in your work, please cite:

```text
Mohammadi Dinani, A., et al. (2026).
ReaxKit: A Modular Python Toolkit for Preparing, Parsing, and Analyzing ReaxFF Molecular Dynamics Simulations.
arXiv:2609.22019. https://arxiv.org/abs/2609.22019

Source code: https://github.com/ali-m-dinani/reaxkit
Software archive: https://doi.org/10.5281/zenodo.18485384
```

---

## Future Directions

* Implement machine learning-based surrogate models for rapid property prediction and active-site identification.
* Develop a plugin system for user-contributed analysis routines and visualization components.
* Integrate with Jupyter notebooks for interactive analysis and visualization.

If you have any feature request, you can submit it through the ReaxKit's GitHub page, or directly sending an email to `Dinani@psu.edu`.

---
## Additional resources:

- [AUTHORS.md](AUTHORS.md) —  Full credits and acknowledgments.
- [LICENSE](LICENSE) — Full license terms under the **MIT License**
