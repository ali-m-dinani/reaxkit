# Architecture Overview

This page gives a top-level mental model of how ReaxKit is organized and how data moves through it.

## 1) File Categories

ReaxKit modules are grouped by role:

- **Engine** (`src/reaxkit/engine/...`): file readers/writers, adapters, and generators tied to specific engines (ReaxFF, LAMMPS, AMS).
- **Analysis** (`src/reaxkit/analysis/...`): task-oriented computations with request-task-result contracts.
- **Workflows** (`src/reaxkit/workflows/...`): CLI orchestration layers that parse args, build requests, run tasks, and present/export outputs.
- **Utils** (`src/reaxkit/utils/...`): shared utility logic (numerical helpers, EOS tools, support functions).
- **Presentation** (`src/reaxkit/presentation/...`): plotting, rendering, and output formatting utilities.
- **WebUI** (`src/reaxkit/webui/...`): graphical interface and callback wiring.

<a id="all_things_in_reaxkit_by_category"></a>

The figure below shows a category-level map of major ReaxKit components.

<div style="text-align:center;" markdown="1">
![all_things_in_reaxkit_by_category](figures/all_things_in_reaxkit_by_category.png){ style="width:90%; max-width:800px;" }

*Figure: ReaxKit components organized by category.*
</div>

How to read this figure:

- Use it as a routing guide when deciding where new code belongs.
- Engine and Analysis form the core data-processing path, while Workflows orchestrate execution and user-facing commands.
- Utils and Presentation provide shared support layers used across multiple categories.
- WebUI sits on top of workflows/analysis for interactive usage patterns.

Examples by category:

- **Engine**: `src/reaxkit/engine/reaxff/io/xmolout_handler.py` is an Engine file because it parses a ReaxFF output format into structured data.
- **Analysis**: `src/reaxkit/analysis/trajectory/msd.py` is an Analysis file because it defines analysis requests/tasks/results and computes MSD from trajectory data.
- **Workflows**: `src/reaxkit/workflows/trajectory_workflow.py` is a Workflow file because it handles CLI arguments and dispatches analysis execution.
- **Utils**: `src/reaxkit/utils/equation_of_states.py` is a Utils file because it provides reusable numerical/helper functions shared by higher-level modules.
- **Presentation**: `src/reaxkit/presentation/plot/renderers/single.py` is a Presentation file because it is responsible for plot rendering/output formatting.
- **WebUI**: `src/reaxkit/webui/ui/analysis/callback_sections/execution_callbacks.py` is a WebUI file because it wires interactive UI callbacks to analysis execution actions.

## 2) How User Requests Are Processed (Engine-Independent)

This section focuses on how a user request for data/information is handled,
independent of whether the backing files came from ReaxFF, LAMMPS, AMS, or another source.

Core flow:

1. User submits a request (CLI or UI), including what to compute and optional filters.
2. Workflow layer validates/normalizes the request and builds a typed **request** object.
3. The request is routed to the matching analysis task based on task contract/type.
4. The analysis executor runs task logic and produces a typed **result** object.
5. Result is presented/exported (table, plot, file), using a consistent output contract.

<a id="how_requests_are_handled_and_analysis_executor_works"></a>

The figure below shows how request routing and analysis execution are connected.

<div style="text-align:center;" markdown="1">
![how_requests_are_handled_and_analysis_executor_works](figures/how_requests_are_handled_and_analysis_executor_works.PNG){ style="width:90%; max-width:800px;" }

*Figure: Request parsing, routing, analysis execution, and result delivery flow.*
</div>

How to interpret this figure:

- The left side is user intent (what information is requested).
- The middle layers are request shaping and task routing.
- The executor layer runs task-specific computation with a uniform interface.
- The right side is output delivery (display/export), independent of source engine format.

Example:

- Command: `reaxkit timeseries --field trajectory[1].z --xaxis time --export atom1_z.csv`
- Why this fits the flow:
  - The request asks for one specific information stream (atom-1 z-coordinate over time).
  - Workflow converts CLI args into a typed request.
  - Executor dispatches to the timeseries task.
  - Task returns a structured series/table result.
  - Presentation/export writes `atom1_z.csv` using the same result contract used by other tasks.

## 3) `reaxkit-workspace` Structure

A typical `ReaxKit` workspace is organized into these folders:

- **analysis**
- **cache**
- **data**
  - **raw**
  - **parsed**
  - **run_index**
- **inputs**
- **figures**
- **logs**
- **reports**

<a id="reaxkit_workspace_overall_folder_structure"></a>

The figure below shows the overall folder layout of a typical `reaxkit-workspace`.

<div style="text-align:center;" markdown="1">
![reaxkit_workspace_overall_folder_structure](figures/reaxkit_workspace_overall_folder_structure.png){ style="width:90%; max-width:800px;" }

*Figure: Overall folder structure for organizing ReaxKit inputs, generated files, and outputs.*
</div>

How to use this structure:

- Keep original simulation artifacts in `data/raw/<run_id>/...` and do not edit them in place.
- Store normalized parsed datasets in `data/parsed/<parsed_id>/` (for example `trajectorydata.h5` + `meta.json`).
- Use `data/run_index/<run_id>.json` for run-level metadata and lookup.
- Keep generated/selected input files under `inputs/<run_id>/...`.
- Write analysis tabular outputs under `analysis/<task>/<analysis_id>/...`.
- Write visual outputs under `figures/<task>/<analysis_id>/...`.
- Use `cache` for handler/analysis caches and cache indexes.
- Keep execution traces in `logs` and higher-level deliverables in `reports`.

Example placement by folder:

- **inputs**: `ReaxKit/inputs/<run_id>/control`
- **data/raw**: `ReaxKit/data/raw/<run_id>/xmolout`
- **data/parsed**: `ReaxKit/data/parsed/<parsed_id>/trajectorydata.h5` and `ReaxKit/data/parsed/<parsed_id>/meta.json`
- **data/run_index**: `ReaxKit/data/run_index/<run_id>.json`
- **analysis**: `ReaxKit/analysis/msd/<analysis_id>/result.csv` and `ReaxKit/analysis/msd/<analysis_id>/settings.json`
- **figures**: `ReaxKit/figures/msd/<analysis_id>/msd_vs_time.png`
- **cache/handlers**: `ReaxKit/cache/handlers/<handler_id>/cache.h5`
- **cache/analysis**: `ReaxKit/cache/analysis/<analysis_id>/cache.h5`
- **cache/index**: `ReaxKit/cache/index/handlers.json` and `ReaxKit/cache/index/analysis.json`
- **logs**: `ReaxKit/logs/reaxkit.log`
- **reports**: `ReaxKit/reports/summary.md`

Recommended practice:

- Keep raw source files immutable.
- Keep generated/processed artifacts in their designated folders.
- Prefer HDF5 + JSON for parsed/cached metadata and CSV/PNG for exported analysis products.
- Keep reports and reproducibility metadata version-controlled.

---

For first usage flow, continue with [Quickstart](quickstart.md) and [Tutorials](tutorials/index.md).
