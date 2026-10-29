# webui

## Purpose
Implements the web application layer, including API/backend logic, UI composition, and web-specific presentation registry/performance settings.

## What Belongs Here
- UI layout/callback modules and Dash app wiring.
- Web backend payload/schema/serialization helpers.
- Web presentation registry/perf config.

## What Does Not Belong Here
- Engine parsing and analysis task implementations.
- Core runtime registry infrastructure.

## Structure
- `backend/`: API schemas, serializers, backend registries.
- `ui/`: page/layout and callback-facing UI composition.
- `presentation/`: web presentation registry and perf config.
- `app.py`, `dash_app.py`, `callbacks.py`, `components.py`, `layouts.py`, `utils.py`.

## Flow
Web UI issues command/workflow requests, receives analyzed/persisted payloads, and renders interactive views/export actions.

## Extension Points
- Add UI feature surfaces under `ui/`.
- Add backend payload transformations under `backend/`.

## Large-result runtime

`backend/artifact_tables.py` stores immutable Parquet tables and serves bounded
pages. `backend/jobs.py` owns separate local compute/query worker lanes and commits
results only when their transitive inputs still match. Expensive calculations,
filtered/sorted queries, snapshot imports, and exports run outside Dash callbacks.
Utilities always consume full inputs in their worker. Active plots query the
complete artifact and prepare bounded geometry in the query worker. Line summaries
preserve per-range bucket endpoints/extrema and disconnect missing-value buckets;
histograms use exact server-side counts. Zoom, resize and frame requests replace
obsolete work. Cached figures share the 16 MiB query/frame cache.

The canvas reports its quality and lets users click a point for its original row,
or refresh an expired/cancelled view. Images save the displayed summary and current
camera; use table exports for all scientific data. Numeric x values are required
for line plots; missing/nonnumeric x positions produce an explicit error rather
than silently joining across unknown positions. Group aggregation happens before
decimation; at most the configured number of groups is shown, with omissions labelled.

The Plotly 3D prototype displays one frame of a coordinate table, with stable point
IDs, a deterministic full-frame sample and spatial extrema. Numeric frame values
are selected from `frame_index`, `timestep`, or `iter` when x/y/z coordinates exist;
`frame_col` can be supplied explicitly. Optional payload `cell` data uses
`{"origin": [x,y,z], "vectors": [[...], [...], [...]]}`. Optional `bonds` are atom-ID
pairs; only supplied bonds with visible endpoints are drawn. No bonds or periodic
cell are inferred. Geometry is batched, counts/bytes are bounded, and camera state
survives frame changes. Full trajectory playback/prefetch and a dedicated GPU
component remain gated on handler frame access and measured browser performance.

Analysis execution uses the core result cache by default at
`<workspace>/.reaxkit_cache/analysis`. Identical requests with unchanged source
identity can reuse a saved result. Different selections or changed inputs may
require reading the source again; a small output table does not imply cheap
input parsing. This cache is separate from the GUI's bounded page/figure caches.

Rendering limits live in `presentation/ui_performance.json`: 12,000 total line
points at the default width (also on zoom), up to 256 histogram bins, 20,000 total
3D geometry vertices including optional cell/bonds, and a 4 MiB figure cap. A point
budget limits transport; complete-result scans still take time in a cancellable
worker. The current GUI requires Dash 3.1 or newer for optional component state.

Install the `webui` extra to include DuckDB. The default per-worker memory safeguard
is 4096 MiB, configurable with `REAXKIT_UI_JOB_MEMORY_MB`. The local service owns
temporary artifacts for its lifetime; portable snapshots use a JSON file plus a
`.tables` sidecar directory that must be kept with it. Existing inline snapshots
remain importable. Browser reload restores a session while its server is alive.

Open the GUI with `?rk_profile=1` to enable a local browser performance download.
The reproducible benchmark, operating limits, and acceptance status are documented
in [the performance report](../../../benchmarks/WEBUI_PERFORMANCE.md).

## Adjustable workspace

Drag the grips between the sidebar and canvas, hierarchy and parameters, or canvas
and activity drawer. A focused grip also accepts arrow keys (16 px), Shift+arrow
(48 px), Home/End (limits), and Enter (default size). Escape cancels an active drag.
Panel movement stays in the browser; only released preferences are published.
Graphs resize through a throttled container observer without replacing their data.

The command bar offers Analysis, Visualization, Results, and Classic sidebar
presets, a sidebar toggle, reset, and a dark/light theme switch. Each panel has a
collapse control; the canvas can be maximized and restored. Layout and theme use
versioned local browser storage. Reset returns to the Analysis layout while keeping
the chosen theme. Narrow windows use an overlay sidebar; use its command-bar toggle
to reveal the canvas. Hierarchy search retains matching nodes and their ancestors;
Up/Down and Home/End move focus, and Enter selects a node. Selection exposes the
node's available creation, configuration, execution, and deletion actions.

Select an analysis and use **Delete analysis** in the hierarchy or parameters
panel to remove it together with its child utilities, presentations, and results.
Work for the deleted subtree is stopped; unrelated analyses remain available.

Analysis fields are browser drafts. **Apply changes** validates and saves them;
**Discard** restores the saved request; **Run analysis** validates and saves the
current draft before submitting it. Status updates keep the editor mounted.
Drafts survive switching nodes within the page through memory persistence, but
unsaved drafts do not survive a page reload. Units and help come from task schemas;
Engine/Output advanced groups start collapsed. Explicit integer selectors are
limited to 100,000 entries to avoid allocating enormous lists on a UI request.
Use sampling parameters such as `every` for larger selections. Utility and
presentation editors retain their existing save behavior.

The bottom drawer shows up to 12 recent jobs (active first), individual Stop
controls, error details, and up to 30 result cards with counts, Open, and available
provenance. Logs read only the final 256 KiB / 400 lines per file. Hidden job/log
drawer polling is disabled. New plots follow the application theme; an explicitly
chosen Plotly template remains independent. Theme changes preserve scientific
colors, view ranges, and the camera.

Theme tokens and component styles live in `assets/workspace.css`; vector icon
masks live in `assets/workspace-icons.css`. Dash 4 colors use its documented
[component theme variables](https://plotly.com/blog/dash-core-components-gets-a-design-driven-refresh-with-dash-4/),
alongside compatibility selectors for Dash 3. New assets are included in wheels.
The [design preview](../../../benchmarks/gui_workspace_preview.html) illustrates
loaded, empty, running, failed, and huge-result states. It is a mockup, not evidence
of browser performance. The user accepted the GUI after local testing on
2026-10-02. Final automated and installed-wheel checks passed; detailed browser
performance, cross-browser, and display-scaling measurements remain unmeasured,
as recorded in the performance report.

Analysis workers initialize the task registry themselves, including internal task
names such as `partial_energy_series` that have no corresponding CLI route.
Registration failures retain their underlying exception in Activity error details.
