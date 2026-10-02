# GUI performance: measurements and validation

## Closeout — 2026-10-02

The GUI improvement work is closed for the agreed local-workstation scope. The
user tested the GUI successfully and accepted its workflow and appearance, and
requested targeted automated verification instead of further extensive manual
testing. The implementation review found no remaining required feature work.
The temporary improvement plan was removed at the user's request; this report
and the webui README retain the operating details and validation evidence.

Final verification on Windows with the configured Python 3.12.10 environment:

- `python -m pytest tests/webui tests/engine/test_quick_frame_counts.py tests/analysis_tasks/test_charge_table.py -q`: **77 passed** in 34.28 seconds. Two existing Dash DataTable deprecation warnings remain; Dash is constrained below version 5.
- `node --test tests/webui/workspace_client.test.cjs tests/webui/plot_queries_client.test.cjs`: **11 passed**.
- A freshly built wheel installed into an isolated target passed `benchmarks/webui_installed_smoke.py`: imports came from that target; Dash routes, packaged CSS/JS/icon assets, and a Windows-spawn plot worker passed.
- Coverage includes legacy and portable snapshots, full-result exports, cancellation and stale-result handling, independent query/compute lanes, bounded queues, shutdown, and artifact cleanup over 20 jobs.

The plot-height follow-up excludes the loading spinner's empty wrapper from flex
growth and lets the graph fill the height left by its controls. Automated checks
pass; this specific rendering correction has not been independently observed in
a connected browser during closeout.

### Representative charge-table run

The user's `fixed_charge_run` run reported about 345 seconds in the GUI. Its
`run_2026_10_02_13_46_06.timing.log` records **343.459 seconds inside the engine**,
including 343.414 seconds for streaming analysis of 3,202 frames from a
14,518,895,842-byte `fort.7`. The difference from the user's reported total is
only an approximate GUI/startup overhead estimate. Nested pipeline timings
overlap and must not be summed.

The GUI uses the core analysis cache at `<workspace>/.reaxkit_cache/analysis`.
The completed result was verified on disk as
`1ae54fb234ff4a4d6b94552dc1bf9939483823242847ecf9d962c7f4415caa4f.pkl`
(536,019 bytes). An unchanged request and source identity can reuse it; a warm
repeat was not timed here. Input caching being enabled does not establish a
cache hit: this run used the `generic_text` streaming reader. Selecting four
output atoms still requires reading the requested source frames. Further
optimization of that engine path is separate from the completed GUI work.

### Scope and measurement limits

The earlier milestone sections below are historical measurements. Their open
browser acceptance gates are superseded by the user's local acceptance and
decision to limit further testing. Cross-browser/high-DPI/accessibility matrices,
browser p95/FPS/peak-memory measurements, alternate Python versions, and a
30-minute interactive soak were not newly performed or claimed to pass. They
are no longer blockers for this closeout. Full trajectory playback, nearby-frame
prefetch, and a dedicated GPU renderer remain conditional future enhancements,
not unfinished requirements for the accepted GUI. No 10x overall GUI speedup or
30–60 FPS claim is made.

## Historical implementation measurements

Measured on 2026-09-24, Windows 11, Python 3.12.10, 22 logical CPUs, approximately 32 GiB RAM. Full environment and individual timing samples are in the adjacent JSON files. The fixture contains deterministic `step`, `value`, and `atom_id` columns.

## Measured data paths

Five repetitions per case. Values below are medians, except where explicitly identified as the first sample.

| One million rows | Original | Updated |
| --- | ---: | ---: |
| Result normalization (updated path also writes Parquet) | 545.57 ms | 128.53 ms |
| Pipeline metadata snapshot | 2749.49 ms | 0.025 ms |
| First table-page call, first sample | 235.12 ms | 36.22 ms |
| Table-page call, median | 0.65 ms | 1.44 ms |
| Whole-result sort and page | Not measured | 43.45 ms |
| Process RSS at end of benchmark | 651 MB | 204 MB |

The old warm table timing reused an in-memory row cache and presented a capped subset with native filtering. The new table path reads a bounded page and supports queries over the complete result. These warm-page figures therefore describe different capabilities; the new path does not claim a speedup for an already cached small table. RSS is an end-of-run measurement, **not peak memory**.

The metadata bottleneck is eliminated, and normalization is about 4.2x faster in this fixture. These measurements do **not** establish a 10x end-to-end GUI speedup, a faster scientific calculation, or browser interaction latency. Do not add independent speedup ratios together.

| Updated ten-million-row fixture | Median |
| --- | ---: |
| Normalize and persist result | 2105.85 ms |
| Metadata snapshot | 0.011 ms |
| Unsorted page | 3.22 ms |
| Global sort and page | 524.98 ms |
| Page response size | 9,871 bytes |
| Process RSS at end | 419 MB |

These timings exclude process startup and browser rendering. Cold filtered/sorted GUI queries use the query worker lane; cached queries and unfiltered pages avoid new worker startup. The ten-million-row original path was not benchmarked. Cases use a narrow numeric table; wide, string-heavy tables and real trajectories need separate measurements.

## Reproduce

Use the project's configured Python interpreter:

```text
python benchmarks/webui_pipeline.py --rows 1000000 --repeats 5 --output benchmarks/local_report.json
python benchmarks/webui_smoke_server.py
python -m pytest tests/webui tests/engine/test_quick_frame_counts.py -q
```

The smoke server uses port 8067 and creates a synthetic million-row result. Open `http://127.0.0.1:8067/?rk_profile=1`, select its table/plot, page, filter, sort, and execute a presentation while navigating. The profiling flag adds a local download button for a bounded browser report containing callback resource timings, long tasks where supported, and pointer-to-paint samples. No telemetry is uploaded. Inspect server stage measurements through `app.reaxkit_service.metrics.snapshot()` in a development harness.

## Implemented behavior

- Results use immutable Parquet references with schemas, counts, and bounded previews. Numeric arrays retain shape metadata; mixed columns use a JSON encoding when Arrow cannot represent them directly. Mixed columns sort/filter as text in table queries, while raw table exports retain decoded values.
- Tables query all rows before paging. Pages default to 100 rows, cap at 500 rows and 2 MiB, and expose column selection. The API returns stable physical row IDs separately from user columns.
- DataFrame-to-record expansion and recursive full-artifact copying are removed from the normal publication path. Existing utilities still materialize their full inputs inside the isolated worker; they never calculate from a preview.
- The UI has a shared 64 MiB representation-cache budget and a separate 16 MiB page-cache budget. Superseded unreferenced artifacts are pruned; jobs protect files they are still using.
- Computation and expensive table queries have separate worker lanes. Native numerical-library threads are limited to one per worker. The queue is bounded, duplicate active submissions are reused, and cancelled/failed jobs preserve the previous result.
- Progress uses existing handler/executor reporter callbacks when available, throttled to four updates per second, plus persistence/query/export stages. Pipeline mutations are committed by the owning service after checking transitive input revisions.
- Workers are monitored against a default 4096 MiB per-process RSS budget. Set `REAXKIT_UI_JOB_MEMORY_MB` before launch to adjust it. This is a sampled safeguard, not a hard OS memory limit; child processes and very brief allocation spikes can exceed it.
- Dataset metadata probing never falls back to loading a whole trajectory. Unsupported counts remain unknown. Re-loading even the same input configuration advances its revision and invalidates older jobs.
- Browser reload reconnects to the same in-memory session while the server is alive. Idle sessions are eligible for expiry after one hour when cleanup runs; active jobs protect their session. Server restart requires importing a saved snapshot.
- New snapshots use format version 2 with portable Parquet sidecars. Keep the JSON file and its `.tables` directory together. Legacy inline snapshots remain readable. Table exports stream complete data; CSV/Excel publication occurs only after successful worker completion.

## Acceptance status and remaining validation

Automated tests cover table correctness, mixed/null values, legacy import, portable export, real Windows worker spawn, duplicate/cancel/failure behavior, stale inputs, query availability during compute, artifact cleanup over 20 jobs, queue bounds, and a bounded Dash table callback round trip. The source also includes a repeatable GUI fixture and browser profiling instrumentation.

At the phase-3 milestone, **33 tests passed** across the GUI suite and existing quick-frame-count tests. A built wheel was installed into an isolated target and its imports, Dash layout, callback definitions, and profiling asset were smoke-tested. Dash emits its existing DataTable deprecation warning.

No connected browser was available during this session, and no representative user dataset path was supplied. Consequently visual interaction, browser p95 latency, real large-system loading, and a 30-minute interactive soak remain unverified. Python GUI data-path measurements and callback tests are the available evidence. Reference-machine browser validation remains a release gate.

Phase 4 now replaces prefix previews with complete-result summaries, described below. Panel resizing and visual redesign are implemented in phases 5–6 below, with browser acceptance still open. Dash DataTable is retained for this rollout and Dash is bounded below version 5; its deprecation should be addressed during later presentation work.

## Phase 4: scalable plots and one-frame 3D prototype

Recorded on 2026-09-24 on the same workstation. Raw timings, versions, fixture details, payload sizes, and sampled process RSS are in [webui-plots-phase4.json](webui-plots-phase4.json). Each timing has five repetitions. These are CPU/data-path measurements, not click-to-paint or GPU timings. First and warm OS-cache samples are retained separately in the JSON; the table reports medians.

| Fixture | Complete overview summary | Narrow-range summary | Histogram | Figure JSON |
| --- | ---: | ---: | ---: | ---: |
| 100k result rows | 38.9 ms | 15.7 ms | 18.5 ms | 181 KB |
| 1M result rows | 359.5 ms | 34.8 ms | 112.1 ms | 326 KB |
| 10M result rows | 4656.4 ms | 33.6 ms | 891.5 ms | 345 KB |
| 10k 3D points | 73.9 ms | — | — | 953 KB |
| 100k 3D points | 159.6 ms | — | — | 1.96 MB |
| 1M 3D points | 579.7 ms | — | — | 2.02 MB |

The range fixture selects 101 observations near the end of the result, including a narrow spike. Native numeric predicates allow Parquet row-group pruning without renumbering source rows. Full-range scans still scale with data size; the 10M-row overview takes several seconds in a cancellable worker. The 3D tiers show all 10k points or at most 20k sampled points, retaining spatial extrema. These reductions change display detail, never stored results or analysis inputs.

Ready figures are cached; Plotly figure construction and encoding also run in the worker. At 1M rows/points, uncached job completion including Windows spawn measured **1735.5 ms for 2D** and **2552.4 ms for 3D**. Cached submission plus Python JSON serialization measured **26.5 ms** and **125.4 ms**, respectively. These measurements exclude HTTP transfer, browser parsing, rendering and interaction; five repetitions do not establish browser p95 targets. Per-figure construction timings remain in the raw report to distinguish rendering preparation from query time.

The largest sampled parent-process RSS in the summary benchmarks was approximately 620 MiB. This includes fixture arrays, native-library allocations and allocator state from previous cases, sampled every 20 ms; it is not incremental artifact memory, an OS-enforced peak, browser memory, or a long-run capacity guarantee. The initial implementation audit is retained in [webui-plots-phase4-initial.json](webui-plots-phase4-initial.json). It motivated predicate pushdown and moving figure construction off callbacks; it is not the pre-improvement GUI baseline.

Implemented safeguards and semantics:

- Complete-source filters and exact group aggregation precede line reduction. Numeric x values are required. Dense range buckets keep first/last/min/max observations. Buckets containing missing y values are conservatively disconnected, with markers exposing isolated extrema; no false continuity is drawn across missing observations.
- Total line points are capped across groups and zoom. Excess groups are explicitly reported. Histograms send server-side bins with exact counts, including the final right edge; missing/nonfinite values are counted separately.
- 180 ms client debounce coalesces viewport events. Obsolete jobs are cancelled per browser view, and a request token rejects late responses. Camera/zoom/legend/selection state is preserved during updates; refresh retries expired/cancelled views. Clicking a retained point reads its exact original source row; aggregated values are identified as aggregates.
- Ready plot/frame results share the existing 16 MiB query cache, keyed by artifact revision, renderer version, request, range, resolution and frame. Geometry and payload budgets apply before publication. The default 20k 3D vertex budget includes cell/bond overlays; the final figure cap is 4 MiB. Numeric arrays are packed by Plotly where supported.
- The molecular prototype uses one frame of an existing coordinate table, explicit cell vectors/origin, supplied atom-ID bond pairs with visible endpoints, and stable physical row/atom IDs. It does not infer a cell or bonds. Full playback and prefetch remain gated on handler frame access and real browser measurements. A dedicated GPU component will only be chosen if measured Plotly interaction misses the target.
- Plot image export uses the displayed summary and current axes/camera. Complete scientific output remains available through table export. Presentation mappings, labels and styles remain supported.

Reproduce with the configured interpreter and installed Node:

```text
python benchmarks/webui_plots.py
python -m pytest tests/webui tests/engine/test_quick_frame_counts.py -q
node --test tests/webui/plot_queries_client.test.cjs
python benchmarks/webui_smoke_server.py
```

The smoke server now includes a million-row line plot, an exact histogram, and three 100k-atom frames with a periodic cell and supplied bond pairs. Open it on port 8067 with `?rk_profile=1`; exercise zoom/reset, legend toggles, point clicks, frame changes, rapid navigation/cancellation and refresh. Frame values are 0, 1 and 2. Browser validation should include camera movement, 30–60 FPS targets, long tasks, memory, and comparison with a representative simulation. No enabled browser was available in this session, so those acceptance gates remain open.

Phase-4 automated coverage includes extrema and gap preservation, full-source filters/aggregates, exact histogram counts, empty ranges, shared budgets, byte limits, reserved-column collisions, stable point IDs, cell/bond geometry, frame isolation, cache expiry, cancellation across browser views, displayed-figure export, and Dash callback publication. Pure JavaScript tests cover debounce, log ranges, stale-response rejection, state preservation and retry; they are not a substitute for browser/GPU testing. The GUI now requires `dash>=3.1,<5` for optional component state.

Final verification: **58 Python tests and 5 client-logic tests passed**. The rebuilt wheel was installed into an isolated target; imports were asserted to come from that target, and Dash routes, both browser assets, and a real Windows-spawn plot job passed the installed-package smoke check. `benchmarks/webui_installed_smoke.py` preserves this check for future releases.

## Phases 5–6: adjustable workspace and visual design

Implemented on 2026-09-24. The shell now has local pointer-capture splitters,
animation-frame coalescing, release-only persistence, keyboard resizing, four
presets, collapse/restore, canvas maximize, and reset. Geometry clamps at smaller
sizes and uses a sidebar overlay on narrow viewports. A throttled ResizeObserver
resizes the graph during dragging and settles it immediately on release.

The [standalone design preview](gui_workspace_preview.html) provides loaded,
empty, running, failed, and huge-result mockups in dark and light themes. The app
uses packaged CSS tokens, vector icon masks, consistent controls, searchable
hierarchy, and the Activity/Results/Logs drawer. Analysis forms keep browser drafts
through status updates and provide validated Apply/Discard/Run actions. Integer
selector expansion is capped before allocation. Activity displays at most 12 jobs,
results at most 30 cards, and each log read at most 256 KiB / 400 lines. Hidden
drawer polling is disabled. Plots can follow the app theme without changing stored
scientific data, explicit colors, camera, or ranges.

Verification includes 16 workspace Python tests in addition to the existing 58
GUI/quick-frame tests, and **11 passing JavaScript tests** (five plot tests and six
workspace tests). Coverage includes real callback registration/assets, unique IDs,
accessible separator metadata, invalid and excessive selectors, atomic draft
save-before-submit, Discard, stable editors, join-source refresh, search ancestors,
bounded logs/activity, version validation, drag cancellation, keyboard controls,
maximize restoration, theme isolation, and suppressed polling. A DOM stub checks
that 159 pointer movements coalesce into one scheduled animation frame and cause
zero server property updates before release. This is a logic test, not a measured
frame rate. Geometry tests cover 1366x768 and 1920x1080 at 100/125/150/200% scaling;
they do not establish actual browser rendering or accessibility acceptance.

Reproduce the checks with the configured environment:

```text
python -m pytest tests/webui tests/engine/test_quick_frame_counts.py -q
node --test tests/webui/workspace_client.test.cjs tests/webui/plot_queries_client.test.cjs
python benchmarks/webui_smoke_server.py
```

Browser acceptance remains open: review both themes and dropdowns, drag all three
boundaries during active jobs, verify keyboard focus and drafts, switch presets,
maximize/restore, reload, cancel/retry, and inspect results/logs. Record 30–60 FPS,
p95 input latency, memory, high-DPI behavior, and a 30-minute soak on a representative
dataset. No enabled browser was available in this session. Phase 7 should start
with this review; no new overall speed or browser smoothness claim is inferred
from these tests.

Final phase-5/6 verification: **74 Python tests and 11 JavaScript tests passed**.
The rebuilt wheel passed an isolated-install smoke check for imports, Dash routes,
workspace CSS/JS/icon assets, and a real Windows-spawn plot worker. The existing
Dash DataTable deprecation warning remains. Git whitespace checks passed.
