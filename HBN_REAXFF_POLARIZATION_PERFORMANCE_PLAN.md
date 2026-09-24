# ReaxFF h-BN polarization: production performance plan

## Review conclusion and required outcome

The previous plan correctly located the slow reader path, but a generic cache
bypass plus worker tuning was not sufficient to substantiate a 10x objective.
This revision makes the repeated cache inventory the first targeted fix,
requires the same efficient parser with and without caching, and defines a
bulk binary reader as the escalation if cache repairs do not meet the target.
Small parser cleanups or a faster calculation alone do not close this task.

The primary acceptance target is **at least 10x lower elapsed time** for the
same ReaxFF-charge command and all 13,334 frames: **at most 2,479.8 s
(41 min 19.8 s)** versus 24,798 s. The stretch target is 20x, or 1,239.9 s.
These are engineering acceptance targets, not measured or guaranteed speedups.
Use the same scientific settings, output profile, allocation, and input data.
The cold run must include all required indexing, conversion, cache creation,
staging, and publication; a warm-cache speedup is a separate result.

An initial engineering budget is 2,150 s for the reader, 100 s for the
polarization kernel, and 150 s for everything else (2,400 s total). This
requires roughly an 11.4x improvement in reader time. Even eliminating the
90 s kernel entirely saves only 0.36% of the present command's wall time.
Four ideal calculation workers save only about 68 s while input is unchanged.
Optimizing just `fort.7` is insufficient if the 4,046 s formal coordinate
reader remains: that time alone exceeds the entire 10x budget. Both source
paths and their shared cache machinery are in scope.

## Production evidence and diagnosis

The completed SLURM job 55785915 analyzed 13,334 selected frames (`0:40001:3`)
with 4 CPUs and 64 GiB allocated. The two `/usr/bin/time -v` reports cover the
analysis commands, not the preceding `replace_geometry_name.py` pass.

| Measure | Formal charges | ReaxFF charges |
| --- | ---: | ---: |
| Wall time | 1:10:44 | 6:53:18 |
| User CPU time | 486 s | 752 s |
| System CPU time | 517 s | 3,876 s |
| CPU utilization | 23% | 18% |
| Filesystem inputs, as reported by GNU time | 1,574,855,624 | 12,112,838,632 |
| Filesystem outputs, as reported by GNU time | 8,112,968 | 15,207,504 |
| Peak RSS | 10.63 GB | 10.65 GB |

The run-specific ReaxKit timing logs
(`human_readable_timing_2026_09_24_02_28_39.log` and
`human_readable_timing_2026_09_24_03_39_40.log`) add the decisive attribution:

| Pipeline phase | Formal charges | ReaxFF charges |
| --- | ---: | ---: |
| Reader (time spent advancing the source iterator) | 4,046 s | 24,586 s |
| Polarization workers (cumulative) | 95 s | 90 s |
| Pipeline total | 4,224 s | 24,760 s |
| Reference preparation | 0.09 s | 0.13 s |
| Selected frames completed | 13,334 | 13,334 |
| Execution policy | 1 serial worker | 1 serial worker |

ReaxFF takes 5.84 times as long. The additional ~5 h 43 min of wall time is
much larger than the additional ~1 h of combined user and system CPU time.
System time is 7.5 times the formal run's value, and reported filesystem
inputs are 7.7 times higher. These are strong signs of I/O amplification and
waiting, though GNU time's block counters do not identify which files or
operations caused it. The ReaxFF reader consumed 99.3% of pipeline wall time;
the additional reader time over formal charges is 20,540 s (5 h 42 min 20 s).
The numerical polarization kernel is not the cause of the large gap. Both
commands completed successfully. The 18 MB `.err` file consists mostly of
progress redraws; the stage timings came from the separate `logs/timing/`
files. The logged `source_bytes` values (57.1 GB formal, 328.9 GB ReaxFF)
identify logical source sizes, not actual disk traffic or cache traffic.

The formal request loads `xmolout` as `TrajectoryData`; the ReaxFF request
loads aligned `xmolout` and `fort.7` as `ElectrostaticsData`. The ReaxFF path is
`adapter_parts/streaming.py` -> `quick_io/charges.py` ->
`Fort7Handler.stream_file_frames(charge_arrays_only=True,
include_atom_types=False)`. The charge reader does not read all of `fort.7`
for each selected frame. For a cold, sorted selection it makes one forward
pass, skips decoding unselected atom rows, and constructs charge arrays for
selected frames. An existing or partial cache can use byte-offset indexing and
`seek`. The `xmolout` and `fort.7` progress bars advance together because the
reader pairs records and the bounded calculation pipeline applies
backpressure; their seconds/frame figures cannot attribute time to either
file. The [OVITO reader example](https://www.ovito.org/manual/python/introduction/examples/file_readers/example_file_format_reader_optimization.html)
already matches ReaxKit's indexed `seek` path; building a second index before
this forward scan is unlikely to accelerate the cold full-range job.

The current code exposes a more promising, testable mechanism within the slow
reader path. Both cold one-pass readers persist parsed selected frames in
16-frame SQLite batches.
`FrameStore.put_frames()` calls `enforce_frame_cache_limit()` after every
batch; that calls `inspect_frame_cache()`, which opens every workspace frame
store and queries its frame count and latest access time. Around 834 batches
per input make this a repeated, growing cache inventory on shared storage.
The default cache limit is 10 GiB, and eviction may also affect the active
generation. This can amplify filesystem reads and metadata traffic. It is a
**high-priority hypothesis**, not a proven explanation of all 6:53:18.

A temporary local SQLite experiment on 2026-09-24 verified this mechanism with
the current `frames` column order, 1,024 rows, and a 128 KiB payload per row.
For `SELECT COUNT(*), COALESCE(MAX(last_accessed_at), '') FROM frames`,
`EXPLAIN QUERY PLAN` reported `SCAN frames`; median query time over five runs
was 0.07139 s. Adding an index on `last_accessed_at` changed the plan to
`SCAN frames USING COVERING INDEX` and the median to 0.0000865 s (about 825x
faster). Both returned identical results. This establishes an expensive query
design, not its fraction of the SLURM runtime. OS caching was uncontrolled;
these local query timings must not be reported as production speedup.
For cache repairs alone to deliver 10x, they must remove roughly 90% of total
elapsed time (about 91% of current reader time). The existing logs cannot yet
establish that fraction; the reader budget and escalation path are essential.
Running a growing table scan after every small insert batch creates
approximately quadratic cumulative inventory work as selected-frame count
grows. A covering index reduces that cost but still scans a growing index;
the final hot path should avoid the inventory query entirely.

The preprocessing output reported 274.4 MiB/s for `xmolout` and 289.4 MiB/s for
`fort.7`. That is evidence that bulk access to this storage can be much faster
than the analysis reader under some conditions, making a large improvement
plausible. It is not a sustained-bandwidth guarantee for the next allocation.
The full 57.1/328.9 GB source sizes also cannot estimate this command's read
throughput: a finite selection stops near source frame 40,000, possibly well
before EOF. Measure the bytes through that boundary before estimating the
I/O floor or claiming a number of whole-file rereads.

There is also memory and scheduling work, but the logs make them secondary for
wall-time improvement. `stream_polarization()` appends every per-frame result
to a Python list before `combine_results()`, even though the
public output is primarily a compact per-frame table and the atom mapping is
the same prepared reference across frames. The ~10.6 GB RSS in both runs is
consistent with large retained result objects, but the attached reports do not
measure their composition. `HBNReferencePolarizationTask` declares a safe
reference-frame map yet sets `automatic_parallel=False`; the four allocated
CPUs provided only one active frame worker in these runs. Adding worker threads
cannot remove the measured ~6 h 50 min reader cost and should not lead this
optimization.

## Implementation order

### 1. Isolate the cost *within* the measured reader bottleneck

The run-specific ReaxKit timing logs establish that reader advancement
dominates. They do not split it into `xmolout` source reads, `fort.7` source
reads, parsing, SQLite payload writes, cache-limit maintenance, and alignment.
Add aggregated, low-frequency counters
for source bytes and time (`xmolout`, `fort.7`), charge parsing, frame-cache
read/write time, cache-limit inventory/eviction time, hits/misses, and result
collection. Report actual reads separately from logical source file size.
Instrument each operation itself: the existing `pipeline_reader` phase is a
wall-time envelope, not proof that all of it is disk I/O. In this serial run
there was no concurrent worker backpressure, but source and cache costs remain
unseparated. Throttle progress updates so they do not produce millions of
redraws. Time active operations, not time spent suspended at a generator's
`yield`. Record source bytes through the last requested frame, actual source
read calls/bytes, cache payload bytes, inventory calls/time, SQLite commits,
cache hits/misses, source passes, and the reader branch used. The current
one-pass `source_bytes=stat().st_size` counters need correction or relabeling.

**Correct the cache A/B test first.** Currently disabling caching sends
`Fort7Handler.stream_file_frames()` to the general text loop, which decodes
skipped lines and checks each line for a header. The cold cached charge path
uses `_iter_selected_charge_frames_sequential()`, a different binary parser.
Thus an unmodified `--no-input-cache` run is useful diagnostically but does
not isolate cache overhead. Route finite ascending charge selections through
the same efficient sequential parser regardless of cache mode; wrap optional
cache operations around it. Then compare cache off, cache on with inventory
deferred, and the existing behavior on identical selections. Do not alter
charge values, atom-type requirements, header validation, or frame ordering.

Use a short bounded investigation: selected counts such as 512, 1,024 and
2,048 at the same stride distinguish growing cache work from linear reading.
Also test one representative middle/late slice. Drain the actual paired
reader and record checksums as well as running the scientific command. Move
to the full selection once the component timings support the 2,400 s budget;
do not spend repeated seven-hour runs on variants already outside the budget.

### 2. Eliminate repeated cache scans and small transaction overhead

Primary implementation sites: `core/storage/frame_store.py`,
`engine/reaxff/io/fort7_handler.py`, and `engine/reaxff/io/xmolout_handler.py`.

- Separate detailed cache inspection from quota enforcement. The reader's
  write loop must not execute `COUNT/MAX` over cached frame payload tables or
  recursively enumerate all databases per 16 frames. Maintain small store
  summaries transactionally and use file sizes for budget reconciliation.
  Batch/aggregate last-access updates rather than adding a write to every
  cache read. Retain detailed inspection as an explicit diagnostic operation.
- Support existing caches safely. A metadata covering index can accelerate
  legacy inventory while migrating to summaries; count any one-time index
  build/reconciliation in startup time. Do not silently discard valid caches.
- Bound quota reconciliation by byte growth/time and run closure, with
  hysteresis. Protect all active generations across processes, not only the
  current writer's own eviction call. If pinned data exhausts the budget,
  stop adding optional payloads and continue streaming; do not delete active
  stores or let protected writes grow without bound.
- Batch payload writes by bounded bytes as well as frame count, and amortize
  connection/transaction setup where measured useful. Do not share one SQLite
  connection unsafely between worker threads. Do not trade cache integrity
  for speed by disabling durability checks indiscriminately.
- Select direct streaming when populating an empty payload cache has no
  demonstrated payoff for the current large forward pass. Reuse a compatible
  warm cache when beneficial. Keep explicit cache controls and an intentional
  population mode. Fix the shared cache first; do not rely solely on turning
  caching off or changing defaults to hide its scaling defect.

Pass gate: inventory work does not grow quadratically with selected frames;
cache-enabled and direct modes use the same parser; component measurements
are inside the reader budget. If not, proceed to the bulk reader below.

### 3. Build a bulk binary source reader if the budget is still missed

Keep one forward source owner feeding the existing bounded frame pipeline.
Read multi-MiB binary blocks (benchmark a bounded 4-32 MiB range), carry partial
lines across block boundaries, and locate frame boundaries with native byte
search/count operations. Skip unselected payloads without Unicode decoding,
tokenization, float conversion, or Python callbacks per atom. Parse only the
selected coordinate fields and charge/ID fields into contiguous NumPy arrays;
do not construct connectivity tables or convert bond orders. Benchmark this
against the already-buffered `readline` implementation: increasing the file
buffer alone is not an assumed 10x fix.

For `xmolout`, use the validated atom count and line/record boundaries; for
`fort.7`, validate header candidates and preserve its variable-width/fused
integer-field handling and optional trailing fields. Handle CRLF, blank lines,
truncated records, and variable atom counts explicitly. Do not compute frame
offsets by multiplying a sampled row width. Use a tested generic fallback
when the input is outside the fast format contract, and log its reason.

Record byte offsets opportunistically during the same source pass, in bounded
metadata batches. The current cold one-pass path stores payloads without
building its offset coverage; a later partial-cache selection can therefore
need another scan. Separate the offset index from payload representations so
coordinate-only and richer views share validated boundaries. Never do a full
index pass followed by a second full parse pass for this cold ascending job.
Stop at the requested boundary, with only bounded lookahead.

Use a valid existing index to read just selected frame ranges, coalescing
nearby requests where useful on shared storage. For repeated commands, an
optional compact chunked array cache can store coordinates, charges, IDs,
cells, iterations and source indices, with a manifest and atomic publication.
Build it only once while streaming, without a second text traversal or one
file per frame. Its construction cost belongs to the cold-run measurement;
report later reuse separately. This is a stronger warm-run route than
repeatedly interpreting wide connectivity text. Prototype this only if the
repaired existing cache cannot meet the measured read/serialization budget.

If selected-row conversion still exceeds budget after I/O repair, benchmark
native/vectorized conversion or a compiled reader against the same reference
parser. Require measured benefit before adding a compiled dependency. Parser
parallelism, if needed, must process bounded already-read blocks through the
shared execution machinery, with one source reader and ordered collection;
workers must not each reopen or rescan the trajectory. Keep calculation
thread tuning separate from parser throughput.

### 4. Reduce per-frame result retention after reader speed improves

Use the shared reducer/artifact contracts from
[SELECTED_FRAME_PIPELINE_PLAN.md](SELECTED_FRAME_PIPELINE_PLAN.md) and
[UNIFIED_EXECUTION_AND_OUTPUT_PLAN.md](UNIFIED_EXECUTION_AND_OUTPUT_PLAN.md).
Append compact polarization rows and source-frame metadata incrementally;
retain the prepared mapping/reference once. Keep optional displacement rows
on the existing lazy `TableSpool`/artifact path. Do not accumulate 13,334
mapping DataFrames or full per-frame result objects. Keep deterministic frame
order, output names, provenance, and the existing numerical result.

This change targets RSS and the ~84 seconds outside measured reader and
worker phases, so it should not be presented as the primary six-hour fix.
After the direct reader and compact collector are measured, benchmark the
reference-frame map with 1, 2, and 4 workers and a bounded queue under the
same SLURM allocation. Enable automatic frame workers for this command only
if the kernel has a measured threading benefit and passes numerical and
thread-safety checks. Keep BLAS/OpenMP limits scoped to worker execution.
Reader/worker overlap can hide at most a small fraction of this job's time
while the kernel remains near 90 seconds; threads cannot repair
shared-filesystem cache thrashing. Do not change the global scheduling policy
based on this single job.

## Validation and speed target

Use a controlled matrix on one node with identical scientific flags and input
files: early, middle, and late slices; a representative multi-thousand-frame
stride; and the full 13,334-frame selection. Compare current cache behavior,
`--no-input-cache`, redesigned direct streaming, warm cache, and a cache on
node-local scratch when available. Run each in a fresh workspace or explicitly
record cache state. The original commands ran sequentially, so the second
could inherit `xmolout` cache state from the first; do not treat them as a
controlled cold-cache comparison. Record wall/user/system time, filesystem
blocks, source and cache bytes, per-stage times, peak RSS, queue occupancy,
frame throughput, and output size. Include the preprocessing pass separately
when reporting whole-job time.

Acceptance requires the <=2,479.8 s full ReaxFF command target above, unchanged
scientific results, lower cache I/O/system overhead, bounded input retention,
and separately reported sparse/warm-cache behavior. A 2x or 5x result is an
intermediate improvement, not completion of the requested 10x goal. The formal
run is also slow and is not a lower bound on the optimized ReaxFF run.

Measure bulk read bandwidth on the actual required source prefixes. If that
measured I/O floor alone exceeds the target, document it and evaluate local
scratch or an already available compact cache, including copy/conversion cost
and available disk space. Do not automatically stage hundreds of GB or claim
10x by moving preparation outside the timed command. Separate analysis-command
speed from total SLURM-script speed: the preprocessing and formal analysis
must also be accounted for before claiming 10x for the whole script. Archive
exact code revision, settings, cache state, logs, outputs and per-phase times.
Repeat the final comparison sufficiently to identify shared-storage variance;
publish the measured speedup and its range, not an extrapolated microbenchmark.

Regression tests should cover cold direct and warm/indexed paths, sparse and
strided selections, `xmolout`/`fort.7` frame alignment, corrupted/missing
cache entries, interrupted cache writes, fixed-width fused fields, changing
atom counts, and ordered output with 1/2/4 workers. Compare charge arrays,
polarization tables, mapping, optional displacements, and artifacts against
the existing implementation within established numerical tolerances.
Add structural I/O tests proving one forward traversal, no unselected numeric
parsing, no hot-path payload-table inventory scans, safe cache quotas, and
identical parser output across cache modes. Exercise binary block boundaries
and a stopped/restarted partial index if that reader is implemented.

## Implementation checkpoint (2026-09-24)

Implemented the cache repair, unified sequential charge parser, opportunistic
offset recording, buffered source-read counters, per-source `input_stream`
timing records, throttled batch progress, and chunked polarization result
collection. The cache now uses transactional summaries, amortized stream
connections, byte/time-based maintenance, cross-process generation leases,
and byte/frame-bounded write batches. Existing valid cache payloads survive
the additive metadata migration. Optional payload writes pause when pinned
data exhausts the reconciled budget; small offset metadata remains writable.
This is a soft workspace budget with bounded reconciliation intervals, not a
strict instantaneous disk cap across concurrent writers and SQLite WAL files.

The local 2,048-frame synthetic cache benchmark improved cold population from
34.094 s to 3.335 s (10.22x). At 1,024 frames it improved from 9.932 s to
0.908 s (10.93x). These are **cache-only samples**, not the full scientific
command. The original trajectory data and SLURM filesystem are not available
locally, so the production 10x acceptance remains open. The default cache and
worker policies are unchanged pending those measurements.

The implementation and reproducible comparison commands are documented in
[benchmarks/HBN_PERFORMANCE.md](benchmarks/HBN_PERFORMANCE.md). Use
[hbn_paired_reader.py](benchmarks/hbn_paired_reader.py) for bounded production
reader comparisons with checksums and
[hbn_polarization.slurm](benchmarks/hbn_polarization.slurm) for the original
full scientific command in a fresh workspace. The block parser, shared
cross-view offset format, compact array cache, and compiled conversion remain
conditional follow-ups if measured residual reader time misses the budget.

## Remaining to do

- [x] Inspect job 55785915's run-specific ReaxKit timing logs: both runs used
  one serial worker; the ReaxFF reader took 24,586 s versus 90 s of worker
  time. The logs do not break reader time down by source/cache operation.
- [x] Review the 10x requirement, set a <=41 min 19.8 s acceptance target, and
  verify the payload-table inventory scan with a temporary SQLite experiment.
- [x] Unify the cached/uncached sequential charge parser and add active reader,
  raw source-read, payload-cache, offset-write and maintenance counters.
- [x] Remove repeated payload-table inventory from the write loop, amortize
  transactions, and protect active stores from peer quota eviction. Document
  the reconciled quota and optional-write suspension behavior.
- [x] Record offsets during forward scans and limit pending payload batches
  by both frame count and estimated bytes.
- [x] Reduce duplicate result retention with a chunked shared collector;
  preserve optional detail spooling and the existing execution policy.
- [x] Pass targeted cache, reader, streaming, and collector regressions,
  including 48 h-BN parity cases covering both charge sources and 1/2/4 workers.
  The broad repository suite was stopped before completion; it is not a pass.
- [ ] Confirm scaling and the cache contribution on the original input with
  the paired-reader checksum benchmark; measure cold, direct, and warm paths.
- [ ] Compare the resulting reader with the 2,150 s engineering budget. If it
  misses, implement/benchmark the bulk binary reader and shared offset index;
  add compact chunk reuse or compiled conversion only as measurements require.
- [ ] Assess calculation workers only after reader throughput meets its budget.
- [ ] Pass correctness and I/O scaling gates and the full cold-run 10x SLURM
  acceptance test. Record warm-run results separately; do not mark the goal
  achieved based on a query benchmark or a marginal end-to-end improvement.
