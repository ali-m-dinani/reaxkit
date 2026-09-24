# ReaxFF h-BN polarization performance plan

## Diagnosis from the current code

The formal-charge command requests `TrajectoryData`, so it reads `xmolout` but
does not need `fort.7`. The ReaxFF-charge command requests `ElectrostaticsData`.
Its ReaxFF adapter streams selected `xmolout` frames and calls
`quick_io/charges.py::iter_fort7_charge_frames`, which delegates to
`Fort7Handler.stream_file_frames(charge_arrays_only=True,
include_atom_types=False)`. It extracts IDs and partial charges without
building the connectivity table. It does **not** load the entire `fort.7` file
for every frame.

On a cold, empty frame cache with sorted selected indices, `Fort7Handler`
performs one forward scan through `fort.7`, skips parsing unselected atom rows,
parses each selected frame once, and persists charge records in 16-frame SQLite
batches. With an existing or partial cache it can use stored byte offsets and
seek to missing frames. `xmolout` has a similar selected-frame cache. This
means the central byte-offset/`seek` optimization in the [OVITO example](https://www.ovito.org/manual/python/introduction/examples/file_readers/example_file_format_reader_optimization.html)
is already implemented for the indexed path. Applying it again to the cold
one-pass path would probably add an index scan and slow that first run. The
other OVITO optimization, bulk property transfer, is specific to the OVITO
Python/C++ boundary; ReaxKit already constructs NumPy charge arrays and does
not assign an OVITO property atom by atom.

The two 88% progress indicators are coupled by `adapter_parts/streaming.py`:
it aligns an `xmolout` record and a `fort.7` record, then yields the pair to a
bounded frame pipeline. A slow parser, cache, polarization kernel, or pipeline
backpressure can hold *both* bars at nearly the same frame. The displayed
seconds per frame therefore do not measure either loader in isolation. The
formal command's `/usr/bin/time` report (1:10:44 elapsed, low CPU use, high
filesystem input count) suggests that I/O may already be substantial without
`fort.7`. The live ReaxFF run's own timing output is needed to attribute its
additional hours. Existing local timing logs for a different run show a
substantial `Fort7ChargeOnlyReader` cost, but do not establish the bottleneck
on this SLURM node.

## Proposed work

1. **Measure the current job accurately.** Capture ReaxKit's `--timing` phases
   (`load_handler`, `load_total`, `analyze`, `pipeline_reader`,
   `pipeline_workers`, `pipeline_collector_wait`, `pipeline_total`) and the
   source paths. Record source file sizes, frame and atom counts, cache path,
   cache hit/miss counts, bytes read from source, index bytes, and cache bytes.
   Add narrowly scoped wall-time counters for `xmolout` source reads, `fort.7`
   source reads/charge parsing, SQLite frame reads/writes, and the per-frame
   polarization kernel. Report each separately; pipeline reader time can
   include coordination effects and must not be treated as pure disk time.
2. **Compare equivalent short slices.** On the same node and storage, run
   formal and ReaxFF charge modes over representative early and late slices
   with the same frames, replication, volume method, workers, and output
   profile. Measure cold cache, warm cache, and `--no-input-cache` separately.
   Put temporary caches on node-local scratch when available. Keep input
   files unchanged. Include a few thousand frames so indexing, cache writes,
   and steady-state parsing can be distinguished. Confirm the `fort.7` header
   count and that requested indices align with `xmolout`.
3. **Optimize the dominant stage found in step 1.** If cold cache writes or
   shared-filesystem SQLite traffic dominate, avoid persisting every charge
   and coordinate frame during a one-pass analysis, or make that persistence
   explicitly optional; retain byte-offset indexing for random access and
   reusable warm caches. If charge parsing dominates, profile
   `_iter_selected_charge_frames_sequential` and replace costly per-row string
   operations with a measured, format-safe parser that fills preallocated
   NumPy arrays. If warm-cache access dominates, improve batching and cache
   placement. If the polarization kernel dominates, profile its per-frame
   hull-volume calculation and reference/displacement work before changing
   mathematics or volume semantics. If `xmolout` dominates, optimize its
   selected-frame path too. Do not assume the OVITO `seek` change alone will
   help this sequential workload.
4. **Validate and benchmark.** Add a regression fixture covering sparse frame
   selection, fused fixed-width atom fields, changing atom counts, and
   `xmolout`/`fort.7` alignment. Compare charges and polarization outputs
   against the current parser within existing numerical tolerances. Benchmark
   cold/warm/no-cache runs with elapsed time, CPU time, filesystem input and
   output, peak RSS, and per-stage timing. Accept an optimization only if it
   improves the measured dominant case without harming correctness or common
   random-frame access.

## Remaining to do

- Obtain the current ReaxFF job's ReaxKit timing log and completed
  `/usr/bin/time -v` report; the progress excerpt alone cannot identify the
  bottleneck.
- Implement the measurement in step 1, run the comparisons in step 2, then
  choose and implement the optimization indicated by those measurements.
- Run the regression and cluster-scale benchmarks in step 4 and record the
  before/after results here.
