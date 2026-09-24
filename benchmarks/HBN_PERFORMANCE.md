# h-BN polarization performance validation

The acceptance target for job 55785915 is **2,479.8 s or less** for the full
ReaxFF analysis command (24,798 s originally). A faster cache benchmark alone
does not establish that target. The supplied local files contain timing logs,
not the original `xmolout` and `fort.7` trajectories.

## Implemented changes

- Cache quota maintenance no longer scans payload tables after every 16 frames.
  Small transactional summaries replace growing `COUNT/MAX` inventories;
  reconciliation uses file sizes after byte growth, elapsed time, and closure.
- Reader sessions reuse a connection and pin their generation across processes.
  Reads touch a small summary at most every 30 seconds instead of rewriting
  large frame rows. Pending writes are bounded by 16 frames or approximately
  8 MiB, with a single oversized frame written on its own.
- Finite increasing charge selections use the same sequential parser with and
  without input caching. Both readers build offset coverage during that pass.
  Source reads use a 1 MiB binary buffer; skipped rows avoid Unicode decoding.
- Polarization collects primary tables in chunks and retains invariant mappings
  once. Optional detailed outputs keep their existing spool behavior.
- Redirected progress output updates at most every two seconds, with a
  ten-second heartbeat for indeterminate operations.

The default cache policy and scientific flags are unchanged. No automatic
worker increase, compiled dependency, source-file copy, or full-file index pass
has been added. A bulk block parser remains a measured escalation if these
changes do not meet the reader budget on the actual input.

## Local cache-only evidence

`hbn_cache_scaling.py` was run before and after the cache repair, on local
Windows storage with 5,120 atoms per synthetic payload and 16-frame batches.
Each case used a new private cache. Raw results are in
`benchmark_results/hbn_cache_before.json` and `hbn_cache_after.json`.

| Frames | Before cold writes (s) | After cold writes (s) | Ratio |
| --- | ---: | ---: | ---: |
| 512 | 1.567 | 0.503 | 3.11x |
| 1,024 | 9.932 | 0.908 | 10.93x |
| 2,048 | 34.094 | 3.335 | 10.22x |

This is one local sample per case, with uncontrolled OS caching. It measures
cache machinery, not trajectory parsing, scientific analysis, or SLURM storage.
The full-command 10x acceptance test is still outstanding.

Local regression validation passed the 197-test cache/streaming/rollout group,
the focused executor/progress/collector tests, and an expanded 48-case h-BN
matrix comparing formal/ReaxFF charges, 1/2/4 workers, global/local results,
and all output profiles against materialized calculations. Cold/direct/warm
paired-reader checksums match on fixtures. Cross-process eviction, interrupted
writes, legacy cache migration, cancelled-stream cleanup, and byte-bounded
batches have dedicated checks. `bash -n` accepts the SLURM script.
The full repository test run was stopped after approximately 15 minutes,
before completion; it is not counted as a passing full suite. ASE emits a
NumPy deprecation warning in the h-BN tests.

## Bounded reader comparison on the cluster

Use the Python interpreter from the environment containing this checkout.
Run from the repository root; substitute the input directory. Cache directory
names below must be unused for the cold run. No script deletes existing caches.

```bash
python benchmarks/hbn_paired_reader.py --input /path/to/run \
  --frames 0:6144:3 --cache-dir /path/to/bench-cache-2048 \
  --no-input-cache --output direct-2048.json

python benchmarks/hbn_paired_reader.py --input /path/to/run \
  --frames 0:6144:3 --cache-dir /path/to/bench-cache-2048 \
  --output cold-2048.json

python benchmarks/hbn_paired_reader.py --input /path/to/run \
  --frames 0:6144:3 --cache-dir /path/to/bench-cache-2048 \
  --output warm-2048.json
```

All three `checksum` fields must match. The benchmark checks frame order and
count and hashes coordinates, charges, cells, iterations, and source indices.
It drains the same paired adapter reader used by the scientific command, with
bounded retention. Its wall time includes checksum computation, but excludes
interpreter startup and imports. Use `/usr/bin/time -v` around it when comparing
whole-process time and RSS. Cache setup/population is inside the measured run.

Also compare 512 (`0:1536:3`) and 1,024 (`0:3072:3`) frames using separate empty
directories, plus a representative later slice such as `30000:36144:3`.
Vary test order when repeating to expose OS-cache and shared-storage variance.
An empty payload cache is not evidence of a cold OS cache. Do not clear the
system page cache or move staging cost outside the measurement.

## Full scientific command

After the bounded tests support the 2,150 s reader budget, submit
`hbn_polarization.slurm` from the original simulation directory. It repeats
the formal-then-ReaxFF command sequence with the original scientific flags and
a new workspace for this job ID. The second command can reuse xmolout frames
read by the first, as in the original job. The script assumes the existing
`xmolout` and `fort.7` already received the geometry-name replacement; it does
not rewrite them. Compare both polarization tables and invariant mappings
against the original outputs, and record the checkout revision and environment
alongside the job logs. The previous `reaxkit_workspace` remains untouched.

The benchmark job's `reaxff.time` is the ReaxFF-command acceptance measurement;
`formal.time` gives the corresponding formal-charge comparison.
The target is **<=41 min 19.8 s** with unchanged results. Report repeated runs
and storage variance. Preprocessing must be included separately before claiming
a speedup for the original entire script. A new workspace clears ReaxKit cache
state; it cannot clear the operating system's file cache, and the old workspace's
state at job 55785915 cannot be reconstructed from its timing logs.

## Reading the new metrics

Look for `phase: input_stream` in
`logs/timing/machine_readable_timing.log`. Each input has one record when its
stream closes. `reader_active_seconds` excludes time suspended while the
consumer calculates polarization. It includes reader startup and cleanup.

- `source_size_bytes`: full logical source-file size.
- `source_bytes`: consumed prefix on a forward scan, or selected frame-range
  bytes on indexed reads. `index_bytes` describes separate index work when
  present; a cold one-pass index needs no second source scan.
- `source_read_bytes`, `source_read_calls`, `source_read_seconds`: raw source
  read calls including bounded buffer lookahead, but excluding identity
  head/tail sampling. These are application reads, not physical disk traffic.
  `source_opens` counts reader opens; a warm reader can open without reading.
  Check `source_read_instrumented`: the generic unbounded fort.7 text fallback
  is not instrumented by the binary-read wrapper.
- `cache_read_seconds`, `cache_write_seconds`, `cache_metadata_seconds`, and
  `cache_maintenance_seconds`: active cache operations. Payload serialization
  belongs to write time. Connection initialization and some metadata lookups
  remain in the reader envelope. These counters should not all be subtracted
  from worker or pipeline times as though the phases were disjoint.
- `cache_payload_bytes_read/written`, `cache_commits`,
  `cache_maintenance_calls`, `cache_offsets_written`, `hits`, `misses`,
  `parsed_frames`, and `reader_branch` explain what the reader actually did.

Forward-scan offset counts are reported by `cache_offsets_written`; legacy
`indexed_frames/index_bytes` refer to a separate index pass. Bytes through a
selected boundary must not be confused with the whole file size. Source read
time excludes parsing and the OS may satisfy reads without physical I/O.

If the reader budget is missed, identify whether source reads, cache operations,
or residual parsing/adapter work dominate. Implement the plan's block parser
only when the remaining cost supports it. Compact cache formats, compiled
conversion, scratch staging, and worker tuning require their own measurements.
