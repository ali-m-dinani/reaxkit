# Frame-cache benchmark

`frame_cache_overlap.py` measures cold, exact-hit, partial-overlap, and
no-overlap xmolout selections. It reports wall time, cache hits/misses, parsed
frames, index bytes, and source-frame bytes.

On 2026-09-22, the motivating 3.05 GB trajectory was measured with:

```text
--stop 3200 --first-step 100 --second-step 40
```

| Case | Hits | Misses | Parsed | Source-frame bytes | Index bytes | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold `0:3200:100` | 0 | 32 | 32 | 30,500,096 | 2,955,649,928 | 128.07 s |
| exact cached selection | 32 | 0 | 0 | 0 | 0 | 0.16 s |
| partial `0:3200:40` | 16 | 64 | 64 | 61,000,192 | 57,187,680 | 7.84 s |
| isolated no-overlap | 0 | 80 | 80 | 76,250,240 | 3,013,790,736 | 134.66 s |

The partial request extended the existing index by 60 frames because its last
requested source index was beyond the first request. It still avoided rescanning
the already covered 3,101 frames and parsed exactly the expected 64 misses.

The complete CLI sequence was also run against that simulation. A dynamic
charge command using `0:3200:100` completed successfully, followed by the
formal-charge command using `0:3200:40`. The second command reported
`xmolout frame cache: 16 hit, 64 miss`, did not open fort.7, and wrote the full
polarization, displacement, mapping, and aligned-reference outputs.

## Selected-frame runtime validation

Run the portable workstation matrix with:

```bash
python -m reaxkit.core.runtime.benchmark --output benchmark_results/workstation.json
```

The matrix compares one, two, and four workers for an in-memory producer and
for first/repeated reads from a generated binary trajectory. It rejects
nondeterministic results and any run whose retained frame payload exceeds the
configured in-flight limit.

Submit `slurm_selected_frame_pipeline.sh` on Roar Collab after activating the
same ReaxKit environment used for production analysis. Set `PYTHON_EXE` when
`python` is not the intended environment. Put the benchmark workspace under
`SLURM_TMPDIR` for local-scratch measurements; set `RESULT_DIR` to a shared
filesystem path so the JSON report and `/usr/bin/time` output persist.

For production comparisons, run the same scientific command twice with an
empty ReaxKit frame cache and then with the populated cache. Record `seff`,
the command timing log, filesystem location, ReaxKit commit, and input hashes.

## Unified rollout corpus and performance gates

The small correctness corpus is generated deterministically by
`tests/core/test_unified_rollout.py` and the existing engine fixtures. Run the
medium scientific comparison with:

```bash
python benchmarks/scientific_frame_maps.py --frames 32 --atoms 128
python -m reaxkit.core.runtime.benchmark --frames 128 --payload-mib 1 --workers 1 2 4 --output benchmark_results/unified_runtime_workstation.json
```

The checked-in scientific report contains 21 serial/thread comparisons across
coordinate, displacement, dihedral, two SciPy Voronoi, and two neighbor kernels.
Every result matched its materialized scientific reference at `1e-12` tolerance.
On the recorded workstation, four-worker Voronoi took 1.26 s versus 1.91 s with
one worker; geometry took 1.83 s versus 3.41 s. Lightweight kernels were slower
with threads; neighbor gains were small. Those tasks retain automatic serial
streaming, with explicit thread overrides for larger workloads.

The runtime matrix runs each case in a separate interpreter and records wall
time, CPU time/efficiency, peak RSS, process I/O bytes where supported, output
size, throughput, and retained payload peaks. Windows uses peak working set;
Linux uses `ru_maxrss`. The memory regression compares 12 and 160 generated
1 MiB frames, permitting 32 MiB of process baseline variation. First/repeated
file reads describe process-level access; they do not claim an OS page-cache
flush. Scientific speedups are fixture-specific, not a production guarantee.

## Private 28,800-atom, 4,000-frame HPC acceptance recipe

Keep the production input private. Record its SHA-256 hashes, atom count, frame
count, source format, engine, ReaxKit commit, environment, Slurm allocation,
filesystem, and scientific selection. Use the same input and flags in each run.

1. Copy the 28,800-atom, 4,000-frame trajectory and required aligned charge or
   connectivity source to an empty run-specific directory on shared storage.
   Prepare an equivalent directory under `$SLURM_TMPDIR` for local-scratch runs.
2. Validate a small selection, including noncontiguous frames and the reference,
   with `--execution serial` and `--execution auto`. Compare identities, frames,
   categories, periodic-image choices, and numeric core outputs at the task's
   tolerance before timing the full trajectory.
3. For each filesystem, use fresh isolated ReaxKit cache directories for cold
   application-cache runs. Repeat the identical command for warm-cache runs.
   Do not delete other users' caches or require privileged OS cache flushing.
4. Run the selected scientific command over `--frames 0:4000:1`, first serial
   and then automatic, and capture `/usr/bin/time -v -o time.txt reaxkit ...`.
   Use the command's normal scientific options, charge source, and reference.
   Repeat with `--output-profile full` to measure requested detail output.
5. Save timing logs, settings, artifact manifests and file sizes. After the job
   completes save `seff JOB_ID` and `sacct` resource information. Report wall
   time, CPU seconds, CPU efficiency, peak RSS, read/write bytes, output bytes,
   and selected frames per second. On platforms without per-process I/O counters,
   mark the field unavailable and attach site-provided filesystem measurements.
6. Check that source-frame parse counts do not increase with worker count and
   that dependency frames are read once. Compare a short prefix with all 4,000
   frames to distinguish bounded input memory from retained primary results.

Submit `slurm_selected_frame_pipeline.sh` for the portable runtime, correctness,
and medium scientific gates. Run the private-data commands in the same recorded
allocation or a separate job with matching resources. Acceptance requires
equivalent science and a measured memory or wall-time benefit, with no increased
source rereads. Production Slurm acceptance has not been run in this workstation
session and remains pending.

## Completion migration gates

```bash
python benchmarks/rollout_completion.py
python -m pytest tests/core/test_rollout_completion.py tests/core/test_command_inventory.py -q
```

`benchmark_results/rollout_completion.json` records 27 materialized-versus-bounded
comparisons for polarity, basal polarization, both strain kernels and diffusivity
at requested worker counts 1, 2 and 4. Ordered/global kernels correctly retain
one worker. Comparisons preserve identities and numerical values; a legacy
reference-first table is compared by source frame because shared execution
emits ascending source-frame order. Newly benchmarked kernels remain automatic
serial pending a repeatable threading benefit on representative workloads.

The isolated scientific memory test uses 8,192 atoms and 16 versus 512 frames.
Recorded peak RSS was 172,863,488 versus 182,312,960 bytes (about 9 MiB growth),
with one source pass. Its four-bin primary table grows with the requested frame
count. Timings in this completion report were collected while the full regression
suite was also running; they are informational and did not enable new automatic
parallel defaults. The original runtime and scientific reports remain available.
