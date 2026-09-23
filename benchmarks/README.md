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
