# Generating Electric-Field Schedules with `gen_eregime`

This tutorial reflects the current ReaxKit generator workflow for creating
`eregime.in` files.

---

## What `eregime.in` contains

`eregime.in` stores sampled electric-field rows with this structure:

`(iteration, V_index, direction, magnitude)`

with header:

```text
#Electric field regimes
#start #V direction Magnitude(V/A)
```

---

## Main command

The general format for this command is as follows:

```bash
reaxkit gen_eregime --type <sin|pulse|func> --iteration-step <int> [profile-specific flags]
```

Common flags:
- `--output` output filename/path (default `eregime.in`)
- `--direction` field axis (`x`, `y`, `z`)
- `--V` voltage index column
- `--start-iter` first iteration number

---

## Profile 1: Sinusoidal field

```bash
reaxkit gen_eregime --type sin --output eregime.in --max-magnitude 0.35 --points-per-cycle 17 --iteration-step 500 --num-cycles 2 --direction z --V 1 --copy-to-dot
```

Meaning:
- `--max-magnitude`: exact positive and negative extrema around the offset
- `--points-per-cycle`: number of rows in one complete piecewise-linear cycle, including its start and end
- `--iteration-step`: MD iterations between consecutive rows
- `--num-cycles`: number of complete cycles

Each cycle returns exactly to `--dc-offset` at its start, half-cycle, and end.
Adjacent cycles share that boundary row, preventing consecutive duplicate
field values. The total row count is
`num_cycles * (points_per_cycle - 1) + 1`.

Within each quarter-cycle, the magnitude changes by an equal amount at every
row. This produces linear ramps between zero and each extremum. It is therefore
a triangular, piecewise-linear sampled profile rather than exact sine values.
Use legacy `--step-angle` when exact sine-value sampling is required.

For equal-duration positive and negative halves, use an odd count such as 9 or
11. With 10 points there are nine time intervals, so one half necessarily has
one additional interval.

Optional sin-only controls:
- `--dc-offset`
- `--phase` (must be zero or a whole multiple of `2*pi` in points-per-cycle mode)
- `--step-angle` (legacy alternative to `--points-per-cycle`)

---

## Profile 2: Smooth pulse field

```bash
reaxkit gen_eregime --type pulse --output eregime.in --amplitude 0.003 --width 50 --period 200 --slope 20 --iteration-step 250 --num-cycles 5 --direction z --V 1
```

Meaning:
- `--amplitude`: pulse height above baseline
- `--width`: flat-top duration
- `--period`: full positive+negative cycle length
- `--slope`: rise/fall duration
- `--num-cycles`: cycle count

Optional pulse controls:
- `--step-size` (sampling resolution)
- `--baseline`

---

## Profile 3: User-defined function

```bash
reaxkit gen_eregime --type func --output eregime.in --expr "0.003*cos(2*pi*t/100)" --t-end 1000 --dt 1 --iteration-step 250 --direction z --V 1
```

Meaning:
- `--expr`: function of `t`
- `--t-end`: final sampled time
- `--dt`: sampling interval
- `--iteration-step`: iteration mapping between samples

---

## Validation behavior

The generator validates inputs before writing:
- direction must be `x`, `y`, or `z`
- `iteration_step` must be positive
- profile-specific required flags must be provided
- pulse constraints must be physically consistent

Invalid settings raise clear errors instead of producing malformed files.

---

## Python API note

If you need programmatic generation, the core API is:

```python
from reaxkit.engine.reaxff.generators.eregime_generator import gen_eregime
```

with `profile_type="sin" | "pulse" | "func"` and the same profile parameters.

---

## Output location

Generated inputs are stored using ReaxKit generator output layout 
(under `reaxkit_workspace/inputs/`), with optional `--copy-to-dot` to place a
copy in the current working directory.

---


## Related next steps

- See the next tutorial [04_gen_plot_workflow](04_gen_plot_workflow.md) to learn how to plots for any data you have.

---

[Back to Tutorials](index.md)
