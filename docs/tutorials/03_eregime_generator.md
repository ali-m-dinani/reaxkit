# Generating Electric-Field Schedules with `eregime.in`

In previous tutorials, we focused on **analyzing existing ReaxFF outputs**.
In this tutorial, we shift to the *input side* and show how ReaxKit helps you
**generate an `eregime.in` file** for electric-field–driven simulations.

The goal is to make electric-field protocols:
- reproducible,
- readable,
- easy to modify,
- and consistent with ReaxFF / AMS expectations.

---

## What is `eregime.in`?

`eregime.in` defines **time-dependent electric fields** applied during a ReaxFF
simulation.

Each line typically specifies:
- the iteration at which the field is applied,
- the applied electric field (`V`),
- the field direction (`x`, `y`, or `z`),
- the field magnitude (in V/Å).

---

## ReaxKit’s approach to `eregime.in` generation

Instead of manually editing `eregime.in`, ReaxKit provides **programmatic
generators** that:

- construct field schedules from physical parameters,
- validate directions and sampling,
- write correctly formatted files,
- keep logic separate from file I/O.

All generators ultimately call a single low-level writer:

```text
write_eregime()
```

This ensures a single, consistent file format.

---

## The common output structure

All generators produce rows of the form:

`(iteration, V_index, direction, magnitude)`

These rows are then written with a standard header:

```text
#Electric field regimes
#start #V direction Magnitude(V/A)
```

This format matches ReaxFF expectations exactly.

---

## Generator 1: Sinusoidal electric fields

A sinusoidal field is commonly used for:

* dielectric response,
* polarization hysteresis,
* frequency-dependent studies.

Mathematically:

```equation
E(t) = dc_offset + A · sin(phase + ωt)
```

#### Using `make_eregime_sinusoidal`

```
make_eregime_sinusoidal(
     "eregime.in",
    max_magnitude=0.05,
    step_angle=0.05,
    iteration_step=100,
    num_cycles=5,
    direction="z",
)
```

Key parameters:

* `max_magnitude` → peak field amplitude (V/Å)
* `step_angle` → angular resolution (radians)
* `iteration_step` → MD iterations between samples
* `num_cycles` → number of sinusoidal cycles
* `direction` → field direction (x, y, or z)

Internally:

* the sine wave is sampled uniformly in angle,
* each sample is mapped to a simulation iteration,
* the result is written as a complete `eregime.in`.

---

## Generator 2: Smooth bipolar pulse fields

Smooth pulses are useful for:

* switching dynamics,
* avoiding numerical artifacts,
* studying transient responses.

Each cycle consists of:

* ramp up,
* flat top,
* ramp down,
* baseline,

followed by a mirrored negative pulse.

#### Using `make_eregime_smooth_pulse`
```
make_eregime_smooth_pulse(
    "eregime.in",
    amplitude=0.04,
    width=50.0,
    period=200.0,
    slope=20.0,
    iteration_step=50,
    num_of_cycles=3,
    direction="z",
)
```

Key parameters:

* `amplitude` → pulse height (V/Å)
* `width` → flat-top duration
* `slope` → ramp duration
* `period` → full pulse cycle
* `step_size` → temporal resolution
* `num_of_cycles` → number of cycles

The generator enforces:

* physically consistent timing,
* symmetric positive/negative pulses,
* smooth transitions (no discontinuities).

---

## Generator 3: Arbitrary user-defined fields

Sometimes the field profile does not match a standard waveform.

ReaxKit allows you to define:

```text
E(t) = f(t)
```

directly.

#### Using `make_eregime_from_function`

```
def my_field(t):
    return 0.02 * t * np.exp(-t / 10.0)

make_eregime_from_function(
    "eregime.in",
    func=my_field,
    t_end=50.0,
    dt=0.2,
    iteration_step=100,
    direction="z",
)
```

This approach:

* samples `func(t)` uniformly in time,
* maps time samples to iteration numbers,
* gives you full freedom over the waveform.

This is especially useful for:

* externally fitted fields,
* machine-learning–generated protocols,
* experiment-informed schedules.

---

## Direction, iteration, and validation

Across all generators:

* directions are validated (x, y, z only),
* iteration steps must be positive,
* invalid timing configurations raise errors early.

This prevents silent generation of invalid `eregime.in` files.

---

## Output location and usage

Generated files are typically written to:

`reaxkit_generated_inputs/eregime.in`

You can then:

* reference `eregime.in` directly in your ReaxFF input,
* version-control it,
* regenerate it parametrically when conditions change.

---

## What you can do next

With eregime generation in place, you can now:

* couple electric fields with trajectory analysis,
* generate polarization vs field loops,
* explore frequency-dependent response,
* integrate experiment-informed field profiles,
* automate full input–simulation–analysis pipelines.

This closes the loop between **input generation** and **output analysis** in ReaxKit.
