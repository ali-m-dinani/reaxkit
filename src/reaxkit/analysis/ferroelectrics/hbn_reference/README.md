# hBN-reference local polarization and geometric polarity

## Al-centered neighborhoods

`get-hbn-reference-local-polarization` now defaults to one neighborhood per Al.
Each Al uses the three nearest coplanar N images in the prepared hBN reference.
Neighbors and reference positions are fixed throughout the trajectory; there is
no instantaneous nearest-neighbor reassignment. Periodic images are retained,
including multiple images of the same N in small simulation cells. Missing or
ambiguous three-neighbor shells fail explicitly instead of using distant atoms.
`neighbor_count` and `atom_count` count image memberships, not necessarily unique
simulation atom IDs in small periodic cells.
The current neighborhood construction requires Al/N reference sites; configured
substitutions can still map onto those reference sites.

Choose `--local-charge-method partition` (default) or `--local-charge-method neutral-center`:

| Method | N contribution to an Al neighborhood | Guarantee |
| --- | --- | --- |
| `partition` | Original N charge-weighted displacement divided among all neighborhood-image memberships of that N | Sum of local dipoles equals the existing global dipole, to numerical precision |
| `neutral-center` | Each of the three N images receives charge `-q_Al/3` | Every Al neighborhood has zero effective net charge |

Both use the existing global reference displacements and charge-sign convention.
Al charges may be ReaxFF charges or user-specified formal charges. `neutral-center`
uses the Al charge and does not use the independent N charge for its selected
dipole; the original N charges remain needed to report partition/global closure
diagnostics. Its local dipoles generally do not sum to the global ReaxFF result.
`raw_dipole_*` always reports the partition result; `neutralized_dipole_*` reports
the neutral-center alternative; `dipole_*` and `P_*` use the selected method.
`effective_net_charge (e)` describes the selected method, while `net_charge (e)`
describes the original partition. `local_charge_method` identifies the selection.
The former `--local-charge-treatment` option is replaced by this explicit choice;
the old mean-charge subtraction is no longer the local model.
The previously introduced `neutral-al` spelling is replaced by `neutral-center`;
update old command lines. This name change does not yet generalize the hBN
reference-neighbor construction to other materials.

The local command defaults to `--volume-method bbox`. With
`--local-volume-method equal`, each Al receives `V/N_Al`. With `deformation`,
Al-centered affine volume weights are normalized to sum to V.
`--volume-method cell` uses the full simulation-cell determinant, including
vacuum; `bbox` uses atomic coordinate extents and `hull` uses the whole-slab
convex hull. Local slab bbox/hull volumes are measured on coordinates lifted to
their matched reference images (`reference_position + displacement`), so
independent periodic image choices cannot enlarge the volume. These can differ
from a raw wrapped-coordinate bbox/hull used by the unchanged total command.

`--volume-method local-hull` uses a separate convex hull of each current Al and
its three N neighbor images. Equivalently, use `--local-volume-method hull`;
the slab volume method then supplies only a comparison volume. With four points
the convex hull is a tetrahedron, with volume `abs(det([v1,v2,v3]))/6` for the
three Al-to-N vectors. These are physical local volumes, NOT weights normalized
to fill the slab. Their sum need not equal the slab volume; they do not define
a space-filling partition. Cell-group output sums its Al neighborhood volumes.

Neighbor vectors follow the fixed reference image, transformed with the current
simulation cell, plus minimum-image relative atomic displacements. An N wrapped
onto the left box edge therefore remains adjacent to its Al at the right edge.
The same treatment is used for geometric polarity and neutral-center dipoles.
Periodic-image multiplicity is retained even in small cells. The `--periodic`
axes must describe the actual simulation; open lateral boundaries without a
complete three-neighbor reference shell still fail explicitly.

Coplanar neighborhoods have zero hull volume, `has_valid_local_volume=false`,
`local_volume_status=degenerate_hull`, and NaN polarization. No artificial
jitter or fallback slab volume is used. Very small nonzero volumes can give
large densities. The geometric polarity and dipole remain available.
Summary tables report invalid-volume counts and volume-partition closure error.

For shared slab volumes, partition gives the volume-weighted relation
`P = sum(V_local * P_local)/V`; equality with the total command's polarization
also requires the same numerical volume convention. Independent neighborhood
hulls do not generally satisfy the slab-volume equality, and zero-volume
neighborhoods have undefined local densities. Dipole-sum conservation in
partition mode is independent of this volume choice.

`--local-grouping al` is the default for local values and plots. `layer` is an
alias for these Al-centered values, not a disjoint two-atom AlN pair. `cell`
aggregates two Al neighborhoods by reference crystallographic-cell identity;
its dipoles and assigned volumes are sums of the Al-centered quantities.
The existing cell/layer CSV filenames are retained with these revised meanings.
Large local/cell/layer CSVs still require `--export-local-csv`.

`hbn_reference_al_neighbors.csv` lists three Al-N image memberships per
`local_neighborhood_id`, simulation atom IDs, reference bond vectors, and both
weighting conventions. This ID also appears in local results and projected
centers. Shared N atoms can appear in several neighborhoods; periodic images
of the same atom remain separate memberships.

In `hbn_reference_local_polarization.extxyz`, `local_neighborhood_id` and
`is_local_center` identify Al centers. N atoms have ID -1, center flag 0, and
NaN local quantities because they belong to multiple neighborhoods.
Only one set of `local_dipole`, `local_polarization`, `local_volume`, geometric
displacement/polarity, and their applicable `_c` components is exported.
`has_valid_local_volume` flags valid center volumes; `neighborhood_net_charge`
reports the selected effective charge. These are always neighborhood values,
even with `--local-grouping cell`; cell aggregates remain in CSVs/plots.
Frame metadata records `local_grouping=neighborhood` and the charge/volume methods.

Redundant `local_al_id`, `local_cell_id`, `local_layer_id`, layer quantities,
and duplicate Al/layer-center flags are removed from the local EXTXYZ.
Local neighborhood tables likewise no longer expose duplicate group IDs.
Crystallographic cell/layer labels remain in reference mapping artifacts and
internal bookkeeping; cell aggregate tables retain their cell ID.
Previously exported files are not rewritten: rerun to produce the new schema.

## Geometric projected polarity

`get-hbn-reference-projected-polarity` now uses geometry only, with one sign per Al:

`s_a = (displacement_Al - mean(displacement_of_three_reference_N_images)) dot component_axis`

Positive means the Al moves in the positive selected direction relative to the
mean N displacement. This is a structural sign convention, not the old sign of
the charge-weighted dipole. For the usual positive formal Al charge, it is
opposite to the previous electrical sign convention used by this code.
Use `--geometric-zero-tolerance` in angstrom (default `1e-6`): magnitudes at or
below it are zero and remain in bin means. The old `--dipole-zero-tolerance`
option is no longer accepted. Missing/ambiguous neighborhoods fail rather than
silently creating a polarity. Non-finite coordinates also fail.

No charge data or local volume is needed or loaded by projected polarity.
Inherited `--charge-source`, `--formal-charge`, and `--volume-method` arguments
are accepted for compatibility but do not affect its calculation. Spatial bins
are fixed at the reference Al positions. Centers tables contain `local_neighborhood_id`,
`representative_atom_id`, `geometric_displacement_<component> (angstrom)`, and
`polarity`; they no longer contain dipole or charge columns. All projection
tables identify `polarity_method=geometric`. `--local-grouping layer` is an alias
for per-Al output; `cell` is no longer accepted for this command.

Existing output directories are not migrated. Recompute projected data before
using the new convention in downstream velocity analyses; do not mix old dipole
signs and new geometric signs. The local/projected task versions have changed
to invalidate old scientific caches/checkpoints. Shared bounded streaming and
recovery mechanisms are retained.

### Projected-polarity Extended XYZ

Add `--write-extxyz` to projected polarity to write
`hbn_reference_projected_polarity.extxyz`, with optional
`--extxyz-precision 8` (significant digits). This is independent of
`--write-centers`: no center CSV is required. Standard output writes it only
when requested, full/legacy profiles enable it, and minimal suppresses it.

Every selected source frame contains all atoms, original coordinates, atom
numbers, current lattice and periodic flags. Al centers carry
`local_neighborhood_id`, `is_local_center`, `geometric_displacement`,
`geometric_polarity`, their `_c` projections, and `polarity` for the selected
`--component`. N atoms have ID -1, center flag 0, and NaN geometric properties.
There are no charge, dipole, volume, or polarization fields. Values are
per-neighborhood, not spatial-bin averages. Metadata records the component,
zero tolerance and geometric sign convention. Frames are streamed in order;
checkpoints commit complete XYZ frames with scientific tables and support
re-export after completion. Existing files require rerunning to gain this export.

## Domain-wall velocity from hBN projected polarity

| Method | Required CSV | Measures | Main advantage | Main limitation |
|---|---|---|---|---|
| Fraction growth | User-defined fractions in a modified grid CSV or x/z matrices | Onset-to-threshold average and interval rates | Per-bin endpoints exclude the late plateau | Effective growth, not necessarily one wall; threshold and sampling matter |
| Crossing times | `…_2d.csv` | Position versus persistent zero-crossing time | Directional estimate for a coherent front | Bin balance is not necessarily a wall; coarse sampling and nucleation can mislead |
| Local tracking | `…_centers.csv` | Position of matched local sign boundaries versus time | Separate spatial tracks without projection averaging | More data; matching, lattice staggering and topology changes complicate interpretation |

Crossing and tracking read an **existing result folder** from
`get-hbn-reference-projected-polarity`. Fraction growth reads exported or user-modified CSV
files directly. They never call that command, load
KF files, or read trajectories. All filenames below begin with
`hbn_reference_projected_polarity_`. Centers must have been exported with
`--write-centers --centers-format csv`; Parquet is not read by these workflows.

## Inputs and units

Use a spatially resolved projection, for example `--local-grouping layer
--projection-plane xz --projection-bins 19 10`. The 2D CSV retains both axes;
the upstream `--profile-axis` affects its kymograph, not these estimators.
The velocity `--axis` selects propagation direction, not dipole component.
One bin along an axis cannot support a crossing-time propagation fit there.

For crossing and tracking, time comes from the input table's `time` column, or is joined from
`whole_slab_summary.csv` using both `frame_index` and `iter`. If neither has
time, supply `--time-per-iteration`; time is then `iter × time_per_iteration`.
Existing time takes precedence. Never substitute a frame interval for an
iteration timestep. Declare its units with `--time-unit fs|ps|ns|s`.
Conversion is explicit: 1 angstrom/ps = 100 m/s.

For crossing and tracking, `--field-start-time` subtracts the field onset and excludes earlier data.
`--time-min` and `--time-max` select an inclusive interval **after** that
subtraction. Select the active propagation interval, rather than including
long plateaus. Subtract onset only once. Missing, conflicting or nonmonotonic
timestamps are rejected. Nothing assumes that frame 100 means iteration 500.

All coordinates are **reference coordinates**: velocities describe propagation
through the reference lattice, not lab-frame motion or elastic deformation.
New projected results represent Al-centered geometric signs (`al`, with `layer`
accepted as an alias). Historical charge-weighted cell/layer results have a
different sign and grouping convention and must not be mixed with these data.
Polarity measures signs, not polarization magnitudes.

## 1. Threshold fraction growth

Command: `get-hbn-domain-wall-fraction-velocity`.

This command uses **your calculated switched fractions** when present. Otherwise,
`--initial-sign +1` (the default) calculates `1 - positive_count/defined_group_count`;
`--initial-sign -1` calculates `1 - negative_count/defined_group_count`.
Use `defined_group_count`: it counts the local groups whose signs contribute to
the polarity counts, for both layer and cell grouping. The exported
`defined_cell_count` is a compatibility alias with the same value and is accepted
when `defined_group_count` is absent. Zero-polarity groups contribute to these
complement fractions; undefined groups do not. Empty grid bins receive zero
weight, and a slice with no defined groups is rejected.
There is no `--time-min`, `--time-max`, regression fit,
or automatic cycle selection. Supply fractions appropriate to the cycle being
measured and set its absolute `--field-start-time`.

### Input formats

- Modified `hbn_reference_projected_polarity_2d.csv`: retain `time`, `u_axis`,
  `v_axis`, `u_bin`, `v_bin` and optionally add `switched_fraction`. The existing header
  `switched fraction` is also accepted, or select a name with `--value-column`.
  An xz grid produces both x-bin and z-bin curves. Fractions are averaged
  using `defined_group_count` weights when present, otherwise equal weights.
  Every time must have the complete rectangular grid.
- Matrix: first column `x_bin/time(ps)` or `x_bin`, followed by numeric time
  headers. Each data cell is your switched fraction for that bin and time.
  `z_bin/time(ps)` and `z_bin` work similarly. Legacy `time_0.025_ps` headers
  are accepted. Use `--second-input` for the other axis's matrix. A single
  added fraction column in a wide matrix cannot describe all its times.

Fractions must be finite and in [0, 1]. Duplicate bin/time samples are rejected.
Existing fractions are never recomputed; a missing custom `--value-column`
is an error. All timestamps in
tables and plots retain original input time; `elapsed_time` is measured from
field onset. `--time-unit` declares the original time unit; it does not rescale
input timestamps.

### Axes and physical lengths

For a 19 by 10 xz grid:

- Each of the **19 x-bin curves** combines its 10 z bins and measures effective
  **z-direction growth** using the physical material thickness `--length-z`.
- Each of the **10 z-bin curves** combines its 19 x bins and measures effective
  **x-direction growth** using `--length-x`.

`--axis both` is the default and requires both lengths in angstrom. For a
single velocity direction, use `--axis z --length 50`, for example, with an
x-bin matrix. One matrix cannot reconstruct the other axis. The clean output
tables include all supplied axes even when only one velocity axis is selected.

### Endpoint average

For each bin, find its baseline fraction f0 at the field onset t0. Use an
exact sample when available, otherwise interpolate between bracketing samples;
never extrapolate. Find the **first recorded sample after onset** whose
fraction f1 exceeds `--threshold`, at time t1. Threshold equality is not a
crossing (a 1e-12 numerical tolerance avoids roundoff-induced crossings).

`average_velocity = length * (f1 - f0) / (t1 - t0) / front_count`

The threshold endpoint is not interpolated: the summary reports its actual
recorded time and fraction. `--front-count` defaults to one and represents
an assumed number of equally advancing fronts. `--threshold 0.98` measures
growth to the first fraction above 0.98; threshold 1 cannot be exceeded by
valid fractions. Bins not reaching the threshold, already above it at onset,
or lacking a bracketed onset are marked with a status and NaN average velocity.
Selecting a later onset alone does not stop the search before another cycle:
prepare cycle-specific fractions/data when that distinction matters.

### Lag-aware fraction growth

Add `--lag-threshold 0.05` to the same command to enable the lag-aware method.
Omitting it preserves the original onset-to-threshold method. For each bin,
the initial fraction f0 is evaluated at `--field-start-time`, with interpolation
only when bracketed. The rise starts at the first recorded sample with fraction
strictly above f0 + 0.05 (an absolute fraction increment, not 5% of f0).
The endpoint remains the first recorded sample strictly above `--threshold`.
Both comparisons use the existing 1e-12 tolerance; neither crossing is interpolated.

- `lag_time = rise_start_time - field_start_time`
- `rise_time = threshold_time - rise_start_time`
- `average_velocity = length * (threshold_fraction - rise_start_fraction) / rise_time / front_count`

The numerator excludes growth during the lag, so distance and time describe the
same interval. `velocities.csv` adds `lag_threshold`, `lag_target_fraction`,
`rise_start_time`, `rise_start_fraction`, `lag_time`, and `rise_time`.
`elapsed_time` continues to mean total onset-to-endpoint duration. All durations
use `--time-unit`; velocities are also exported in m/s. Summary statistics use
the rise-only velocities. Fraction plots mark the lag target and rise start.

If the lag target is not reached, status is `lag_not_reached`. If it is at or
above the final threshold, status is `lag_target_at_or_above_threshold`.
Crossing both levels in the same sample gives `unresolved_rise`: the saved data
cannot resolve a nonzero rise time. These cases have NaN average velocity.
A measured lag is retained when the endpoint threshold is not reached.
First crossings can be triggered by noise; no persistence rule is imposed.

### Instantaneous rates and plots (both methods)

`velocity_intervals.csv` contains finite differences between consecutive
post-onset samples, including samples **after** the threshold:

`instantaneous_velocity = length * (f_next - f_previous) / delta_time / front_count`

When onset lies between samples, its interpolated baseline is inserted for
the first partial interval. Each rate is located at the interval midpoint.
Rates are signed: back-switching gives negative values, plateaus give zero.
These are discrete interval averages, not an exact instantaneous derivative.

Every analyzed bin produces two publication-style PNGs, with white backgrounds
and no grids:

- Fraction versus original time, including onset, threshold and endpoint markers.
- Velocity versus original time: **blue dots** for interval rates and a **red
  horizontal line** for the constant onset-to-threshold average across the
  plotted post-onset time range. The late plateau remains visible as zero dots.
  If no valid endpoint average exists, no average line is fabricated.

Per-bin fraction plots are stored in `plots/fraction/` and per-bin velocity
plots in `plots/velocity/`, retaining filenames such as `x_bin_0_fraction.png`
and `x_bin_0_velocity.png`. Combined plots `plots/x_all_bins_fraction.png`
and `plots/z_all_bins_fraction.png` overlay all curves for each analyzed bin
axis, colored from blue (low bin number) to red (high bin number), with a
bin-number colorbar. They retain the original time axis, onset and threshold
guides, white background and no grid.

When the grid CSV provides `u_min/max/center (angstrom)` and
`v_min/max/center (angstrom)`, the fraction-table headers and per-bin plot
titles include each bin's center and bounds. The velocity and interval CSVs
include `bin_min_angstrom`, `bin_max_angstrom`, `bin_center_angstrom` and
`bin_position`. Combined plots show center coordinates on their colorbars.
Coordinates must be fixed across the input times; these are projection-bin
coordinates, not instantaneous wall positions. Bottom means the lowest-z bin
and top the highest-z bin in the supplied grid; x uses low-x/high-x labels.
This is a coordinate convention, not a claim about viewing orientation.
Matrix inputs without coordinates retain their original headers and plots
explicitly mark coordinates unavailable; positions are never guessed from IDs.

Outputs are `x_bins_fraction_vs_time.csv` and `z_bins_fraction_vs_time.csv`
(time rows, bin columns), `velocities.csv` (both endpoint times/fractions,
elapsed time, length, front count, velocity and status), `velocity_intervals.csv`,
`settings.json`, and `plots/`. Only available input-axis matrices are written.

`velocity_summary.csv` has one row per analyzed velocity direction (x/z),
reporting the unweighted mean, median, minimum and maximum of the per-bin
onset-to-threshold velocities in m/s. Only bins with status `ok` contribute;
total, valid and excluded bin counts are included. Statistics are blank when
no bins qualify. These summarize endpoint averages across bins, not the
instantaneous rates across time.

The summary also reports `std_velocity_m_per_s` (population standard deviation,
`ddof=0`), `velocity_cv = std / mean`, and
`speed_heterogeneity_index = std / (mean + std)`. These use the same valid bins
with equal weighting. Identical positive speeds give zero CV and heterogeneity;
larger values indicate more variation. CV is unbounded, while the heterogeneity
index approaches one for strong relative variation in nonnegative speeds.
CV and heterogeneity are blank when the mean is zero or no bins qualify.
One valid positive-speed bin gives zero variation but cannot establish spatial
uniformity; always inspect valid/excluded counts. This is a descriptive index
of switching-speed heterogeneity, not a measure of domain-wall flatness.

Every run also writes `README.md` into its output directory, explaining the
summary formulas, numerical examples of the heterogeneity index, exclusion
rules, missing values and interpretation limitations.

These remain **effective growth speeds**. Nucleation, multiple fronts and
growth along the omitted projection axis affect the fractions. A horizontal
average line summarizes the two endpoints; it does not mean motion is constant.

## 2. Crossing times

Command: `get-hbn-domain-wall-crossing-velocity`.

Recompute `mean_polarity = (positive_count - negative_count)/defined_group_count`.
For each spatial bin, find a crossing from the initial-sign side to the target
side of zero. Interpolate its time linearly between adjacent saved samples.
Require `--persistence` consecutive target-side samples (default two, including
the crossing sample). Missing frames and undefined bins are not interpolated
across. Bins already at/past zero at the start are not assigned an onset time.
Multiple persistent crossings are marked ambiguous and excluded.

Fit reference bin-center position against crossing time separately for each
transverse slice. Signed slopes indicate propagation direction. Nonmonotonic
or simultaneous arrival sequences, and gaps between bins with valid crossings,
are flagged rather than reported as valid
front velocities. Inspect `samples.csv` and the spatial maps: monotonic arrivals
alone cannot rule out independent nucleation. This method does not automatically
identify separate fronts or split a reversing front into segments.

Use actual bin-center distances, not the full box length. Ten evenly spaced
centers span nine bin widths. No additional length multiplier is needed after
fitting position versus time. `--aggregate-transverse` sums counts across all
transverse bins before detecting crossings; it can obscure curved fronts.

## 3. Local sign-boundary tracking

Command: `get-hbn-domain-wall-tracking-velocity`.

Partition centers into lines with the same two transverse reference coordinates,
rounded to `--coordinate-decimals` (default four). Sort each line along the
requested axis. A boundary is the midpoint between adjacent groups of opposite
nonzero polarity. Zero/undefined sites are not bridged. No dipole-magnitude
interpolation is implied. Site staggering may produce sparse, distinct lines;
inspect the exported lane labels before interpreting the tracks. Rounding is
for coordinate noise, not arbitrary spatial coarse-graining.

Between consecutive available saved frames, match boundaries of the same sign
orientation within the required `--max-step` displacement. Accept only mutually
unique matches. Ambiguous candidates start new tracks; disappearing boundaries
end tracks. Missing sites invalidate that line for the frame. Tracks never
bridge a missing line/frame. A per-track least-squares slope gives velocity;
single-frame tracks remain marked `insufficient_points`.

For periodic motion along the selected axis, explicitly supply
`--periodic-length`. The algorithm includes the seam boundary and accumulates
minimum-image displacements. `max_step` must be less than half this length.
Too small a gate fragments tracks; too large a gate risks identity swaps or
ambiguity. Even unique matches do not prove physical identity after nucleation
or annihilation. The method tracks one-dimensional intersections with walls,
not a reconstructed 3D surface or its normal velocity. Under-resolved motion
cannot be recovered by any matching algorithm.

## Examples (Bash)

`results` is an existing projected-polarity run folder. Example lengths and
step gates below must be replaced with values appropriate to your simulation.

```bash
reaxkit get-hbn-domain-wall-fraction-velocity \
  --input "$results/hbn_reference_projected_polarity_2d.csv" \
  --value-column "switched fraction" --threshold 0.98 \
  --axis both --length-x 100 --length-z 50 \
  --field-start-time 0.125 --time-unit ps

reaxkit get-hbn-domain-wall-crossing-velocity --input "$results" \
  --axis z --time-unit ps --field-start-time 0.125 --persistence 2

reaxkit get-hbn-domain-wall-tracking-velocity --input "$results" \
  --axis x --time-unit ps --field-start-time 0.125 \
  --max-step 5 --periodic-length 100
```

For crossing/tracking only, if physical time is absent, add, for example, `--time-per-iteration 0.00025`
only if each iteration really represents 0.00025 ps.

## Outputs and interpretation

Each invocation writes directly to a unique
`reaxkit_workspace/other/<command>/run_*/` folder, or `--output-dir` when given:

Crossing/tracking export the following; fraction-growth outputs are listed in its section above.

- `samples.csv`: crossing events or boundary tracks.
- `velocities.csv`: signed slopes in angstrom per chosen time unit and m/s,
  point counts, fit windows, R-squared and status. Fraction-growth signs describe
  increasing/decreasing target population, not an independently identified
  direction of wall motion.
- `diagnostics.csv`: missing/ambiguous crossings or tracking conditions.
- `settings.json`: source folder, method, units, coordinate convention and options.

### Time- and position-resolved speeds

The crossing workflow also writes `velocity_intervals.csv`, alongside fitted
velocities. Each interval includes time, axis, signed velocity in m/s, absolute
speed and validity status. Fraction-growth interval outputs are described above.

**Crossing times:** one row per neighboring axial-bin pair and transverse lane
reports both positions, both interpolated crossing times, the saved frames
bracketing each crossing, and spatial/time midpoints. The local velocity is

`v_pair = (position_upper - position_lower) / (crossing_time_upper - crossing_time_lower)`.

This retains the sign for reverse propagation. `time_start/end` are ordered
chronologically; `lower/upper` refer to spatial positions. No extra length
factor is applied. Missing or simultaneous crossings and invalid overall
arrival sequences produce flagged NaNs. These are **event-interval speeds**,
not measurements at every saved frame; no values are extrapolated before or
after the events. Two crossings within the same saved-frame interval are
especially sensitive to temporal resolution despite linear interpolation.

Use the midpoint columns for time/position maps and filter `status=ok`.
Finite-difference speeds are noisier than the fitted averages. Averaging local
speeds does not in general reproduce a least-squares velocity over the full
fit window.

Only interpret rows with `status=ok` as successful numerical fits. High
R-squared does not establish a physical wall identity. Inspect sample counts,
time resolution and spatial maps; compare different bin sizes and fit windows.
Crossing/tracking export tables. Fraction growth additionally generates its own per-bin plots.
