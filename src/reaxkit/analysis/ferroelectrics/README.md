# Wurtzite polarity analysis

The four-folded and three-folded implementations use the same signed c-axis
bond projection. For center atom \(i\) and neighbor \(j\),

\[
b_{ij,c} = (\mathbf r_j^{\mathrm{image}}-\mathbf r_i)\cdot\hat{\mathbf c},
\]

where \(\hat{\mathbf c}\) is the normalized configured c-axis and
\(\mathbf r_j^{\mathrm{image}}\) is the nearest periodic image of the neighbor.
Thus a positive projection points along the configured c-axis ("up") and a
negative projection points in the opposite direction ("down"). Distances and
projections use the minimum periodic image when cell data are available.

## `four_folded_wurtzite`

The four-folded implementation finds the four nearest unique neighbor atoms
inside `neighbor_cutoff`. A site is complete only when all four are present.
Among those four bonds, the bond with the largest absolute c-axis projection is
classified as apical:

\[
j_{\mathrm{apical}} = \underset{j}{\operatorname{argmax}}\ |b_{ij,c}|.
\]

The remaining three bonds are basal. Their signed mean is

\[
b_{i,\mathrm{basal}} = \frac{1}{3}\sum_{j\in B_i} b_{ij,c}.
\]

The four-folded geometric displacement is

\[
\Delta_i = b_{i,\mathrm{apical}} - b_{i,\mathrm{basal}}.
\]

Given the configured tolerance \(\epsilon\), the discrete polarity is

\[
s_i =
\begin{cases}
+1 & \Delta_i > \epsilon \quad (\mathrm{UP}),\\
-1 & \Delta_i < -\epsilon \quad (\mathrm{DOWN}),\\
0 & |\Delta_i|\le\epsilon \quad (\mathrm{UNASSIGNED}).
\end{cases}
\]

The charge-weighted local vector and its c-axis component are

\[
\mathbf p_i = \sum_j q_j(\mathbf r_j^{\mathrm{image}}-\mathbf r_i),
\qquad
\eta_{i,c}=\mathbf p_i\cdot\hat{\mathbf c}.
\]

The code reports \(\mathbf p_i\) and \(\eta_{i,c}\) in e·Å and converts them
to debye using 1 e·Å = 4.80320427 D. It also reports the change in mean
basal projection relative to the same atom in `reference_frame` (frame 0 by
default):

\[
\delta b_{i,\mathrm{basal}}(t) =
b_{i,\mathrm{basal}}(t)-b_{i,\mathrm{basal}}(t_{\mathrm{ref}}).
\]

### Limitations for surfaces and three-coordinate sites

The four-folded method has two coupled limitations for the intended surface
analysis:

1. A center with only three neighbors is marked incomplete, so its displacement,
   polarity, dipole, and local order are left unassigned.
2. Apical assignment happens before polarity is known and uses only
   \(|b_{ij,c}|\). At a top surface, a lone neighbor below the center can therefore
   be called apical even when an UP site requires an apical neighbor above it.
   With candidates on both sides, the absolute-value rule also has no polarity
   criterion for choosing the physically consistent side.
3. Because \(\Delta_i\) includes the apical bond, a missing or incorrectly chosen
   apical atom directly changes the polarity classification.

## `three_folded_wurtzite`

The three-folded implementation makes the three basal bonds sufficient for a
valid site. It first collects every unique selected neighbor inside the cutoff,
including nearest periodic images. It then chooses the three candidates with
the smallest \(|b_{ij,c}|\), with distance as the tie-breaker, because basal
bonds have smaller c-axis separation than bonds to another layer.

For these three basal neighbors it defines the requested displacement using
the negative signed mean basal projection:

\[
\Delta_i = -b_{i,\mathrm{basal}}
= -\frac{1}{3}\sum_{j=1}^{3} b_{ij,c}.
\]

The minus sign accounts for the bond-vector convention: every bond points from
the center atom to its neighbor. Basal N atoms below an Al center therefore
have negative \(b_{ij,c}\), which gives positive \(\Delta_i\), UP polarity, and
a dipole pointing from N toward Al. The same tolerance rule above converts
\(\Delta_i\) to UP, DOWN, or UNASSIGNED.
Consequently, a center with exactly three basal neighbors receives polarity,
dipole, local-order, and reference-change values; an apical neighbor is not
required for the site to be complete.

Only after polarity has been calculated does the code select an apical neighbor:

- for UP polarity, eligible apical candidates must have \(b_{ij,c}>\epsilon\);
- for DOWN polarity, eligible candidates must have \(b_{ij,c}< -\epsilon\);
- for zero polarity, no apical neighbor is selected.

If several candidates occur on the permitted side, the one with the largest
\(|b_{ij,c}|\) is selected because it is most separated from the basal layer.
Candidates on the opposite side are labeled `ignored`. Therefore an UP surface
site with only a lower-layer candidate keeps its polarity but reports
`has_apical_neighbor = false`; that lower candidate is not misidentified as
apical. If both upper- and lower-layer candidates exist, the upper one is chosen
for UP polarity and the lower one for DOWN polarity.

The neighbor table uses three role states during the calculation:
`basal`, `apical_candidate`, and, after polarity is known, `apical` or `ignored`.
The polarity table records `has_three_basal_neighbors` separately from
`has_apical_neighbor`, making surface coordination explicit. The trajectory
export writes both flags along with `polarity`, `eta_c`, and `delta_eff`.

The charge-weighted vector uses the three basal neighbors plus the selected
apical neighbor, when one exists. Ignored candidates do not contribute.

### Spatially binned polarization

`get-three-folded-wurtzite-polarization` reuses the local dipole vectors above.
The user chooses the number of bins independently along x, y, and z. Bin edges
are fixed from the reference-frame coordinate range so that the same spatial
regions are compared across frames. For bin \(k\), the code sums all valid site
dipoles assigned to that bin:

\[
\boldsymbol{\mu}_k = \sum_{i\in k}\mathbf p_i.
\]

It converts the dipole density to polarization using

\[
\mathbf P_k = \frac{\boldsymbol{\mu}_k}{V_k}
\times C_{\mathrm{e/angstrom^2\ to\ micro C/cm^2}},
\]

where \(V_k\) is selected with `--volume-method`:

- `hull`: convex-hull volume of all finite atoms located in the bin;
- `bbox`: axis-aligned bounding-box volume of those atoms;
- `cell`: simulation-cell volume divided equally among all requested 3D bins.

Hull and bounding-box volumes can be undefined or zero in sparsely populated or
geometrically degenerate bins. Their polarization values are reported as `NaN`.

With `--heatmaps`, the workflow projects the selected polarization component
onto `xy`, `xz`, or `yz`. If bins exist along the omitted axis, their dipoles and
volumes are summed before the projected polarization is calculated. Use
`--global-scaling` to share symmetric color limits across all selected frames,
or `--no-global-scaling` for independent per-frame limits.

## `basal_plane_displacement_for_dipole_moment`

This method implements the basal-plane displacement construction described by
John Hayden, Mohammad Delower Hossain, Yihuang Xiong, Kevin Ferri, Wanlin Zhu,
Mario Vincenzo Imperatore, Noel Giebink, Susan Trolier-McKinstry, Ismaila Dabo,
and Jon-Paul Maria, "Ferroelectricity in boron-substituted aluminum nitride thin
films," *Physical Review Materials* **5**, 044412 (2021),
[https://doi.org/10.1103/PhysRevMaterials.5.044412](https://doi.org/10.1103/PhysRevMaterials.5.044412).

Hayden et al. calculate spontaneous polarization from charge-weighted ionic
displacements and use the base of the nitrogen tetrahedron as the displacement
reference. The cited first-principles construction is

\[
P_3=\frac{|e|}{\Omega}\sum_k \overline{Z}^{*}_{k,33}
\Delta u_{k,3},
\]

where every ion has one displacement from a nonpolar hexagonal reference and
\(\overline{Z}^{*}_{k,33}\) is the longitudinal Born effective charge averaged
along the switching path. Reproducing that calculation exactly requires the
nonpolar reference coordinates and Born effective charges; neither formal
charges nor `fort.7` partial charges are Born effective charges.

The trajectory implementation is a local structural approximation. It fixes
the nitrogen framework as the reference and measures every Al/B center once
from the mean plane of its three basal N neighbors. It uses either
user-provided formal charges or per-frame charges from `fort.7`.

The polarity-aware neighbor roles come directly from `three_folded_wurtzite`.
For the three basal neighbor image positions, the reference point is

\[
\overline{\mathbf r}_{i,\mathrm{basal}}=
\frac{1}{3}\sum_{j=1}^{3}\mathbf r_{ij}^{\mathrm{image}}.
\]

The center displacement is

\[
\Delta\mathbf u_i=
\mathbf r_{i,\mathrm{center}}-\overline{\mathbf r}_{i,\mathrm{basal}}.
\]

The local contribution is then

\[
\boldsymbol{\mu}_i=e Z_i\Delta\mathbf u_i,
\qquad e=-|e|.
\]

The electron charge is negative by the package convention. Every atom appears
once in `basal_plane_ions.csv`. Al/B centers carry the displacement above;
the fixed N reference framework and other non-center ions have zero displacement
and zero contribution. This is how all supercell ions remain represented
without assigning one N atom several incompatible local reference planes. The
apical neighbor remains a useful structural diagnostic, but its full height
above another N plane is not an additional ferroelectric displacement.

### Local dipole and polarization

`get-basal-plane-displacement-dipole` reports one local dipole vector per valid
Al/B center. `get-basal-plane-displacement-local-polarization` converts each
center dipole
to a local polarization, \(\mathbf P_i=\boldsymbol{\mu}_i/\Omega_i\), using
one of two local-volume conventions:

- `--local-volume-method equal` (default) divides the selected frame `hull`,
  `bbox`, or `cell` volume equally among all valid centers. The assigned local
  volumes sum to the frame volume, so their volume-weighted polarization
  reproduces the frame polarization.
- `--local-volume-method coordination` uses the tetrahedron whose vertices are
  the three basal N neighbors and the apical N neighbor. A center without four
  valid neighbor vertices, or with a degenerate tetrahedron, is retained with
  `has_valid_local_volume=false` and `NaN` local polarization.

The command writes one row per selected center to
`basal_plane_local_polarization.csv`, including the source dipole, assigned
volume, and x, y, and z polarization components.

The basal-plane dipole, binned-polarization, and local-polarization commands
also write a compact `*_poled_counts.csv`. It has one x, y, and z row per
frame with the iteration, time, total entry count, positive (poled-up) count,
negative (poled-down) count, and their percentages. Zero and unavailable
values remain in the total count but are neither poled up nor poled down.

The default combination is `--local-volume-method equal --volume-method hull`.
The convex hull is calculated from occupied atomic coordinates and therefore
excludes empty slab vacuum before it is divided among valid centers. `bbox`
also excludes surrounding vacuum using occupied coordinate extents. `cell`
includes the complete simulation cell and should be selected only when that is
the intended normalization.

For spatial visualization, `--plot-2d` bins centers in the selected
`--plot-plane` and aggregates along the omitted coordinate. Polarization maps
sum dipoles and local volumes separately and then divide, rather than averaging
already-normalized center polarizations. `--plot-3d` writes a per-frame 3D
center scatter plot. `--plot-quantity polarization|dipole` and
`--plot-component x|y|z` select the displayed value. Shared color limits across
frames are available with `--global-scaling`.

`--write-extxyz` writes `basal_plane_local_polarization.extxyz` for OVITO. Each
frame retains every atom and provides `is_local_center`,
`has_valid_local_volume`, `local_volume`, `local_dipole`, and
`local_polarization` atom properties. Vector properties have three Cartesian
components.

Add `--include-electric-field --field-direction z` to read `fort.78`, align the
selected field component by iteration, and write it only in each frame header
as `electric_field`, `electric_field_direction`, and `electric_field_units`.

### Projected polarity fractions

`get-basal-plane-displacement-projected-polarity` converts the selected local
dipole component into a discrete polarity for every valid center:

\[
s_i=\operatorname{sign}(\mu_{i,c})\in\{-1,0,+1\}.
\]

Values with magnitude at or below `--dipole-zero-tolerance` receive polarity
zero and remain in the population. Only centers without a valid finite dipole
are excluded. In every spatial bin, the reported value is the plain arithmetic
mean,

\[
\overline{s}=\frac{1}{N}\sum_i s_i
=\frac{N_+-N_-}{N_++N_0+N_-},
\]

which ranges from -1 (all negative) through 0 (equal populations) to +1 (all
positive). This operation does not weight by dipole magnitude or local volume.
The CSV tables also contain positive, zero, and negative counts and fractions.

`--projection-plane xz --projection-bins 40 40 --plot-2d` bins in x and z and
projects through y. The command constructs the bin edges at `--reference-frame`
(frame 0 by default), assigns every center atom to one bin by atom ID, and keeps
that assignment fixed in every later frame even if the atom crosses a spatial
boundary.

`--plot-kymograph --profile-axis z` writes a **kymograph**: source frame is on
the horizontal axis, reference-frame z position is on the vertical axis, and
mean polarity is the color. A kymograph is the general name used here for any
position-versus-frame heatmap and can be adopted by other workflows. The
profile reuses the matching projection bins. For `--projection-plane xz
--projection-bins 1 10 --profile-axis z`, the kymograph therefore has 10 z
bins. `--plot-evolution` remains an alias for `--plot-kymograph`.

This projected command streams selected frames and calculates only the chosen
center-dipole component. It does not build the full per-ion basal-plane table.
Use `--workers N` for bounded frame-level parallelism and `--chunk-size N` to
limit queued frames. Detailed center rows are off by default; request them
with `--write-centers`. They use Parquet by default, or CSV with
`--centers-format csv`.

```powershell
reaxkit get-basal-plane-displacement-projected-polarity `
  --periodic xy `
  --charge-source formal `
  --formal-charge Al=3 N=-3 `
  --frames 0:3200:50 `
  --reference-frame 0 `
  --component z `
  --projection-plane xz `
  --projection-bins 40 40 `
  --profile-axis z `
  --workers 4 `
  --chunk-size 16 `
  --plot-2d `
  --plot-kymograph
```
This option does not add charge or charge-difference atom properties.

Example:

```powershell
reaxkit get-basal-plane-displacement-local-polarization `
  --periodic xy `
  --charge-source formal `
  --formal-charge Al=3 N=-3 `
  --local-volume-method equal `
  --volume-method hull `
  --plot-2d --plot-plane xy --plot-component z `
  --plot-3d --write-extxyz `
  --include-electric-field --field-direction z
```

For a slab, `cell` includes vacuum and therefore lowers the reported material
polarization. Use `hull` (the default) or `bbox` when the intended denominator
is the occupied material volume. Use `cell` only when polarization per complete
simulation-cell volume, including vacuum, is desired.

### Difference from the other wurtzite methods

- `four_folded_wurtzite` requires a four-neighbor coordination and derives its
  local charge-weighted vector from center-to-neighbor bond vectors. Its
  structural polarity uses the apical-minus-basal displacement.
- `three_folded_wurtzite` determines polarity from the negative mean of three
  basal bond projections, then selects an optional apical neighbor on the side
  required by that polarity. Its existing dipole sums charge-weighted
  center-to-neighbor bond vectors for the basal neighbors and selected apical.
- `basal_plane_displacement_for_dipole_moment` reuses the three-folded basal
  selection but changes the dipole origin. The mean basal position becomes the
  local zero-displacement plane for each Al/B center. The N framework is fixed;
  the apical bond height is diagnostic and is not added to the dipole.

The polarization workflow bins and sums the one-row-per-ion contributions in
user-selected x, y, and z bins and divides by `hull`, `bbox`, or `cell` bin
volumes. It also supports 2D heatmaps with shared global scaling or independent
per-frame scaling.

## `get-hbn-reference-polarization`

This method calculates vector polarization from the displacement of a
trajectory relative to a nonpolar, layered hexagonal AlN structure. The
reference is bundled as
`hbn_reference/AlN_hbn.cif`, so an installed ReaxKit package does not depend on
the repository's `examples_to_test` directory. A different CIF can be selected
with `--reference`.

The calculation follows the displacement expression used by Hayden et al.,
with the selected formal or ReaxFF charge and the electron-charge sign:

\[
\mathbf{P}=\frac{|e|}{\Omega}\sum_{k=1}^{N}
\left(-q_k\Delta\mathbf{u}_k\right).
\]

The output contains Cartesian x, y, and z components. The c component is the
projection onto the normalized direction selected by `--c-axis`.

### Which atoms contribute

**Every trajectory atom contributes to the total dipole.** For each selected
frame, the implementation calculates

\[
\boldsymbol{\mu}_k=-q_k\Delta\mathbf{u}_k,
\qquad
\boldsymbol{\mu}=\sum_{k=1}^{N}\boldsymbol{\mu}_k.
\]

Thus Al and N both contribute, and B contributes when it is present and has a
formal charge supplied with `--formal-charge B=...`, or when ReaxFF charges are
selected. Pass `--write-displacements` to write the per-atom dipole components
to `hbn_reference_displacements.csv`; this large table is disabled by default.
The x, y, z, and c-projected dipole and polarization components in
`hbn_reference_polarization.csv` are their sums.
`hbn_reference_polarization_poled_counts.csv` reports the sign of each total
Cartesian component for every frame with its iteration and time.
With equal and
opposite cation and anion charges, this all-ion sum is equivalent to a
relative cation-anion sublattice displacement, and a common rigid translation
cancels by charge neutrality.

`--reference-species B=Al` affects site matching only: it permits a B atom in
the trajectory to occupy an Al site in the AlN reference. It does not replace
B by Al, assign B a charge, or cause only B/Al atoms to enter the dipole sum.
For an Al/N-only trajectory this option is unnecessary.

### Reference construction and matching

The reference preparation proceeds as follows:

1. Read the bundled CIF or the file selected by `--reference`.
2. Apply ReaxKit's hexagonal-to-orthogonal transformation when
   `--orthogonalize-reference` is present. Use
   `--no-orthogonalize-reference` to retain the CIF cell; if neither is given,
   the trajectory and reference cell angles select the closer representation.
3. Repeat the oriented structure by the required explicit counts from
   `--replication NX NY NZ`. The resulting reference atom count must equal the
   trajectory atom count.
4. Align the reference lattice-vector directions with the trajectory cell.
   Length mismatches no larger than `--max-reference-strain` (0.15 by default)
   are treated as homogeneous material strain. Larger mismatches are preserved
   as empty space, which prevents slab vacuum from stretching the reference.
5. Remove a common periodic translation so a change of cell origin does not
   appear as ionic displacement.

ReaxFF normally preserves GEO atom order. The fast path accepts that identity
mapping only when the resulting reference alignment has an RMS displacement no
larger than 2 angstrom. If that validation fails, a species-restricted periodic
KD-tree search constructs a one-to-one spatial assignment. Mapping details are
written to `hbn_reference_mapping.csv`, and the replicated, aligned structure
is written to `AlN_hbn_replicated_aligned.xyz`.

The frame-zero reference fractional coordinates are carried into every later
instantaneous cell. A common translation is refitted in every frame, followed
by minimum-image displacement along the axes selected by `--periodic`.

### Polarization volume for slabs

For a bulk periodic cell, the material volume and simulation-cell volume are
the same. For a slab, dividing by the full cell volume includes vacuum and
dilutes a bulk-like polarization. `--volume-method` selects the denominator in
the same way as the other ferroelectric polarization workflows:

- `hull` (default): volume of the three-dimensional convex hull of all finite
  atom coordinates in the frame;
- `bbox`: product of the occupied x, y, and z coordinate extents;
- `cell`: determinant of the complete simulation cell, including vacuum.

The selected value is written as `volume (angstrom^3)`, its name is written as
`volume_method`, and the full box volume remains available as
`simulation_cell_volume (angstrom^3)` for comparison. Hull and bounding-box
methods exclude empty slab vacuum automatically. If fewer than four finite
points are available or the coordinates are geometrically degenerate, hull
volume and polarization are reported as `NaN`.

Both occupied-coordinate methods define the surface at the outermost atomic
centers. A three-dimensional slab polarization therefore still depends on a
thickness convention; `hull`, `bbox`, and `cell` make that choice explicit.
The total dipole per in-plane area is the corresponding thickness-independent
slab quantity.

### Charge source and interpretation

`--charge-source auto` uses ReaxFF partial charges from `fort.7` when available
and otherwise uses formal charges. `--charge-source reaxff` requires `fort.7`;
`--charge-source formal` uses `--formal-charge` values. ReaxKit supplies the
formal defaults `Al=3` and `N=-3`. All trajectory elements need a configured
formal value when formal charges are selected. Charges should sum closely to
zero over the complete cell so that the dipole is independent of origin.

This displacement calculation is a linearized approximation to polarization
relative to the selected nonpolar reference. It is not a Berry-phase
calculation, and it does not select a different polarization branch. Results
can consequently differ from first-principles values because formal or ReaxFF
charges replace Born effective charges, and because of finite-temperature or
strained coordinates, surfaces, defects, and nonlinear electronic
contributions.

For the 19 by 19 by 10 orthogonal reference used in the example trajectory:

```powershell
reaxkit get-hbn-reference-polarization `
  --engine reaxff `
  --input . `
  --xmolout .\xmolout `
  --charge-source formal `
  --formal-charge Al=3 N=-3 `
  --replication 19 19 10 `
  --orthogonalize-reference `
  --volume-method hull `
  --output-dir .\hbn_polarization `
  --frames 0
```

### Local dipole and polarization

`get-hbn-reference-local-polarization` partitions the replicated reference into
its stoichiometric primitive cells. For the bundled structure, each local cell
contains two Al sites and two N sites. Membership is fixed by the reference
construction and atom-to-reference-site mapping. It is not rebuilt from the
instantaneous basal or apical coordination, so an apical-neighbor switch does
not move an atom discontinuously between local cells. The switch instead
appears through the sign and magnitude of the mapped atomic displacements.

For reference cell \(g\), the raw local dipole is

\[
\boldsymbol{\mu}^{\mathrm{raw}}_g=
\sum_{i\in g}-q_i\Delta\mathbf{u}_i.
\]

Raw local dipoles sum exactly to the global dipole. With the default formal
charges, every Al2N2 cell is neutral because
\(2(+3)+2(-3)=0\). Instantaneous ReaxFF charges need not sum to zero inside
each cell, even when the complete simulation is neutral. A charged local group
has an origin-dependent dipole. The default
`--local-charge-treatment auto` therefore replaces its charges by

\[
q_i^{\mathrm{local}}=q_i-\frac{Q_g}{N_g},\qquad
Q_g=\sum_{i\in g}q_i,
\]

when \(|Q_g|>10^{-10}\ e\). This makes the effective local charge exactly zero
and prevents a rigid translation of a charged cell from creating local
polarization. `raw_dipole_*` columns retain the exact decomposition of the
global result; `neutralized_dipole_*` columns retain the neutral result; and
`dipole_*` plus `P_*` contain the value selected by
`--local-charge-treatment`. Use `raw` to force exact global decomposition or
`neutralize` to force neutralization. Formal Al=3, N=-3 results are identical
under all three choices.

Local polarization divides each selected local dipole by a local volume:

- `--local-volume-method equal` (default) divides the selected frame volume
  equally among all reference cells;
- `--local-volume-method deformation` fits a local affine deformation from
  neighboring reference-cell centers and uses the absolute determinant as a
  relative volume weight. The weights are normalized so that all local volumes
  add to the selected frame volume. Rank-deficient local fits fall back to the
  median valid weight, or an equal weight if none is valid.

`hbn_reference_local_polarization_poled_counts.csv` counts positive and
negative local x, y, and z polarization values for the selected cell or layer
grouping. It includes frame, iteration, time, total count, and percentages.

The frame volume still comes from `--volume-method`. Its default is `hull`, so
empty slab vacuum is excluded before the volume is split among cells. Select
`cell` only when polarization should use the complete simulation box including
vacuum.

```powershell
reaxkit get-hbn-reference-local-polarization `
  --replication 19 19 10 `
  --periodic xy `
  --charge-source reaxff `
  --volume-method hull `
  --local-volume-method equal `
  --plot-2d --plot-plane xz --plot-bins 1 40 `
  --plot-3d `
  --plot-quantity polarization --plot-component c `
  --global-scaling `
  --write-extxyz
```

The command writes `hbn_reference_local_polarization.csv`, a frame summary that
reports raw/global closure, and the source displacement and mapping tables. In
2D polarization plots, dipoles and local volumes are summed along the omitted
coordinate before division. Dipole plots sum dipoles directly. The 3D plot
colors each reference-cell center.

`--write-extxyz` retains every trajectory atom. Every atom receives
`local_cell_id`; one representative cation per cell receives
`is_local_cell_center=1`, `local_volume`, `local_dipole`,
`local_polarization`, `local_dipole_c`, and `local_polarization_c`. These local
values are `NaN` on the other atoms so OVITO does not display duplicate vectors
for a single cell.

### Local polarity projections and kymographs

`get-hbn-reference-projected-polarity` converts the selected local dipole
component to

\[
s_g=\operatorname{sign}(\mu_{g,\alpha})\in\{-1,0,+1\}.
\]

Zero dipoles remain in the arithmetic mean. A bin value therefore measures the
signed population fraction: +1 means all cells are positive, -1 means all are
negative, and 0 can mean all are nonpolar or an equal positive/negative
population. The projected-bin CSV tables report positive, negative, and zero
counts and fractions so those cases remain distinguishable. The dedicated
`hbn_reference_projected_polarity_whole_slab_summary.csv` table aggregates all
spatial bins into one row per selected frame and reports counts, fractions, and
0--100 percentages for the complete slab.

Cell-to-bin membership is calculated once from the fixed reference-cell
centers associated with `--reference-frame` and reused for every selected
frame. Atomic motion cannot move a cell between bins. `--plot-2d` writes one
projection-plane heatmap per frame. `--plot-kymograph` (also available as
`--plot-evolution`) averages over the other projection coordinate and writes
mean polarity versus source frame and the selected `--profile-axis`.
The profile uses the corresponding count from `--projection-bins`; for
`--projection-plane xz --projection-bins 1 40 --profile-axis z`, it has 40 z
bins.

The command streams selected ReaxFF frames, aggregates only the requested
cell or layer grouping with NumPy, and keeps frame work in bounded chunks.
`--workers N` processes up to N frames concurrently; set BLAS/OpenMP thread
counts to 1 when using multiple workers. The detailed per-group centers table
is off by default. Request it with `--write-centers`; it is written as Parquet
by default, or as CSV with `--centers-format csv`.

```powershell
reaxkit get-hbn-reference-projected-polarity `
  --replication 19 19 10 `
  --periodic xy `
  --frames 0:3200:50 `
  --reference-frame 0 `
  --charge-source reaxff `
  --local-charge-treatment auto `
  --component c `
  --projection-plane xz `
  --projection-bins 1 40 `
  --profile-axis z `
  --workers 4 `
  --chunk-size 16 `
  --plot-kymograph
```
