# ReaxFF internal, external, and total local electrostatics

This package evaluates the internal shielded ReaxFF Coulomb kernel at every selected atom coordinate. The atom occupying the target coordinate is excluded as a source. A hypothetical `+1 e` probe supplies `gamma_i`; trajectory atoms retain their saved charges and `gamma_j` values. The iteration-aligned applied field is read from `fort.78`. At position `r`, the external potential is `-E_external dot (r - r_reference)`, and total local values are the sums of their internal and external contributions.

The default reference is the center of the occupied atomic bounding box in each frame. Use `--potential-reference-mode fixed-midpoint` to retain the first selected frame's midpoint as the trajectory expands or moves, or `--potential-reference-position X Y Z` to provide a fixed Cartesian reference explicitly.

The default electric field is the analytic negative gradient of the shielded and tapered potential. Potential is reported in volts. Electric field is reported in both `V/angstrom` and `MV/cm`, using `1 V/angstrom = 100 MV/cm`.

```powershell
reaxkit get-potential-and-electric-field `
  --run-dir . --xmolout xmolout --fort7 fort.7 --fort78 fort.78 --ffield ffield `
  --periodic xyz --frames ::20 `
  --bin-axes z --bins 50 --plot-bins --plot-kymograph
```

The output contains:

- `coulomb_totals.csv`
- Only with `--export-per-atom-csv`: `coulomb_per_atom.csv` and `voltages_and_electric_fields/` with one per-atom CSV per probe and an average CSV
- `binned_data_and_plots/` with binned CSVs and one plot per selected frame and probe
- `binned_data_and_plots/kymographs/` with frame- or iteration-versus-position voltage and field maps

Each probe CSV contains internal, external, and total local potential and electric-field columns. Use `write-trajectory-with-potential-and-electric-field` to write the same quantities as Extended XYZ atom properties.

Both commands skip per-atom CSV export by default. This does not disable checkpoint tables, requested binning/kymographs, or Extended XYZ output. Existing CSVs from earlier runs are not deleted. The trajectory command currently recalculates the electrostatics; it does not reuse another command's results. In particular, results calculated with `--periodic xy` cannot substitute for `--periodic xyz` calculations.

Add `--write-extxyz` to `get-potential-and-electric-field` to also write
`trajectory_with_local_electrostatics.extxyz` in its analysis/output directory.
This uses the existing trajectory task: each selected frame is calculated once,
and its values are reused for the trajectory, tables, bins, and kymographs.
With checkpointing, complete XYZ frame segments are committed alongside the
scientific tables and published atomically; resume retains committed segments
without duplicating frames. Without checkpointing, the normal incremental XYZ
writer is used. Trajectory serialization still takes time and disk space.
Resume with the same `--write-extxyz` setting; a checkpoint created without this
flag is not compatible because it does not contain the trajectory segments.

Kymographs require one bin axis (`x`, `y`, or `z`). By default they show total local potential and total local field magnitude versus source frame. Use `--kymograph-values` to select internal, external, or total-local potential/field layers, `--plot-field-component` to select `x`, `y`, `z`, magnitude, or mean magnitude for field layers, and `--kymograph-time-axis iteration` to use iteration on the horizontal axis.
