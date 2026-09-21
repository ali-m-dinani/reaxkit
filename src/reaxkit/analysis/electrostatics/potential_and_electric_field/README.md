# ReaxFF potential and local electric field

This package evaluates the shielded ReaxFF Coulomb kernel at every selected atom coordinate. The atom occupying the target coordinate is excluded as a source. A hypothetical `+1 e` probe supplies `gamma_i`; trajectory atoms retain their saved charges and `gamma_j` values. Results are written for each requested probe species and as an equal arithmetic average across probes.

The default electric field is the analytic negative gradient of the shielded and tapered potential. Potential is reported in volts. Electric field is reported in both `V/angstrom` and `MV/cm`, using `1 V/angstrom = 100 MV/cm`.

```powershell
reaxkit get-potential-and-electric-field `
  --run-dir . --xmolout xmolout --fort7 fort.7 --ffield ffield `
  --periodic xyz --frames ::20 `
  --bin-axes z --bins 50 --plot-bins
```

The output contains:

- `coulomb_per_atom.csv` and `coulomb_totals.csv`
- `voltages_and_electric_fields/` with one per-atom CSV per probe and an average CSV
- `binned_data_and_plots/` with binned CSVs and one plot per selected frame and probe

Use `write-trajectory-with-potential-and-electric-field` to write the same local values as Extended XYZ atom properties.
