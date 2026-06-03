# Li/S Box Pipeline Example

This example demonstrates a small simulation-controller pipeline for preparing
a lithium/sulfur (`Li/S`) packed structure from a ReaxFF-style `fort.90`
geometry/connectivity file. It shows how users can use ReaxKit's codes for parsing,
analysis, etc., and then make their own custom code for specific tasks like random packing.

The controller reads molecular connectivity, derives the sulfur inventory,
computes a target Li/S composition, randomly packs `S2`, atomic `S`, and `Li`
species into a cubic box, and writes new structure files that can be inspected
or passed into later simulation steps.

## What This Example Does

1. Reads `fort.90` using ReaxKit's ReaxFF geometry parser.
2. Finds connected molecular fragments and filters sulfur-containing species.
3. Counts sulfur atoms after excluding bare `SFx` fragments.
4. Computes the required Li count and Li/S box size from density and composition settings.
5. Randomly places `S2`, `S`, and `Li` fragments with a minimum-distance tolerance.
6. Writes a packed XYZ structure.
7. Writes representative XYZ files for each detected Li/S/F isomer group.

This is intentionally written as an example controller rather than a core
library workflow. It coordinates several reusable ReaxKit modules and produces
new files from an input structure.

## Files

- `fort.90`: example ReaxFF/XTLGRF input geometry and connectivity file.
- `prepare_lis_box.py`: Li/S composition math and random in-memory packing.
- `run_fort90_to_xmolout.py`: controller script that parses `fort.90`, extracts isomer representatives, calls the Li/S box preparation step, and writes generated XYZ files.
- `README.md`: this guide.

## Generated Files

Running the example creates files in the directory passed to `--output` and
`--isomer-xyz-dir`.

Typical generated files are:

- `output.xyz`: final packed Li/S structure.
- `F1Li1S1_1.xyz`, `F2Li1S1_1.xyz`, etc.: representative isomer structures extracted from `fort.90`.

These files are generated artifacts. They do not need to be committed unless
you intentionally want them as fixtures or reference outputs.

## Run

From the repository root:

```powershell
python docs/examples/lifs_box_pipeline/run_fort90_to_xmolout.py `
  --fort90 docs/examples/lifs_box_pipeline/fort.90 `
  --output docs/examples/lifs_box_pipeline/generated/output.xyz `
  --density 0.007645 `
  --xS 0.12 `
  --tolerance 4.0 `
  --seed 42 `
  --max-attempts 1000 `
  --li-per-s 2.0 `
  --mass-s 32.0 `
  --mass-li 7.0 `
  --s2-bond-length 1.9 `
  --isomer-xyz-dir docs/examples/lifs_box_pipeline/generated/isomers
```

The `generated/` directory is created automatically.

## Parameters

- `--fort90`: input `fort.90` file containing atom records and connectivity.
- `--output`: packed XYZ structure written by the controller.
- `--isomer-xyz-dir`: directory for representative isomer XYZ files.
- `--density`: target density used to size the cubic box.
- `--xS`: monoatomic sulfur fraction used when splitting sulfur into `S2` and `S`.
- `--tolerance`: minimum allowed atom-atom distance during random placement.
- `--seed`: random seed for reproducible placement.
- `--max-attempts`: maximum placement attempts per fragment before failing.
- `--li-per-s`: Li:S stoichiometric multiplier.
- `--mass-s`: sulfur atomic mass used for density sizing.
- `--mass-li`: lithium atomic mass used for density sizing.
- `--s2-bond-length`: bond length used for generated `S2` fragments.

## Notes

- Sulfur counting keeps sulfur-containing molecules and excludes bare `SFx`
  species where the element set is `{S, F}` and the fragment has exactly one
  sulfur atom.
- Isomer filenames use the formula plus an isomer index. For example,
  `F1Li1S1_1.xyz` is isomer `1` for formula `F1Li1S1`.
- The script name includes `xmolout` for historical reasons, but this example
  currently writes XYZ files.
