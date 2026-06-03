# ROMP Active-Control Example

This example demonstrates a config-driven controller for a ROMP simulation
step. It reads the current ReaxFF geometry/connectivity state, applies a
small active-control edit, and writes files that can be used as the next
simulation input.

The controller is useful as a template for simulation loops where each step
uses analysis results to decide how the next structure should be modified.
In this example, the edit removes selected molecular fragments and inserts a
monomer near a chosen anchor atom while enforcing a minimum-distance check.

## What This Example Does

1. Reads `fort.90` with ReaxKit's `GeoHandler`.
2. Detects connected molecular components from the parsed connectivity table.
3. Selects molecules for removal by exact formula or generalized rules.
4. Removes the selected atoms from the working structure.
5. Inserts one or more monomers near random anchor atoms.
6. Rejects insertion points that overlap existing atoms.
7. Writes a next-step ReaxFF `geo` file.
8. Writes JSON and text reports describing the edit.

This is intentionally written as an example controller rather than a single
core API call. It shows how ReaxKit parsers can be composed with
controller-specific edit operations to drive simulation-state changes.

## Files

- `data/fort.90`: example ReaxFF/XTLGRF geometry and connectivity input.
- `data/monomer.bgf`: monomer structure inserted by the controller.
- `config.yaml`: input, output, removal, insertion, and box settings.
- `run_active_step.py`: controller script for one active-control step.
- `romp_edit_ops.py`: reusable edit operations used by the controller.
- `README.md`: this guide.

## Generated Files

Running the example writes files under `generated/`:

- `geo_next`: next-step ReaxFF `geo` structure.
- `active_step_report.json`: structured report with atom counts, formulas, box settings, and insertion diagnostics.
- `active_step_report.txt`: compact human-readable report.

Generated files are outputs of the controller. They normally should not be
committed unless you want reference fixtures.

## Run

From the repository root:

```powershell
python docs/examples/romp_active_control/run_active_step.py `
  --config docs/examples/romp_active_control/config.yaml
```

The `generated/` directory is created automatically.

## Configuration

The example is controlled by `config.yaml`.

Important sections:

- `input`: paths to the current `fort.90` state and the monomer file.
- `output`: paths for the generated next-step `geo` file and reports.
- `removal`: molecule deletion settings.
- `insertion`: monomer placement settings.
- `box`: output cell lengths and angles.

## Removal Modes

`removal.mode` supports:

- `formula`: remove molecules whose formula exactly matches `target_formulas`.
- `rules`: remove molecules using generalized element and motif filters.

`removal.selection_strategy` supports:

- `first`: deterministic first matches.
- `random`: deterministic random selection when `insertion.seed` is set.

Rule fields in `removal.rules`:

- `include_any_elements`: molecule must contain at least one listed element.
- `include_all_elements`: molecule must contain all listed elements.
- `exclude_any_elements`: molecule must not contain any listed element.
- `allowed_elements`: molecule element set must be a subset of this list.
- `include_motifs`: at least one formula-like motif must match.
- `exclude_motifs`: no listed formula-like motif may match.

With `mode: rules`, if every rule list is empty, no molecules are removed.

Motif syntax is formula-like:

- `S1F6`: exactly one `S` and exactly six `F`.
- `SFx`: at least one `S` and at least one `F`.
- `Li1S1F1`: exact counts for all listed elements.

Example rule-based removal:

```yaml
removal:
  enabled: true
  mode: rules
  selection_strategy: first
  max_molecules: null
  rules:
    include_any_elements: [S]
    exclude_any_elements: [Li]
    include_motifs: [SFx]
```

## Insertion Settings

The insertion step places the monomer relative to randomly selected anchor
atoms:

- `place_count`: number of monomers to insert.
- `anchor_element`: atom type used as the placement anchor.
- `nearest_element`: nearby atom type used for local context selection.
- `search_radius`: radius around the anchor for candidate placement points.
- `min_distance`: minimum allowed distance between inserted and existing atoms.
- `candidate_samples`: number of random candidate points to test per insertion.
- `seed`: random seed for reproducible placement.

If no valid non-overlapping position is found, insertion is reported as partial
or failed instead of silently accepting an overlapping structure.

## Notes

- The example writes a new ReaxFF/XTLGRF `geo` file directly from the edited coordinate table.
- The output report records before/after atom counts, removed molecules, insertion status, and observed minimum insertion distance.
- The controller can be adapted into a larger simulation loop by running it after an analysis step and using `generated/geo_next` as the next input geometry.
