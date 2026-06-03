# data

## Purpose
Contains packaged static YAML data used by runtime and analysis helpers.

## What Belongs Here
- Global constants and units metadata.
- Alias maps for tolerant variable/column resolution.

## What Does Not Belong Here
- User runtime outputs.
- Analyzer- or workflow-specific metadata (keep those under owning package `data/` folders).

## Structure
- `constants.yaml`
- `units.yaml`
- `variable_aliases.yaml`

## Flow
Loaded by core resolve/platform utilities at runtime through package resources.

## Extension Points
- Add new global metadata only when used across multiple layers/packages.
