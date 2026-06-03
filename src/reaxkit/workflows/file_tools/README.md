# workflows/file_tools

## Purpose
Generator/file-transformation workflows for common ReaxFF/LAMMPS/utility file operations.

## What Belongs Here
- Workflows that generate, transform, or repair input/output files.

## What Does Not Belong Here
- Scientific analyzer computations.
- Core runtime registry internals.

## Structure
- Modules for `control`, `geo`, `ffield`, `fort7`, `fort83`, `trainset`, `xmolout`, and related templates/helpers.

## Flow
Called via generator commands; they normalize args, run file operation logic, and persist run metadata/outputs.

## Extension Points
- Add command workflow modules for new file tool operations and register generator routes.
