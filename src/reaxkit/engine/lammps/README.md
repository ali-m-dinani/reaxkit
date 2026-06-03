# engine/lammps

## Purpose
LAMMPS-specific adapter and handler implementations.

## What Belongs Here
- LAMMPS adapter detection/load logic.
- Dump/log file handlers.

## What Does Not Belong Here
- ReaxFF or AMS parsing behavior.

## Structure
- `adapter.py`
- `dump_handler.py`
- `lammps_log_handler.py`

## Flow
Selected by engine resolver for LAMMPS inputs and converts them into typed domain data.

## Extension Points
- Add handler support for additional LAMMPS file variants.
