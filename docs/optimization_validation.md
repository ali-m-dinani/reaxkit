# Optimization support and validation

This tracked document describes general contracts and limitations. Detailed
project-specific investigation notes belong in the optional, gitignored local
file `src/reaxkit/optimization/AUDIT.md`. That file is not required to build,
test, install or run ReaxKit. Do not publish datasets, native result directories,
personal paths or scheduler job identifiers as validation fixtures.

## Integration boundary

ReaxKit orchestrates separate programs through commands and files; it does not
import or bundle optimizer or engine implementations. Each component retains its
own licensing requirements. A repository-level license does not override more
restrictive notices in bundled engine files. Users must supply appropriately
licensed engines and compatible isolated environments.

Adapter contracts in `src/reaxkit/optimization/adapters/*/adapter.yaml` define
available settings, defaults, seed support and restart semantics. Use
`reaxkit optimize-ffield --list-optimizers` to inspect the installed catalog.
See [the user guide](ffield_optimization.md) for launch and collection commands.

## Evidence levels

- Configuration validation checks the YAML contract, not scientific correctness.
- Dry-run preparation checks staging, not executable or cluster compatibility.
- Synthetic converter tests check parsing and provenance, not engine behavior.
- A completed job and zero exit status are necessary but insufficient evidence
  of a scientifically valid optimization.
- Engine smoke tests must verify actual evaluations, native/normalized scores,
  parameter identity and usable force-field artifacts before longer runs.
- Formal cluster acceptance additionally requires the documented replica,
  interruption, collection and resubmission checks. A successful individual
  smoke test does not automatically change an adapter's cluster-validation flag.

## Scientific limits

Progress steps may represent iterations, generations, evaluations or checkpoints;
they must not be treated as interchangeable. Best-so-far checkpoints do not
recover every candidate evaluation. Prediction comparisons require verified
candidate identities, target definitions, units, weights and dataset coverage.
Source-local residual arrays must not be joined across optimizers by position.

Parameter-only force fields are not exact optimizer restart states. Requested
seeds, forwarded seeds and demonstrated reproducibility are separate concepts.
Time/evaluation limits are not evidence of scientific convergence. Engines and
dataset-specific objectives may differ, so raw losses are not automatically
comparable across adapters.

See [GloMPO](GLOMPO.md) and [Flocky](FLOCKY.md) for their integration gates.
Flocky's native build and engine execution require verification; parser support
for saved upstream output does not demonstrate a new engine run.

## Reproducible checks

From an installed ReaxKit checkout, run:

```bash
python -m pytest tests/optimization -q -rs
```

Review skips explicitly: checks requiring private local samples or a particular
platform are not evidence of validation when skipped. Keep private evidence
local and publish only reusable tests, format contracts and general limitations.
