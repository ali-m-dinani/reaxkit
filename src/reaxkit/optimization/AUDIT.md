# Optimizer contracts and native launch audit

Audit date: 2026-11-05. Source inventory:
`C:/Users/alimo/PycharmProjects/REAXFF_OPTIMIZERS.md`.
Revision identifiers below describe local checkouts, including observed local
changes; they are not claims of upstream release compatibility.

No optimizer code is imported into ReaxKit, vendored or shipped with it. ReaxKit's new code remains
under its existing MIT license. This inventory records observed license files;
it does not resolve redistribution rights for projects with missing notices.
An external program may generate its own runtime bundle (the OGolem bridge copies
its scripts into its bundle); those artifacts retain that program's license
obligations and are not relicensed as ReaxKit code.

| Adapter | Source/revision | License evidence | Runtime and input requirements |
| --- | --- | --- | --- |
| ams_mcff | SCM OldReaxFF docs; iopt=4 | User-provided licensed installation | SCM standalone executable, common training inputs, control, min/max params; native replic is inner trials |
| ams_cmaes | SCM OldReaxFF docs; iopt=7 | User-provided licensed installation | Same executable; sigma/min/max params, optional cmaes restart |
| standalone_soppi | PyReaxFF/reaxff_fortran_source_code/ffopt.f; hash in adapter.yaml | Source headers explicitly request no redistribution | User-built Fortran executable, explicit unit mapping, max/min bounds, explicit accerr; no verified build here |
| jax_soppi | JAX-SOPPI 061b9e8 | No top-level license found | Python >=3.10; JAX 0.4.26–0.4.30, NumPy 1.x; separate JAX-ReaxFF/JAX-MD source paths |
| jax_reaxff | JAX-ReaxFF 144ee1c | GPL-3.0 text in LICENSE | Compatible JAX/JAX-MD environment; native driver CLI; parameters alone do not resume optimizer state |
| indeedopt | INDEEDopt 3112946 | MIT LICENSE | Native ReaxFF plus prepared folder/ and params-ml; full workflow has dataset-specific script constants; explicit existing short JAX mode |
| ogolem | ogolem a66dae3 | BSD-style LICENSE including advertising acknowledgement | Java build plus Python >=3.10 bridge; prepared fit.ogo/evaluator bundle; external engine |
| pibo | pibo_mos2 1ebd482 | MIT declared in pyproject.toml; no top-level LICENSE found | Python >=3.11, NumPy >=2.1,<3, separate ReaxFF LAMMPS executable; dataset-specific MoS2 or cobalt driver |
| ezff | EZFF 6d39f8e | MIT LICENSE | Python script API; NumPy 1.26.4 in setup.py, selected algorithm libraries, optional MPI; user-defined objective/engine |

Read all applicable notices before any future redistribution. OGolem's license
contains an advertising acknowledgement requirement; do not call it plain MIT or
assume a generic three-clause BSD license. JAX-SOPPI licensing needs resolution
before redistribution. Legacy Fortran headers in ffopt.f, reac.f and version.f
explicitly request no redistribution; the absence of a separate LICENSE must
not be interpreted as permission.

## Inspected formats and limits

| Adapter | Evidence inspected | Converter coverage | Remaining uncertainty |
| --- | --- | --- | --- |
| ams_mcff | Official results documentation | Explicit user-mapped whitespace columns; field/report artifacts | No local AMS 2023.103 log sample; no guessed default columns or fort.99 parity |
| ams_cmaes | Official results documentation | Explicit mapped progress columns; best field and resume artifact | Native progress layout and fort.99 require version-specific examples |
| standalone_soppi | ffopt.f ffsav, lines 1301–1314 | Error force field markers from unit 83 | Runtime unit filename must be configured; no compiled-run sample; final field may be incomplete |
| jax_soppi | soppi/checkpoint.py, search.py; outputs/smoke_full/history.jsonl | Row/current/best history; candidate residuals; checkpoint artifacts | Residuals have source-local array identities, not structure labels; engine-call counts not inferred |
| jax_reaxff | driver.py, optimizer.py, helper.py; ffields/report_1_1326.txt | Rounded stdout loss, labeled report parity, field artifacts; optional evaluation snapshot | Machine-report items omit identities and are not positionally joined to another file |
| indeedopt | main-parallel.py, bruteforce2.py, data_extract3.py; short_run/short_results/results.csv | Short sample losses, full-mode best NPZ snapshots, best artifacts | Generic full dl-data tables do not encode a universal scalar fitness history |
| ogolem | tools/reaxff_bridge.py; converted/cobalt_jax archive/progress | Completed candidate archives, fitness, available JAX reports, best fields | Older archives omit completion metadata; prediction-only responses need original target manifest |
| pibo | benchmark.py, staged_bo.py, cobalt_pibo.py; results/cobalt_quick_pibo/history.csv | Cobalt evaluation/penalty rows; MoS2 NPZ/JSON best snapshots | Best snapshots do not recover missing history; benchmark RMSE is a separate post-hoc metric |
| ezff | ezff/__init__.py _write_out_forcefields/save_evaluated; example smoke results/1/errors | Epoch recommendation vectors, latest cumulative all_evaluated_TIMESTAMP.npz, and force-field artifacts | Archive counts may include loaded history; objective names/units are user-defined |

Fixtures in ReaxKit are synthetic, based on these format contracts, without
copying restricted datasets, optimizer code, or proprietary outputs. Real local
reports are used for read-only validation where available. All adapters remain
**cluster-unvalidated**. AMS mapped-column support is contract-tested only;
Fortran parsing is source-format-tested only.

The JAX-SOPPI input-conversion tool was reviewed. It sorts bounds and can generate
OGolem/EZFF/PIBO bundles with JAX evaluation. Calling it wholesale would change
some engines/objectives, contrary to this phase's scope. ReaxKit therefore does
not run it automatically. Prepared bundles must retain the user's intended engine,
objective, and dataset coverage.

## Settings mapping

Each adapter's `adapter.yaml` records its exposed setting types, defaults,
native setting family, seed support, and checkpoint semantics.
`--list-optimizers` exposes this catalog and the generated settings template
contains every adapter. These settings are passed to the explicit external runner
contract, or mapped by the new per-adapter launch modules where supported.

Global paths map to JAX-SOPPI's --ffield/--params/--trainset/--geo and JAX-ReaxFF's
--init_FF/--params/--train_file/--geo. Native output directories are owned by each
run, not shared globally. Global seed maps to JAX-ReaxFF --seed when its external
runner supports that mapping. Native mode forwards the documented seed flag;
worker metadata distinguishes forwarding from verified scientific application.
JAX-SOPPI and legacy SOPPI have no RNG flag in the inspected interfaces.
OGolem bridge --seed includes the initial vector and is exposed as
`include_initial`, never as an RNG seed.

INDEEDopt engine/short_samples/jax_python/jaxreaxff_source/jaxmd_source correspond
to INDEEDOPT_ENGINE, INDEEDOPT_SHORT_SAMPLES, INDEEDOPT_JAX_PYTHON,
INDEEDOPT_JAXREAXFF_SOURCE, and INDEEDOPT_JAXMD_SOURCE. nos, nos_2, lm_iter, bn,
and num_prop are script constants; a separately maintained runner must implement
configuration for them. The original script also rewrites bounds using gpf=5,
so automatic native mode is rejected rather than silently changing the search
space. ReaxKit never edits the optimizer's source.

PIBO internal replicates must equal one and its internal optimizer list must have
one member. The global job grid owns replicas. Its current MoS2 force-weight
setting has no implemented force term, so nonzero w_forces is rejected. The cobalt
objective is explicitly different. Cross-method numerical loss rankings are not
inferred from similar names.

EZFF's settings map to FFParam, read_variable_bounds, read_forcefield_template,
set_algorithm, parameterize, generate_pool, and normalize_errors. The external
script owns those calls and the objective callback. No direct EZFF import occurs.

## User job-file evidence

The supplied Roar script loads AMS/Intel/MPI modules and sets NSCM, OpenMP,
SCM_USE_THREADS, and OMP_STACKSIZE before invoking the engine. These are user job
configuration, not universal defaults. Keep them in the supplied job script;
replace the final direct engine invocation with the documented worker handoff.
No account, partition, hardware constraint, module version, rank count, or AMS
path from that example is hard-coded into ReaxKit.

## Stage 5 launch evidence and limits

- AMS mappings stage common files, fixed-width control overrides and iopt 4/7.
  CMA restart is copied explicitly; version-specific progress mapping is still
  required. No licensed executable was launched here.
- Legacy ffopt reads units 3 (geometry), 4 (field), 20 (iopt), 21 (params),
  22/23 (couplings), and 35 (random-file input), plus named control/trainset.in.
  Native launch supports unlinked parameters only, supplies empty coupling
  inputs, requires random_file, and rejects near-zero bounds that native code
  widens. The native fourth-column step clamp is disclosed. No build was run.
- JAX-SOPPI and JAX-ReaxFF map inspected CLI arguments with separate interpreters.
  Shared preflight rejects unsupported/incomplete training sections, RMSG-only
  entries, invalid weights and missing geometry references, not a full engine
  validation. Validation files and auxiliary inputs are staged/checksummed.
- OGolem invokes its separately installed bridge in the allocation before Java.
  Its custom evaluator command, timeout, backend and include-initial option
  are explicit. Generated bridge example Slurm files are never submitted.
- PIBO native support is only its existing cobalt/LAMMPS driver, with explicit
  subset acknowledgement, budget and seed mapping. MoS2-native mode fails because
  its own dataset/configuration/working-file contract differs. No objective or
  engine is substituted.
- INDEEDopt and EZFF retain explicit external-runner support. Their exposed
  settings are runner contracts, not a claim that unmodified native scripts
  accept run-config.json. External runner development remains in their projects.
- Native launch plans, auxiliary input hashes, worker commands, timeout state
  and allocation variables are recorded. Source/fixture validation is distinct
  from Linux process validation and actual Roar scientific smoke tests.

## External references

### GloMPO (2026-11-08)

- Upstream `mfgustavo/glompo` ac76da2, GPL-3.0, maintained separately in
  `ali-m-dinani/glompo`. No upstream implementation is copied into ReaxKit.
- Inspected real local CMA/sphere run: 12 evaluations; HDF5 iter_hist columns
  call_id/x/fx, YAML summary, trajectory PNG. Exporter runs in GloMPO's environment.
- Hash-bound JSONL conversion preserves per-optimizer candidate identities and
  parameter-space/profile metadata. No HDF5 optimizer objects are deserialized.
- Initial external runner uses the upstream AMS/ParAMS ReaxFF SSE interface,
  one CMA worker, strict classic-input conversion and parameter-only best field.
- ReaxFF execution, SCM API compatibility and Roar acceptance remain unverified.
  No parity predictions, reproducible RNG forwarding or exact restart claimed.
  See `docs/GLOMPO.md` for the prerequisite gate and subsequent submission path.

### Reference links

- [SCM MCFF documentation](https://www.scm.com/doc/OldReaxFF/MCFFOptimizer.html)
- [SCM CMA-ES documentation](https://www.scm.com/doc/OldReaxFF/CMAESFFOptimizer.html)
- [Roar batch jobs](https://docs.icds.psu.edu/running-jobs/batch-jobs/)
- [GNU separate-program guidance](https://www.gnu.org/licenses/gpl-faq.en.html#MereAggregation)
