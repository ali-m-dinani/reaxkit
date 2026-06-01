# Engine API

This section documents the engine layer in ReaxKit: adapters, file handlers, and
input generators used by AMS, LAMMPS, and ReaxFF flows.

## Structure

Engine docs are organized into:

- `base` (shared engine abstractions)
- `ams` (AMS adapter + RKF handlers)
- `lammps` (LAMMPS adapter + dump/log handlers)
- `common` (engine-agnostic IO and generators)
- `reaxff` (ReaxFF adapters, adapter parts, generators, and IO handlers)

## Module Index

### Core

- [base](base_doc.md)

### AMS

- [adapter](ams/adapter_doc.md)
- [rkf_handler](ams/rkf_handler_doc.md)

### LAMMPS

- [adapter](lammps/adapter_doc.md)
- [dump_handler](lammps/dump_handler_doc.md)
- [lammps_log_handler](lammps/lammps_log_handler_doc.md)

### Common

#### Generators

- [ffield_generator](common/generators/ffield_generator_doc.md)
- [structure_transformers](common/generators/structure_transformers_doc.md)
- [xyz_generator](common/generators/xyz_generator_doc.md)

#### IO

- [ffield_handler](common/io/ffield_handler_doc.md)
- [geo_io](common/io/geo_io_doc.md)

### ReaxFF

#### Adapter

- [adapter](reaxff/adapter_doc.md)

#### Adapter Parts

- [loaders_dynamics](reaxff/adapter_parts/loaders_dynamics_doc.md)
- [loaders_forcefield](reaxff/adapter_parts/loaders_forcefield_doc.md)
- [loaders_properties](reaxff/adapter_parts/loaders_properties_doc.md)

#### Generators

- [addmol_generator](reaxff/generators/addmol_generator_doc.md)
- [charges_generator](reaxff/generators/charges_generator_doc.md)
- [control_generator](reaxff/generators/control_generator_doc.md)
- [eregime_generator](reaxff/generators/eregime_generator_doc.md)
- [fort7_repair](reaxff/generators/fort7_repair_doc.md)
- [geo_generator](reaxff/generators/geo_generator_doc.md)
- [kopple2_generator](reaxff/generators/kopple2_generator_doc.md)
- [trainset_elastic_energy](reaxff/generators/trainset_elastic_energy_doc.md)
- [trainset_elastic_geometry](reaxff/generators/trainset_elastic_geometry_doc.md)
- [trainset_heatfo](reaxff/generators/trainset_heatfo_doc.md)
- [trainset_mp](reaxff/generators/trainset_mp_doc.md)
- [trainset_source_adapter](reaxff/generators/trainset_source_adapter_doc.md)
- [trainset_yaml](reaxff/generators/trainset_yaml_doc.md)
- [tregime_generator](reaxff/generators/tregime_generator_doc.md)
- [vregime_generator](reaxff/generators/vregime_generator_doc.md)
- [xmolout_generator](reaxff/generators/xmolout_generator_doc.md)

#### IO

- [base](reaxff/io/base_doc.md)
- [control_handler](reaxff/io/control_handler_doc.md)
- [eregime_handler](reaxff/io/eregime_handler_doc.md)
- [fort13_handler](reaxff/io/fort13_handler_doc.md)
- [fort57_handler](reaxff/io/fort57_handler_doc.md)
- [fort73_handler](reaxff/io/fort73_handler_doc.md)
- [fort74_handler](reaxff/io/fort74_handler_doc.md)
- [fort76_handler](reaxff/io/fort76_handler_doc.md)
- [fort78_handler](reaxff/io/fort78_handler_doc.md)
- [fort79_handler](reaxff/io/fort79_handler_doc.md)
- [fort7_handler](reaxff/io/fort7_handler_doc.md)
- [fort99_handler](reaxff/io/fort99_handler_doc.md)
- [geo_handler](reaxff/io/geo_handler_doc.md)
- [molfra_handler](reaxff/io/molfra_handler_doc.md)
- [params_handler](reaxff/io/params_handler_doc.md)
- [summary_handler](reaxff/io/summary_handler_doc.md)
- [trainset_handler](reaxff/io/trainset_handler_doc.md)
- [vels_handler](reaxff/io/vels_handler_doc.md)
- [xmolout_handler](reaxff/io/xmolout_handler_doc.md)
