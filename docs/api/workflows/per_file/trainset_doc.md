# Trainset Workflow

CLI namespace: `reaxkit trainset <task> [flags]`

Trainset workflow for ReaxKit.

This workflow provides tools for inspecting, categorizing, generating, and
exporting ReaxFF trainset files used in force-field training and validation.

It supports:
- Reading an existing trainset file and exporting individual sections
  (e.g. charge, heat of formation, geometry, cell parameters, energy)
  as CSV tables for inspection or downstream analysis.
- Extracting and listing unique group comments (categories) defined in
  trainset sections, with optional sorting and CSV export.
- Generating a template trainset settings YAML file populated with
  default values for elastic and structural targets.
- Generating complete elastic-energy trainsets and associated tables
  from either:
    • a user-provided YAML settings file, or
    • Materials Project data via a material ID and API key.
- Optionally generating and post-processing strained geometry files
  associated with elastic trainset construction.

The workflow is designed to bridge high-level training specifications
(YAML, Materials Project data) with concrete ReaxFF trainset inputs in a
reproducible, CLI-driven manner.

## Available tasks

### `category`

#### Examples

- `reaxkit trainset category --section all --export trainset_categories.csv`
- `reaxkit trainset category --section all --sort`
- `reaxkit trainset category --section energy --export energy_categories.csv`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to trainset/fort.99 file |
| `--section SECTION` | Section to analyze: all, charge, heatfo, geometry, cell_parameters, energy |
| `--export EXPORT` | Optional CSV file to write categories into (e.g. trainset_categories.csv) |
| `--sort` | Sort labels alphabetically (default: off) |

### `gen-settings`

#### Examples

- `reaxkit trainset gen-settings`
- `reaxkit trainset gen-settings --out reaxkit_outputs/trainset/trainset_settings.yaml`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--out OUT` | Output YAML filename/path (resolved under reaxkit_outputs/trainset/ if relative). |

### `generate`

YAML mode:<br>
  reaxkit trainset generate --yaml trainset_settings.yaml<br>
<br>
Materials Project mode:<br>
  reaxkit trainset generate --mp-id mp-661 --api-key YOUR_KEY

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--yaml YAML` | Path to an existing trainset_settings.yaml file. |
| `--mp-id MP_ID` | Materials Project material id (e.g., mp-661). |
| `--api-key API_KEY` | Materials Project API key (or set MP_API_KEY env var). |
| `--bulk-mode {voigt,reuss,vrh}` | Which MP bulk modulus to use (default: vrh). |
| `--out-yaml OUT_YAML` | Where to write the generated YAML in MP mode (resolved under outputs if relative). |
| `--structure-dir STRUCTURE_DIR` | Directory to write MP-downloaded structure files (default: next to out-yaml). |
| `--verbose` | Verbose MP fetching/logging. |
| `--out-dir OUT_DIR` | Directory to write elastic-energy trainset + tables (resolved under outputs if relative). |

### `get`

#### Examples

- `reaxkit trainset get --section all --export reaxkit_outputs/trainset`

#### Options

| Flag | Description |
|---|---|
| `-h, --help` | show this help message and exit |
| `--file FILE` | Path to trainset/fort.99 file |
| `--section SECTION` | Section to export: all, charge, heatfo, geometry, cell_parameters, energy |
| `--export EXPORT` | Directory to save CSVs into (default: trainset_analysis/) |
